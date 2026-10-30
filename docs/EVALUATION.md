# 评测

## 发音扫描

`data/pron_sentences.jsonl` 含 100 个词（多音字 + 生僻词）的自然句。流程：

1. `scripts/gen_pinyin_eval_inputs.py` 生成 pinyin token IDs
2. 用目标 checkpoint/ONNX 生成音频
3. `scripts/asr_pinyin_early.py` 或 Qwen3-ASR 转写
4. 比较目标词是否出现在转写中

注意：ASR 词面命中是筛查指标，不是最终人耳结论；同音字可能误报。

## 当前结果（pinyin224）

- full float：7 个缺失（参差 + 6 个生僻词）
- streaming float encoder + float decoder：8 个（含 1 个 ASR 误判）
- 长短句时长 MAE：长 0.094 / 短 0.232（v2 espeak 0.141/0.250）
- 长句 >600 音素时长比：0.928（espeak 0.864）

## static-int8 encoder 的缺失词

`encoder static QDQ int8（所有可量化激活） + decoder float` 共 13 个缺失：

- float 也缺的 6 个：参差、龃龉、踟蹰、迤逦、恣睢、忸怩
- static 额外引入的 7 个：记得、调换、种子、蹒跚、氤氲、缱绻、魑魅

扩大校准的实验：

- 355 句（pron100 + long55 + pilot200）：13 个缺失
- 加入 1,845 条训练句（共 2,200 条，覆盖长短句）：**18 个缺失，更差**
- 排除 `/dp`（duration predictor）后再量化：56 个缺失，明显破坏
- 结论：问题不是校准覆盖不足；全量激活量化会同时缩短输出时长（49 字样例 5.19s→5.02s，约 -3%），并引入常用词错误

## Conv-only 静态量化（当前推荐）

只对 `Conv` 做静态 QDQ（激活 QUInt8、权重 QInt8、per-channel MinMax），attention/LayerNorm 等非 Conv 激活保持 float：

- 100 词缺失：**6**（与 float/dynamic 相同的 6 个生僻词，无新增常用词错误）
- 49 字 latent 时长 5.17s（float 5.19s），不再出现全量静态量化的缩短
- 长句 ASR（55 句）：CER 2.33%，float encoder 2.23%，dynamic-QUInt8 2.23%
- K30：首音 387ms、encoder 237ms、RTF 0.86、Total PSS 197MB、VmHWM 286MB
- 体积：encoder 16.0MB + decoder 6.7MB ≈ 22.7MB

结论：全量静态量化的问题主要来自非 Conv 激活，而不是 Conv 权重/激活本身；Conv-only 在保持 23MB 体积和低首音的同时，把质量拉回 float/dynamic 水平。

## K30 性能/内存

设备：Redmi K30，Android 10，QA debug，4 threads，同一 49 字句。

| 版本 | 磁盘 | 首音 | RTF | VmHWM | Native Heap PSS | Total PSS | 100 词缺失 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原版 xiao_ya streaming | 21MB | 549ms | 0.90 | 283MB | 119MB | 198MB | 基准 |
| our float encoder + float decoder | 63MB | 634ms | 0.88 | 317MB | 159MB | 239MB | 8 |
| our static-int8（全量激活）encoder + float decoder | 23MB | 323ms | 0.83 | 294MB | 133MB | 212MB | 13 |
| our static Conv-only encoder + float decoder | 22.7MB | 387ms | 0.86 | 286MB | 128MB | 197MB | 6 |
| our dynamic-int8 (QInt8) encoder + float decoder | 22MB | 1590ms | 0.98 | 284MB | 124MB | 204MB | 7 |
| our dynamic-QUInt8 encoder + float decoder | 22MB | 631ms | 0.83 | 261MB | 123MB | 190MB | 6 |

## 为什么 INT8 动态量化慢

原版 xiao_ya 的 int8 archive 使用 **UINT8** ConvInteger 权重（dtype 2）；我们最初用 `QuantType.QInt8`（dtype 3）。ORT CPU 的 ConvInteger 对 UINT8 有优化，INT8 路径慢 3–6 倍。改 `QuantType.QUInt8` 后，K30 encoder 1.45s → 445ms，首音 1.59s → 631ms，质量仍为 6 个生僻词缺失。

## QAT 实验结果（未采用）

从 `epoch=224` 的 checkpoint 出发，对 `enc_p` 与 `dp` 的 Conv1d 插入 weight/activation fake-quant（QUInt8），lr 2e-5、batch 16、10 epoch；decoder 保持 float。分别导出 epoch 4 与最终 epoch 10 权重，再做 Conv-only 静态 QDQ：

| 权重 | 100 词缺失（epoch 4） | 100 词缺失（epoch 10） |
|---|---:|---:|
| QAT float | 7 | 7 |
| QAT static Conv-only（QInt8 权重） | 6 | 7 |
| QAT static Conv-only（QUInt8 权重） | 6 | 7 |
| QAT static 全量激活 | 12 | 16 |
| QAT dynamic QUInt8 | 6 | 6 |

- QAT 没有把 Conv-only 的 6 个生僻词继续往下压；epoch 10 反而多 1 个（缱绻），全量静态量化退化更明显
- 剩余 6 个（龃龉/觊觎/踟蹰/迤逦/恣睢/忸怩）在 float、dynamic、static Conv-only 上完全重合，属于教师/ASR 词表与生僻词覆盖的公共下限，不是量化误差
- 结论：Conv-only PTQ 已经足够，QAT 不作为发布路径；保留脚本与结果供后续更换 backbone/更大模型时复验

## 建议

- 首版：**streaming static Conv-only encoder + float decoder**（22.7MB，首音 387ms，RTF 0.86，PSS 197MB，100 词 6 个生僻词缺失）
- 内存更紧且可接受 ~0.63s 首音：dynamic-QUInt8 encoder + float decoder（22MB，PSS 190MB）
- 不建议全量激活 static-int8：会多错常用词，并把输出时长缩短约 3%
- 需要更高吞吐时：full float32（61MB，RTF 0.18）
- 剩余 6 个生僻词应从前端词典/教师发音覆盖解决，而不是继续量化调参
