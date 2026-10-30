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

`encoder static QDQ int8 + decoder float` 共 13 个缺失：

- float 也缺的 6 个：参差、龃龉、踟蹰、迤逦、恣睢、忸怩
- static 额外引入的 7 个：记得、调换、种子、蹒跚、氤氲、缱绻、魑魅

扩大校准的实验：

- 355 句（pron100 + long55 + pilot200）：13 个缺失
- 加入 1,845 条训练句（共 2,200 条，覆盖长短句）：**18 个缺失，更差**
- 排除 `/dp`（duration predictor）后再量化：56 个缺失，明显破坏
- 结论：问题不是校准覆盖不足，而是 encoder 的 attention/dp 对 int8 精度敏感；单纯增加校准数据不能修复

## K30 性能/内存

设备：Redmi K30，Android 10，QA debug，4 threads，同一 49 字句。

| 版本 | 磁盘 | 首音 | RTF | VmHWM | Native Heap PSS | Total PSS | 100 词缺失 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原版 xiao_ya streaming | 21MB | 549ms | 0.90 | 283MB | 119MB | 198MB | 基准 |
| our float encoder + float decoder | 63MB | 634ms | 0.88 | 317MB | 159MB | 239MB | 8 |
| our static-int8 encoder + float decoder | 23MB | 323ms | 0.83 | 294MB | 133MB | 212MB | 13 |
| our dynamic-int8 encoder + float decoder | 22MB | 1590ms | 0.98 | 284MB | 124MB | 204MB | 7 |

## 建议

- 首版：**streaming float encoder + float decoder**（质量/首音平衡）
- 内存硬约束且可接受 ~1.5s 首音：dynamic-int8 encoder + float decoder
- static-int8 encoder 虽然 23MB/0.32s，但额外 7 个常用词错误，不建议上线
- 需要更高吞吐时：full float32（61MB，RTF 0.18）
