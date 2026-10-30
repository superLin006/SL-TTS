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

## K30 性能

设备：Redmi K30，Android 10，QA debug，4 threads。

| 版本 | 首音 | 26字/49字总耗时 | 备注 |
|---|---:|---:|---|
| streaming float encoder+decoder | 373/565ms | 3.9/9.3s | 无 underrun |
| streaming static-int8 encoder+float decoder | 246/314ms | 3.7/8.3s | 质量下降 |
| streaming dynamic-int8 encoder+float decoder | 845ms | 4.4/9.3s | 质量好、算力大 |
| full float | ~0.94s | 0.9/1.7s | 吞吐最好 |
| full static int8 | ~0.5s | ~1.0s | 长句不稳 |
| full dynamic int8 | ~4.4s | 4.4/8.0s | 太慢 |

## 建议

- 首版：streaming float encoder + float decoder（质量/首音平衡）
- 需要更高吞吐时：full float32
- 需要更小体积时：先解决 encoder 静态量化误差，再考虑 static int8
