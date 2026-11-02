# data

- `corpus_v2.jsonl` — 9,726 句自然对话训练文本（无音频）
- `long_sft_2000.jsonl` — 2,000 句长句 SFT 文本
- `pilot_corpus_200.jsonl` — 200 句 held-out 测试集
- `long_eval_50.jsonl` — 55 句长句 held-out
- `pron_sentences.jsonl` — 100 词发音扫描集
- `calib_pinyin.jsonl` — 355 条 pinyin token IDs，静态量化校准集（pron100 + long55 + pilot200）
- `pinyin_metadata.csv` — 11,725 条 pinyin 训练文本（原 LJSpeech metadata，不含音频）
- `poly_drill.jsonl` — 127 条多音词修正 SFT 文本
- `poly_contexts.jsonl` — 46 条多音词上下文审计句（目标词 + 期望读音）

训练文本由 LLM 生成并过滤；ASR/人工发现的问题句已剔除。音频、教师权重、checkpoint 和第三方 lexicon 不在仓库内。
