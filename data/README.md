# data

- `corpus_v2.jsonl` — 9,726 句自然对话训练文本（无音频）
- `long_sft_2000.jsonl` — 2,000 句长句 SFT 文本
- `pilot_corpus_200.jsonl` — 200 句 held-out 测试集
- `long_eval_50.jsonl` — 55 句长句 held-out
- `pron_sentences.jsonl` — 100 词发音扫描集

训练文本由 LLM 生成并过滤；ASR/人工发现的问题句已剔除。音频、教师权重、checkpoint 和第三方 lexicon 不在仓库内。
