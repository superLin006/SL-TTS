# 数据

## 训练文本

- `data/corpus_v2.jsonl`：9,726 句自然对话，77 场景、33 类数字、25 组多音字；avg 23.2 字
- `data/long_sft_2000.jsonl`：2,000 句自然长句（35–80 字），平均 44.6 字
- 合计 11,725 句用于教师合成；音频不入库

## 评测文本

- `data/pilot_corpus_200.jsonl`：held-out 200 句，不参与训练
- `data/long_eval_50.jsonl`：55 句长句 held-out（37–62 字）
- `data/pron_sentences.jsonl`：100 个多音字/生僻词句，用于发音扫描

## 生成方式

- v2 对话：DeepSeek LLM，50 批 × 50 句；`gen_dialogues.py` + `matrix.py`，带类别、数字、多音字、英文覆盖校验
- 长句 SFT：50 批 × 50 句，`gen_long_sft.py`；含数字/多音字/中英混排专项
- 所有句子经过去重、与 held-out 集查重、长度/标点/隐私过滤
- 教师音频：Kokoro v1.1-zh zf_001；用 App lexicon 47k 短语修正多音字后合成

## 不在仓库

- 音频、checkpoint、ONNX 权重
- `lexicon.txt` / `tokens.txt`（源自 xiao_ya/BZNSYP，见 NOTICE）
- API key、训练服务器凭据
