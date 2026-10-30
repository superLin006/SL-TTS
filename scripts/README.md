# scripts

按流水线顺序：

## 语料
- `matrix.py`：场景/数字/多音字覆盖矩阵
- `gen_dialogues.py`：LLM 双人对话生成（DeepSeek OpenAI 兼容）
- `build_corpus.py`：清洗、去重、与 held-out 查重
- `gen_long_sft.py` / `build_long_sft.py`：长句 SFT 生成与筛选
- `gen_pron_sentences.py`：100 词发音扫描集

## 前端
- `frontend_zh.py`：中文数字/单位/英文术语归一化 + 多音字短语词典
- `pinyin_frontend.py`：App lexicon pinyin token 输出（longest match）

## 教师
- `build_lexicon_phrases.py`：把 App lexicon 短语转成 pypinyin 短语表
- `gen_teacher_corpus.py`：Kokoro 教师合成（加载 jieba/pypinyin 修正）
- `gen_teacher_long.py`：早期长句教师合成（参考）

## 预处理/训练
- `preprocess_pinyin.py`：pinyin 标签 Piper 预处理（覆盖 config token map）
- `sft_train.py`：从 checkpoint 热启动 SFT（实验）
- `train_pinyin.sh`：参考训练命令（路径需按环境修改）

## 导出/量化
- `build_stream_cut_assets.py`：从 full ONNX 切 `/Mul_7_output_0` streaming 资源，并生成
  Conv-only 静态 QDQ（`encoder.onnx`，推荐）、全量激活静态（`encoder_static_full.onnx`）和 dynamic QUInt8
- `qat_train.py`：Conv1d fake-quant QAT（实验，未超过 Conv-only PTQ）
- `qat_clean.py`：把 Lightning QAT checkpoint 去掉 fake-quant 包装后导出 clean checkpoint
- `quantize_encoder.py` / `quantize_static.py`：早期单模型量化实验
- `soup_ckpt.py`：espeak 版本权重插值（实验保留）

## 评测
- `poly_scan.py`：教师 G2P vs App lexicon 多音词扫描（输入 `data/poly_contexts.jsonl`）
- `asr_cer.py` / `asr_dir_pron.py`：目录级 ASR 转写与 CER/目标词命中统计
- `pron_table.py`：pinyin 与 espeak 音素对照
- `gen_pinyin_eval_inputs.py`：pinyin eval 输入
- `test_onnx_pinyin.py`：ONNX float/int8 推理自检
- `asr_pinyin_early.py` / `asr_sensevoice.py`：ASR 回读
- `validate_pinyin.py` / `validate_g2p_all.py`：前端覆盖与崩溃检查
- `gen_eval_200.py` / `gen_eval_long.py` / `gen_sft_input.py` / `build_sft_dataset.py`：历史评测/数据打包脚本

脚本中的 `/mnt/data`、`/mnt/d` 是研究服务器/本机路径，迁移时按 `configs/paths.example.env` 修改。
