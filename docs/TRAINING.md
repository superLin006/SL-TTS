# 训练流程

## 1. 语料

```bash
export DEEPSEEK_API_KEY=...
python scripts/gen_long_sft.py --workers 8      # 生成 50 批自然长句
python scripts/build_long_sft.py                # 去重/查重 → long_sft_2000.jsonl
```

v2 自然对话语料已包含在 `data/corpus_v2.jsonl`；字段：`id, text, kind, scene, chars, split`。

## 2. 教师合成

需要 Kokoro v1.1-zh 权重与 `zf_001` voice：

```bash
python scripts/build_lexicon_phrases.py         # 解析 App lexicon 短语表
python scripts/gen_teacher_corpus.py \
  --corpus data/corpus_v2.jsonl \
  --out  $SLTTS_ROOT/train/teacher_v2
```

`gen_teacher_corpus.py` 在 KPipeline 创建后会加载 47k 短语到 jieba + pypinyin，修正教师读音。补充短语见 `configs/custom_phrases.json`。

## 3. pinyin 标签与预处理

`scripts/pinyin_frontend.py` 使用 App 的 `tokens.txt`/`lexicon.txt`（不随仓库分发）做最长匹配，输出与 App g2pW/pinyin 前端一致的 token IDs。

```bash
python scripts/preprocess_pinyin.py \
  --input-dir  $SLTTS_ROOT/train/pinyin/input \
  --output-dir $SLTTS_ROOT/train/pinyin/piper_dataset \
  --language cmn --sample-rate 22050 \
  --dataset-format ljspeech --single-speaker --max-workers 16
```

脚本会把 `config.json` 的 `phoneme_id_map` 覆盖为 App pinyin token 表，`num_symbols=73`。

## 4. 训练

```bash
cd $PIPER_SRC   # piper/src/python
python -m piper_train \
  --dataset-dir $SLTTS_ROOT/train/pinyin/piper_dataset \
  --accelerator gpu --devices 1 \
  --batch-size 16 --num-workers 16 --max-phoneme-ids 600 \
  --quality medium --max_epochs 300 --checkpoint-epochs 25 \
  --validation-split 0.02 --check_val_every_n_epoch 25 \
  --precision 32 --num_sanity_val_steps 0 \
  --default_root_dir $SLTTS_ROOT/train/runs/medium_pinyin
```

`num-workers` 需要在 Piper 的 `__main__.py` 注册；仓库脚本 `scripts/train_pinyin.sh` 为参考。

## 5. 评测

```bash
python scripts/gen_pron_sentences.py            # 100 词发音集（若 data 已有可跳过）
python scripts/gen_pinyin_eval_inputs.py
python scripts/pron_table.py                    # pinyin/espeak 对照
python scripts/test_onnx_pinyin.py              # ONNX float/int8 自检
```

ASR 回读可使用 Qwen3-ASR 或 SenseVoice；`scripts/asr_pinyin_early.py` 为参考实现。
