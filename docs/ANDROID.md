# Android streaming

## 现有 App 契约

原生 `PiperStreamingTts` 读取目录内：

- `encoder.onnx`：输入 `input,input_lengths,scales`，输出中间 latent `/Mul_7_output_0`
- `decoder-float.onnx`：输入 `/Mul_7_output_0`，输出 `output`
- `frontend-ids-only.onnx`：sherpa 前端 echo，返回 token IDs
- `tokens.txt` / `lexicon.txt` / `phone.fst,date.fst,number.fst`

调试入口（QA debug 包）：

```bash
adb shell am start -n com.livebabel.android.qa/com.livebabel.android.MainActivity \
  --es test_tts_text "我明天去图书馆还书。" \
  --es test_tts_language "简体中文" \
  --es test_tts_model_path "/sdcard/Android/data/com.livebabel.android.qa/files/sltts_stream/frontend-ids-only.onnx" \
  --ez test_tts_streaming true --ei test_tts_threads 4
```

## 从 full ONNX 构建 streaming

`tools/android/probe_vits_streaming.py` 的规则：以 `/Mul_7_output_0` 为切点，切出 encoder 与 decoder。`scripts/build_stream_cut_assets.py` 是对 pinyin224 的实现：

```bash
python scripts/build_stream_cut_assets.py \
  --src export/pinyin224.onnx \
  --out export/stream_cut \
  --calib data/calib_pinyin.jsonl
```

输出 `encoder_float.onnx`、`decoder-float.onnx`、`frontend-ids-only.onnx`（含 `sample_rate=22050` 等 metadata），以及 `encoder.onnx`（Conv-only 静态 QDQ，推荐）、`encoder_static_full.onnx`（对照）和 `encoder_dynamic_u8.onnx`（备选）。

## 量化结论

- **推荐首版**：模型名 **SL-TTS-zh-v1**；encoder **static Conv-only** + decoder float，约 22.7MB；K30 首音 349–387ms、RTF 0.86、PSS 197–204MB，100 词 6–7 个生僻词缺失。ckpt224/ckpt299 两个候选见 `release/SL-TTS-zh-v1-ckpt224|ckpt299`，试听后定版。
  - 只对 Conv 做 static QDQ（激活 QUInt8、权重 QInt8、per-channel MinMax），attention/LayerNorm 等非 Conv 激活保持 float。
  - 全量激活 static-int8 会多错常用词并把输出时长缩短约 3%，不要用。
- encoder dynamic-**QUInt8** + decoder float：约 22MB，同为 6 个生僻词缺失；首音 0.63s、RTF 0.83、PSS 190MB，是低内存备选。
- encoder dynamic-**QInt8** + decoder float：同样 22MB，但 ORT CPU ConvInteger 的 INT8 内核慢 3–6 倍（首音 1.59s），不要用 QInt8。
- encoder FP16：K30 native ORT 不支持 `com.microsoft.Gelu` FP16，加载失败。
- full static int8：长句后半段不稳；full dynamic int8：保真但太慢。
- QAT：10 epoch 未超过 Conv-only PTQ，详见 `docs/EVALUATION.md`。

发布资源 pack 包含 `manifest.json`（name、checkpoint、sample_rate、文件 sha256）与 `MODEL_CARD.md`；当前候选 pack 为 `release/SL-TTS-zh-v1-ckpt224` 与 `release/SL-TTS-zh-v1-ckpt299`。不要把模型权重提交进 Git。
