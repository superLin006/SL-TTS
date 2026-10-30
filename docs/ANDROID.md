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
python scripts/build_stream_cut_assets.py
```

输出 `encoder_float.onnx`、`decoder-float.onnx`、`frontend-ids-only.onnx`（含 `sample_rate=22050` 等 metadata）。

## 量化结论

- **推荐首版**：encoder float + decoder float，约 63MB；K30 首音 373ms/565ms，无 underrun。
- encoder static-int8 + decoder float：约 23MB，首音 246ms，但 100 词多错 5 个常用词，不采用。
- encoder dynamic-int8 + decoder float：约 22MB，质量好；encoder 719ms，首音 845ms，不划算。
- encoder FP16：K30 native ORT 不支持 `com.microsoft.Gelu` FP16，加载失败。
- full static int8：长句后半段不稳；full dynamic int8：保真但太慢。

发布资源 pack 包含 `manifest.json`（sample_rate、文件 sha256）与 `MODEL_CARD.md`；不要把 63MB 权重提交进 Git。
