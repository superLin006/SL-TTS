# tools/android

`probe_vits_streaming.py` / `build_streaming_assets.py` 是 LiveBabel App 现有 streaming 资源构建脚本的参考副本：

- 以 `/Mul_7_output_0` 为切点
- encoder：输入 `input,input_lengths,scales`，输出 latent
- decoder：输入 latent，输出 PCM
- frontend-ids-only：sherpa 前端 echo

这些脚本原属 App 仓库 `android/tools/`；在 SL-TTS 中仅用于生成/复现 streaming ONNX。构建发布资产时请使用 `scripts/build_stream_cut_assets.py`（针对 pinyin 模型）。
