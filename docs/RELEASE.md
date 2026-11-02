# SL-TTS-zh-v1 release

最终模型：ckpt299 + 127 条多音词修正 drill + 400 replay，10 epoch SFT；static Conv-only encoder + float decoder。

## Android streaming pack

`SL-TTS-zh-v1/`（22.7MB 模型）：

- encoder.onnx SHA-256 `3fad402554bfdd003729e65ab45ceae2117fc659f6bf9b4f20c6492e839efa72`
- decoder-float.onnx SHA-256 `b46fb1fd97caa5e3ebdd220bbd8f38289f210c7f3d2cb515f9f7a524b25b17c0`
- frontend-ids-only.onnx SHA-256 `ace1e835aa525c1b99a238f1aa892feb5fcd360754c71d836f20a807ffab5e99`
- lexicon.txt SHA-256 `51fbaa276d84770c05df5797ec5cff9b58e88f597fa694946978d90794fa2519`

## iOS full voice

- `piper/zh/sltts-zh-v1.onnx`，63,375,662 B，SHA-256 `614ccf77c3ef056d15433b96f5885904fc7c627e57b85f09f793ee4274fbc87d`
- 同一份修正 lexicon；旧 xiao_ya 完整模型不再使用。

## 指标

100 词缺失 6；200 held-out CER 1.47%；55 长句 2.04%；18 超长 OOD 2.42%；K30 首音 372–419ms、PSS 203–204MB。
