# SL-TTS

轻量中文 TTS 训练与端侧流式部署工程：Kokoro 教师 + Piper/VITS 学生 + App pinyin 前端。

## 状态

- 语料：9,726 句自然对话（14.3h）+ 2,000 句长句 SFT；合计 11,725 句 / 约 20h 教师语音
- 教师：Kokoro v1.1-zh `zf_001`，使用 App lexicon 修正多音字后重新合成
- 学生：Piper/VITS medium，pinyin token 标签（对齐 App g2pW/pinyin 前端），从零训练
- 当前验收 checkpoint：`epoch=224`（300 epoch 训练中）
- 首发端侧形态：**streaming encoder float + decoder float**，23MB 级静态版本保留为实验

## 客观结果

100 词多音字/生僻词 ASR 扫描：

| 模型 | 缺失词数 | 备注 |
|---|---:|---|
| full pinyin224 float | 7 | 参差 + 6 个生僻词 |
| streaming float encoder + float decoder | 8 | 基本相同，另 1 个 ASR 误判 |
| streaming static-int8 encoder + float decoder | 13 | encoder 静态量化多错 5 个常用词 |
| streaming dynamic-int8 encoder + float decoder | 7 | 质量好但首音慢 |

K30 端侧 streaming 实测（4 threads）：

| 版本 | 资源 | 首音 | 26 字/49 字总耗时 | 备注 |
|---|---:|---:|---:|---|
| float encoder + float decoder | ~63MB | 373ms / 565ms | 3.9s / 9.3s | 无 underrun，推荐首版基线 |
| static-int8 encoder + float decoder | ~23MB | 246ms / 314ms | 3.7s / 8.3s | 首音最快，但 100 词多错 5 个 |
| dynamic-int8 encoder + float decoder | ~22MB | 845ms | 4.4s / 9.3s | 质量好，动态 ConvInteger 算力大 |
| full float | 61MB | ~0.94s | 0.9s / 1.7s | 吞吐最好，非流式 |
| full static int8 | 18.5MB | ~0.5s | ~1.0s | 长句后半段不稳 |
| full dynamic int8 | 19.3MB | ~4.4s | 4.4s / 8.0s | 保真但太慢 |
| encoder FP16 | ~35MB | — | — | K30 native ORT 不支持 Gelu FP16，未采用 |

## 仓库内容

- `scripts/`：语料生成、pinyin 前端、教师合成、预处理、训练、导出、评测脚本
- `data/`：训练/评测文本 JSONL（不含音频）
- `configs/`：补充短语与 token 表
- `tools/android/`：Piper 全模型切分 streaming 资源的参考脚本
- `docs/`：模型卡、训练、Android、评测、数据与 license 说明

**不包含**：模型权重、音频、checkpoint、第三方 lexicon、API key。发布资产单独构建，见 `docs/ANDROID.md`。

## 快速导航

- [模型卡](docs/MODEL_CARD.md)
- [训练流程](docs/TRAINING.md)
- [Android streaming](docs/ANDROID.md)
- [评测方法](docs/EVALUATION.md)
- [数据说明](docs/DATA.md)
- [第三方与 license](NOTICE.md)

## License

代码默认仅限 LiveBabel 内部使用，除非另行书面授权。第三方组件与数据条款见 `NOTICE.md`；发布模型/lexicon 前必须单独确认授权。
