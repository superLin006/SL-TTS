# SL-TTS

轻量中文 TTS 训练与端侧流式部署工程：Kokoro 教师 + Piper/VITS 学生 + App pinyin 前端。模型对外名 **SL-TTS-zh-v1**。

## 状态

- 语料：9,726 句自然对话（14.3h）+ 2,000 句长句 SFT；合计 11,725 句 / 约 20h 教师语音
- 教师：Kokoro v1.1-zh `zf_001`，使用 App lexicon 修正多音字后重新合成
- 学生：Piper/VITS medium，pinyin token 标签（对齐 App g2pW/pinyin 前端），从零训练
- 全量 pinyin 重训已完成：300 epoch（`Trainer.fit stopped: max_epochs=300 reached`），最终 checkpoint `epoch=299`；`epoch=224` 是中途验收快照。
- **最终发布候选：`SL-TTS-zh-v1` = ckpt299 + 多音词修正 drill SFT（10 epoch）+ 修正 lexicon**
  - 修正：还款/还了/还了书/还了钱/行行/头发/关系；又长又黑强制 cháng
  - 100 词 ASR 缺失 6（生僻词）；200 held-out CER 1.47%；55 长句 2.04%；18 超长 OOD 2.42%
  - 上一版 ckpt299（未修正）分别是 7 / 1.74% / 2.15% / 2.48%
- 当前最佳端侧形态：**streaming static Conv-only encoder + float decoder**（22.7MB）
  - encoder 只对 Conv 做静态 QDQ（QUInt8 激活 / QInt8 权重），attention/LayerNorm 等非 Conv 激活保持 float
  - K30：独立首音 372–419ms、RTF ~0.82、Total PSS ~203–204MB；并发 ASR suite 首音 464–603ms、underrun=0、取消恢复通过
  - 注意：教师音频不是发音金标准；P09“长”、P12“数”已确认教师读错、学生读对，教师坏样本需替换
- QAT（10 epoch）已验证：未超过 Conv-only PTQ，详见 `docs/EVALUATION.md`；不纳入发布

## 客观结果

100 词多音字/生僻词 ASR 扫描：

| 模型 | 缺失词数 | 备注 |
|---|---:|---|
| **SL-TTS-zh-v1 final（ckpt299 + polyfix SFT）** | **6** | **仅生僻词，当前发布候选** |
| SL-TTS-zh-v1 · ckpt224 static Conv-only | 6 | 仅生僻词 |
| SL-TTS-zh-v1 · ckpt299 static Conv-only | 7 | 多 1 个缱绻；长句/OOD 更好 |
| full pinyin224 float | 7 | 参差 + 6 个生僻词 |
| streaming float encoder + float decoder | 8 | 基本相同，另 1 个 ASR 误判 |
| streaming dynamic-QUInt8 encoder + float decoder | 6 | 仅生僻词，首音较慢 |
| streaming static-int8（全量激活）encoder + float decoder | 13 | 多错 5 个常用词 |
| streaming dynamic-int8 encoder + float decoder | 7 | 质量好但首音慢 |

K30 端侧 streaming 实测（4 threads）：

| 版本 | 资源 | 首音 | 26 字/49 字总耗时 | 备注 |
|---|---:|---:|---:|---|
| **static Conv-only · final SFT** | **~22.7MB** | **372–419ms**（含 1 次 1.2s 抖动） | **8.3s** | **PSS 203–204MB / HWM 277–299MB，发布候选** |
| static Conv-only · ckpt224 | ~22.7MB | 349–387ms | 9.0–9.1s | PSS 197–199MB / HWM 281–286MB |
| static Conv-only · ckpt299 | ~22.7MB | 371–375ms（含 558ms 抖动） | 8.3s | PSS 203–204MB / HWM 270–282MB |
| dynamic-QUInt8 encoder + float decoder | ~22MB | 631ms | 8.6s | PSS 190MB |
| float encoder + float decoder | ~63MB | 373ms / 565ms | 3.9s / 9.3s | 质量对照 |
| static-int8 encoder + float decoder | ~23MB | 246ms / 314ms | 3.7s / 8.3s | 首音最快，但 100 词多错 5 个 |
| dynamic-int8 encoder + float decoder | ~22MB | 845ms | 4.4s / 9.3s | 质量好，动态 ConvInteger 算力大 |
| full float | 61MB | ~0.94s | 0.9s / 1.7s | 吞吐最好，非流式 |
| full static int8 | 18.5MB | ~0.5s | ~1.0s | 长句后半段不稳 |
| full dynamic int8 | 19.3MB | ~4.4s | 4.4s / 8.0s | 保真但太慢 |
| encoder FP16 | ~35MB | — | — | K30 native ORT 不支持 Gelu FP16，未采用 |

## 环境

```bash
pip install -r requirements.txt
```

详见 [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)。

## 仓库内容

- `scripts/`：语料生成、pinyin 前端、教师合成、预处理、训练、导出、量化、QAT、评测脚本
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
