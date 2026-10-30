# Notice / third-party

本仓库代码默认仅限 LiveBabel 内部使用。以下第三方组件/数据有各自条款：

- **Piper / piper_train**：MIT；训练代码来自 piper 项目。
- **Kokoro v1.1-zh**：Apache-2.0；教师权重不随仓库分发。
- **misaki**：MIT；教师 G2P。
- **pypinyin**：MIT；拼音。
- **jieba**：MIT；分词。
- **ONNX Runtime**：MIT；导出/量化运行时。
- **zh_CN-xiao_ya-medium lexicon/tokens**：来自 BZNSYP 相关 Piper 模型，**非商用条款**；本仓库不包含该文件。训练标签和 App 前端依赖该词表，商用/再分发前必须确认授权或替换为自建词表。
- **LLM 生成语料**：由 DeepSeek API 生成，仓库仅保存文本；发布前请按你的服务条款确认可再分发性。
- **教师合成音频**：由 Kokoro 生成，不再分发。

模型/lexicon 的 license 与代码 license 分开；不要把 `lexicon.txt`、`tokens.txt`、ONNX、音频提交到公开仓库。
