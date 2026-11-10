# ThinkingEars-ASR

基于 **578,895 条词库**的中文 ASR 增强插件。支持拼音热词纠错、已知词保护、停顿断句和语音指标提取，可接在已有语音识别系统后使用，无需重训声学模型。

**仅限非商业用途。商业使用须事先取得书面授权。** 详见 [LICENSE](LICENSE)。

从 [AI开放剧本模拟器 V5](https://github.com/superalp1985/AI-Open-Scenario-Simulator) 的实际 ASR 实现中独立整理，原作者：王秉钦（王大粘）、金皓若（老金）。

## 安装

需要 Python 3.10 或更新版本。

```bash
git clone https://github.com/superalp1985/ThinkingEars-ASR.git
cd ThinkingEars-ASR
python -m pip install .
```

需要直接使用本地 Paraformer 模型或麦克风时，安装音频依赖：

```bash
python -m pip install ".[audio]"
```

文字增强只需要 `numpy` 和 `pypinyin`，使用时不会加载声学模型，也不会访问网络。

## 增强已有 ASR 的文本

```python
from thinkingears_asr import ASREnhancer

enhancer = ASREnhancer(hotwords=["共振峰", "梅尔频谱"])

# 将其他 ASR 的输出传入；默认保留空格及已有标点。
text = enhancer.enhance("测量供振峰")
print(text)  # 测量共振峰
print(enhancer.lexicon_count)  # 578895
```

默认启用全部词库，用于保护已知词。纠错目标来自显式配置的热词。
例如“继承”和“集成”同音时，词库中的“继承”会被保留。
多个候选热词完全同音时，插件保留原文，避免任意选择一个结果。

调用 `set_hotwords()` 可在对话切换领域时替换热词，无需重新加载模型：

```python
enhancer.set_hotwords(["集成电路", "晶体管"])
enhancer.set_hotwords([])  # 清空热词
```

热词也可以来自 JSON 文件，支持词语数组或分组对象：

```json
{
  "语音技术": ["共振峰", "梅尔频谱"],
  "芯片": ["集成电路", "晶体管"]
}
```

```python
enhancer = ASREnhancer(hotwords="my_hotwords.json")
```

自动音似匹配针对 2 至 10 字的中文热词。按不带声调的拼音精确匹配，属于规则式纠错。

## 自定义词库和纠错规则

```python
enhancer = ASREnhancer(
    hotwords=["集成电路"],
    lexicon_path="my_words.txt",
    corrections={"因该": "应该"},
)
```

自定义词库会替代内置保护词库。支持每行一词的 `.txt`、以词为键的 `.json` 对象、
JSON 词语数组，以及这些格式的 `.gz` 压缩文件。
也可以设置 `use_builtin_lexicon=False` 关闭内置词库。

显式 `corrections` 优先执行，会覆盖原文，即使原文属于已知词。
原项目的游戏热词和替换规则放在 `examples/`，按需加载。

## 停顿断句

```python
enhancer = ASREnhancer()
print(enhancer.enhance("你好 今天过得怎么样", final=True))
```

`final=True` 启用规则标点，两段中文之间的空格表示停顿边界，可生成逗号、句号和问号。
已有标点、数字小数点、英文单词之间的空格会被保留，适合带停顿边界的 ASR 文本。
普通文本增强默认不启用此模式。

## 测量语音指标

```python
import numpy as np

# audio 为 16 kHz、单声道、浮点波形，通常取值在 -1 到 1。
audio = np.zeros(16000, dtype=np.float32)
metrics = enhancer.analyze(audio, text="你好")
print(metrics["duration"])  # {"raw": 1.0, "score": 2}
```

当前输出 8 个指标：平均音量、峰值音量、语速、低能量帧频率、低能量帧占比、
能量变化、句末基频变化、时长。每项包含 `raw` 和 1 至 10 的 `score`。
音量使用线性幅值；低能量帧指标为停顿近似值，详见 [实现说明](docs/architecture.md)。

## 本地语音识别

本地后端使用 `sherpa-onnx` 的离线 Paraformer。准备模型目录：

```text
models/sherpa-onnx-paraformer-zh-2023-09-14/
  model.int8.onnx
  tokens.txt
```

模型可从 [sherpa-onnx 官方 Paraformer 模型文档](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/paraformer-models.html) 获取，
模型文件遵守其原始许可，需单独下载。

```python
enhancer = ASREnhancer(
    model_dir="models/sherpa-onnx-paraformer-zh-2023-09-14",
    hotwords=["共振峰"],
)
enhancer.initialize()
result = enhancer.transcribe(audio)  # 16 kHz 单声道浮点数组
print(result["text"])
```

`transcribe()` 会对低于当前能量阈值的音频返回空文本，减少静音误识别。
可通过 `set_threshold_level(1)` 至 `set_threshold_level(10)` 调整阈值。

麦克风识别：

```bash
thinkingears listen --model-dir models/sherpa-onnx-paraformer-zh-2023-09-14 --seconds 10
```

Python 中也可使用 `initialize()`、`start(seconds=10)`、`snapshot()`、`stop()`。
`snapshot()["text"]` 保留标点，`displayText` 提供以空格分隔的显示版本。

环境变量 `THINKINGEARS_INPUT_DEVICE` 可指定输入设备编号或名称片段。
Windows 模型路径含中文时，可通过 `THINKINGEARS_CACHE` 指定可写的纯英文缓存目录。

## 命令行

```bash
thinkingears info
thinkingears enhance "测量供振峰" --hotwords my_hotwords.json
thinkingears enhance "你好 今天过得怎么样" --final
python -m thinkingears_asr enhance "hello world"
```

## 词库文件

| 文件 | 内容 |
| --- | --- |
| `src/thinkingears_asr/data/lexicon.json.gz` | 随安装包提供的全部 578,895 个词条，用于已知词保护 |
| `resources/master_dict.json.gz` | 原始完整标注词库的无损压缩，保留拼音、频率、分类等字段 |
| `resources/manifest.json` | 原始文件及运行词库的 SHA-256、词条数、来源说明 |

运行时只加载词条集合，避免每次解析完整标注。压缩仅改变存储方式，保留全部词条。

## 验证

```bash
python -m pip install ".[dev]"
python -m unittest discover -s tests -v
python -m build
```

自动检查覆盖热词纠错、词库保护、同音歧义、英文空格、热词切换、无模型调用和安装包构建。

## 技术范围与许可

当前实现是 ASR 文本后处理和本地 Paraformer 集成。词库保护与拼音热词匹配可直接复用。
白皮书中的完整解码器替换、12 维声学测量及 FAME/BVC 属于架构设计，当前发布不宣称已实现这些能力。
识别准确率、纠错收益和延迟需在实际模型、硬件及目标语料上测量。

本项目采用自定义**非商业许可**，源码可见，非商业修改和再分发须保留署名及相同限制。
第三方依赖、声学模型和第三方数据各自遵守原始许可，详见 [NOTICE](NOTICE) 与 [LICENSE](LICENSE)。
