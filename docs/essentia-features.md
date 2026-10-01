# 原生 Essentia 分析模块

前端注册 29 个新模块，实际计算由原生 Essentia C++ 完成：Windows 使用独立
C++ worker，macOS Intel / Apple Silicon 使用官方 C++ Python wheel 的隔离 worker。
Python 负责解码、重采样、输入验证和磁盘缓存，不使用 WASM 或 librosa 替代算法。
VA 是另一个既有模块，需要模型权重；以下 DSP 特征不需要模型权重。

## 输出与对应界面

| 类型 | 模块 | 前端复用 |
| --- | --- | --- |
| 连续曲线（19） | RMS、Energy、Stevens Loudness、Zero-crossing rate、Spectral centroid / rolloff / flatness / crest / flux / entropy / complexity / spread / skewness / kurtosis、HFC、Dissonance、YIN FFT Pitch、Pitch confidence、Onset strength | LibrosaVectorModule → VectorImageProcessor，复用现有 Canvas 曲线 |
| 时间×特征（6） | Mel / Bark / ERB bands、MFCC、GFCC、HPCP | LibrosaMatrixModule → MatrixImageProcessor / MatrixWebGLRenderer，原有色图、颜色范围与透明度 |
| 点与区域（4） | Onsets、Silence regions、Stable pitch regions、Key regions | LibrosaMarkerModule → ModuleUsingMarker，原有选区、改名、移动与删除 |

这些公共类名保留 Librosa 前缀以减少旧模块迁移；每个新模块有独立的
`essentia.*` ID，分析请求携带 `engine: "essentia"`，不会复用 librosa 的结果缓存。

## 运行

在后端根目录运行普通 `start`，首次自动准备、校验并试算全部 29 项：

```powershell
./start.cmd
```

macOS：`bash start.command`。默认也准备 VA 权重并试算真实模型；`--basic`
仅准备 DSP（不下载 CLAP/VA 权重）。Windows 自动构建不依赖 Visual Studio，
依赖存于后端 `.tools/`、`vendor/`；macOS wheel 存于后端 `.venv/`。
现有 `.env` 不会被覆盖，后续启动复用已验证的环境。设置与第三方授权见
[原生说明](../native/essentia/README.md)。Windows 的 worker 与 TensorFlow DLL
仍需同时存在，即使只使用基础 DSP。手动 MSVC 构建脚本作为开发者选项保留。

接口需要 Bearer token：

```text
GET  /v1/essentia/capabilities
POST /v1/interactive-assets/{assetId}:essentia
```

示例请求：

```json
{"algorithm":"mfcc","options":{"sampleRate":44100,"frameLength":2048,"hopLength":512,"bands":40,"coefficients":20},"cachePolicy":"use"}
```

参数使用 camelCase，未知参数会被拒绝。FFT 大小必须为 256–8192 的二次幂；
hopLength 不能超过 frameLength；coefficients 不能超过 bands。
最多三小时、50 万帧、200 万矩阵单元。较长音频可增加 hopLength、减少频带数。

## 数值含义与限制

- 输出 vectors 为通道×时间；matrix 为时间×特征；点标记与区域时间单位为秒。
  前端映射到当前解码音频时间轴，Marker 使用整数采样索引。
- Mel / Bark / ERB 以本次分析最大频带能量为 0 dB，范围 −100–0 dB，
  **不是绝对 dBFS**。MFCC / GFCC 保留正负值；HPCP 为强度，12 音级从 A 开始。
- Loudness 是 Stevens 能量幂律指标，不是 LUFS；Spectral crest 是频谱峰值/均值，
  不是波形峰值/RMS；Spectral spread 是频率方差，单位 Hz²。
- Pitch 使用单音 YIN FFT，低置信度帧显示 0；稳定音高区域为量化、连续音高段，
  不是复调转录。调性区域为窗口平均 HPCP + Temperley 大小调模板，
  不是五声、教会调式识别，也不保证正确。两者界面明确标记“估计”。
- 静音区域由 RMS 阈值和最小时长推断；Onsets 用 Essentia 的检测函数和峰值选择。
  标记都可手动修改；仅明确重新分析或修改分析参数才重建结果。
- API 缓存包含引擎、源音频哈希、参数、worker / DLL 签名和适配器版本。
  前端 `.audio_toolkit` 文件夹保存结果与模块状态，继续支持重开与导入导出。

## 验证

集成测试在已安装 Windows worker 上通过真实 HTTP API 执行全部 29 项，
检查有限数值、矩阵维度、缓存及刷新、参数拒绝，并用含中间静音的 440 Hz
合成音频检查 RMS、音高和静音区间。无原生环境时跳过实际运行测试，不自动下载。
浏览器烟雾页面 `app/tests/essentia-editor.html` 使用隔离测试后端 49322，
验证 Vector、HPCP Matrix、Marker 以及分列/叠加切换。
这些是功能/数值约束测试，不代表复杂音乐的音乐学识别准确率评估。
