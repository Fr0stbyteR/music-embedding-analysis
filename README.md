# Music Embedding Analysis

本地音乐分析与人工标注后端，基于 FastAPI、librosa 和可选的音频／文本嵌入模型。提供音高、节拍、和声、音色、结构分析，以及 CLAP / MuQ 语义检索、异步任务和人工审核标注。

当前版本 **0.2.0**，仓库包含后端服务与客户端接入协议。独立图形界面和 VS Code 音频编辑器不在本仓库内；启动后可使用浏览器 API 文档，或接入已有客户端。模型相似度不是概率，检测结果默认是建议，需人工审核。

## 一键启动（Windows / macOS）

下载或克隆仓库，解压到一个可写的目录。**首次运行需要联网**，无需预先安装 Python、uv、Homebrew 或模型。

Windows 双击根目录的 **`start.cmd`**；macOS 双击 **`start.command`**。两个入口统一命名为 `start`，Windows 的 PowerShell 实现是 `start.ps1`。脚本会：

1. 使用现有 uv，或从官方来源安装到项目的 `.tools/uv`（不需要管理员密码，也不修改 shell 配置）。
2. 自动准备 Python 3.11、`.venv` 和锁定版本的依赖。
3. 首次从 `.env.example` 创建 `.env`；已有配置不会被覆盖。
4. 安装所选模型依赖，默认从 ModelScope 下载音乐 CLAP（包含文本编码器与分词器）。
5. 在本机启动服务，终端显示访问地址和会话令牌。

首次下载包含数 GB 的模型／依赖，请预留空间并等待终端出现 `Application startup complete`。后续启动复用环境和缓存。macOS 默认使用 CPU；Apple Silicon 和 Intel 的模型依赖使用兼容版本，不自动启用 MPS。

如果 ZIP 解压后双击提示权限不足，在项目目录打开终端，执行一次：

```bash
chmod +x start.command
./start.command
```

也可以始终使用以下命令，无需可执行权限：

```bash
bash start.command
```

只使用 librosa 分析，不安装／加载嵌入模型：Windows 在 PowerShell 中执行 `./start.cmd --basic`，macOS 执行：

```bash
bash start.command --basic
```

基础模式首次仍需联网安装 Python 和基础依赖，但不下载模型，也不会修改 `.env`。不提供真实的语义识别结果。

服务就绪后打开 <http://127.0.0.1:49321/docs>。先展开 `GET /v1/health`，点击 **Try it out → Execute**；返回 200 表示服务已启动。其余 API 需点击 **Authorize**，填入终端 JSON 中的 `token`（只填令牌，不加 `Bearer`）。默认每次启动生成新令牌。停止服务按 **Ctrl+C**。

## 开发环境

普通用户使用上面的一键入口即可。需要手动管理环境的开发者，安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 后在项目目录中运行：

```powershell
uv sync --locked --python 3.11
uv run --no-sync python scripts/launch.py
```

基础模式追加 `--basic`。Windows 也可直接执行 `powershell -NoProfile -ExecutionPolicy Bypass -File ./start.ps1`，基础模式追加 `-Basic`。执行策略设置仅对当前进程生效，不修改系统策略。原 `start-clap.ps1` 已统一替换为 `start.ps1`。

开发与测试：

```bash
uv sync --locked --python 3.11 --extra dev
uv run --no-sync pytest
```

`uv.lock` 已纳入版本控制；更新依赖后运行 `uv lock` 并一并提交。CI 在 Windows 和 macOS 上测试基础 API 与启动流程，并检查 macOS CLAP 依赖导入。

## 配置与模型

环境变量优先于 `.env`。修改 `.env` 后重启服务。

| 配置 | 用途 |
| --- | --- |
| `MAB_PORT` | 默认 `49321`，端口被占用时改为其他可用端口 |
| `MAB_SESSION_TOKEN` | 可选固定令牌；不设置则启动时随机生成 |
| `MAB_DATA_ROOT` | 默认 `.music-annotation-data`，保存本地音频、缓存和数据库 |
| `MAB_AUTO_LOAD_PROVIDER` | 首次默认 `clap_music`；留空只启用基础分析 |
| `MAB_MODEL_DOWNLOAD_SOURCE` | `clap_music` 的下载源：默认 `modelscope`，可选 `huggingface` |
| `MAB_AUTO_LOAD_DEVICE` | 默认 `auto`，有 CUDA 时使用 CUDA，否则 CPU |
| `MAB_AUTO_LOAD_ALLOW_DOWNLOAD` | 首次默认 `true`；缓存完整后可设为 `false` |
| `MAB_AUTO_LOAD_CHECKPOINT_PATH` | 可选的本地模型路径 |
| `MAB_HUGGINGFACE_CACHE` | 可选的专用 Hugging Face 缓存目录 |
| `MAB_CORS_ORIGIN_REGEX` | 允许的浏览器来源，默认 localhost / 127.0.0.1 |

| 模式 | 配置值 | 准备方式 |
| --- | --- | --- |
| 音乐 CLAP（Transformers 格式） | `clap_music` | 默认，从 ModelScope 一键安装和下载 |
| 旧版 LAION-CLAP（原始 `.pt`） | `laion_clap_music_htsat_base` | 保留兼容，仍从 Hugging Face 下载权重与 RoBERTa |
| MuQ-MuLan 中英文语义相似度 | `muq_mulan_large` | 启动器自动安装 `muq` extra；允许下载或提供本地路径 |
| M2D 时序嵌入 | `m2d_clap_2025` | 启动器安装 `m2d` extra；需另行准备 vendor 源码和 checkpoint，见模型文档 |
| librosa 基础分析 | 空值 | 无需模型权重 |

### 大陆用户与下载源

新配置默认使用 ModelScope 的 [laion/larger_clap_music](https://modelscope.cn/models/laion/larger_clap_music)。它使用 Transformers 格式，包含音频／文本编码器和分词器；加载本地快照时禁止隐式访问 Hugging Face。只下载推理所需文件，不自动回退到另一下载源。

已有 `.env` 不会被覆盖。希望使用新默认模型时，在 `.env` 设置：

```dotenv
MAB_AUTO_LOAD_PROVIDER=clap_music
MAB_MODEL_DOWNLOAD_SOURCE=modelscope
MAB_AUTO_LOAD_ALLOW_DOWNLOAD=true
```

改用 Hugging Face：把 `MAB_MODEL_DOWNLOAD_SOURCE` 改为 `huggingface`，仍加载同一个 `laion/larger_clap_music` 模型。不同下载源的快照分开缓存在 `models/downloads/`，切换源可能重新下载。配置也可通过同名环境变量覆盖。

原始 `.pt` 的 `laion_clap_music_htsat_base` 与 `muq_mulan_large` 保留原有行为，仍可能访问 Hugging Face，包括嵌套编码器；此下载源选项不改变它们。暂未确认与 MuQ 匹配的 ModelScope 镜像，不使用名称相似但未经验证的权重替代。新的 `clap_music` 使用独立 provider ID，预处理实现与旧版不同，旧版检测阈值需重新验证。

离线使用：成功下载后将 `MAB_AUTO_LOAD_ALLOW_DOWNLOAD=false`，`clap_music` 仅查所选源的本地缓存；也可用 `MAB_AUTO_LOAD_CHECKPOINT_PATH` 指向完整的 Transformers 模型目录（不接受原始 `.pt`）。旧版模型还需要缓存所有嵌套编码器。依赖、uv 和 Python 的安装仍可能访问 PyPI / GitHub，此选项只控制新音乐 CLAP 的模型下载源。

## API 使用流程

交互分析无需先建项目：

```text
POST /v1/interactive-assets
POST /v1/interactive-assets/{assetId}:describe
POST /v1/interactive-assets/{assetId}:librosa
```

上传请求体为原始音频字节，`X-File-Name` 是 URL 编码的文件名。`:describe` 接受 `startSeconds`、`endSeconds`、可选 `providerId` 和 `maximumResults`；`:librosa` 接受 `algorithm`、`options` 和 `cachePolicy`（`use` / `refresh`）。完整请求字段见 `/docs`。

项目标注流程：注册可信音频路径 → 编译检测计划 → 验证计划 → 小范围预览 → 运行 → 人工确认。预览保存证据但不创建标注；运行结果以 `suggested` / `model` 保存，不覆盖人工确认结果。模型加载、探测和检测任务通过 `/v1/jobs/{jobId}` 查询进度。

## 文档与目录

- [后端架构](docs/backend-architecture.md)：存储、异步任务与标注循环。
- [模型选择](docs/model-selection.md)与[已有模型测试](docs/model-smoke-test.md)：模型能力、限制与本地 CPU 测量。
- [检测编排](docs/v1-detection-orchestration.md)：检测计划、路由、证据与审核策略。
- [客户端接入](docs/frontend-bridge.md)：Webview / 扩展宿主边界。
- `src/music_annotation_backend/`：服务实现；`tests/`：测试。
- `spec/`：OpenAPI、项目／检测计划 schema、消息类型与模型适配接口。
- `examples/`：检测计划样例；`scripts/`：环境与模型启动辅助。
- [贡献指南](CONTRIBUTING.md)与[发布检查](docs/releasing.md)。

## 常见问题

- **下载失败**：检查访问 PyPI、GitHub、Hugging Face 的网络／代理，然后重新启动。可先用 `--basic` 验证基础功能。
- **端口被占用**：关闭已有服务，或修改 `.env` 的 `MAB_PORT`，然后使用新的 `/docs` 地址。
- **提示找不到模型**：已有 `.env` 不会自动更新；确认 provider、checkpoint 路径和下载开关。
- **API 返回 401**：使用本次启动 JSON 输出中的令牌重新 Authorize。健康检查无需令牌。
- **macOS 阻止打开下载文件**：根据系统提示在“隐私与安全性”中允许已确认来源的文件，或在终端使用 `bash start.command`；无需关闭系统安全保护。

## 许可证

本仓库代码使用 [MIT License](LICENSE)。第三方库、模型代码、模型权重和音频数据遵循各自许可证；本项目的 MIT 授权不替代这些条款。模型与用户音频不会随仓库发布。
