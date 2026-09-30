# 发布检查

## 仓库内容

- 包含源码、测试、`uv.lock`、协议、示例、README 和 MIT LICENSE。
- 排除 `.env`、`.venv`、`.tools`、`.uv-cache`、`models`、`vendor`、用户音频与 `.music-annotation-data`。`.gitignore` 不会自动移除已经跟踪的文件，发布前检查 `git ls-files`。
- 不把整个本地工作目录直接压缩上传；使用 GitHub 源码下载或 `git archive`，确保只包含已提交的发布文件。
- 确认第三方源码、模型权重及示例音频的独立授权。仓库代码的 MIT 不涵盖这些内容。

## 验证

```bash
uv lock --check
uv sync --locked --python 3.11 --extra dev
uv run --no-sync pytest
```

在没有预装 Python / uv 的干净 Windows 环境中双击 `start.cmd`，在干净 macOS checkout 中执行 `bash start.command`。先使用 `--basic` 验证安装和健康检查（应返回 200），再验证默认模型下载、语义分析、停止与再次启动。macOS 应覆盖 Apple Silicon 和 Intel 实机。默认 CLAP 验证需要网络和足够磁盘空间。

检查 `start.command` 的 Git 文件模式是 `100755`，行尾为 LF。Windows 维护者可执行：

```bash
git update-index --add --chmod=+x start.command
```

这会暂存该文件；检查暂存区后再提交。源码 ZIP 在某些解压工具中会丢失可执行权限，README 已提供 `bash start.command` 的入口。

## 发布

确认版本号（`pyproject.toml` 和 `src/music_annotation_backend/__init__.py`）一致，CI 通过，再创建版本 tag / GitHub Release。说明当前交付的是后端服务、首次联网下载要求、已有功能和已知限制。

不要将单台机器的模型加载结果当作 macOS 全平台验证。CI 中的无模型启动与依赖导入也不能替代实际模型下载和推理测试。
