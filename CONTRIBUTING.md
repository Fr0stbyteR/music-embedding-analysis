# Contributing

欢迎提交问题和 Pull Request。请附操作系统、Python 版本、provider、复现步骤，以及去除令牌和私人路径后的日志。模型问题需附 checkpoint 名称与设备信息，不要上传私人音频、模型权重或 `.env`。

## Development

```bash
uv sync --locked --python 3.11 --extra dev
uv run --no-sync pytest
```

按需增加 `--extra laion`、`--extra muq` 或 `--extra m2d`。基础测试不依赖真实模型，也不下载权重。新模型适配器遵循 `spec/model-provider.md`；API 或消息协议变更应同步更新 `spec/` 与相关文档。

依赖变更后运行 `uv lock`，提交 `pyproject.toml` 和 `uv.lock`。保持 macOS 的兼容依赖约束。启动脚本使用 LF 换行，`start.command` 在 Git 中保留可执行位。

## Pull requests

说明具体问题、改后的行为和验证结果。添加能覆盖行为的测试；分析结果必须保留来源，不将相似度描述为概率，也不能自动覆盖人工确认标注。

贡献按本仓库 MIT 许可证提供。引用第三方代码时保留其署名与许可证。
