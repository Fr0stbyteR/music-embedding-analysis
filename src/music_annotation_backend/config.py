from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MAB_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_root: Path = Field(default=Path(".music-annotation-data"))
    model_root: Path = Field(default=Path("models"))
    vendor_root: Path = Field(default=Path("vendor"))
    huggingface_cache: Path | None = None
    model_download_source: Literal["modelscope", "huggingface"] = "modelscope"
    host: str = "127.0.0.1"
    port: int = 49321
    session_token: str | None = None
    auto_load_provider: str | None = None
    auto_load_device: str = "auto"
    auto_load_checkpoint_path: str | None = None
    auto_load_allow_download: bool = False
    cors_origin_regex: str = r"https?://(localhost|127\.0\.0\.1)(:\d+)?"
    maximum_upload_bytes: int = 2 * 1024 * 1024 * 1024
    log_level: str = "info"
    mood_embedding_graph: Path | None = None
    mood_regression_graph: Path | None = None
    essentia_native_executable: Path | None = None
    essentia_native_timeout_seconds: float = Field(default=1800, ge=30, le=10800)
    essentia_setup: Literal["auto", "off"] = "auto"
    essentia_setup_mood: bool = True
    omr_python: Path | None = None
    omr_auto_install: bool = True
    omr_timeout_seconds: float = Field(default=1800, ge=30, le=10800)

    def prepare(self) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.model_root.mkdir(parents=True, exist_ok=True)
        self.vendor_root.mkdir(parents=True, exist_ok=True)
        if self.huggingface_cache is not None:
            self.huggingface_cache.mkdir(parents=True, exist_ok=True)
