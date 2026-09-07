from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MAB_", extra="ignore")

    data_root: Path = Field(default=Path(".music-annotation-data"))
    model_root: Path = Field(default=Path("models"))
    vendor_root: Path = Field(default=Path("vendor"))
    huggingface_cache: Path | None = None
    host: str = "127.0.0.1"
    port: int = 49321
    session_token: str | None = None
    log_level: str = "info"

    def prepare(self) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.model_root.mkdir(parents=True, exist_ok=True)
        self.vendor_root.mkdir(parents=True, exist_ok=True)
        if self.huggingface_cache is not None:
            self.huggingface_cache.mkdir(parents=True, exist_ok=True)
