"""Music CLAP in Transformers format, with explicit regional download routing."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import Settings


MODEL_ID = "laion/larger_clap_music"
MODEL_FILES = ["*.json", "*.txt", "*.safetensors", "pytorch_model.bin"]


def resolve_snapshot(settings: Settings, checkpoint_path: str | None, allow_download: bool) -> Path:
    if checkpoint_path:
        directory = Path(checkpoint_path).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError("clap_music requires a Transformers model directory, not a legacy .pt file")
        return directory
    source = settings.model_download_source
    cache = settings.model_root / "downloads" / source
    if source == "modelscope":
        from modelscope import snapshot_download
        return Path(snapshot_download(
            MODEL_ID, cache_dir=str(cache), allow_patterns=MODEL_FILES,
            local_files_only=not allow_download,
        ))
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(
        MODEL_ID, cache_dir=str(settings.huggingface_cache or cache),
        allow_patterns=MODEL_FILES, local_files_only=not allow_download,
    ))


class TransformersClap:
    """Expose the embedding methods used by the existing CLAP scoring adapter."""

    def __init__(self, directory: Path, device: str):
        from transformers import ClapModel, ClapProcessor
        self.device = device
        # Both encoders and the tokenizer are bundled. Never query HF implicitly.
        self.processor = ClapProcessor.from_pretrained(str(directory), local_files_only=True)
        self.model = ClapModel.from_pretrained(str(directory), local_files_only=True).to(device).eval()

    def get_audio_embedding_from_data(self, x: np.ndarray, use_tensor: bool = False) -> Any:
        import torch
        with torch.inference_mode():
            inputs = self.processor(audios=list(x), sampling_rate=48000, return_tensors="pt").to(self.device)
            features = self.model.get_audio_features(**inputs)
        return features if use_tensor else features.detach().cpu().numpy()

    def get_text_embedding(self, texts: list[str], use_tensor: bool = False) -> Any:
        import torch
        with torch.inference_mode():
            inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            features = self.model.get_text_features(**inputs)
        return features if use_tensor else features.detach().cpu().numpy()
