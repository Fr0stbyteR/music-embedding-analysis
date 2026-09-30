from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import sys
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from .config import Settings
from .schemas import ProviderCapability


class Provider(ABC):
    provider_id: str
    display_name: str
    sample_rate: int
    supports_frame_embeddings: bool
    supports_text_embeddings: bool
    supports_training: bool
    weight_license: str | None
    commercial_use: bool | None
    package_name: str | None

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model: Any = None
        self.state = "available" if self.installed() else "not_installed"
        self.device: str | None = None
        self.error: str | None = None
        self._text_embedding_cache: dict[tuple[str, ...], Any] = {}
        self._inference_lock = threading.RLock()

    def installed(self) -> bool:
        return self.package_name is None or importlib.util.find_spec(self.package_name) is not None

    def capability(self) -> ProviderCapability:
        return ProviderCapability(
            provider_id=self.provider_id, display_name=self.display_name, installed=self.installed(),
            loaded=self.model is not None, state=self.state, supports_frame_embeddings=self.supports_frame_embeddings,
            supports_text_embeddings=self.supports_text_embeddings, supports_training=self.supports_training,
            sample_rate=self.sample_rate, weight_license=self.weight_license, commercial_use=self.commercial_use,
            device=self.device, error=self.error,
        )

    async def load(self, device: str, checkpoint_path: str | None, allow_download: bool) -> dict[str, Any]:
        self.state, self.error = "loading", None
        started = time.perf_counter()
        try:
            resolved = self._resolve_device(device)
            self.model = await asyncio.to_thread(self._load_sync, resolved, checkpoint_path, allow_download)
            self.device, self.state = resolved, "loaded"
            return {"providerId": self.provider_id, "device": resolved, "seconds": time.perf_counter() - started}
        except Exception as exc:
            self.state, self.error, self.model = "failed", f"{type(exc).__name__}: {exc}", None
            raise

    def unload(self) -> None:
        self.model = None
        self.device = None
        self.state = "available" if self.installed() else "not_installed"
        self._text_embedding_cache.clear()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @abstractmethod
    def _load_sync(self, device: str, checkpoint_path: str | None, allow_download: bool) -> Any: ...

    async def probe(self) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("provider is not loaded")
        return await asyncio.to_thread(self._probe_sync)

    @abstractmethod
    def _probe_sync(self) -> dict[str, Any]: ...

    def score_audio_text(self, waveforms: np.ndarray, texts: list[str]) -> np.ndarray:
        raise NotImplementedError(f"{self.display_name} does not support audio/text scoring")

    def embed_audio_for_text(self, waveforms: np.ndarray) -> np.ndarray:
        """Return normalized audio embeddings when the provider exposes them."""
        raise NotImplementedError

    def score_audio_embeddings_text(self, audio_embeddings: np.ndarray, texts: list[str]) -> np.ndarray:
        """Score previously computed normalized audio embeddings against text."""
        raise NotImplementedError


class MockProvider(Provider):
    provider_id = "mock"
    display_name = "Deterministic mock"
    sample_rate = 16000
    supports_frame_embeddings = True
    supports_text_embeddings = True
    supports_training = True
    weight_license = "internal-test"
    commercial_use = True
    package_name = None

    def _load_sync(self, device: str, checkpoint_path: str | None, allow_download: bool) -> Any:
        return {"dimension": 32, "device": device}

    def encode(self, waveforms: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("provider is not loaded")
        rng = np.random.default_rng(42)
        projection = rng.standard_normal((4, 32), dtype=np.float32)
        stats = np.stack((waveforms.mean(1), waveforms.std(1), waveforms.min(1), waveforms.max(1)), axis=1)
        return stats @ projection

    def _probe_sync(self) -> dict[str, Any]:
        started = time.perf_counter()
        result = self.encode(np.zeros((1, self.sample_rate), dtype=np.float32))
        return {"providerId": self.provider_id, "device": self.device or "cpu", "sampleRate": self.sample_rate,
                "audioShape": list(result.shape), "audioSeconds": time.perf_counter() - started}

    def score_audio_text(self, waveforms: np.ndarray, texts: list[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("provider is not loaded")
        energy = np.sqrt(np.mean(np.square(waveforms), axis=1, keepdims=True)).astype(np.float32)
        text_bias = np.array([
            int.from_bytes(hashlib.blake2s(text.encode("utf-8"), digest_size=2).digest(), "big") / 65535
            for text in texts
        ], dtype=np.float32)[None, :]
        return np.clip(0.25 + energy * 2 + text_bias * 0.35, 0, 1)


class M2DClapProvider(Provider):
    provider_id = "m2d_clap_2025"
    display_name = "M2D-CLAP 2025 temporal"
    sample_rate = 16000
    supports_frame_embeddings = True
    supports_text_embeddings = False
    supports_training = True
    weight_license = "See upstream LICENSE.pdf"
    commercial_use = None
    package_name = "torch"

    def _load_sync(self, device: str, checkpoint_path: str | None, allow_download: bool) -> Any:
        checkpoint = Path(checkpoint_path) if checkpoint_path else (
            self.settings.model_root
            / "m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025"
            / "checkpoint-30.pth"
        )
        portable = self.settings.vendor_root / "m2d" / "examples"
        if not portable.exists():
            raise FileNotFoundError(f"M2D source missing: {portable}; run scripts/setup_models.ps1")
        if not checkpoint.exists():
            raise FileNotFoundError(f"M2D checkpoint missing: {checkpoint}; run scripts/setup_models.ps1")
        sys.path.insert(0, str(portable))
        try:
            from portable_m2d import PortableM2D
            model = PortableM2D(str(checkpoint)).to(device).eval()
        finally:
            sys.path.remove(str(portable))
        return model

    def _probe_sync(self) -> dict[str, Any]:
        import torch
        started = time.perf_counter()
        with torch.inference_mode():
            result = self.model(torch.zeros(1, self.sample_rate, device=self.device))
        return {"providerId": self.provider_id, "device": self.device or "cpu", "sampleRate": self.sample_rate,
                "audioShape": list(result.shape), "audioSeconds": time.perf_counter() - started}


class LaionClapProvider(Provider):
    provider_id = "laion_clap_music_htsat_base"
    display_name = "LAION-CLAP HTSAT-base"
    sample_rate = 48000
    supports_frame_embeddings = False
    supports_text_embeddings = True
    supports_training = True
    weight_license = "Checkpoint-specific"
    commercial_use = None
    package_name = "laion_clap"

    def _load_sync(self, device: str, checkpoint_path: str | None, allow_download: bool) -> Any:
        import laion_clap
        model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=device)
        checkpoint = Path(checkpoint_path) if checkpoint_path else (
            self.settings.model_root / "laion_clap_music" / "music_audioset_epoch_15_esc_90.14.pt"
        )
        if not checkpoint.exists() and allow_download:
            from huggingface_hub import hf_hub_download
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = Path(hf_hub_download(
                "lukewys/laion_clap",
                "music_audioset_epoch_15_esc_90.14.pt",
                local_dir=checkpoint.parent,
            ))
        if not checkpoint.exists():
            raise FileNotFoundError(f"LAION music checkpoint missing: {checkpoint}")
        model.load_ckpt(str(checkpoint))
        return model

    def _probe_sync(self) -> dict[str, Any]:
        started = time.perf_counter()
        result = self.model.get_audio_embedding_from_data(
            x=np.zeros((1, self.sample_rate), dtype=np.float32), use_tensor=False
        )
        audio_seconds = time.perf_counter() - started
        started = time.perf_counter()
        text = self.model.get_text_embedding(["古琴泛音", "guqin harmonic"], use_tensor=False)
        return {"providerId": self.provider_id, "device": self.device or "cpu", "sampleRate": self.sample_rate,
                "audioShape": list(result.shape), "audioSeconds": audio_seconds,
                "textShape": list(text.shape), "textSeconds": time.perf_counter() - started}

    def embed_audio_for_text(self, waveforms: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("provider is not loaded")
        with self._inference_lock:
            audio = np.asarray(self.model.get_audio_embedding_from_data(x=waveforms, use_tensor=False), dtype=np.float32)
        audio /= np.maximum(np.linalg.norm(audio, axis=1, keepdims=True), 1e-8)
        return audio

    def score_audio_embeddings_text(self, audio_embeddings: np.ndarray, texts: list[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("provider is not loaded")
        key = tuple(texts)
        with self._inference_lock:
            text = self._text_embedding_cache.get(key)
            if text is None:
                text = np.asarray(self.model.get_text_embedding(texts, use_tensor=False), dtype=np.float32)
                text /= np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-8)
                self._text_embedding_cache[key] = text
        return np.clip((audio_embeddings @ text.T + 1) / 2, 0, 1)

    def score_audio_text(self, waveforms: np.ndarray, texts: list[str]) -> np.ndarray:
        return self.score_audio_embeddings_text(self.embed_audio_for_text(waveforms), texts)


class ClapMusicProvider(LaionClapProvider):
    provider_id = "clap_music"
    display_name = "Music CLAP (Transformers / ModelScope)"
    package_name = "transformers"
    supports_training = False
    weight_license = "Apache-2.0 (laion/larger_clap_music model card)"
    commercial_use = True

    def _load_sync(self, device: str, checkpoint_path: str | None, allow_download: bool) -> Any:
        from .clap_music import TransformersClap, resolve_snapshot
        directory = resolve_snapshot(self.settings, checkpoint_path, allow_download)
        return TransformersClap(directory, device)


class MuQMulanProvider(Provider):
    provider_id = "muq_mulan_large"
    display_name = "MuQ-MuLan large"
    sample_rate = 24000
    supports_frame_embeddings = False
    supports_text_embeddings = True
    supports_training = False
    weight_license = "CC-BY-NC-4.0"
    commercial_use = False
    package_name = "muq"

    def _load_sync(self, device: str, checkpoint_path: str | None, allow_download: bool) -> Any:
        from muq import MuQMuLan
        source = checkpoint_path or "OpenMuQ/MuQ-MuLan-large"
        if not checkpoint_path and not allow_download:
            raise FileNotFoundError("checkpointPath is required unless allowDownload=true")
        kwargs: dict[str, Any] = {}
        if self.settings.huggingface_cache is not None:
            self.settings.huggingface_cache.mkdir(parents=True, exist_ok=True)
            kwargs["cache_dir"] = str(self.settings.huggingface_cache)
        return MuQMuLan.from_pretrained(source, **kwargs).to(device).eval()

    def _probe_sync(self) -> dict[str, Any]:
        import torch
        started = time.perf_counter()
        with torch.inference_mode():
            audio = self.model(wavs=torch.zeros(1, self.sample_rate, device=self.device))
        audio_seconds = time.perf_counter() - started
        started = time.perf_counter()
        with torch.inference_mode():
            text = self.model(texts=["古琴泛音", "guqin harmonic"])
        return {"providerId": self.provider_id, "device": self.device or "cpu", "sampleRate": self.sample_rate,
                "audioShape": list(audio.shape), "audioSeconds": audio_seconds,
                "textShape": list(text.shape), "textSeconds": time.perf_counter() - started}

    def embed_audio_for_text(self, waveforms: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("provider is not loaded")
        import torch
        with self._inference_lock:
            with torch.inference_mode():
                audio = self.model(wavs=torch.from_numpy(waveforms).to(self.device))
                audio = torch.nn.functional.normalize(audio.float(), dim=-1)
                return audio.cpu().numpy()

    def score_audio_embeddings_text(self, audio_embeddings: np.ndarray, texts: list[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("provider is not loaded")
        import torch
        key = tuple(texts)
        with self._inference_lock:
            text = self._text_embedding_cache.get(key)
            if text is None:
                with torch.inference_mode():
                    text = torch.nn.functional.normalize(self.model(texts=texts).float(), dim=-1).cpu().numpy()
                self._text_embedding_cache[key] = text
        return np.clip((audio_embeddings @ text.T + 1) / 2, 0, 1)

    def score_audio_text(self, waveforms: np.ndarray, texts: list[str]) -> np.ndarray:
        return self.score_audio_embeddings_text(self.embed_audio_for_text(waveforms), texts)


class ProviderRegistry:
    def __init__(self, settings: Settings):
        providers = (MockProvider(settings), M2DClapProvider(settings), LaionClapProvider(settings), ClapMusicProvider(settings), MuQMulanProvider(settings))
        self.providers = {provider.provider_id: provider for provider in providers}

    def capabilities(self) -> list[ProviderCapability]:
        return [provider.capability() for provider in self.providers.values()]

    def get(self, provider_id: str) -> Provider:
        try:
            return self.providers[provider_id]
        except KeyError as exc:
            raise KeyError(f"unknown provider: {provider_id}") from exc
