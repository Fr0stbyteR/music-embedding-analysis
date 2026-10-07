"""Optional Essentia DEAM regression; no CLAP/energy proxy or implicit downloads."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import platform
from threading import Lock
from uuid import uuid4

import numpy as np

from .config import Settings
from .schemas import Asset, MoodCurveRequest
from .native_essentia import NativeEssentia
from .essentia_runtime import native_executable
from .essentia_macos import require_macos_sdl2
from .result_statistics import result_statistics


class MoodUnavailable(RuntimeError):
    pass


class MoodAnalyzer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedding_path = settings.mood_embedding_graph or settings.model_root / "essentia" / "msd-musicnn-1.pb"
        self.regression_path = settings.mood_regression_graph or settings.model_root / "essentia" / "deam-msd-musicnn-2.pb"
        self.lock = Lock()
        self.models = None
        self.identity = None
        executable = native_executable(settings)
        self.native = NativeEssentia(executable, settings.essentia_native_timeout_seconds,
            probe_timeout_seconds=settings.essentia_probe_timeout_seconds)

    @property
    def native_selected(self) -> bool:
        return self.settings.essentia_native_executable is not None or self.native.executable.is_file()

    def capabilities(self) -> dict:
        reason = None
        runtime = "essentia-python"
        native_info = None
        if self.native_selected:
            runtime = "essentia-cpp"
            try:
                native_info = self.native.probe()
            except RuntimeError as error:
                reason = str(error)
        elif platform.system() == "Windows":
            reason = "Restart with start.cmd to prepare and verify the native Essentia runtime and VA weights."
        elif importlib.util.find_spec("essentia") is None:
            reason = "Restart with start.command to prepare and verify the native Essentia wheel and VA weights."
        elif reason is None:
            try:
                require_macos_sdl2()
                from essentia.standard import TensorflowPredictMusiCNN, TensorflowPredict2D
            except (ImportError, OSError, RuntimeError) as error:
                reason = f"Essentia TensorFlow algorithms could not load: {error}"
        if reason is None and (not self.embedding_path.is_file() or not self.regression_path.is_file()):
            reason = "Missing Essentia weights: msd-musicnn-1.pb and deam-msd-musicnn-2.pb in MAB_MODEL_ROOT/essentia."
        return {"model": "deam-msd-musicnn-2", "available": reason is None, "reason": reason, "scale": "normalized-minus-one-to-one", "weightLicense": "CC BY-NC-SA 4.0", "loaded": self.models is not None, "runtime": runtime, "native": native_info}

    def _load(self) -> None:
        paths = (self.embedding_path, self.regression_path)
        signature = tuple((str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size) for path in paths)
        if self.native_selected:
            signature += (json.dumps(self.native.probe(), sort_keys=True), self.native.signature)
        if self.identity and self.identity[0] == signature:
            return
        if self.native_selected:
            self.models = None  # Native models live only inside each isolated job.
        else:
            require_macos_sdl2()
            from essentia.standard import TensorflowPredictMusiCNN, TensorflowPredict2D
            self.models = (
                TensorflowPredictMusiCNN(graphFilename=str(paths[0]), output="model/dense/BiasAdd", lastPatchMode="repeat"),
                TensorflowPredict2D(graphFilename=str(paths[1]), output="model/Identity"),
            )
        hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
        self.identity = (signature, hashes)

    def predict(self, audio: np.ndarray) -> tuple[float, float]:
        embedding, regression = self.models
        minimum = 16000 * 3
        if len(audio) < minimum:
            audio = np.tile(audio, math.ceil(minimum / max(1, len(audio))))[:minimum]
        raw = np.asarray(regression(embedding(np.asarray(audio, dtype=np.float32))), dtype=np.float64)
        if raw.size == 0 or raw.ndim != 2 or raw.shape[1] != 2 or not np.isfinite(raw).all():
            raise RuntimeError("DEAM returned invalid valence/arousal predictions")
        values = np.clip((raw.mean(axis=0) - 5.0) / 4.0, -1, 1)
        return float(values[0]), float(values[1])

    def analyze(self, asset: Asset, request: MoodCurveRequest) -> dict:
        duration = request.timeline_duration_seconds or asset.duration_seconds
        if not math.isfinite(duration) or duration <= 0 or duration > 10800 or asset.duration_seconds > 10800:
            raise ValueError("Mood analysis supports audio timelines up to 3 hours")
        count = math.floor(duration / request.hop_seconds) + 1
        if count > 10000:
            raise ValueError("Too many mood windows; increase hopSeconds (maximum 10000 points)")
        capability = self.capabilities()
        if not capability["available"]:
            raise MoodUnavailable(capability["reason"])
        with self.lock:
            try:
                self._load()
            except (ImportError, OSError, RuntimeError) as error:
                raise MoodUnavailable(f"Could not load Essentia TensorFlow models: {error}") from error
            descriptor = json.dumps({"version": 2, "runtime": capability.get("runtime"), "native": capability.get("native"), "asset": asset.content_hash, "weights": self.identity[1], "window": request.window_seconds, "hop": request.hop_seconds, "duration": duration}, sort_keys=True)
            cache_root = self.settings.data_root / "mood-cache"
            cache_root.mkdir(parents=True, exist_ok=True)
            cache_path = cache_root / f"{hashlib.sha256(descriptor.encode()).hexdigest()}.json"
            if request.cache_policy == "use" and cache_path.is_file():
                try:
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                    if cached.get("model") == capability["model"] and len(cached["points"]) == count:
                        cached["metadata"] = result_statistics({"vectors": [[point[field] for point in cached["points"]] for field in ("valence", "arousal")], "metadata": cached.get("metadata", {})})["metadata"]
                        return {**cached, "cached": True}
                except (ValueError, TypeError, KeyError, OSError):
                    pass
            import librosa
            audio, _ = librosa.load(asset.path, sr=16000, mono=True)
            if not len(audio) or not np.isfinite(audio).all():
                raise ValueError("Audio has no finite samples")
            native_duration = len(audio) / 16000
            scale = native_duration / duration
            if self.native_selected:
                points = self.native.curve(audio, self.embedding_path, self.regression_path, request.window_seconds, request.hop_seconds, duration)
            else:
                points = []
                for index in range(count):
                    time = min(duration, index * request.hop_seconds)
                    start = max(0, min(time - request.window_seconds / 2, duration - request.window_seconds))
                    end = min(duration, start + request.window_seconds)
                    left = max(0, min(len(audio) - 1, round(start * scale * 16000)))
                    right = min(len(audio), max(left + 1, round(end * scale * 16000)))
                    valence, arousal = self.predict(audio[left:right])
                    points.append({"timeSeconds": round(time, 6), "valence": valence, "arousal": arousal})
            result = {"model": capability["model"], "scale": capability["scale"], "windowSeconds": request.window_seconds, "hopSeconds": request.hop_seconds, "points": points, "cached": False}
            result["metadata"] = result_statistics({"vectors": [[point[field] for point in points] for field in ("valence", "arousal")]})["metadata"]
            temporary = cache_path.with_name(f"{cache_path.stem}.{uuid4().hex}.tmp")
            try:
                temporary.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
                temporary.replace(cache_path)
            finally:
                temporary.unlink(missing_ok=True)
            return result
