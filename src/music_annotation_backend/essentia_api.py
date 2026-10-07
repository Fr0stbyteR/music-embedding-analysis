"""Cached native Essentia analyses; Python only decodes/resamples the asset."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from threading import Lock
from tempfile import TemporaryFile
from uuid import uuid4

import numpy as np
from pydantic import ConfigDict, Field, StrictInt, model_validator

from .config import Settings
from .essentia_runtime import feature_runtime
from .result_statistics import result_statistics
from .schemas import ApiModel, Asset, InteractiveLibrosaRequest

VECTOR_ALGORITHMS = {"rms", "energy", "loudness", "zeroCrossingRate", "spectralCentroid", "spectralRolloff", "spectralFlatness", "spectralCrest", "spectralFlux", "spectralEntropy", "spectralComplexity", "hfc", "spectralSpread", "spectralSkewness", "spectralKurtosis", "dissonance", "pitch", "pitchConfidence", "onsetStrength"}
MATRIX_ALGORITHMS = {"melBands", "barkBands", "erbBands", "mfcc", "gfcc", "hpcp"}
MARKER_ALGORITHMS = {"onsets", "silenceRegions", "pitchNotes", "keyRegions"}
ALGORITHMS = VECTOR_ALGORITHMS | MATRIX_ALGORITHMS | MARKER_ALGORITHMS


class EssentiaOptions(ApiModel):
    model_config = ConfigDict(extra="forbid")
    sample_rate: StrictInt = Field(default=44100, ge=16000, le=96000)
    frame_length: StrictInt = Field(default=2048, ge=256, le=8192)
    hop_length: StrictInt = Field(default=512, ge=64, le=8192)
    bands: StrictInt = Field(default=40, ge=12, le=128)
    coefficients: StrictInt = Field(default=13, ge=1, le=64)
    roll_percent: float = Field(default=.85, ge=.1, le=1, allow_inf_nan=False)
    threshold_db: float = Field(default=-60, ge=-120, le=0, allow_inf_nan=False)
    minimum_duration: float = Field(default=.1, ge=.01, le=10, allow_inf_nan=False)
    confidence: float = Field(default=.6, ge=0, le=1, allow_inf_nan=False)
    key_window: float = Field(default=8, ge=2, le=60, allow_inf_nan=False)
    tuning: float = Field(default=440, ge=400, le=480, allow_inf_nan=False)

    @model_validator(mode="after")
    def valid_dimensions(self):
        if self.frame_length & (self.frame_length - 1):
            raise ValueError("frameLength must be a power of two: 256, 512, 1024, 2048, 4096 or 8192")
        if self.hop_length > self.frame_length:
            raise ValueError("hopLength must not exceed frameLength")
        if self.coefficients > self.bands:
            raise ValueError("coefficients must not exceed bands")
        return self


class EssentiaUnavailable(RuntimeError):
    pass


class EssentiaAnalyzer:
    def __init__(self, settings: Settings):
        self.native = feature_runtime(settings)
        self.cache_root = settings.data_root / "essentia-cache"
        self.lock = Lock()

    def capabilities(self) -> dict:
        try:
            info = self.native.probe()
            available = sorted(ALGORITHMS.intersection(info.get("features", [])))
            return {"available": bool(available), "algorithms": available, "runtime": info["runtime"], "reason": None if available else "Restart with the start script to prepare Essentia feature analysis", "native": info}
        except RuntimeError as error:
            return {"available": False, "algorithms": [], "runtime": self.native.expected_runtime, "reason": str(error)}

    def analyze(self, asset: Asset, request: InteractiveLibrosaRequest) -> dict:
        if request.algorithm not in ALGORITHMS:
            raise ValueError(f"Unsupported Essentia algorithm: {request.algorithm}")
        options = EssentiaOptions.model_validate(request.options)
        if asset.duration_seconds > 10800:
            raise ValueError("Essentia analysis supports audio up to 3 hours")
        # Bound JSON/renderer memory before decoding and before invoking the worker.
        frames = math.ceil(asset.duration_seconds * options.sample_rate / options.hop_length) + 2
        bins = 12 if request.algorithm in {"hpcp", "keyRegions"} else min(28, options.bands) if request.algorithm == "barkBands" else options.coefficients if request.algorithm in {"mfcc", "gfcc"} else options.bands
        if frames > 500000 or (request.algorithm in MATRIX_ALGORITHMS | {"keyRegions"} and frames * bins > 2000000):
            raise ValueError("Analysis exceeds result size budget; increase hopLength or reduce bands")
        with self.lock:
            capability = self.capabilities()
            if not capability["available"] or request.algorithm not in capability["algorithms"]:
                raise EssentiaUnavailable(capability.get("reason") or "Rebuild the native worker for this algorithm")
            version = hashlib.sha256(b"".join(Path(__file__).with_name(name).read_bytes() for name in ("essentia_api.py", "native_essentia.py", "essentia_runtime.py", "result_statistics.py"))).hexdigest()
            descriptor = {"version": version, "asset": asset.content_hash, "algorithm": request.algorithm, "options": options.model_dump(), "native": capability["native"], "executable": self.native.signature}
            key = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
            self.cache_root.mkdir(parents=True, exist_ok=True)
            cached_path = self.cache_root / f"{key}.json"
            if request.cache_policy == "use" and cached_path.is_file():
                try:
                    cached = json.loads(cached_path.read_text(encoding="utf-8"))
                    if cached.get("algorithm") == request.algorithm:
                        self.validate_result(cached, request.algorithm, options, cached["duration"])
                        return {**cached, "cache": {**cached["cache"], "status": "hit"}}
                except (OSError, ValueError, TypeError, KeyError, RuntimeError):
                    pass
            import librosa
            audio, _ = librosa.load(asset.path, sr=options.sample_rate, mono=True)
            if not len(audio) or not np.isfinite(audio).all():
                raise ValueError("Audio must contain finite samples")
            with TemporaryFile() as pcm:
                np.asarray(audio, dtype="<f4").tofile(pcm)
                pcm.seek(0)
                arguments = ["--features", request.algorithm, options.sample_rate, options.frame_length, options.hop_length, options.bands, options.coefficients, options.roll_percent, options.threshold_db, options.minimum_duration, options.confidence, options.key_window, options.tuning]
                result = self.native._run(arguments, stdin=pcm, timeout=self.native.timeout_seconds)
            self.validate_result(result, request.algorithm, options, len(audio) / options.sample_rate)
            result_statistics(result)
            result["cache"] = {"status": "refresh" if request.cache_policy == "refresh" else "miss", "createdAt": datetime.now(timezone.utc).isoformat()}
            temporary = cached_path.with_name(f"{key}.{uuid4().hex}.tmp")
            try:
                temporary.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
                temporary.replace(cached_path)
            finally:
                temporary.unlink(missing_ok=True)
            return result

    @staticmethod
    def validate_result(result: dict, algorithm: str, options: EssentiaOptions, duration: float):
        native_duration = result.get("duration") if isinstance(result, dict) else None
        if not isinstance(result, dict) or result.get("protocol") != 1 or result.get("algorithm") != algorithm or result.get("sampleRate") != options.sample_rate or not isinstance(native_duration, (int, float)) or not math.isfinite(native_duration) or native_duration <= 0 or abs(native_duration - duration) > .001:
            raise RuntimeError("Invalid native Essentia result header")
        if algorithm in VECTOR_ALGORITHMS | MATRIX_ALGORITHMS:
            try:
                data = np.asarray(result.get("vectors" if algorithm in VECTOR_ALGORITHMS else "matrix"), dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise RuntimeError("Invalid native Essentia feature data") from error
            if data.ndim != 2 or not data.size or not np.isfinite(data).all() or data.size > 2000000:
                raise RuntimeError("Invalid native Essentia feature data")
            expected_bins = 12 if algorithm == "hpcp" else min(28, options.bands) if algorithm == "barkBands" else options.coefficients if algorithm in {"mfcc", "gfcc"} else options.bands
            if (algorithm in VECTOR_ALGORITHMS and data.shape[0] != 1) or (algorithm in MATRIX_ALGORITHMS and data.shape[1] != expected_bins):
                raise RuntimeError("Invalid native Essentia feature dimensions")
        elif algorithm == "onsets":
            times = np.asarray(result.get("values"), dtype=np.float64)
            if times.ndim != 1 or not np.isfinite(times).all() or np.any(times < 0) or np.any(times > duration + .001) or np.any(np.diff(times) < 0):
                raise RuntimeError("Invalid native Essentia onset times")
        else:
            intervals = result.get("intervals")
            labels = result.get("labels")
            if not isinstance(intervals, list) or not isinstance(labels, list) or len(intervals) != len(labels) or not all(isinstance(label, str) for label in labels):
                raise RuntimeError("Invalid native Essentia marker labels")
            for pair in intervals:
                if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in pair) or pair[0] < 0 or pair[1] > duration + .001 or pair[1] <= pair[0]:
                    raise RuntimeError("Invalid native Essentia marker range")
