"""Cached librosa analysis for browser and desktop clients."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .librosa_engine import analyze
from .signal_statistics import SIGNAL_STATISTICS_ALGORITHMS
from .roughness import ROUGHNESS_ALGORITHMS
from .schemas import Asset, InteractiveLibrosaRequest


SUPPORTED_ALGORITHMS = {
    "beats", "onsets", "nonSilent", "rms", "zeroCrossingRate", "onsetStrength",
    "spectralCentroid", "spectralBandwidth", "spectralRolloff", "spectralFlatness",
    "pitch", "melSpectrogram", "chroma", "mfcc",
} | SIGNAL_STATISTICS_ALGORITHMS | ROUGHNESS_ALGORITHMS


def engine_version() -> str:
    digest = hashlib.sha256()
    for name in ("librosa_engine.py", "signal_statistics.py", "roughness.py"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def analyze_interactive_asset(asset: Asset, request: InteractiveLibrosaRequest, cache_root: Path) -> dict:
    if request.algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"unsupported analysis algorithm: {request.algorithm}")
    cache_root.mkdir(parents=True, exist_ok=True)
    engine_hash = engine_version()
    descriptor = json.dumps({
        "asset": asset.content_hash,
        "algorithm": request.algorithm,
        "options": request.options,
        "engine": engine_hash,
    }, sort_keys=True, separators=(",", ":"))
    cache_path = cache_root / f"{hashlib.sha256(descriptor.encode()).hexdigest()}.json"
    if request.cache_policy == "use" and cache_path.exists():
        try:
            result = json.loads(cache_path.read_text(encoding="utf-8"))
            if result.get("algorithm") == request.algorithm:
                return {**result, "cache": {**result.get("cache", {}), "status": "hit"}}
        except (OSError, ValueError, TypeError):
            pass  # An incomplete or invalid cache entry must not prevent analysis.
    result = analyze({"path": asset.path, "algorithm": request.algorithm, "options": request.options})
    result["cache"] = {
        "status": "refresh" if request.cache_policy == "refresh" else "miss",
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    temporary = cache_root / f"{cache_path.stem}.{uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
        temporary.replace(cache_path)
    finally:
        temporary.unlink(missing_ok=True)
    return result
