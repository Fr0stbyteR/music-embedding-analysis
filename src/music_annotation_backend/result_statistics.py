"""Whole-result summaries shipped with analysis, never selection requests."""
from __future__ import annotations

import base64
import math

import numpy as np


def summarize(values, valid=None) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array) & (valid if valid is not None else True)]
    if not array.size:
        return {"count": 0}
    mean, std = float(np.mean(array)), float(np.std(array))
    return {"count": int(array.size), "mean": mean, "min": float(np.min(array)),
            "max": float(np.max(array)), "std": std, "rms": math.hypot(mean, std)}


def _validity(metadata: dict, index: int, length: int):
    encoded = metadata.get(f"validity.{index}")
    if not isinstance(encoded, str):
        return None
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) * 8 < length:
        raise ValueError("invalid statistics validity mask")
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little", count=length).astype(bool)


def result_statistics(result: dict) -> dict:
    """Add compact primitive metadata; existing workspace serialization preserves it.

    Matrices have one mixed-audio channel. Their aggregate and each bin have
    separate namespaces, avoiding confusion with LPCC coefficient summaries.
    """
    if "vectors" not in result and "matrix" not in result:
        return result
    metadata = result.setdefault("metadata", {})
    if metadata.get("statistics.version") == 1:
        return result

    def write(prefix, summary):
        for key in ("count", "mean", "min", "max", "std", "rms"):
            metadata.pop(f"{prefix}.{key}", None)
        metadata.update({f"{prefix}.{key}": value for key, value in summary.items()})

    for index, vector in enumerate(result.get("vectors", [])):
        write(f"statistics.{index}", summarize(vector, _validity(metadata, index, len(vector))))
    if "matrix" in result:
        matrix = np.asarray(result["matrix"], dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("statistics require a time-major matrix")
        summaries = []
        for index in range(matrix.shape[1]):
            summary = summarize(matrix[:, index], _validity(metadata, index, len(matrix)))
            summaries.append(summary)
            write(f"statistics.matrix.0.bin.{index}", summary)
        active = [item for item in summaries if item["count"]]
        count = sum(item["count"] for item in active)
        aggregate = {"count": count}
        if count:
            mean = sum(item["mean"] * (item["count"] / count) for item in active)
            variance = sum((item["std"] ** 2 + (item["mean"] - mean) ** 2) * (item["count"] / count) for item in active)
            std = math.sqrt(max(0., variance))
            aggregate.update(mean=mean, min=min(item["min"] for item in active),
                             max=max(item["max"] for item in active), std=std, rms=math.hypot(mean, std))
        write("statistics.matrix.0", aggregate)
    metadata["statistics.version"] = 1
    return result
