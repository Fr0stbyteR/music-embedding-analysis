"""Whitelisted official models, shared native embeddings and durable score caches."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryFile
from threading import Lock
from uuid import uuid4
import numpy as np
from pydantic import ConfigDict, Field
from .schemas import ApiModel
from .essentia_api import EssentiaUnavailable
from .essentia_runtime import feature_runtime

MANIFEST = json.loads(Path(__file__).with_name("essentia_tf_models.json").read_text(encoding="utf-8"))
FILES = {entry["name"]: entry for entry in MANIFEST}
BACKBONES = {"musicnn": "msd-musicnn-1", "effnet": "discogs-effnet-bs64-1", "tempo": "deepsquare-k16-3"}

@dataclass(frozen=True)
class Model:
    family: str
    stem: str
    target: str | None = None

MODELS = {
    "tfInstrument": Model("effnet", "mtg_jamendo_instrument-discogs-effnet-1"),
    "tfMoodTheme": Model("effnet", "mtg_jamendo_moodtheme-discogs-effnet-1"),
    "tfGenre": Model("effnet", "mtg_jamendo_genre-discogs-effnet-1"),
    "tfTimbre": Model("effnet", "timbre-discogs-effnet-1", "bright"),
    "tfVoice": Model("musicnn", "voice_instrumental-msd-musicnn-1", "voice"),
    "tfAcoustic": Model("musicnn", "mood_acoustic-msd-musicnn-1", "acoustic"),
    "tfElectronic": Model("musicnn", "mood_electronic-msd-musicnn-1", "electronic"),
    "tfTonal": Model("musicnn", "tonal_atonal-msd-musicnn-1", "tonal"),
    "tfDanceability": Model("musicnn", "danceability-msd-musicnn-1", "danceable"),
    **{f"tf{name.capitalize()}": Model("musicnn", f"mood_{name}-msd-musicnn-1", name) for name in ("happy", "sad", "relaxed", "aggressive", "party")},
    "tfTags": Model("musicnn", "msd-musicnn-1"),
    "tfTempo": Model("tempo", "deepsquare-k16-3"),
}
ALIASES = {"tfInstrumentCurve": "tfInstrument", "tfInstrumentRegions": "tfInstrument", "tfMoodThemeCurve": "tfMoodTheme", "tfTagsCurve": "tfTags", "tfTempoCandidates": "tfTempo"}
ALGORITHMS = set(MODELS) | set(ALIASES)
MATRICES = {"tfInstrument", "tfMoodTheme", "tfGenre", "tfTags", "tfTempoCandidates"}

class TFOptions(ApiModel):
    model_config = ConfigDict(extra="forbid")
    hop_seconds: float = Field(default=2, ge=1, le=30, allow_inf_nan=False)
    label: str | None = Field(default=None, min_length=1, max_length=120)
    threshold: float = Field(default=.6, ge=0, le=1, allow_inf_nan=False)
    minimum_duration: float = Field(default=2, ge=.1, le=60, allow_inf_nan=False)

def matrix(value, rows, columns, *, scores=False):
    try:
        data = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise RuntimeError("Invalid TensorFlow result") from error
    if data.shape != (rows, columns) or not np.isfinite(data).all():
        raise RuntimeError("Invalid TensorFlow result dimensions or values")
    if scores and (np.any(data < -1e-6) or np.any(data > 1 + 1e-6)):
        raise RuntimeError("TensorFlow scores outside 0-1")
    return np.clip(data, 0, 1) if scores else data

def candidate_regions(scores, hop, duration, threshold, minimum_duration, label):
    # Hysteresis avoids boundary chatter; output remains a user-editable candidate.
    active, start = False, 0.
    intervals, labels = [], []
    for index in range(len(scores) + 1):
        value = float(scores[index]) if index < len(scores) else -1.
        boundary = duration if index == len(scores) else min(duration, index * hop)
        if not active and value >= threshold:
            active, start = True, boundary
        elif active and value < threshold * .8:
            if boundary - start >= minimum_duration:
                intervals.append([round(start, 6), round(boundary, 6)])
                labels.append(f"{label} · candidate")
            active = False
    return intervals, labels

class TensorflowAnalyzer:
    def __init__(self, settings):
        self.settings = settings
        self.native = feature_runtime(settings)
        self.root = settings.model_root / "essentia"
        self.cache_root = settings.data_root / "essentia-tf-cache"
        self.lock = Lock()
        self.hashes = {}

    def capabilities(self):
        reason, info, available = None, None, []
        try:
            info = self.native.probe()
            if info.get("tensorflowFeatures") != 1:
                raise EssentiaUnavailable("Restart with the updated start script to rebuild Essentia TensorFlow support")
            for algorithm in sorted(ALGORITHMS):
                spec = MODELS[ALIASES.get(algorithm, algorithm)]
                required = {f"{stem}.{ext}" for stem in (BACKBONES[spec.family], spec.stem) for ext in ("json", "pb")}
                if all((self.root / name).is_file() for name in required):
                    available.append(algorithm)
            if not available:
                reason = "Restart with the start script to prepare the official Essentia TensorFlow models"
        except RuntimeError as error:
            reason = str(error)
        return {"available": bool(available), "algorithms": available, "reason": reason, "runtime": self.native.expected_runtime, "weightLicense": "CC BY-NC-SA 4.0 (MTG models; check TempoCNN upstream license)", "native": info}

    def weight(self, name):
        path = self.root / name
        if not path.is_file():
            raise EssentiaUnavailable(f"Missing {name}; restart with the start script")
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if self.hashes.get(name) != signature:
            with path.open("rb") as stream:
                sha = hashlib.file_digest(stream, "sha256").hexdigest()
            if sha != FILES[name]["sha256"]:
                raise EssentiaUnavailable(f"Checksum mismatch: {name}. Move it aside and run the start script; it was not overwritten.")
            self.hashes[name] = signature
        return path

    def run(self, arguments, data):
        with TemporaryFile() as pcm:
            np.asarray(data, dtype="<f4").tofile(pcm); pcm.seek(0)
            result = self.native._run(arguments, stdin=pcm, timeout=self.native.timeout_seconds)
        if not isinstance(result, dict) or result.get("protocol") != 1:
            raise RuntimeError("Invalid TensorFlow worker protocol")
        return result

    def save_arrays(self, path, **arrays):
        temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as output:
                np.savez_compressed(output, **arrays)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def load_arrays(self, path):
        if not path.is_file(): return None
        try:
            with np.load(path, allow_pickle=False) as stored:
                return {name: stored[name] for name in stored.files}
        except (OSError, ValueError, EOFError):
            return None

    def analyze(self, asset, request, progress=None):
        algorithm = request.algorithm
        if algorithm not in ALGORITHMS: raise ValueError("Unknown Essentia TensorFlow algorithm")
        base = ALIASES.get(algorithm, algorithm)
        spec = MODELS[base]
        options = TFOptions.model_validate({"hopSeconds": 6 if base == "tfTempo" else 2, **request.options})
        if asset.duration_seconds <= 0 or asset.duration_seconds > 10800:
            raise ValueError("Essentia TensorFlow supports audio up to 3 hours")
        sr, frame_hop, patch = (11025, 512, 256) if spec.family == "tempo" else (16000, 256, 128 if spec.family == "effnet" else 187)
        hop_frames = max(1, round(options.hop_seconds * sr / frame_hop))
        hop = hop_frames * frame_hop / sr
        if math.ceil(asset.duration_seconds / hop) > 11000: raise ValueError("Too many prediction points")
        embedding_width = 256 if spec.family == "tempo" else 1280 if spec.family == "effnet" else 200
        if math.ceil(asset.duration_seconds / hop) * embedding_width > 2000000:
            raise ValueError("Embedding cache exceeds memory budget; increase hopSeconds (prediction interval)")
        notify = progress or (lambda *_: None)
        with self.lock:
            capability = self.capabilities()
            if algorithm not in capability["algorithms"]:
                raise EssentiaUnavailable(capability["reason"] or "Required model missing; restart with the start script")
            stem = BACKBONES[spec.family]
            graph = self.weight(f"{stem}.pb")
            meta_path = self.weight(f"{spec.stem}.json")
            self.weight(f"{spec.stem}.pb")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            labels = [str(i + 30) for i in range(256)] if base == "tfTempo" else meta["classes"]
            target = options.label or spec.target or {"tfInstrument": "piano", "tfMoodTheme": "calm", "tfTags": "instrumental"}.get(base)
            if target is not None and target not in labels: raise ValueError(f"Unknown target label: {target}")
            descriptor = {"version": 2, "asset": asset.content_hash, "family": spec.family, "hopFrames": hop_frames,
                "backbone": FILES[f"{stem}.pb"]["sha256"], "runtime": capability["native"], "signature": self.native.signature}
            key = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
            self.cache_root.mkdir(parents=True, exist_ok=True)
            embeddings_path = self.cache_root / f"{key}-embeddings.npz"
            embeddings = self.load_arrays(embeddings_path) if request.cache_policy == "use" else None
            embedding_hit = embeddings is not None
            width = 256 if spec.family == "tempo" else 1280 if spec.family == "effnet" else 200
            if embeddings is not None:
                try:
                    duration = float(embeddings["duration"])
                    count = math.ceil(duration / hop)
                    if not math.isfinite(duration) or duration <= 0 or duration > 10800 or count > 11000: raise RuntimeError("Invalid cache duration")
                    matrix(embeddings["matrix"], count, width, scores=spec.family == "tempo")
                    if spec.family == "musicnn": matrix(embeddings["tags"], count, 50, scores=True)
                except (RuntimeError, ValueError, KeyError):
                    embeddings = None; embedding_hit = False
            if embeddings is None:
                import librosa
                audio, _ = librosa.load(asset.path, sr=sr, mono=True)
                if not len(audio) or not np.isfinite(audio).all(): raise ValueError("Audio must contain finite samples")
                duration, count = len(audio) / sr, math.ceil(len(audio) / (hop_frames * frame_hop))
                notify(.15, "Audio decoded")
                raw = self.run(["--tf-backbone", spec.family, graph.resolve(), hop_frames], audio)
                embeddings = {"matrix": matrix(raw.get("matrix"), count, width, scores=spec.family == "tempo"), "duration": np.asarray(duration)}
                if spec.family == "musicnn": embeddings["tags"] = matrix(raw.get("tags"), count, 50, scores=True)
                self.save_arrays(embeddings_path, **embeddings)
            duration = float(embeddings["duration"])
            count = len(embeddings["matrix"])
            notify(.7, "Shared embeddings loaded" if embedding_hit else "Shared embeddings computed")
            score_key = hashlib.sha256(f"{key}:{FILES[f'{spec.stem}.pb']['sha256']}:{FILES[f'{spec.stem}.json']['sha256']}".encode()).hexdigest()
            score_path = self.cache_root / f"{score_key}-scores.npz"
            saved = self.load_arrays(score_path) if request.cache_policy == "use" else None
            scores, score_hit = None, False
            if saved is not None:
                try:
                    scores = matrix(saved["scores"], count, len(labels), scores=True); score_hit = True
                except (KeyError, RuntimeError): pass
            if scores is None:
                if base == "tfTags": scores = embeddings["tags"]
                elif base == "tfTempo": scores = embeddings["matrix"]
                else:
                    output = next(item["name"] for item in meta["schema"]["outputs"] if item.get("output_purpose") == "predictions")
                    inputs = meta["schema"]["inputs"][0]["name"]
                    raw = self.run(["--tf-head", (self.root / f"{spec.stem}.pb").resolve(), inputs, output, width, len(labels)], embeddings["matrix"])
                    scores = matrix(raw.get("matrix"), count, len(labels), scores=True)
                self.save_arrays(score_path, scores=scores)
            notify(.9, "Model scores loaded" if score_hit else "Model scores computed")
            result = {"algorithm": algorithm, "sampleRate": sr, "duration": duration,
                "metadata": {"hopLength": hop_frames * frame_hop, "minValue": 0, "maxValue": 1,
                    "engine": "essentia-tf", "model": spec.stem, "classLabels": json.dumps(labels), "contextSeconds": patch * frame_hop / sr,
                    "scoreKind": "model-score-not-calibrated-confidence", "timeAnchor": "cell-center", "embeddingCache": "hit" if embedding_hit else "miss", "targetLabel": target},
                "cache": {"status": "refresh" if request.cache_policy == "refresh" else "hit" if score_hit else "miss", "createdAt": datetime.now(timezone.utc).isoformat()}}
            if algorithm in MATRICES:
                result.update(matrix=scores.tolist(), labels=labels)
            elif algorithm == "tfTempo":
                values = np.argmax(scores, axis=1) + 30
                result["vectors"] = [values.tolist()]
                result["metadata"].update(minValue=30, maxValue=285, globalTempo=int(np.bincount(values, minlength=286).argmax()))
            else:
                values = scores[:, labels.index(target)]
                if algorithm == "tfInstrumentRegions":
                    result["intervals"], result["labels"] = candidate_regions(values, hop, duration, options.threshold, options.minimum_duration, target)
                else: result["vectors"] = [values.tolist()]
            notify(1., "Display data prepared")
            return result
