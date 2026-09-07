from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from uuid import UUID, uuid4

_numba_cache = Path(os.environ.get("MAB_DATA_ROOT", tempfile.gettempdir())) / "numba-cache"
_numba_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(_numba_cache))

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

from .providers import ProviderRegistry
from .schemas import Asset, DetectionCandidate, DetectionLabelPlan, DetectionPlan
from .store import utc_now


HOP_LENGTH = 512
PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def inspect_asset(path_value: str, name: str | None = None) -> Asset:
    path = Path(path_value).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"asset is not a file: {path}")
    info = sf.info(path)
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError("asset has no decodable audio frames")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return Asset(
        id=uuid4(), name=(name or path.name).strip(), path=str(path), content_hash=digest.hexdigest(),
        sample_rate=info.samplerate, channels=info.channels, samples=info.frames,
        duration_seconds=info.frames / info.samplerate, created_at=utc_now(),
    )


def analyze_range(
    asset: Asset,
    plan: DetectionPlan,
    run_id: UUID,
    start_sample: int,
    end_sample: int,
    providers: ProviderRegistry,
    maximum_candidates: int,
) -> tuple[list[DetectionCandidate], list[str]]:
    end_sample = min(end_sample, asset.samples)
    if end_sample <= start_sample:
        raise ValueError("analysis range is empty")
    data, sample_rate = sf.read(asset.path, start=start_sample, stop=end_sample, dtype="float32", always_2d=True)
    audio = np.mean(data, axis=1, dtype=np.float32)
    if not len(audio):
        raise ValueError("analysis range contains no audio")

    labels_by_family: dict[str, list[DetectionLabelPlan]] = {}
    for label in plan.labels:
        labels_by_family.setdefault(label.family, []).append(label)
    candidates: list[DetectionCandidate] = []
    warnings: list[str] = []

    measured = {
        "pitch": _analyze_pitch,
        "harmony": _analyze_harmony,
        "form": _analyze_form,
        "rhythm": _analyze_rhythm,
        "dynamics": _analyze_dynamics,
        "timbre": _analyze_timbre,
    }
    for family, analyzer in measured.items():
        if family not in labels_by_family:
            continue
        try:
            events = analyzer(audio, sample_rate)
            candidates.extend(_events_to_candidates(events, labels_by_family[family], run_id, asset.id, start_sample))
        except Exception as exc:
            warnings.append(f"{family} analyzer failed: {type(exc).__name__}: {exc}")

    semantic_labels = [label for label in plan.labels if label.family in {"instrument", "technique", "voice", "affect", "production", "custom"}]
    semantic_groups: dict[tuple[str, float, float], list[DetectionLabelPlan]] = {}
    for label in semantic_labels:
        detector = next((detector for detector in label.detectors if detector.mode == "zero-shot-text"), None)
        if detector:
            semantic_groups.setdefault((detector.provider_id, label.window.context_seconds, label.window.hop_seconds), []).append(label)
    for labels in semantic_groups.values():
        try:
            candidates.extend(_semantic_candidates(audio, sample_rate, start_sample, asset.id, run_id, labels, providers))
        except Exception as exc:
            warnings.append(f"{', '.join(label.display_name for label in labels)} abstained: {type(exc).__name__}: {exc}")

    candidates.sort(key=lambda candidate: (candidate.start_sample, candidate.end_sample, candidate.label_id))
    if len(candidates) > maximum_candidates:
        warnings.append(f"Candidate limit reached; returned {maximum_candidates} of {len(candidates)} results")
        candidates = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)[:maximum_candidates]
        candidates.sort(key=lambda candidate: (candidate.start_sample, candidate.label_id))
    return candidates, warnings


def _events_to_candidates(events: list[dict], labels: list[DetectionLabelPlan], run_id: UUID, asset_id: UUID, offset: int) -> list[DetectionCandidate]:
    results: list[DetectionCandidate] = []
    for event in events:
        event_label = str(event.pop("label"))
        label = _target_for_event(event_label, labels)
        score = float(np.clip(event.pop("score", 0.5), 0, 1))
        local_start = int(event.pop("start"))
        local_end = int(event.pop("end"))
        start = offset + local_start
        end = offset + max(local_end, local_start + 1)
        decision = "review" if score >= label.thresholds.review else "abstain"
        results.append(DetectionCandidate(
            id=uuid4(), run_id=run_id, asset_id=asset_id, start_sample=start, end_sample=end,
            label_id=event_label, score=score, uncertainty=1 - score, decision=decision,
            evidence={"targetLabelId": label.label_id, "detector": "librosa", "scoreKind": "heuristic-not-calibrated", "detectorScores": {"librosa": score}, "descriptors": event, "constraints": [], "warnings": []},
        ))
    return results


def _target_for_event(event_label: str, labels: list[DetectionLabelPlan]) -> DetectionLabelPlan:
    if event_label.startswith("harmony.key"):
        return next((label for label in labels if label.label_id == "harmony.key"), labels[0])
    if event_label.startswith("harmony.chord"):
        return next((label for label in labels if label.label_id == "harmony.chords"), labels[0])
    return labels[0]


def _analyze_pitch(audio: np.ndarray, sample_rate: int) -> list[dict]:
    frame_length = min(2048, max(512, 2 ** int(np.floor(np.log2(len(audio))))))
    if frame_length < 512:
        return []
    f0 = librosa.yin(audio, fmin=librosa.note_to_hz("C1"), fmax=min(librosa.note_to_hz("C8"), sample_rate * 0.45), sr=sample_rate, frame_length=frame_length, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=HOP_LENGTH, center=True)[0][:len(f0)]
    voiced = np.isfinite(f0) & (rms > max(float(np.percentile(rms, 20)), 1e-4))
    midi = np.rint(librosa.hz_to_midi(np.maximum(f0, 1))).astype(int)
    events: list[dict] = []
    index = 0
    while index < len(midi):
        if not voiced[index]:
            index += 1
            continue
        end = index + 1
        while end < len(midi) and voiced[end] and abs(midi[end] - midi[index]) <= 1:
            end += 1
        if end - index >= 2:
            note = int(np.median(midi[index:end]))
            hz = float(np.median(f0[index:end]))
            start_sample = int(librosa.frames_to_samples(index, hop_length=HOP_LENGTH))
            end_sample = min(len(audio), int(librosa.frames_to_samples(end, hop_length=HOP_LENGTH)))
            stability = float(np.exp(-np.std(librosa.hz_to_midi(f0[index:end]))))
            events.append({"start": start_sample, "end": end_sample, "label": f"pitch.note.{librosa.midi_to_note(note, unicode=False)}", "score": 0.55 + 0.35 * stability, "medianHz": hz, "midi": note})
        index = end
    return events


def _analyze_harmony(audio: np.ndarray, sample_rate: int) -> list[dict]:
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate, hop_length=HOP_LENGTH)
    mean = np.mean(chroma, axis=1)
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    profiles = [(root, mode, np.roll(profile, root)) for mode, profile in (("major", major), ("minor", minor)) for root in range(12)]
    correlations = [float(np.corrcoef(mean, profile)[0, 1]) for _, _, profile in profiles]
    best = int(np.nanargmax(correlations))
    root, mode, _ = profiles[best]
    confidence = float(np.clip((correlations[best] + 1) / 2, 0, 1))
    events = [{"start": 0, "end": len(audio), "label": f"harmony.key.{PITCH_CLASSES[root]}.{mode}", "score": confidence, "keyCorrelation": correlations[best]}]

    templates: list[tuple[int, str, np.ndarray]] = []
    for chord_root in range(12):
        for quality, thirds in (("maj", (0, 4, 7)), ("min", (0, 3, 7))):
            template = np.zeros(12)
            template[[(chord_root + interval) % 12 for interval in thirds]] = 1
            templates.append((chord_root, quality, template))
    normalized = chroma / np.maximum(np.linalg.norm(chroma, axis=0, keepdims=True), 1e-8)
    scores = np.stack([template @ normalized / np.linalg.norm(template) for _, _, template in templates])
    chord_indexes = np.argmax(scores, axis=0)
    index = 0
    while index < len(chord_indexes):
        end = index + 1
        while end < len(chord_indexes) and chord_indexes[end] == chord_indexes[index]:
            end += 1
        if (end - index) * HOP_LENGTH / sample_rate >= 0.35:
            chord_root, quality, _ = templates[int(chord_indexes[index])]
            chord_score = float(np.mean(scores[chord_indexes[index], index:end]))
            events.append({"start": index * HOP_LENGTH, "end": min(len(audio), end * HOP_LENGTH), "label": f"harmony.chord.{PITCH_CLASSES[chord_root]}.{quality}", "score": np.clip(chord_score, 0, 1), "templateSimilarity": chord_score})
        index = end
    return events


def _analyze_form(audio: np.ndarray, sample_rate: int) -> list[dict]:
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate, hop_length=HOP_LENGTH)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=8, hop_length=HOP_LENGTH)
    features = np.vstack((librosa.util.normalize(chroma, axis=1), librosa.util.normalize(mfcc, axis=1)))
    delta = np.linalg.norm(np.diff(features, axis=1), axis=0)
    smooth_frames = max(1, round(sample_rate / HOP_LENGTH))
    novelty = np.convolve(delta, np.ones(smooth_frames) / smooth_frames, mode="same")
    distance = max(1, round(3 * sample_rate / HOP_LENGTH))
    prominence = max(float(np.percentile(novelty, 65)), 1e-5)
    peaks, _ = find_peaks(novelty, distance=distance, prominence=prominence)
    boundaries = [0, *[int((peak + 1) * HOP_LENGTH) for peak in peaks], len(audio)]
    events: list[dict] = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), 1):
        strength = 1.0 if start == 0 else float(np.clip(novelty[max(0, start // HOP_LENGTH - 1)] / (np.max(novelty) + 1e-8), 0, 1))
        events.append({"start": start, "end": end, "label": f"form.section.{index}", "score": 0.55 + strength * 0.3, "boundaryStrength": strength})
    return events


def _analyze_rhythm(audio: np.ndarray, sample_rate: int) -> list[dict]:
    tempo, beats = librosa.beat.beat_track(y=audio, sr=sample_rate, hop_length=HOP_LENGTH)
    bpm = float(np.asarray(tempo).reshape(-1)[0])
    events = [{"start": 0, "end": len(audio), "label": f"rhythm.tempo.{round(bpm)}bpm", "score": 0.65, "bpm": bpm}]
    for beat in beats:
        sample = int(librosa.frames_to_samples(beat, hop_length=HOP_LENGTH))
        events.append({"start": sample, "end": min(len(audio), sample + max(1, HOP_LENGTH // 4)), "label": "rhythm.beat", "score": 0.7, "beatFrame": int(beat)})
    return events


def _analyze_dynamics(audio: np.ndarray, sample_rate: int) -> list[dict]:
    rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)[0]
    db = librosa.amplitude_to_db(np.maximum(rms, 1e-8), ref=np.max)
    return _change_events(db, len(audio), sample_rate, "dynamics", "increase", "decrease")


def _analyze_timbre(audio: np.ndarray, sample_rate: int) -> list[dict]:
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate, hop_length=HOP_LENGTH)[0]
    normalized = (centroid - np.median(centroid)) / (np.std(centroid) + 1e-8)
    return _change_events(normalized, len(audio), sample_rate, "timbre", "brighter", "darker")


def _change_events(values: np.ndarray, audio_length: int, sample_rate: int, family: str, positive: str, negative: str) -> list[dict]:
    smooth_frames = max(1, round(0.5 * sample_rate / HOP_LENGTH))
    smooth = np.convolve(values, np.ones(smooth_frames) / smooth_frames, mode="same")
    derivative = np.diff(smooth)
    threshold = max(float(np.percentile(np.abs(derivative), 85)), 1e-6)
    peaks, _ = find_peaks(np.abs(derivative), height=threshold, distance=max(1, round(sample_rate / HOP_LENGTH)))
    events = []
    for peak in peaks:
        center = int((peak + 1) * HOP_LENGTH)
        radius = round(0.5 * sample_rate)
        direction = positive if derivative[peak] >= 0 else negative
        score = float(np.clip(abs(derivative[peak]) / (np.max(np.abs(derivative)) + 1e-8), 0, 1))
        events.append({"start": max(0, center - radius), "end": min(audio_length, center + radius), "label": f"{family}.{direction}", "score": 0.5 + score * 0.4, "change": float(derivative[peak])})
    return events


def _semantic_candidates(audio: np.ndarray, sample_rate: int, offset: int, asset_id: UUID, run_id: UUID, labels: list[DetectionLabelPlan], providers: ProviderRegistry) -> list[DetectionCandidate]:
    label = labels[0]
    detector = next(detector for detector in label.detectors if detector.mode == "zero-shot-text")
    provider = providers.get(detector.provider_id)
    if provider.model is None:
        raise RuntimeError(f"provider {provider.provider_id} is not loaded")
    prompts: list[str] = []
    prompt_ranges: list[tuple[int, int, int]] = []
    for target in labels:
        if not target.prompts.positive:
            raise ValueError(f"{target.display_name} has no positive prompts")
        start = len(prompts)
        prompts.extend(target.prompts.positive)
        positive_end = len(prompts)
        prompts.extend(target.prompts.negative)
        prompt_ranges.append((start, positive_end, len(prompts)))
    context_samples = max(1, round(label.window.context_seconds * sample_rate))
    hop_samples = max(1, round(label.window.hop_seconds * sample_rate))
    starts = list(range(0, max(1, len(audio) - context_samples + 1), hop_samples)) or [0]
    if starts[-1] + context_samples < len(audio):
        starts.append(max(0, len(audio) - context_samples))
    windows = np.zeros((len(starts), context_samples), dtype=np.float32)
    for index, start in enumerate(starts):
        clip = audio[start:start + context_samples]
        windows[index, :len(clip)] = clip
    if sample_rate != provider.sample_rate:
        windows = np.stack([librosa.resample(window, orig_sr=sample_rate, target_sr=provider.sample_rate) for window in windows])
    scores = provider.score_audio_text(windows, prompts)
    results: list[DetectionCandidate] = []
    for target, (prompt_start, positive_end, prompt_end) in zip(labels, prompt_ranges):
        for index, start in enumerate(starts):
            positives = scores[index, prompt_start:positive_end]
            negatives = scores[index, positive_end:prompt_end]
            positive_score = float(np.mean(positives))
            negative_score = float(np.max(negatives)) if len(negatives) else 0.5
            margin = positive_score - negative_score
            compatibility = float(np.clip(0.5 + margin, 0, 1))
            decision = "review" if compatibility >= target.thresholds.review else "abstain"
            end = min(len(audio), start + context_samples)
            results.append(DetectionCandidate(
                id=uuid4(), run_id=run_id, asset_id=asset_id, start_sample=offset + start, end_sample=offset + end,
                label_id=target.label_id, score=compatibility, uncertainty=1 - abs(margin), decision=decision,
                evidence={"targetLabelId": target.label_id, "detector": detector.provider_id, "scoreKind": "prompt-margin-not-calibrated", "detectorScores": {detector.provider_id: compatibility}, "descriptors": {"positiveMean": positive_score, "negativeMax": negative_score, "margin": margin}, "constraints": [], "warnings": ["Prompt margin is not a calibrated probability"], "prompts": prompts[prompt_start:prompt_end]},
            ))
    return results
