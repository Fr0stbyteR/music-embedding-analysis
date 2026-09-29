from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import threading

import librosa
import numpy as np
import soundfile as sf

from .planner import CATALOGUE
from .providers import Provider, ProviderRegistry
from .schemas import (
    Asset, InteractiveDescribeRequest, InteractiveDescriptionResult, RawSemanticMatch,
    SemanticCurvePoint, SemanticCurveRequest, SemanticCurveResult, SemanticDescription,
)


CURVE_CACHE_VERSION = "clap-relevance-v1"
_CACHE_LOCKS: dict[str, threading.Lock] = {}
_CACHE_LOCKS_GUARD = threading.Lock()


def _cache_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _CACHE_LOCKS_GUARD:
        return _CACHE_LOCKS.setdefault(key, threading.Lock())


@dataclass(frozen=True)
class DescriptionPrompt:
    label_id: str
    text: str
    family: str
    prompts: tuple[str, ...]


EXTRA_PROMPTS: tuple[DescriptionPrompt, ...] = (
    DescriptionPrompt("texture.solo", "独奏", "texture", ("solo music", "a single instrument playing")),
    DescriptionPrompt("texture.ensemble", "合奏", "texture", ("music played by an ensemble", "multiple instruments playing together")),
    DescriptionPrompt("texture.sparse", "稀疏织体", "texture", ("sparse musical texture", "few notes with plenty of space")),
    DescriptionPrompt("texture.dense", "密集织体", "texture", ("dense layered musical texture", "many simultaneous musical layers")),
    DescriptionPrompt("affect.calm", "平静", "affect", ("calm peaceful music", "安静平和的音乐")),
    DescriptionPrompt("affect.energetic", "有活力", "affect", ("energetic exciting music", "充满活力的音乐")),
    DescriptionPrompt("affect.dark", "阴暗", "affect", ("dark tense music", "阴暗紧张的音乐")),
    DescriptionPrompt("affect.bright", "明亮", "affect", ("bright cheerful music", "明亮愉快的音乐")),
    DescriptionPrompt("production.acoustic", "原声", "production", ("acoustic natural recording", "unprocessed acoustic music")),
    DescriptionPrompt("production.electronic", "电子", "production", ("electronic synthesized music", "music with electronic synthesizers")),
    DescriptionPrompt("rhythm.slow", "舒缓", "rhythm", ("slow relaxed music",)),
    DescriptionPrompt("rhythm.fast", "快速", "rhythm", ("fast driving music",)),
    DescriptionPrompt("genre.classical", "古典", "genre", ("classical music", "Western classical music")),
    DescriptionPrompt("genre.traditional", "传统民乐", "genre", ("traditional Chinese music", "Chinese folk instrumental music")),
    DescriptionPrompt("genre.jazz", "爵士", "genre", ("jazz music",)),
    DescriptionPrompt("genre.rock", "摇滚", "genre", ("rock music",)),
    DescriptionPrompt("genre.electronic", "电子音乐", "genre", ("electronic music",)),
)


def description_prompts() -> tuple[DescriptionPrompt, ...]:
    catalogue = tuple(
        DescriptionPrompt(item.label_id, item.name, item.family, tuple(item.positive_prompts))
        for item in CATALOGUE if item.positive_prompts
    )
    return catalogue + EXTRA_PROMPTS


def describe_asset(asset: Asset, request: InteractiveDescribeRequest, providers: ProviderRegistry) -> InteractiveDescriptionResult:
    provider = _choose_provider(request.provider_id, providers)
    scale = asset.duration_seconds / request.timeline_duration_seconds if request.timeline_duration_seconds else 1.0
    requested_start = request.start_seconds * scale
    requested_end = request.end_seconds * scale
    requested_length = max(0.001, requested_end - requested_start)
    end = min(requested_end, asset.duration_seconds)
    start = min(requested_start, max(0.0, end - min(requested_length, asset.duration_seconds)))
    if end <= start:
        raise ValueError("analysis range is outside the audio asset")
    waveforms = _read_windows(asset, start, end, provider.sample_rate)
    definitions = description_prompts()
    prompts = [prompt for item in definitions for prompt in item.prompts]
    # Long selections are represented by a bounded set of 10-second observations.
    # Mean aggregation describes the range as a whole instead of reporting one peak.
    prompt_scores = np.mean(provider.score_audio_text(waveforms, prompts), axis=0)
    descriptions: list[SemanticDescription] = []
    raw_matches: list[RawSemanticMatch] = []
    offset = 0
    for item in definitions:
        count = len(item.prompts)
        item_scores = prompt_scores[offset:offset + count]
        score = float(np.max(item_scores))
        raw_matches.extend(
            RawSemanticMatch(
                prompt=prompt, label_id=item.label_id, family=item.family,
                score=float(prompt_score), cosine_similarity=float(prompt_score * 2 - 1),
            )
            for prompt, prompt_score in zip(item.prompts, item_scores, strict=True)
        )
        offset += count
        descriptions.append(SemanticDescription(label_id=item.label_id, text=item.text, family=item.family, score=score))
    descriptions.sort(key=lambda item: item.score, reverse=True)
    descriptions = descriptions[:request.maximum_results]
    raw_matches.sort(key=lambda item: item.score, reverse=True)
    summary_items: list[str] = []
    seen_families: set[str] = set()
    for item in descriptions:
        if item.family in seen_families:
            continue
        summary_items.append(item.text)
        seen_families.add(item.family)
        if len(summary_items) == 3:
            break
    return InteractiveDescriptionResult(
        asset_id=asset.id, start_seconds=start, end_seconds=end,
        provider_id=provider.provider_id, provider_name=provider.display_name,
        summary=" · ".join(summary_items) or "没有匹配的描述", descriptions=descriptions,
        raw_matches=raw_matches,
    )


def _read_windows(asset: Asset, start: float, end: float, target_rate: int) -> np.ndarray:
    selection_seconds = end - start
    context_seconds = min(10.0, max(1.0, selection_seconds))
    if selection_seconds <= context_seconds:
        starts = np.array([start])
    else:
        count = min(12, max(2, int(np.ceil(selection_seconds / context_seconds))))
        starts = np.linspace(start, end - context_seconds, count)
    target_samples = round(context_seconds * target_rate)
    windows: list[np.ndarray] = []
    for window_start in starts:
        data, sample_rate = sf.read(
            Path(asset.path), start=round(float(window_start) * asset.sample_rate),
            stop=round(float(window_start + context_seconds) * asset.sample_rate),
            dtype="float32", always_2d=True,
        )
        audio = np.mean(data, axis=1, dtype=np.float32)
        if sample_rate != target_rate:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_rate)
        if len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))
        windows.append(np.asarray(audio[:target_samples], dtype=np.float32))
    return np.stack(windows)


def relevance_curve(
    asset: Asset, request: SemanticCurveRequest, providers: ProviderRegistry, cache_root: Path,
) -> SemanticCurveResult:
    provider = _choose_provider(request.provider_id, providers)
    prompts = request.prompts or [request.keyword]
    cache_descriptor = json.dumps({
        "version": CURVE_CACHE_VERSION, "asset": asset.content_hash, "provider": provider.provider_id,
        "keyword": request.keyword, "prompts": prompts, "timelineDuration": request.timeline_duration_seconds,
        "window": request.window_seconds, "hop": request.hop_seconds, "aggregation": request.aggregation,
    }, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"{hashlib.sha256(cache_descriptor.encode('utf-8')).hexdigest()}.json"
    if request.cache_policy != "refresh" and cache_path.exists():
        cached = SemanticCurveResult.model_validate_json(cache_path.read_text(encoding="utf-8"))
        return cached.model_copy(update={"cached": True})

    context_seconds = min(request.window_seconds, max(1.0, asset.duration_seconds))
    if asset.duration_seconds <= context_seconds:
        starts = np.array([0.0], dtype=np.float64)
    else:
        starts = np.arange(0.0, asset.duration_seconds - context_seconds + request.hop_seconds * 0.001, request.hop_seconds)
    batch_size = 24
    supports_embedding_cache = type(provider).embed_audio_for_text is not Provider.embed_audio_for_text
    if supports_embedding_cache:
        embedding_descriptor = json.dumps({
            "version": CURVE_CACHE_VERSION, "asset": asset.content_hash, "provider": provider.provider_id,
            "window": request.window_seconds, "hop": request.hop_seconds,
        }, sort_keys=True, separators=(",", ":"))
        embedding_root = cache_root / "audio-embeddings"
        embedding_root.mkdir(parents=True, exist_ok=True)
        embedding_path = embedding_root / f"{hashlib.sha256(embedding_descriptor.encode('utf-8')).hexdigest()}.npy"
        with _cache_lock(embedding_path):
            if request.cache_policy != "refresh" and embedding_path.exists():
                audio_embeddings = np.load(embedding_path, allow_pickle=False)
            else:
                batches: list[np.ndarray] = []
                for offset in range(0, len(starts), batch_size):
                    windows = np.stack([
                        _read_window(asset, float(start), context_seconds, provider.sample_rate)
                        for start in starts[offset:offset + batch_size]
                    ])
                    batches.append(provider.embed_audio_for_text(windows))
                audio_embeddings = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
                temporary_embeddings = embedding_path.with_suffix(".tmp")
                with temporary_embeddings.open("wb") as file:
                    np.save(file, audio_embeddings, allow_pickle=False)
                temporary_embeddings.replace(embedding_path)
        prompt_scores = provider.score_audio_embeddings_text(audio_embeddings, prompts)
    else:
        score_batches: list[np.ndarray] = []
        for offset in range(0, len(starts), batch_size):
            windows = np.stack([
                _read_window(asset, float(start), context_seconds, provider.sample_rate)
                for start in starts[offset:offset + batch_size]
            ])
            score_batches.append(provider.score_audio_text(windows, prompts))
        prompt_scores = np.concatenate(score_batches, axis=0)
    aggregated = np.mean(prompt_scores, axis=1) if request.aggregation == "mean" else np.max(prompt_scores, axis=1)
    mapped_scores = [float(score) for score in aggregated]

    timeline_scale = request.timeline_duration_seconds / asset.duration_seconds if request.timeline_duration_seconds else 1.0
    points = [
        SemanticCurvePoint(
            time_seconds=float(min(asset.duration_seconds, start + context_seconds / 2) * timeline_scale),
            score=score, cosine_similarity=float(score * 2 - 1),
        )
        for start, score in zip(starts, mapped_scores, strict=True)
    ]
    result = SemanticCurveResult(
        asset_id=asset.id, keyword=request.keyword, prompts=prompts,
        provider_id=provider.provider_id, provider_name=provider.display_name,
        window_seconds=request.window_seconds, hop_seconds=request.hop_seconds,
        aggregation=request.aggregation, points=points,
    )
    temporary = cache_path.with_suffix(f".{threading.get_ident()}.tmp")
    temporary.write_text(result.model_dump_json(by_alias=True), encoding="utf-8")
    temporary.replace(cache_path)
    return result


def _read_window(asset: Asset, start: float, seconds: float, target_rate: int) -> np.ndarray:
    data, sample_rate = sf.read(
        Path(asset.path), start=round(start * asset.sample_rate),
        stop=round((start + seconds) * asset.sample_rate), dtype="float32", always_2d=True,
    )
    audio = np.mean(data, axis=1, dtype=np.float32)
    if sample_rate != target_rate:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_rate)
    target_samples = round(seconds * target_rate)
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))
    return np.asarray(audio[:target_samples], dtype=np.float32)


def _choose_provider(provider_id: str | None, providers: ProviderRegistry):
    if provider_id:
        try:
            provider = providers.get(provider_id)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        if provider.model is None:
            raise RuntimeError(f"provider {provider_id} is not loaded")
        if not provider.supports_text_embeddings:
            raise ValueError(f"provider {provider_id} does not support text embeddings")
        return provider
    for candidate_id in ("laion_clap_music_htsat_base", "muq_mulan_large", "mock"):
        provider = providers.get(candidate_id)
        if provider.model is not None and provider.supports_text_embeddings:
            return provider
    raise RuntimeError("no loaded audio/text provider; load laion_clap_music_htsat_base or muq_mulan_large first")
