from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from uuid import uuid4

from .schemas import (
    DetectionCompiler, DetectionLabelPlan, DetectionPlan, DetectionPlanCompileRequest,
    DetectionPlanValidation, DetectionTarget, DetectionThresholds, DetectionWindow,
    DetectorSpec, FeatureExtractorSpec, PromptSet, ProviderCapability, ValidationFinding,
)


CATALOGUE: tuple[DetectionTarget, ...] = (
    DetectionTarget(label_id="instrument.guqin", name="古琴", family="instrument", positive_prompts=["solo guqin", "中国古琴独奏", "a seven-string Chinese guqin"], negative_prompts=["guzheng", "古筝", "other plucked string instrument"]),
    DetectionTarget(label_id="instrument.piano", name="钢琴", family="instrument", positive_prompts=["solo piano", "钢琴演奏"], negative_prompts=["harpsichord", "guitar"]),
    DetectionTarget(label_id="instrument.violin", name="小提琴", family="instrument", positive_prompts=["solo violin", "小提琴演奏"], negative_prompts=["viola", "erhu"]),
    DetectionTarget(label_id="instrument.erhu", name="二胡", family="instrument", positive_prompts=["solo erhu", "二胡演奏"], negative_prompts=["violin", "cello"]),
    DetectionTarget(label_id="instrument.pipa", name="琵琶", family="instrument", positive_prompts=["solo pipa", "中国琵琶演奏"], negative_prompts=["guitar", "guqin"]),
    DetectionTarget(label_id="instrument.flute", name="笛/长笛", family="instrument", positive_prompts=["solo flute", "竹笛或长笛演奏"], negative_prompts=["recorder", "voice"]),
    DetectionTarget(label_id="instrument.guitar", name="吉他", family="instrument", positive_prompts=["acoustic or electric guitar", "吉他演奏"], negative_prompts=["pipa", "harp"]),
    DetectionTarget(label_id="instrument.drums", name="鼓/打击乐", family="instrument", positive_prompts=["drums and percussion", "鼓或打击乐"], negative_prompts=["plucked strings"]),
    DetectionTarget(label_id="voice.singing", name="人声/歌唱", family="voice", positive_prompts=["a person singing", "歌唱人声"], negative_prompts=["instrumental music", "spoken voice"]),
    DetectionTarget(label_id="technique.harmonic", name="泛音", family="technique", positive_prompts=["a clear natural or artificial harmonic", "清晰的泛音演奏", "bell-like overtone-rich harmonic tone"], negative_prompts=["ordinary stopped tone", "普通按音"]),
    DetectionTarget(label_id="technique.glissando", name="滑音/滑奏", family="technique", positive_prompts=["audible pitch slide or glissando", "连续滑音或滑奏"], negative_prompts=["discrete scale notes", "没有滑音的分离音符"]),
    DetectionTarget(label_id="technique.vibrato", name="颤音/揉弦", family="technique", positive_prompts=["expressive pitch vibrato", "持续的颤音或揉弦"], negative_prompts=["steady pitch without vibrato"]),
    DetectionTarget(label_id="technique.pizzicato", name="拨弦", family="technique", positive_prompts=["a plucked string articulation", "清楚的拨弦发音"], negative_prompts=["bowed sustained strings"]),
    DetectionTarget(label_id="technique.tremolo", name="震音/轮指", family="technique", positive_prompts=["rapid repeated tremolo articulation", "快速重复的震音或轮指"], negative_prompts=["single isolated attack"]),
    DetectionTarget(label_id="technique.staccato", name="断奏", family="technique", positive_prompts=["short detached staccato notes", "短促分离的断奏"], negative_prompts=["smooth legato"]),
    DetectionTarget(label_id="technique.legato", name="连奏", family="technique", positive_prompts=["smooth connected legato phrase", "平滑连接的连奏"], negative_prompts=["detached staccato"]),
)

ALIASES: dict[str, tuple[str, ...]] = {
    "instrument.guqin": ("古琴", "guqin"), "instrument.piano": ("钢琴", "piano"),
    "instrument.violin": ("小提琴", "violin"), "instrument.erhu": ("二胡", "erhu"),
    "instrument.pipa": ("琵琶", "pipa"), "instrument.flute": ("笛子", "竹笛", "长笛", "flute"),
    "instrument.guitar": ("吉他", "guitar"), "instrument.drums": ("鼓", "打击乐", "drum", "percussion"),
    "voice.singing": ("人声", "歌唱", "演唱", "voice", "singing"),
    "technique.harmonic": ("泛音", "harmonic"), "technique.glissando": ("滑音", "滑奏", "glissando", "portamento"),
    "technique.vibrato": ("颤音", "揉弦", "吟", "猱", "vibrato"), "technique.pizzicato": ("拨弦", "pizzicato"),
    "technique.tremolo": ("震音", "轮指", "tremolo"), "technique.staccato": ("断奏", "staccato"),
    "technique.legato": ("连奏", "legato"),
}


MEASURED_TARGETS: dict[str, DetectionTarget] = {
    "pitch": DetectionTarget(label_id="pitch.notes", name="音高与音符", family="pitch"),
    "harmony.key": DetectionTarget(label_id="harmony.key", name="调性", family="harmony"),
    "harmony.chords": DetectionTarget(label_id="harmony.chords", name="和弦", family="harmony"),
    "form": DetectionTarget(label_id="form.sections", name="曲式段落与边界", family="form"),
    "rhythm": DetectionTarget(label_id="rhythm.tempo", name="速度与节拍", family="rhythm"),
    "dynamics": DetectionTarget(label_id="dynamics.changes", name="力度变化", family="dynamics"),
    "timbre": DetectionTarget(label_id="timbre.changes", name="音色变化", family="timbre"),
}


def compile_plan(request: DetectionPlanCompileRequest, providers: list[ProviderCapability]) -> DetectionPlan:
    instruction = request.instruction.strip()
    lower = instruction.casefold()
    targets: dict[str, DetectionTarget] = {target.label_id: target for target in request.targets}

    for item in CATALOGUE:
        if any(alias.casefold() in lower for alias in ALIASES.get(item.label_id, ())):
            targets.setdefault(item.label_id, item)

    general_instruments = any(term in lower for term in ("乐器", "配器", "instrument", "instrumentation"))
    general_techniques = any(term in lower for term in ("演奏法", "奏法", "技法", "technique", "articulation"))
    if general_instruments and not any(target.family in {"instrument", "voice"} for target in targets.values()):
        for item in CATALOGUE:
            if item.family in {"instrument", "voice"}:
                targets[item.label_id] = item
    if general_techniques and not any(target.family == "technique" for target in targets.values()):
        for item in CATALOGUE:
            if item.family == "technique":
                targets[item.label_id] = item

    measured_terms = {
        "pitch": ("音高", "音符", "旋律", "pitch", "note", "melody"),
        "harmony.key": ("调性", "调式", "key", "tonality", "mode"),
        "harmony.chords": ("和声", "和弦", "harmony", "chord"),
        "form": ("曲式", "结构", "段落", "乐句", "form", "section", "phrase", "structure"),
        "rhythm": ("速度", "节拍", "节奏", "tempo", "beat", "rhythm"),
        "dynamics": ("力度", "响度", "强弱", "dynamic", "loudness"),
        "timbre": ("音色", "明亮", "频谱", "timbre", "brightness", "spectral"),
    }
    for key, terms in measured_terms.items():
        if any(term in lower for term in terms):
            target = MEASURED_TARGETS[key]
            targets.setdefault(target.label_id, target)

    if not targets:
        request_is_detection = any(term in lower for term in ("识别", "检测", "标注", "找出", "detect", "annotate", "find"))
        if request_is_detection:
            digest = hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:12]
            target = DetectionTarget(
                label_id=f"custom.request.{digest}", name=instruction[:120], family="custom",
                positive_prompts=[instruction, f"a music excerpt that matches this request: {instruction}"],
                negative_prompts=[f"a music excerpt that does not match this request: {instruction}"],
            )
            targets[target.label_id] = target
        else:
            for key in ("pitch", "harmony.key", "harmony.chords", "form", "rhythm", "dynamics", "timbre"):
                target = MEASURED_TARGETS[key]
                targets[target.label_id] = target

    semantic_provider = _choose_semantic_provider(request, providers)
    labels = [_label_plan(target, semantic_provider) for target in targets.values()]
    outputs = sorted({label.family for label in labels if label.detectors[0].mode == "descriptor-rule"})
    extractors = [FeatureExtractorSpec(provider_id="librosa", outputs=outputs, config={"hopLength": 512})] if outputs else []
    return DetectionPlan(
        id=uuid4(), revision=1, name=instruction[:80], status="draft", source_instruction=instruction,
        taxonomy_version=request.taxonomy_version, preferred_languages=request.preferred_languages,
        labels=labels, feature_extractors=extractors,
        compiler=DetectionCompiler(kind="deterministic", prompt_template_version="builtin-musicology-v1", compiled_at=datetime.now(timezone.utc)),
    )


def validate_plan(plan: DetectionPlan, providers: list[ProviderCapability]) -> DetectionPlanValidation:
    capabilities = {provider.provider_id: provider for provider in providers}
    errors: list[ValidationFinding] = []
    warnings: list[ValidationFinding] = []
    runnable: list[str] = []
    for label in plan.labels:
        usable = False
        for detector in label.detectors:
            if detector.provider_id == "librosa":
                usable = True
                continue
            provider = capabilities.get(detector.provider_id)
            if provider is None:
                finding = ValidationFinding(code="provider.unknown", message=f"Unknown provider {detector.provider_id}", label_id=label.label_id)
                (errors if detector.required else warnings).append(finding)
                continue
            if not provider.installed:
                finding = ValidationFinding(code="provider.not_installed", message=f"{provider.display_name} is not installed", label_id=label.label_id)
                (errors if detector.required else warnings).append(finding)
            elif detector.mode == "zero-shot-text" and not provider.supports_text_embeddings:
                errors.append(ValidationFinding(code="provider.no_text", message=f"{provider.display_name} cannot compare text prompts", label_id=label.label_id))
            elif not provider.loaded:
                warnings.append(ValidationFinding(code="provider.not_loaded", message=f"Load {provider.display_name} before previewing {label.display_name}", label_id=label.label_id))
            else:
                usable = True
        if usable:
            runnable.append(label.label_id)
        elif plan.fusion.missing_analyzer_policy == "fail":
            errors.append(ValidationFinding(code="label.unrunnable", message=f"No runnable detector for {label.display_name}", label_id=label.label_id))
        else:
            warnings.append(ValidationFinding(code="label.will_abstain", message=f"{label.display_name} will abstain until a detector is available", label_id=label.label_id))
        if label.thresholds.auto_accept is not None and label.label_id not in plan.policy.allow_auto_accept_labels:
            errors.append(ValidationFinding(code="policy.auto_accept_locked", message="Auto-accept requires an explicitly allowed and calibrated label", label_id=label.label_id))
    return DetectionPlanValidation(valid=not errors, errors=errors, warnings=warnings, runnable_labels=runnable)


def _choose_semantic_provider(request: DetectionPlanCompileRequest, providers: list[ProviderCapability]) -> str:
    allowed = set(request.available_provider_ids) if request.available_provider_ids else {provider.provider_id for provider in providers}
    by_id = {provider.provider_id: provider for provider in providers}
    for provider_id in ("muq_mulan_large", "laion_clap_music_htsat_base", "mock"):
        provider = by_id.get(provider_id)
        if provider_id in allowed and provider and provider.loaded and provider.supports_text_embeddings:
            return provider_id
    for provider_id in ("muq_mulan_large", "laion_clap_music_htsat_base", "mock"):
        provider = by_id.get(provider_id)
        if provider_id in allowed and provider and provider.supports_text_embeddings:
            return provider_id
    return "mock"


def _label_plan(target: DetectionTarget, semantic_provider: str) -> DetectionLabelPlan:
    measured = target.family in {"pitch", "rhythm", "dynamics", "timbre", "harmony", "form"}
    if measured:
        detector = DetectorSpec(mode="descriptor-rule", provider_id="librosa", weight=1.0)
    else:
        detector = DetectorSpec(mode="zero-shot-text", provider_id=semantic_provider, weight=1.0)
    granularity = "note" if target.family == "pitch" else "event" if target.family == "technique" else "segment"
    context = 2.0 if target.family == "pitch" else 4.0 if target.family == "technique" else 10.0 if target.family == "form" else 6.0
    hop = 0.25 if target.family == "pitch" else 1.0 if target.family == "technique" else 3.0
    return DetectionLabelPlan(
        label_id=target.label_id, display_name=target.name, family=target.family, granularity=granularity,
        detectors=[detector], window=DetectionWindow(context_seconds=context, hop_seconds=hop, minimum_event_ms=80, merge_gap_ms=120),
        prompts=PromptSet(positive=target.positive_prompts, negative=target.negative_prompts),
        thresholds=DetectionThresholds(review=0.55, auto_accept=None, abstain_below=0.35),
    )
