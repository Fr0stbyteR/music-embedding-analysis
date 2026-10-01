from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ApiModel(BaseModel):
    model_config = ConfigDict(alias_generator=lambda s: s.split("_")[0] + "".join(p.title() for p in s.split("_")[1:]), populate_by_name=True)


class Health(ApiModel):
    status: Literal["ok"] = "ok"
    protocol_version: str = "music-annotation/1"
    service_version: str
    librosa_engine_version: str | None = None


class ProviderCapability(ApiModel):
    provider_id: str
    display_name: str
    installed: bool
    loaded: bool
    state: Literal["not_installed", "available", "loading", "loaded", "failed"]
    supports_frame_embeddings: bool
    supports_text_embeddings: bool
    supports_training: bool
    sample_rate: int
    weight_license: str | None = None
    commercial_use: bool | None = None
    device: str | None = None
    error: str | None = None


class Capabilities(ApiModel):
    devices: list[str]
    providers: list[ProviderCapability]
    score_formats: list[str] = ["musicxml", "midi", "pdf", "image"]
    training_modes: list[str] = ["prototype", "head", "partial-backbone", "contrastive-adapter"]


class InteractiveDescribeRequest(ApiModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    timeline_duration_seconds: float | None = Field(default=None, gt=0)
    maximum_results: int = Field(default=8, ge=1, le=20)
    provider_id: str | None = None

    @model_validator(mode="after")
    def valid_range(self) -> "InteractiveDescribeRequest":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("endSeconds must be greater than startSeconds")
        return self


class SemanticDescription(ApiModel):
    label_id: str
    text: str
    family: str
    score: float = Field(ge=0, le=1)


class RawSemanticMatch(ApiModel):
    prompt: str
    label_id: str
    family: str
    score: float = Field(ge=0, le=1)
    cosine_similarity: float = Field(ge=-1, le=1)


class InteractiveDescriptionResult(ApiModel):
    asset_id: UUID
    start_seconds: float
    end_seconds: float
    provider_id: str
    provider_name: str
    summary: str
    descriptions: list[SemanticDescription]
    raw_matches: list[RawSemanticMatch]
    score_kind: Literal["cosine-similarity-not-probability"] = "cosine-similarity-not-probability"
    cached: bool = False


class SemanticCurveRequest(ApiModel):
    keyword: str = Field(min_length=1, max_length=300)
    prompts: list[str] = Field(default_factory=list, max_length=16)
    timeline_duration_seconds: float | None = Field(default=None, gt=0)
    window_seconds: float = Field(default=5.0, ge=1.0, le=30.0)
    hop_seconds: float = Field(default=0.5, ge=0.1, le=10.0)
    aggregation: Literal["mean", "max"] = "mean"
    provider_id: str | None = None
    cache_policy: Literal["use", "refresh"] = "use"

    @model_validator(mode="after")
    def valid_prompts(self) -> "SemanticCurveRequest":
        self.keyword = self.keyword.strip()
        self.prompts = list(dict.fromkeys(prompt.strip() for prompt in self.prompts if prompt.strip()))
        if not self.keyword:
            raise ValueError("keyword must not be blank")
        return self


class SemanticCurvePoint(ApiModel):
    time_seconds: float = Field(ge=0)
    score: float = Field(ge=0, le=1)
    cosine_similarity: float = Field(ge=-1, le=1)


class SemanticCurveResult(ApiModel):
    asset_id: UUID
    keyword: str
    prompts: list[str]
    provider_id: str
    provider_name: str
    window_seconds: float
    hop_seconds: float
    aggregation: Literal["mean", "max"]
    points: list[SemanticCurvePoint]
    score_kind: Literal["cosine-similarity-not-probability"] = "cosine-similarity-not-probability"
    cached: bool = False


class InteractiveLibrosaRequest(ApiModel):
    algorithm: str = Field(min_length=1, max_length=80)
    options: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    cache_policy: Literal["use", "refresh"] = "use"


class MoodCurveRequest(ApiModel):
    window_seconds: float = Field(default=6, ge=3, le=60, allow_inf_nan=False)
    hop_seconds: float = Field(default=1, ge=0.1, le=30, allow_inf_nan=False)
    timeline_duration_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    cache_policy: Literal["use", "refresh"] = "use"


class ProjectCreate(ApiModel):
    name: str = Field(min_length=1, max_length=200)
    directory: str
    description: str = Field(default="", max_length=5000)


class Project(ApiModel):
    id: UUID
    name: str
    directory: str
    description: str = ""
    active_taxonomy_version: str | None = None
    active_model_version_id: UUID | None = None
    revision: int
    created_at: datetime
    updated_at: datetime


AnnotationState = Literal["suggested", "confirmed", "rejected", "ambiguous"]
AnnotationProvenance = Literal["human", "model", "llm", "import"]


class AnnotationCreate(ApiModel):
    asset_id: UUID
    start_sample: int = Field(ge=0)
    end_sample: int = Field(ge=1)
    label_ids: list[str] = Field(min_length=1)
    state: AnnotationState
    provenance: AnnotationProvenance
    confidence: float | None = Field(default=None, ge=0, le=1)
    candidate_scores: dict[str, float] = {}
    model_version_id: UUID | None = None
    score_position: dict[str, Any] | None = None
    note: str = Field(default="", max_length=5000)

    @model_validator(mode="after")
    def valid_range(self) -> "AnnotationCreate":
        if self.end_sample <= self.start_sample:
            raise ValueError("endSample must be greater than startSample")
        if len(set(self.label_ids)) != len(self.label_ids):
            raise ValueError("labelIds must be unique")
        return self


class AnnotationPatch(ApiModel):
    start_sample: int | None = Field(default=None, ge=0)
    end_sample: int | None = Field(default=None, ge=1)
    label_ids: list[str] | None = None
    state: AnnotationState | None = None
    note: str | None = Field(default=None, max_length=5000)


class Annotation(AnnotationCreate):
    id: UUID
    revision: int
    created_at: datetime
    updated_at: datetime


class ProviderLoadRequest(ApiModel):
    device: Literal["auto", "cpu", "cuda"] = "auto"
    checkpoint_path: str | None = None
    allow_download: bool = False


class ProviderProbeResult(ApiModel):
    provider_id: str
    device: str
    sample_rate: int
    audio_shape: list[int]
    audio_seconds: float
    text_shape: list[int] | None = None
    text_seconds: float | None = None


class JobAccepted(ApiModel):
    job_id: UUID
    state: str


class Job(ApiModel):
    id: UUID
    project_id: UUID | None = None
    kind: str
    state: Literal["queued", "running", "cancel_requested", "cancelled", "succeeded", "failed", "interrupted"]
    phase: str
    progress: float | None = None
    message: str = ""
    warnings: list[str] = []
    resumable: bool = False
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime


class AssetImport(ApiModel):
    path: str = Field(min_length=1, max_length=32768)
    name: str | None = Field(default=None, max_length=500)


class Asset(ApiModel):
    id: UUID
    name: str
    path: str
    content_hash: str
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    samples: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    created_at: datetime


AnalysisFamily = Literal[
    "instrument", "technique", "voice", "pitch", "rhythm", "dynamics",
    "timbre", "harmony", "form", "affect", "production", "custom",
]


class DetectionTarget(ApiModel):
    label_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    name: str = Field(min_length=1, max_length=200)
    family: AnalysisFamily
    positive_prompts: list[str] = Field(default_factory=list, max_length=32)
    negative_prompts: list[str] = Field(default_factory=list, max_length=32)


class DetectionPlanCompileRequest(ApiModel):
    instruction: str = Field(min_length=1, max_length=10000)
    taxonomy_version: str | None = None
    preferred_languages: list[Literal["zh", "en"]] = Field(default_factory=lambda: ["zh", "en"])
    available_provider_ids: list[str] = Field(default_factory=list)
    targets: list[DetectionTarget] = Field(default_factory=list, max_length=200)


class DetectorSpec(ApiModel):
    mode: Literal["zero-shot-text", "audio-prototype", "supervised-head", "descriptor-rule", "score-rule"]
    provider_id: str
    weight: float = Field(ge=0, le=1)
    required: bool = False
    model_version_id: str | None = None
    prototype_set_id: UUID | None = None


class DetectionWindow(ApiModel):
    context_seconds: float = Field(gt=0, le=30)
    hop_seconds: float = Field(gt=0, le=10)
    minimum_event_ms: int = Field(ge=10, le=30000)
    maximum_event_ms: int | None = Field(default=None, ge=10)
    merge_gap_ms: int = Field(ge=0, le=10000)
    smoothing_frames: int = Field(default=1, ge=0, le=100)


class PromptSet(ApiModel):
    positive: list[str] = Field(default_factory=list, max_length=32)
    negative: list[str] = Field(default_factory=list, max_length=32)


class DetectionThresholds(ApiModel):
    review: float = Field(ge=0, le=1)
    auto_accept: float | None = Field(default=None, ge=0, le=1)
    abstain_below: float = Field(ge=0, le=1)


class DetectionLabelPlan(ApiModel):
    label_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    display_name: str
    family: AnalysisFamily
    granularity: Literal["clip", "segment", "frame", "event", "note"]
    instrument_scope: list[str] = Field(default_factory=list)
    detectors: list[DetectorSpec] = Field(min_length=1)
    window: DetectionWindow
    prompts: PromptSet
    thresholds: DetectionThresholds


class FeatureExtractorSpec(ApiModel):
    provider_id: str
    outputs: list[str] = Field(min_length=1)
    config: dict[str, str | int | float | bool] = Field(default_factory=dict)


class FusionSpec(ApiModel):
    method: Literal["calibrated-logistic", "calibrated-mlp", "rule-only"] = "rule-only"
    calibration_profile_id: UUID | None = None
    missing_analyzer_policy: Literal["fail", "warn-and-continue", "abstain"] = "warn-and-continue"


class DetectionPolicy(ApiModel):
    default_decision: Literal["review", "abstain"] = "review"
    protect_confirmed: Literal[True] = True
    persist_preview: Literal[False] = False
    allow_auto_accept_labels: list[str] = Field(default_factory=list)
    maximum_suggestions_per_minute: int = Field(default=120, ge=1, le=10000)


class DetectionCompiler(ApiModel):
    kind: Literal["deterministic", "llm"] = "deterministic"
    provider_id: str | None = None
    model: str | None = None
    prompt_template_version: str | None = None
    compiled_at: datetime


class DetectionPlan(ApiModel):
    schema_version: Literal["1.0"] = "1.0"
    id: UUID
    revision: int = Field(ge=1)
    name: str = Field(min_length=1, max_length=200)
    status: Literal["draft", "validated", "retired"]
    source_instruction: str = Field(max_length=10000)
    taxonomy_version: str | None = None
    preferred_languages: list[Literal["zh", "en"]] = Field(default_factory=lambda: ["zh", "en"])
    labels: list[DetectionLabelPlan] = Field(min_length=1)
    feature_extractors: list[FeatureExtractorSpec] = Field(default_factory=list)
    fusion: FusionSpec = Field(default_factory=FusionSpec)
    policy: DetectionPolicy = Field(default_factory=DetectionPolicy)
    compiler: DetectionCompiler


class ValidationFinding(ApiModel):
    code: str
    message: str
    json_pointer: str | None = None
    label_id: str | None = None


class DetectionPlanValidation(ApiModel):
    valid: bool
    errors: list[ValidationFinding] = Field(default_factory=list)
    warnings: list[ValidationFinding] = Field(default_factory=list)
    runnable_labels: list[str] = Field(default_factory=list)


class DetectionPlanRange(ApiModel):
    asset_id: UUID
    start_sample: int = Field(ge=0)
    end_sample: int = Field(ge=1)

    @model_validator(mode="after")
    def valid_range(self) -> "DetectionPlanRange":
        if self.end_sample <= self.start_sample:
            raise ValueError("endSample must be greater than startSample")
        return self


class DetectionPlanRunRequest(ApiModel):
    plan_revision: int = Field(ge=1)
    asset_ids: list[UUID] = Field(min_length=1)
    ranges: list[DetectionPlanRange] = Field(default_factory=list)
    maximum_candidates: int = Field(default=10000, ge=1, le=100000)

    @model_validator(mode="after")
    def valid_assets(self) -> "DetectionPlanRunRequest":
        if len(set(self.asset_ids)) != len(self.asset_ids):
            raise ValueError("assetIds must be unique")
        available = set(self.asset_ids)
        if any(item.asset_id not in available for item in self.ranges):
            raise ValueError("every range assetId must be present in assetIds")
        return self


class DetectionCandidate(ApiModel):
    id: UUID
    run_id: UUID
    asset_id: UUID
    start_sample: int
    end_sample: int
    label_id: str
    score: float = Field(ge=0, le=1)
    uncertainty: float | None = Field(default=None, ge=0, le=1)
    decision: Literal["auto_accept", "review", "abstain", "suppress"]
    evidence: dict[str, Any] = Field(default_factory=dict)


class DetectionRun(ApiModel):
    id: UUID
    plan_id: UUID
    plan_revision: int
    state: Literal["queued", "running", "cancel_requested", "cancelled", "succeeded", "failed", "interrupted"]
    preview: bool
    candidate_count: int = 0
    decision_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime
