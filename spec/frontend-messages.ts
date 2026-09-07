/** Canonical Webview <-> VS Code extension-host bridge contract. */

export const BRIDGE_PROTOCOL = "music-annotation/1" as const;

export type BridgeMethod =
  | "project.open"
  | "asset.import"
  | "annotation.query"
  | "annotation.create"
  | "annotation.update"
  | "annotation.delete"
  | "annotation.bulk"
  | "inference.start"
  | "detectionPlan.compile"
  | "detectionPlan.get"
  | "detectionPlan.update"
  | "detectionPlan.validate"
  | "detectionPlan.preview"
  | "detectionPlan.run"
  | "detectionRun.get"
  | "detectionRun.candidates"
  | "prototypeSet.rebuild"
  | "review.next"
  | "alignment.start"
  | "alignment.updateAnchors"
  | "training.start"
  | "job.cancel";

export interface RequestMessage<P = unknown> {
  protocol: typeof BRIDGE_PROTOCOL;
  kind: "request";
  requestId: string;
  projectId?: string;
  method: BridgeMethod;
  params: P;
}

export interface CancelMessage {
  protocol: typeof BRIDGE_PROTOCOL;
  kind: "cancel";
  requestId: string;
  targetRequestId?: string;
  jobId?: string;
}

export interface ResponseMessage<R = unknown> {
  protocol: typeof BRIDGE_PROTOCOL;
  kind: "response";
  requestId: string;
  ok: boolean;
  result?: R;
  error?: BridgeError;
  etag?: string;
}

export interface ProgressMessage {
  protocol: typeof BRIDGE_PROTOCOL;
  kind: "progress";
  requestId?: string;
  jobId: string;
  phase: string;
  completed?: number;
  total?: number;
  message?: string;
}

export interface EventMessage<P = unknown> {
  protocol: typeof BRIDGE_PROTOCOL;
  kind: "event";
  eventId: string;
  projectId?: string;
  eventType: string;
  occurredAt: string;
  payload: P;
}

export interface BridgeError {
  code:
    | "invalid_request"
    | "not_found"
    | "conflict"
    | "workspace_untrusted"
    | "model_unavailable"
    | "job_failed"
    | "cancelled"
    | "internal_error";
  message: string;
  retryable: boolean;
  details?: unknown;
  currentValue?: unknown;
  currentEtag?: string;
}

export type WebviewToHostMessage = RequestMessage | CancelMessage;
export type HostToWebviewMessage =
  | ResponseMessage
  | ProgressMessage
  | EventMessage;

export interface SampleRange {
  assetId: string;
  startSample: number;
  endSample: number;
}

export interface AnnotationQueryParams extends SampleRange {
  labelIds?: string[];
  reviewStates?: Array<"unreviewed" | "accepted" | "corrected" | "rejected">;
  cursor?: string;
  limit?: number;
}

export interface AnnotationMutationParams extends SampleRange {
  annotationId?: string;
  labelIds: string[];
  confidence?: number;
  note?: string;
  source: "human" | "model" | "llm_proposal" | "import";
  expectedEtag?: string;
}

export interface StartInferenceParams extends SampleRange {
  modelVersionId?: string;
  taxonomyVersionId: string;
  contextBeforeSeconds?: number;
  contextAfterSeconds?: number;
  thresholdProfileId?: string;
}

export interface CompileDetectionPlanParams {
  instruction: string;
  taxonomyVersion?: string;
  preferredLanguages?: Array<"zh" | "en">;
  availableProviderIds?: string[];
  targets?: DetectionTarget[];
}

export interface DetectionTarget {
  labelId: string;
  name: string;
  family:
    | "instrument"
    | "technique"
    | "voice"
    | "pitch"
    | "rhythm"
    | "dynamics"
    | "timbre"
    | "harmony"
    | "form"
    | "affect"
    | "production"
    | "custom";
  positivePrompts?: string[];
  negativePrompts?: string[];
}

export interface DetectionPlanRange {
  assetId: string;
  startSample: number;
  endSample: number;
}

export interface PreviewDetectionPlanParams {
  planId: string;
  planRevision: number;
  ranges: DetectionPlanRange[];
  maximumCandidates?: number;
}

export interface RunDetectionPlanParams {
  planId: string;
  planRevision: number;
  assetIds: string[];
  ranges?: DetectionPlanRange[];
}

export interface StartTrainingParams {
  datasetSnapshotId: string;
  baseModelVersionId: string;
  strategy: "prototype" | "linear_head" | "temporal_mlp" | "adapter";
  hyperparameters?: Record<string, number | string | boolean>;
}
