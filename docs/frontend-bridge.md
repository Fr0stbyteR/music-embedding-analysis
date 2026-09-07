# Frontend bridge contract

## Boundary

The Webview must not receive the Python sidecar port or bearer token. It talks
only to the VS Code extension host through `postMessage`. The extension host:

1. validates each message against a small method allowlist;
2. checks workspace trust before file import, model download, or training;
3. translates logical methods to fixed backend routes;
4. attaches the sidecar session token;
5. validates the backend response before forwarding it to the Webview.

The bridge must never accept an arbitrary URL, filesystem path, HTTP verb, or
header from the Webview.

## Envelope

All messages use protocol `music-annotation/1` and one of these shapes:

- `request`: Webview to extension host. Has a unique `requestId`, a method, and
  validated parameters.
- `response`: final success or failure for one request.
- `progress`: zero or more updates for a long request.
- `event`: project changes that may not have been initiated by this panel.
- `cancel`: asks the extension host to abort a pending request or backend job.

The canonical TypeScript definitions are in `../spec/frontend-messages.ts`.

## Initial method allowlist

| Bridge method | Backend operation | Notes |
|---|---|---|
| `project.open` | `GET /v1/projects/{id}` | Returns initial editor state identifiers, not all annotations. |
| `asset.import` | `POST /v1/projects/{id}/assets:import` | Extension host chooses/normalizes the URI. |
| `annotation.query` | `GET /v1/projects/{id}/annotations` | Windowed by sample range and optional label/review state. |
| `annotation.create` | `POST /v1/projects/{id}/annotations` | Sends sample-index boundaries. |
| `annotation.update` | `PUT /v1/projects/{id}/annotations/{id}` | Must include the last known ETag. |
| `annotation.delete` | `DELETE /v1/projects/{id}/annotations/{id}` | Must include the last known ETag. |
| `annotation.bulk` | `POST /v1/projects/{id}/annotations:bulk` | One undoable logical operation. |
| `inference.start` | `POST /v1/projects/{id}/inference-jobs` | Returns `jobId` immediately. |
| `detectionPlan.compile` | `POST /v1/projects/{id}/detection-plans:compile` | Compiles natural language into a draft plan; returns a job. |
| `detectionPlan.get` | `GET /v1/projects/{id}/detection-plans/{planId}` | Loads the exact plan revision shown in the UI. |
| `detectionPlan.update` | `PUT /v1/projects/{id}/detection-plans/{planId}` | Saves an edited plan with ETag conflict protection. |
| `detectionPlan.validate` | `POST /v1/projects/{id}/detection-plans/{planId}:validate` | Checks schema, provider availability, taxonomy IDs, and calibration. |
| `detectionPlan.preview` | `POST /v1/projects/{id}/detection-plans/{planId}:preview` | Runs on selected ranges without persisting annotations. |
| `detectionPlan.run` | `POST /v1/projects/{id}/detection-plans/{planId}:run` | Starts a full detection run and returns `jobId`. |
| `detectionRun.get` | `GET /v1/projects/{id}/detection-runs/{runId}` | Returns run state, frozen plan revision, and summary. |
| `detectionRun.candidates` | `GET /v1/projects/{id}/detection-runs/{runId}/candidates` | Windowed candidate/evidence retrieval. |
| `prototypeSet.rebuild` | `POST /v1/projects/{id}/prototype-sets/{id}:rebuild` | Recomputes prototypes from a frozen annotation snapshot. |
| `review.next` | `GET /v1/projects/{id}/review-queue` | Uses model uncertainty plus audit sampling. |
| `alignment.start` | `POST /v1/projects/{id}/alignments` | Asynchronous. |
| `alignment.updateAnchors` | `PATCH /v1/projects/{id}/alignments/{id}/anchors` | Human correction with ETag. |
| `training.start` | `POST /v1/projects/{id}/training-jobs` | Requires an immutable dataset snapshot. |
| `job.cancel` | `POST /v1/jobs/{id}:cancel` | Cooperative cancellation. |

## Loading strategy

The frontend should open a project in this order:

1. Load capabilities, project metadata, taxonomy, tracks, and active model.
2. Request annotations only for the visible time range plus a small prefetch
   margin.
3. Subscribe to normalized events through the extension host.
4. Load waveform/spectrogram tiles independently from annotation metadata.
5. Request score/alignment data only if the score pane is visible.

This keeps large works responsive and prevents tens of thousands of annotation
objects from entering Webview state at once.

## Optimistic UI and conflict handling

Create, move, resize, relabel, and delete operations may render optimistically.
Each update carries the latest ETag. On `409 conflict`, the extension host sends
the current server value; the UI offers `keep mine`, `use current`, or a manual
merge. Confirmed human edits are never silently replaced by inference or an
LLM proposal.

## Job UX

Inference, alignment, feature extraction, snapshot creation, and training are
jobs. The UI receives monotonic progress with `phase`, `completed`, `total`,
and an optional human-readable message. Closing the panel does not corrupt a
job. Cancellation is cooperative and ends in `cancelled`, never an ambiguous
success state.

## LLM proposals

The LLM may return structured operations such as relabel, split, merge, or add
a note. The UI must show a diff and obtain an explicit apply action. The apply
request contains proposal IDs and expected annotation revisions so stale
proposals fail safely.

## Forward compatibility

- Unknown event types are ignored and logged.
- Unknown response fields are retained where practical but not rendered.
- Breaking envelope changes require a new protocol string.
- Additive backend changes stay within `/v1`.
