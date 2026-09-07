# Music Annotation Backend Design

This repository contains the backend and design contract for a local,
human-in-the-loop music annotation system. The intended client is the existing
VS Code audio editor, but the protocol is deliberately UI-agnostic so the same
backend can later serve a standalone desktop application.

## Documents

- `docs/backend-architecture.md` — responsibilities, processes, storage, jobs,
  safety, and the end-to-end annotation loop.
- `docs/model-selection.md` — model choice and the staged fine-tuning policy.
- `docs/v1-detection-orchestration.md` — next-version LLM-controlled detection
  plans, model routing, evidence, and acceptance policy.
- `spec/openapi.yaml` — local service HTTP API, including asynchronous jobs and
  the event stream.
- `spec/project.schema.json` — portable project manifest.
- `spec/model-provider.md` — Python interface required from every embedding or
  annotation model adapter.
- `spec/detection-plan.schema.json` — safe, declarative contract produced by a
  deterministic compiler or an LLM.
- `spec/frontend-messages.ts` — typed Webview/extension-host request, response,
  progress, event, and cancellation envelopes.
- `docs/frontend-bridge.md` — security boundary and UI integration rules.

## Proposed default

- Python 3.11
- FastAPI + Pydantic v2
- SQLite for authoritative metadata and audit history
- NumPy/Zarr-compatible binary feature storage outside SQLite
- M2D-CLAP 2025 temporal checkpoint as the default audio backbone
- Frozen-backbone linear/MLP temporal head for fast interactive iteration
- Optional partial fine-tuning only after a validation gate passes

The browser Webview never talks to the Python process directly. The VS Code
extension launches the sidecar on `127.0.0.1` with an ephemeral port and random
session token, proxies requests, and terminates it with the editor session.

## Run the service

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
.venv\Scripts\music-annotation-backend.exe
```

The first stdout line is the extension-host handshake containing the local port
and random bearer token. Interactive API documentation is available at
`http://127.0.0.1:49321/docs` while the service is running.

Model load and probe are asynchronous:

```text
POST /v1/providers/{providerId}:load
POST /v1/providers/{providerId}:probe
GET  /v1/jobs/{jobId}
```

## V0.2 analysis workflow

V0.2 adds the first complete demand-driven annotation loop:

1. Register a trusted audio path with `POST /v1/projects/{projectId}/assets`.
2. Compile a request such as “标注古琴泛音和滑音，并分析音高、和声和曲式”
   with `POST /v1/projects/{projectId}/detection-plans:compile`.
3. Inspect the returned plan and call `:validate`. Every detector, time scale,
   prompt ensemble, threshold, and fallback is explicit and versioned.
4. Use `:preview` on selected ranges. Preview stores evidence candidates but
   never creates annotations.
5. Use `:run` after reviewing the plan. Results are persisted only as
   `suggested` / `model` annotations and never overwrite confirmed human work.

Measured analyzers currently cover note-level pitch, global key and local chord
templates, beats/tempo, dynamics changes, timbre changes, and structural
sections. These outputs carry `heuristic-not-calibrated` provenance. Instrument,
voice, playing-technique, affect, and production labels use prompt ensembles
through a loaded MuQ-MuLan or LAION-CLAP provider; if no compatible provider is
loaded, those labels abstain with a warning. The deterministic compiler also
accepts explicit custom targets, so specialist vocabularies are not limited to
the built-in Chinese/English catalogue. An unknown “识别/检测/标注…” request is
preserved as a stable custom semantic target instead of being silently replaced
with unrelated generic measurements; a client or optional LLM control plane can
later expand it into more precise positive/negative targets.

This version deliberately disables automatic acceptance. A score is evidence
for review, not a calibrated probability, until a later prototype/training
version supplies cross-recording validation and calibration profiles.

See `docs/model-smoke-test.md` for the tested checkpoints and this machine's
CPU measurements.

Set `MAB_HUGGINGFACE_CACHE` only when the sidecar should use a dedicated cache;
otherwise MuQ uses the normal Hugging Face user cache.
