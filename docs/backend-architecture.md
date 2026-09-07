# Python backend architecture

## 1. Product boundary

The backend is a local annotation engine, not the editor UI. It owns:

- project metadata, taxonomy, annotation revisions, and audit history;
- audio inspection, resampling, cached feature extraction, and model inference;
- score ingestion and score–audio alignment;
- uncertainty-based review queues;
- immutable training snapshots, training jobs, evaluation, model versions, and
  rollback;
- optional LLM proposals expressed as reviewable diffs.

The frontend owns:

- waveform, spectrogram, score, marker, and playback rendering;
- selection, dragging, keyboard shortcuts, and undo presentation;
- optimistic local interaction while a revision is being persisted;
- all user confirmations.

An LLM may explain a suggestion, normalize terminology, propose aliases, or
flag contradictions. It must not silently change a confirmed annotation,
taxonomy definition, alignment anchor, dataset split, or published model.

## 2. Process topology

```text
VS Code Webview
    │ postMessage / typed frontend commands
    ▼
VS Code extension host
    │ HTTP on 127.0.0.1 + bearer session token
    ▼
Python sidecar (FastAPI)
    ├── project service
    ├── alignment service
    ├── annotation and review service
    ├── inference scheduler
    ├── training worker
    ├── model registry
    └── SQLite + feature/model files
```

The extension starts the process with an ephemeral port. The sidecar prints one
JSON handshake line to stdout:

```json
{
  "protocol": "music-annotation/1",
  "port": 49321,
  "token": "random-256-bit-token",
  "pid": 12345
}
```

The service binds only to `127.0.0.1`. Every request except health and startup
handshake requires `Authorization: Bearer <token>`. The token never enters the
Webview; the extension host proxies calls. This prevents an untrusted Webview
or another local page from acquiring model and filesystem capabilities.

## 3. Project layout

```text
example.ma-project/
├── project.json                 portable manifest
├── project.sqlite3              authoritative mutable metadata
├── assets/                      optional managed copies
├── scores/                      normalized MusicXML/MIDI and alignment maps
├── features/
│   ├── audio/<asset>/<model>/   chunked float16 frame features
│   └── classical/<asset>/       pitch, onset, chroma, loudness, etc.
├── snapshots/<snapshot-id>/     immutable training manifests
├── models/<model-version>/      heads/adapters, metrics, and manifests
├── exports/
└── tmp/                         recoverable job intermediates
```

SQLite stores identifiers, ranges, provenance, revisions, and paths, but not
large embeddings. Frame features are chunked by time so range requests do not
load an entire recording. Cache keys include asset content hash, decoder
version, sample rate, model identity, weight checksum, layer selection, and
feature parameters.

## 4. Core domain objects

### Asset

An immutable logical input with a stable ID and content hash. A new file
revision creates a new asset. Local paths remain extension-host concerns; the
backend receives a resolved path after workspace-trust checks.

### Score and alignment

MusicXML and MIDI are native score inputs. PDF/image OMR is an optional adapter,
not part of the first milestone. An alignment is a monotonic piecewise mapping
between score positions and audio time:

```text
(score measure/beat/note) <-> (audio sample/time)
```

Every anchor has source (`automatic` or `human`), confidence, and revision.
Human anchors are hard constraints on recomputation unless explicitly removed.

### Taxonomy

Taxonomies are versioned. A label contains:

- stable ID and parent ID;
- preferred Chinese and English names;
- aliases and descriptions;
- positive acoustic cues and exclusion cues;
- confusable labels and allowed co-occurrence;
- minimum useful duration and recommended context;
- annotation instructions and examples.

Changing a label name does not change its ID. Merging or splitting labels
creates an explicit migration reviewed by the user.

### Annotation

An annotation is an interval, not a colored marker string. Required fields are:

- `asset_id`, `start_sample`, `end_sample`;
- one or more `label_id` values;
- lifecycle state: `suggested`, `confirmed`, `rejected`, or `ambiguous`;
- provenance: `human`, `model`, `llm`, or `import`;
- creator/model version and optional score target;
- confidence and candidate scores for model suggestions;
- monotonically increasing revision.

Samples are authoritative; seconds are derived using the asset sample rate.
This avoids accumulated floating-point and resampling errors. Overlapping and
multi-label annotations are supported.

### Review item

A review item points to an annotation or unlabeled candidate and records why it
was selected:

- entropy/margin uncertainty;
- disagreement between models or prompts;
- taxonomy coverage deficit;
- out-of-distribution distance;
- random audit of high-confidence predictions;
- score–audio inconsistency.

The queue is a view over project state, not another source of truth.

### Dataset snapshot and model version

Training never consumes a live mutable query. It consumes an immutable snapshot
containing annotation IDs and revisions, taxonomy version, group-aware splits,
preprocessing configuration, base-weight checksum, and random seed.

A model version contains a parent model, adapter/head weights, training snapshot,
metrics, calibration data, decision thresholds, and publish state. Publishing is
an explicit operation and rollback never deletes the newer version.

## 5. End-to-end workflow

1. Import audio and optional score.
2. Decode metadata, hash the asset, and generate lightweight waveform tiles.
3. Align score and audio; expose low-confidence areas for anchor correction.
4. Create or import a versioned taxonomy.
5. Run baseline inference and create suggestions, never confirmed labels.
6. Build an active-learning review queue.
7. Persist each human decision with optimistic concurrency and audit history.
8. When enough useful corrections accumulate, create a training snapshot.
9. Train the lightweight head first; evaluate on performer/session-disjoint data.
10. Publish only if validation gates pass; otherwise retain the result as a
    rejected candidate with diagnostics.
11. Re-run inference only on invalidated regions and unseen assets.

## 6. Job model

Alignment, feature extraction, inference, training, and export are asynchronous
jobs. Command endpoints return `202 Accepted` and a job ID. Clients receive
progress through `/v1/events` as Server-Sent Events and may poll `/v1/jobs/{id}`
after reconnecting.

Job states:

```text
queued -> running -> succeeded
                  -> failed
                  -> cancel_requested -> cancelled
```

Jobs report a stable phase, numeric progress when meaningful, human-readable
status, warnings, and resumability. Cancellation is cooperative. A killed
process marks unfinished jobs interrupted at next startup; resumable jobs may be
requeued by the user.

## 7. Concurrency and editing

All writes use optimistic concurrency. Annotation responses include an ETag
derived from their revision. Update/delete requests require `If-Match`. A stale
write receives `409 Conflict` with the current representation, allowing the UI
to offer keep-mine/keep-current/merge rather than silently losing work.

Bulk edits are atomic at the metadata level. Long feature or model jobs write
to temporary versioned paths and publish with an atomic rename only after the
manifest and checksums are complete.

## 8. Score–audio alignment

The first implementation should support symbolic scores, not OMR:

1. Parse MusicXML/MIDI to note events and score positions.
2. Synthesize or derive score chroma/onset features.
3. Extract audio chroma/onsets using librosa or a replaceable provider.
4. Use coarse global alignment followed by constrained subsequence DTW.
5. Fit a monotonic piecewise mapping and calculate local confidence.
6. Recompute only affected intervals when a user moves an anchor.

The interface must preserve alternative takes, repeats, cuts, and fermatas.
Therefore an alignment is not assumed to be one constant tempo. Repeated score
sections require an explicit performed-order mapping.

## 9. LLM boundary

LLM requests receive structured evidence rather than raw authority:

- taxonomy definitions and aliases;
- candidate labels and model scores;
- score context;
- classical measurements;
- neighboring confirmed annotations.

The result is a `Proposal` containing a JSON Patch-like diff, explanation,
evidence references, and uncertainty. Applying a proposal is a separate human
action. LLM output never enters a training snapshot unless the resulting
annotation has been human-confirmed; pseudo-label experiments must use a
separate provenance and explicit snapshot policy.

## 10. Packaging and deployment

For non-technical users, ship one platform-specific sidecar rather than asking
them to install Python. A development installation may use `uv`, while releases
bundle the interpreter and Python dependencies with a tool such as PyInstaller
or Nuitka. Model weights are downloaded separately on first use after displaying
size, source, checksum, license, and storage location.

CPU mode supports project editing, score alignment, cached inference, and small
linear-head training. CUDA and Apple acceleration are optional capabilities
reported by `/v1/capabilities`; unsupported actions are disabled in the UI
rather than failing after submission.

## 11. Initial milestones

### M0 — protocol and project integrity

- launch handshake, health/capabilities, project CRUD;
- assets, taxonomy, annotation CRUD, audit log;
- job and event infrastructure;
- no machine-learning dependency.

### M1 — baseline assisted annotation

- symbolic score alignment and editable anchors;
- model-provider interface;
- cached M2D-CLAP features;
- prototype/linear/MLP heads;
- suggestions and uncertainty review queue.

### M2 — safe iterative learning

- immutable snapshots and group-aware split builder;
- head training, calibration, metrics, publish/rollback;
- incremental invalidation and batch inference.

### M3 — advanced semantics

- contrastive audio/text adapters;
- optional MuQ-MuLan provider;
- LLM taxonomy proposals;
- OMR adapter and additional score formats.

