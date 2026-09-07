# V1 detection orchestration

## Product goal

A non-programmer can say:

> 在这首古琴曲中找出泛音、按音、吟、猱和明显的滑音，同时标出力度突变、
> 速度变化与可能的乐句边界。只自动接受把握很高的结果。

The backend compiles this request into a versioned `DetectionPlan`, shows the
plan in readable form, runs several analyzers, and creates reviewable
suggestions with evidence. It never treats the LLM's prose as an audio result.

## What “LLM controls CLAP” means

The LLM is a semantic control plane, not the acoustic detector. It may:

- map specialist terminology and aliases into a taxonomy;
- expand each concept into Chinese and English positive/negative prompts;
- describe acoustic cues and common confusions;
- select an allowed detector family and sensible time scale;
- propose co-occurrence, exclusion, instrument, and score constraints;
- explain model evidence to the reviewer.

It may not:

- assert that an event occurred without model evidence;
- supply Python, SQL, shell, regular expressions, or arbitrary predicates;
- invent provider IDs, files, checkpoints, or label IDs;
- change confirmed annotations;
- lower acceptance thresholds outside configured policy.

The same plan can also be built without an LLM using forms and deterministic
templates. This is essential for reproducibility and offline use.

## User experience

The main entry is a **识别目标** command bar above the timeline. The user can
type a request, choose labels from a taxonomy, or combine both. Compilation
opens a plan card rather than immediately scanning the work. The card answers:

- what will be recognized and at what time granularity;
- which examples, prompts, models, and measured features will be used;
- which distinctions are currently weak or unsupported;
- what will be suggested, withheld, or eligible for automatic acceptance;
- roughly how long the selected-range preview and full run will take.

The primary action is **在选区试跑**. Results appear as translucent lanes below
the waveform, grouped by label family. Clicking a candidate plays target audio
with context and shows a short reason such as “接近 7 个已确认泛音样本；古琴
上下文成立；与按音的差距偏小”. The reviewer can accept, correct the label,
resize, reject, or mark ambiguous with one keystroke.

After a review batch, the UI shows what was learned before offering **更新识别器**:
new positive/negative examples, validation impact, threshold changes, and labels
whose evidence is still too narrow across performers or recordings. Updating
creates a new prototype/calibration version; it never changes past run results.

## Detection hierarchy

### 1. Context pass

Run 4–15 second windows to estimate instrument, voice presence, ensemble type,
recording conditions, and broad musical character. Use LAION-CLAP or MuQ-MuLan
for prompt similarity and optionally Essentia music classifiers. Context scores
are priors, not final short-event labels.

### 2. Event pass

Run overlapping windows with M2D-CLAP frame embeddings. Technique detection is
ranked in this order:

1. a calibrated supervised temporal head;
2. confirmed-example prototypes plus hard-negative prototypes;
3. zero-shot prompt evidence as a weak prior;
4. descriptor and score rules as supporting evidence.

Exact annotation boundaries are decoded from frame and onset evidence. A
one-second target can still be evaluated inside four seconds of context.

### 3. Measured-feature pass

Features that have a physical or conventional estimator should not be guessed
from CLAP text similarity:

| Feature family | Primary measurement | Semantic model role |
|---|---|---|
| onset, articulation density | onset detector / Essentia | name and explain patterns |
| pitch, contour, vibrato, portamento | pitch tracker, note transcription | add instrument/technique prior |
| tempo, beat, rhythmic density | beat tracker / score alignment | describe changes |
| loudness, dynamics | loudness envelope and change points | label musical meaning |
| spectral centroid, noisiness, brightness | spectral descriptors | combine into timbre labels |
| key, chords, tonal tension | tonal analysis plus score | semantic summary |
| notes and pitch bends | score alignment or Basic Pitch | constrain candidate events |
| genre, mood, broad instrumentation | Essentia/CLAP/MuQ | primary semantic evidence |

Essentia is an analyzer provider, parallel to CLAP providers. Every numeric
result stores algorithm/version, units, frame timing, and configuration.

### 4. Constraint and fusion pass

Fusion is declarative and calibrated. It can use:

- required or excluded instrument context;
- positive and negative prompt margins;
- prototype/head probabilities;
- score part, note, pitch, articulation, and position;
- minimum/maximum duration and temporal smoothing;
- allowed co-occurrences and mutually confusing labels;
- out-of-distribution and disagreement scores.

No weighted sum is assumed to be a probability. A calibration profile learned
on held-out recordings converts raw evidence to a probability and records the
profile version.

### 5. Policy pass

Each candidate becomes one of:

- `auto_accept`: only for calibrated labels above a per-label threshold;
- `review`: normal result with evidence and suggested boundaries;
- `abstain`: insufficient evidence or out of distribution;
- `suppress`: violates a validated constraint.

V1 defaults all newly defined specialist techniques to `review`. Auto-accept is
unlocked only after enough cross-recording validation data exists.

## Label specification

Each label is more than a name. It contains:

- family and parent label;
- bilingual names, aliases, and a domain definition;
- observable acoustic cues and exclusion cues;
- positive and negative prompt ensembles;
- likely confusions and allowed co-occurrences;
- instrument/voice scope;
- time granularity, context, hop, minimum event, and merge gap;
- detector ranking and fallback behavior;
- acceptance, review, and abstention thresholds;
- provenance for every LLM-generated field.

For `古琴·泛音`, an initial plan might use four-second M2D context, frame
prototypes, a temporal head when available, onset/spectral evidence, and a
guqin context gate. Prompts such as “isolated guqin harmonic with a bell-like,
high overtone-rich attack” are paired with negatives for ordinary stopped
tones, sliding tones, and unrelated plucked strings. These prompts seed review;
they do not justify high-confidence automatic labels by themselves.

## Prompt ensembles

One prompt per label is too fragile. Store separate templates for:

- object: “a guqin playing ...”;
- event: “a short instance of ...”;
- acoustic description without specialist vocabulary;
- phrase context;
- Chinese terminology;
- English terminology;
- hard negatives and confusable techniques.

Compute and retain every prompt score, the positive mean, negative maximum,
and margin. Prompt revisions create a new plan revision so results remain
reproducible.

## Few-shot learning loop

Confirmed audio examples create a `PrototypeSet` grouped by performer,
instrument, session, and source recording. A label needs both positive examples
and hard negatives. The normal update path is:

1. update robust class centroids and distance distributions;
2. recalibrate thresholds;
3. train a frozen-backbone head once enough independent groups exist;
4. compare with the previous model on an immutable validation snapshot;
5. publish only if gates pass.

Do not compute validation splits by random short clip. Slices derived from the
same performance must stay in the same group.

## Evidence returned to the UI

Every suggestion exposes a compact evidence bundle:

- final calibrated confidence and decision;
- detector/model/checkpoint versions;
- top positive and negative prompt scores;
- nearest confirmed examples and distances;
- supervised-head score when present;
- supporting measured features;
- score position and constraint results;
- disagreement/OOD reason;
- plan revision and calibration profile.

The default UI shows a one-line reason. An expandable panel shows the full
bundle and lets the user play the target with context, compare prototypes, and
accept, relabel, resize, reject, or mark ambiguous.

## API surface

### Compile and manage plans

- `POST /v1/projects/{projectId}/detection-plans:compile`
- `GET /v1/projects/{projectId}/detection-plans/{planId}`
- `PUT /v1/projects/{projectId}/detection-plans/{planId}`
- `POST /v1/projects/{projectId}/detection-plans/{planId}:validate`

Compilation is an asynchronous job. Its result is a draft plan and a list of
questions/warnings. Validation is deterministic and never calls an LLM.

### Preview and execute

- `POST /v1/projects/{projectId}/detection-plans/{planId}:preview`
- `POST /v1/projects/{projectId}/detection-plans/{planId}:run`
- `GET /v1/projects/{projectId}/detection-runs/{runId}`
- `GET /v1/projects/{projectId}/detection-runs/{runId}/candidates`

Preview analyzes a bounded selected range and does not persist annotations.
Run writes only `suggested` annotations unless the project's acceptance policy
explicitly allows auto-accept for that label and calibration version.

### Learn from examples

- `POST /v1/projects/{projectId}/prototype-sets`
- `POST /v1/projects/{projectId}/prototype-sets/{setId}:rebuild`
- `GET /v1/projects/{projectId}/labels/{labelId}/evidence`

Prototype rebuilding and calibration are jobs. Confirming one annotation never
silently initiates expensive backbone training.

## Core execution object

```text
natural-language request / form
    -> plan compiler (deterministic templates + optional LLM)
    -> JSON Schema validation + capability/license checks
    -> human-readable preview
    -> context analyzers
    -> frame/event analyzers
    -> measured features and score constraints
    -> calibrated fusion and policy
    -> reviewable candidates with evidence
    -> confirmed examples -> prototypes/head training
```

## V1 delivery slices

### V1.1 — safe plan compiler

DetectionPlan storage/versioning, deterministic compiler, optional structured
LLM compiler, plan editor, validation report, and three example taxonomies.

### V1.2 — selected-range preview

Audio decoding/resampling, embedding cache, LAION prompt ensemble, M2D frame
features, measured loudness/onset/spectral features, evidence bundle, and no
automatic writes.

### V1.3 — batch suggestions

Window scheduler, overlap merge, candidate persistence, cancellation/resume,
review queue integration, and per-label acceptance policy.

### V1.4 — interactive learning

Prototype sets, hard-negative mining, calibration, frozen temporal head,
evaluation report, version publishing, and rollback.

## Acceptance criteria

- The same plan revision and model versions produce equivalent results within
  declared numeric tolerance.
- An invalid LLM response cannot invoke code or a provider outside the allowlist.
- Every suggestion is traceable to audio range, plan, model, and evidence.
- Confirmed annotations cannot be overwritten by a run.
- Preview of a selected range can be cancelled and never writes annotations.
- A specialist label without calibration never auto-accepts.
- Failure of one analyzer degrades to a warning/fallback instead of losing the
  complete run.
