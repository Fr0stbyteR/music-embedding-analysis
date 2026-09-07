# Model selection and fine-tuning policy

## Decision

Use the recommended **M2D-CLAP 2025** checkpoint as the default audio backbone.
Its tested one-second output has seven frame vectors, which is useful for event
localization but is not a 20/40 ms fine-time model. Compare the separate M2D
40 ms checkpoint when boundary accuracy becomes the bottleneck; that checkpoint
does not provide the same CLAP alignment. Do not fine-tune the entire backbone
during normal annotation sessions.

The default learning ladder is:

1. frozen embeddings plus class prototypes;
2. frozen embeddings plus calibrated logistic or small MLP temporal head;
3. adapter/LoRA or unfreeze the last encoder blocks;
4. contrastive audio/text adaptation only when open-vocabulary retrieval is a
   demonstrated product requirement.

## Why M2D-CLAP

- Its training combines masked audio modeling and language alignment, making it
  useful both for zero-shot semantics and supervised transfer.
- The official repository exposes frame-level features and checkpoints intended
  for further pre-training/fine-tuning.
- The 2025 recommended checkpoint reports AudioSet mAP 0.490.
- Frame-level output is available, unlike a single global CLAP clip vector.
- A frozen-backbone head can update quickly and avoids catastrophic forgetting.

The backend must benchmark actual latency, VRAM, temporal receptive field, and
accuracy on the target machine. A nominal frame interval is not the same as the
minimum recognizable event duration.

## Provider profiles

### `m2d_clap_2025_temporal` — default

Role: frame embeddings, technique detection, supervised transfer.

Recommended first use:

- resample to the checkpoint's required sample rate;
- process overlapping context windows rather than isolated one-second clips;
- retain target-region masks inside each context window;
- project large frame embeddings to a smaller learned dimension in the temporal
  head rather than storing every raw layer indefinitely;
- cache float16 features with exact receptive-field timestamps.

### `laion_clap_music_htsat_base` — compatibility baseline

Role: global zero-shot tagging and prompt experiments.

Advantages: mature ecosystem, music checkpoint, permissive repository license,
and relatively simple audio/text similarity API. Limitations: the public music
weights date from 2023, global pooling is weak for brief events, and specialist
Chinese terminology is unlikely to be represented reliably.

### `muq_mulan_large` — optional research provider

Role: Chinese/English music-text retrieval and semantic comparison.

Advantages: explicitly music-oriented and bilingual. Limitations: approximately
700M parameters, slower/heavier, strict 24 kHz input, fp32 recommended by the
authors, and released weights are CC-BY-NC 4.0. It must not silently enter a
commercial distribution.

## Task heads

The first production head should not be a single clip classifier. It should
consume frame features and emit:

- per-frame multi-label logits;
- event onset/offset likelihoods;
- pooled clip logits for selected-region classification;
- an embedding used by prototypes and active learning;
- calibrated probabilities and an out-of-distribution score.

A lightweight implementation can be:

```text
frame embeddings
 -> LayerNorm
 -> Linear projection (e.g. 3840 -> 256/512)
 -> temporal depthwise Conv1d or 2-layer Transformer
 -> multi-label logits + boundary logits
```

The exact dimensions are experiment configuration, not API guarantees.

## Fast iteration modes

### Prototype update

Update class centroids from confirmed examples. Expected to finish in seconds
and useful before a class has enough examples for supervised training.

### Head training

Train only projection and temporal heads from cached embeddings. This is the
normal interactive update and should finish in seconds to a few minutes.

### Partial backbone adaptation

Run as an explicit advanced job. Unfreeze only the final blocks or attach an
adapter, use a lower learning rate, and keep a replay set of general audio.
This requires a GPU and a larger immutable dataset snapshot.

## Context policy for short techniques

Store the human annotation at its exact event boundary, but train and infer with
context. Each sample consists of:

- exact target interval;
- configurable left/right acoustic context;
- target mask identifying frames belonging to the technique;
- neighboring labels and score position when available.

For example, a 0.7-second event may be embedded inside a 4-second model input.
The loss is computed primarily over the target frames, while the context helps
identify instrument, pitch, phrase, and competing techniques.

## Data split and validation gates

Random clip splitting is forbidden. The snapshot builder groups by, in order of
availability:

1. performer;
2. recording session;
3. physical instrument/microphone;
4. source recording or work;
5. derived clips from the same original event.

Publishing requires configured gates such as:

- macro-F1 or macro-AUPRC does not regress beyond tolerance;
- every critical class has adequate recall or is explicitly waived;
- event-F1 improves for temporal detection;
- calibration error remains acceptable;
- performance on random high-confidence audits remains acceptable;
- no test-group leakage is detected.

## Licensing rule

Every provider manifest records source URL, code license, weight license,
commercial-use flag, required attribution, weight checksum, and acceptance
timestamp. Project exports record the model versions used. Model availability is
a runtime capability, not assumed by the API.
