# Model smoke test

## ModelScope music CLAP (2026-10-01)

The `clap_music` provider was downloaded from ModelScope's
`laion/larger_clap_music` repository and loaded on Windows CPU while
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Download plus cold load took
54.92 seconds in this single observation (about 776 MB of weights).

One second of silence produced finite `[1, 512]` audio embeddings. English and
Chinese prompts produced finite `[1, 2]` similarity scores. A subsequent
`allow_download=False` snapshot lookup successfully reused the ModelScope
cache. This validates the download and inference path, not semantic accuracy
or macOS inference. The new provider uses Transformers preprocessing and a
separate provider ID; do not assume thresholds are interchangeable with the
legacy `.pt` adapter.

## Legacy providers

Test date: 2026-09-07. Device: CPU; no NVIDIA runtime was visible. Timings are
single cold-load/single-call observations, not a formal benchmark.

| Provider | Checkpoint | Load | Probe result |
|---|---|---:|---|
| M2D-CLAP | `m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth` | 14.84 s | 1 s audio -> `[1, 7, 3840]` in 0.16 s |
| LAION-CLAP | `music_audioset_epoch_15_esc_90.14.pt`, HTSAT-base | 53.74 s | 1 s audio -> `[1, 512]` in 0.53 s |
| MuQ-MuLan | `OpenMuQ/MuQ-MuLan-large` | 30.95 s cached | 1 s audio -> `[1, 512]` in 1.22 s; two texts -> `[2, 512]` in 15.81 s |

## Findings

- M2D-CLAP is the clear default for interactive temporal annotation on this
  machine. It emits frame features and was fastest in this smoke test.
- LAION-CLAP music is useful for zero-shot audio/text comparison, but its output
  is global rather than frame-level.
- MuQ-MuLan successfully handles Chinese and English text, but it is too heavy
  for a default per-window path on CPU. It remains an optional research provider.
- `muq==0.1.0` must currently use `transformers<5`; Transformers 5.x changes the
  masking API expected by MuQ's EasyDict-based Conformer configuration.
- The LAION default auto-download checkpoint is not compatible with HTSAT-base.
  The music checkpoint must be passed explicitly.

These figures measure one-second silence only. Before product decisions, run a
warm benchmark on representative guqin recordings, multiple window sizes, and
the deployment GPU.
