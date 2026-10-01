# Essentia TensorFlow modules

Normal `start.ps1` / `start.command` prepares the native runtime, pinned official
weights and real inference self-tests. No separate TF service is needed.
`MAB_ESSENTIA_SETUP_TENSORFLOW=false` skips automatic TF preparation; `--basic`
skips learned-model weight preparation as well. Existing installed models can
still be used. An unavailable runtime/model produces an explicit module error.
Windows x64 uses the C++ worker and TensorFlow C API. macOS Intel/ARM uses the
pinned official Essentia TensorFlow native wheel in the backend environment.
All 21 views have passed real Windows inference; macOS is covered by configured
integration CI, not a local macOS run from the Windows development machine.

## Catalog and rendering

| Views | Official model | UI |
| --- | --- | --- |
| Instrument distribution / selected relevance / candidate regions | MTG Jamendo instrument, Discogs EffNet; 40 labels | Matrix / Vector / Markers |
| Mood/theme distribution / selected relevance | MTG Jamendo moodtheme, Discogs EffNet; 56 labels | Matrix / Vector |
| Genre | MTG Jamendo genre, Discogs EffNet; 87 labels | Matrix |
| Bright timbre | timbre, Discogs EffNet | Vector |
| Voice, acoustic, electronic, tonal, danceability | corresponding MusiCNN classification heads | Vector |
| Happy, sad, relaxed, aggressive, party | corresponding MusiCNN mood heads | Vector |
| Automatic tags / selected relevance | MSD MusiCNN; 50 labels | Matrix / Vector |
| Audio-estimated tempo / tempo candidates | TempoCNN deepsquare-k16-3; 256 bins, 30–285 BPM | Vector / Matrix |

21 views share 16 model specifications. The frontend reuses existing Vector,
Matrix Canvas2D/WebGL and editable Marker components; there is no new renderer.
Class names and positive-class columns come from official model metadata.
TempoCNN is not the MusicXML/DTW performed-tempo module and has no score input.

## Timing, scores and candidates

Prediction interval defaults to 2 seconds (6 seconds for frontend tempo views).
It is quantized to frontend frame hops: 256/16000 seconds for MusiCNN/EffNet and
512/11025 for TempoCNN. Uniform prediction cells cover successive hop-length
intervals; curves are drawn at cell centers. `metadata.timeAnchor` is
`cell-center`, and `metadata.hopLength` is expressed at the result sample rate.

Context windows remain model-native: MusiCNN uses 187 mel frames (~3 seconds),
EffNet 128 (~2 seconds), TempoCNN 256 (~12 seconds). Native frame cutting and
mel frontends are reused; TempoCNN normalizes each patch. Context at file edges
is clamped to available frames, with repeated frames for short files. Dense
sampling does not make these recording/window-level models note-accurate.

Scores are raw model outputs in [0,1], not calibrated confidence or ground truth.
Instrument regions use threshold entry and 0.8×threshold exit hysteresis plus
minimum duration. Their names explicitly say candidate, and manual edits are
preserved when reopening. Reanalysis intentionally replaces candidate results.
Tempo uses the maximum-scoring BPM bin; half/double-tempo errors are possible.

## Cache, limits and API

`GET /v1/essentia-tf/capabilities` and
`POST /v1/interactive-assets/{assetId}:essentia-tf` require the existing session
token. POST accepts the existing analysis request (`algorithm`, `options`,
`cachePolicy`). `options`: `hopSeconds` 1–30, optional `label`, `threshold` 0–1,
`minimumDuration` 0.1–60 seconds. Algorithms and official model files are
whitelisted; arbitrary graph paths are not accepted.

Backend `.music-annotation-data/essentia-tf-cache` stores atomic NPZ embeddings
and classifier scores. Keys include audio content, native runtime, preprocessing
version, quantized interval and model checksums. Heads share backbone embeddings;
target selection and region thresholds reuse existing scores. Invalid cache
contents are recomputed without trusting pickle data. `cachePolicy: refresh`
recomputes embeddings/scores. The frontend also persists outputs, class labels,
settings and edited markers through its existing `.audio_toolkit` folder store.

Maximum audio duration is three hours, at most 11,000 prediction cells and two
million embedding values per request. Increase `hopSeconds` when the embedding
budget is exceeded. Native inference is serialized and processed in batches of
64, including EffNet's fixed batch shape. Audio decoding may still require
substantial memory for long recordings. Requests are not progressive streams;
the frontend shows only the module's own loading indicator.

## Sources and licenses

Model metadata, checksums and URLs are pinned in `essentia_tf_models.json`.
See [official model catalog](https://essentia.upf.edu/models.html),
[EffNet algorithm](https://essentia.upf.edu/reference/std_TensorflowPredictEffnetDiscogs.html)
and [TempoCNN](https://essentia.upf.edu/reference/std_TempoCNN.html).
Essentia software and each model's weights have separate licenses. Many weights
are CC BY-NC-SA 4.0; do not assume this application's MIT license permits
commercial use of third-party software or model weights.
