# Valence / arousal curves

The optional backend uses the actual Essentia DEAM regression head, not CLAP
cosine matches or a loudness surrogate. Output order is **valence, arousal**.
Original DEAM scale [1,9] maps to [-1,1] by `(value - 5) / 4`.
Window predictions are contextual estimates, not per-note emotion ground truth.

## Setup

Normal `start.cmd` (Windows x64) or `bash start.command` (macOS Intel / Apple
Silicon) now automatically prepares the native runtime and the official weights,
then verifies real inference before starting HTTP. Windows uses a project-local
LLVM-MinGW build without Visual Studio; macOS uses a pinned official native wheel.
Repeated starts reuse verified artifacts. `--basic` or
`MAB_ESSENTIA_SETUP_MOOD=false` skips VA weights while keeping DSP.
See [first-run implementation](../native/essentia/README.md) for dependencies,
platform verification and licensing. The manual options below are for developers.

### Native Windows C++ (tested)

From the backend root, run this **once**:

```powershell
./scripts/build-essentia-native.ps1 -DownloadModels
```

Then start the backend normally. It auto-detects the native worker; no WSL,
Essentia Python binding, or extra service is needed. Uses MSVC + TensorFlow
C CPU 2.18.1. See [native implementation](../native/essentia/README.md) for
toolchain requirements, actual verification, limitations and licensing.
Set `MAB_ESSENTIA_NATIVE_EXECUTABLE` only for a custom executable location.

### Linux/macOS Python

Alternatively install the optional `essentia-tensorflow` package with pip
in that backend environment. First check that an upstream wheel exists for
the selected Python version. The Python binding still does not support
native Windows; the executable above bypasses that restriction.

Place these files under `MAB_MODEL_ROOT/essentia/`:

- [msd-musicnn-1.pb](https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.pb)
- [deam-msd-musicnn-2.pb](https://essentia.upf.edu/models/classification-heads/deam/deam-msd-musicnn-2.pb)

Or set `MAB_MOOD_EMBEDDING_GRAPH` and `MAB_MOOD_REGRESSION_GRAPH` in `.env`.
Weights load independently of CLAP/MuLan. The analysis API never downloads them;
the normal `start` preparation phase now downloads and verifies them by default.
Weights are CC BY-NC-SA 4.0, not covered by this project's MIT license;
review the upstream commercial licensing terms before commercial use.

## API

Authenticated `GET /v1/mood/capabilities` reports runtime/version and missing runtime or weights.
`POST /v1/interactive-assets/{id}:mood-curve` accepts `windowSeconds` (3-60),
`hopSeconds` (0.1-30), optional `timelineDurationSeconds`, and `cachePolicy`.
Output contains timestamped `valence` and `arousal` pairs. Cache keys include
audio hash, model checksums, native revision/runtime and analysis parameters. Jobs are limited to
10,000 points and 3-hour timelines. Existing frontend curve results persist
in the normal workspace document and reopen without recomputation.

Sources: [models](https://essentia.upf.edu/models.html),
[DEAM metadata](https://essentia.upf.edu/models/classification-heads/deam/deam-msd-musicnn-2.json),
[platform support](https://essentia.upf.edu/installing.html).
