# Native Essentia runtime

Windows uses upstream Essentia C++ standard algorithms and the official
TensorFlow C CPU library. macOS uses the official native C++ Python wheel in
an isolated DSP subprocess. No WASM, WSL or separately running service is required.
The existing backend launches
an isolated worker per VA or DSP job. The frontend keeps using the existing
mood API, plus the native feature API described in [feature modules](../../docs/essentia-features.md).

## Automatic first-run setup

Run `start.cmd` (Windows x64) or `bash start.command` (macOS Intel/Apple Silicon).
The startup workflow prepares the native runtime, checks all 29 DSP algorithms
on real PCM, and downloads/checks the official VA weights with real inference.
`--basic` prepares all DSP but skips CLAP/VA weights. Repeated starts reuse the
verified artifacts; failures stop startup before HTTP reports ready.

Windows downloads checksum-pinned LLVM-MinGW, Eigen, Essentia source and
TensorFlow C, installs pinned CMake/Ninja into the backend venv, and builds
`.tools/essentia-runtime/essentia-worker.exe`. Visual Studio, admin privileges,
system PATH changes and global toolchain installs are not required. TensorFlow's
required MSVC runtime DLLs are installed app-locally from an official Microsoft
archive. The full portable build, all 29 DSP analyses and VA inference have been
tested on Windows.

macOS uses architecture-specific official `essentia-tensorflow==2.1b6.dev1110`
CPython 3.11 wheels with NumPy 1.26. Intel and ARM archives are SHA-256 checked;
no Xcode/Homebrew build is needed. Both architectures have first-run/API CI
coverage configured, but must be verified by running those macOS jobs.

`MAB_ESSENTIA_SETUP=off` disables preparation, so availability is then the user's
responsibility. `MAB_ESSENTIA_SETUP_MOOD=false` leaves VA optional while ensuring
DSP. Existing `.env` is preserved. All binaries, downloads and models stay in
the backend and are git-ignored. Expect several GB of free disk for initial setup.

## Manual MSVC developer build

From the backend root in PowerShell, after the normal backend environment setup:

```powershell
./scripts/build-essentia-native.ps1 -DownloadModels
```

Requires CMake, Git, a Visual Studio C++ toolchain and Windows SDK. The default
generator matches this machine's Visual Studio 2026. For VS 2022, pass
`-Generator 'Visual Studio 17 2022'` (not tested on this machine).
The script downloads dependencies into `vendor/`, verifies archive SHA-256,
builds into `build/essentia/Release/`, and runs the native self-test. Model
downloads are opt-in through `-DownloadModels` in this developer script; normal
`start` now prepares VA by default. No global package installation is performed.

Start the backend normally afterwards. It auto-detects
`.tools/essentia-runtime/essentia-worker.exe` first, then the developer worker
`build/essentia/Release/essentia-worker.exe`. A custom location can be selected
with `MAB_ESSENTIA_NATIVE_EXECUTABLE` in `.env`; keep `tensorflow.dll` beside
the executable. `GET /v1/mood/capabilities` reports runtime, exact Essentia
revision and TensorFlow version. The worker accepts mono 16 kHz float32 PCM
on stdin; Python/librosa handles audio decoding and resampling.

## What is implemented and tested

- Essentia revision `7320015a1cad3ac1dc038b52ef94803587d09986`.
- MSVC 19.51 / Visual Studio 2026, Windows x64, Release, C++14.
- Eigen 3.4.0, bundled KISS FFT, official TensorFlow C CPU 2.18.1.
- Native self-test checks RMS, 13 MFCCs and 96 MusiCNN mel bands.
- 29 DSP analyses are exposed by `GET /v1/essentia/capabilities` and
  `POST /v1/interactive-assets/{assetId}:essentia`. Actual API tests exercise
  all 29, including spectral vectors, 6 matrices and 4 marker analyses.
  DSP does not need model weights; the linked TensorFlow DLL is still required.
- Actual `msd-musicnn-1` embedding + `deam-msd-musicnn-2` regression ran
  on the first 20 seconds of `k545-1.mp3`: 11 points with window 6 seconds,
  hop 2 seconds, about 1.14 seconds for native inference on this machine.
  This excludes decoding, is one smoke test, and is not a WASM benchmark.
- Real backend API tests verify model inference, timeline coordinates,
  normalized V/A bounds, disk cache and cache refresh. Mocked tests cover
  process failures, timeouts, invalid JSON, invalid output and missing runtime.

This is a **reduced standard-algorithm build**, not the entire Essentia suite.
Essentia's old MSVC streaming `RogueVector` implementation accesses removed
STL internals. The build generator omits streaming namespace definitions in
copies of the selected algorithm sources, without modifying the vendor
checkout or using unsafe vector-layout patches. The feature extraction math
is the original `TensorflowInputMusiCNN`. A small native adapter handles the
187-frame/93-hop patches and repeated final patch, then uses upstream
`TensorflowPredict` for both graphs. Standard FrameCutter retains silent
frames (the upstream streaming default adds tiny noise). Full numerical
parity with a Linux streaming/Python build has not yet been tested.

Models load once per job; sessions are reused across windows. Job exceptions
do not crash the API process. Inference is CPU-only in this Windows build.
The official TensorFlow DLL is large (about 910 MiB); do not commit binaries,
archives or model weights. The backend wheel does not bundle this worker.

## Licensing

The backend's MIT license does **not** cover Essentia or model weights.
Essentia is AGPL-3.0 (or an upstream commercial license); this executable
statically links it. Running it as a separate process does not automatically
remove license obligations. Preserve upstream source/patch availability,
notices and dependency licenses when distributing it, and review the
upstream terms for deployment. The DEAM/MusiCNN weights are CC BY-NC-SA 4.0,
with separate proprietary licensing available from the authors.

Official references:
[Windows support](https://essentia.upf.edu/installing.html#building-essentia-on-windows),
[TensorFlow C packages](https://www.tensorflow.org/install/lang_c),
[Essentia license](https://github.com/MTG/essentia/blob/master/COPYING.txt),
[model terms](https://essentia.upf.edu/models.html).
