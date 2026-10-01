# Score images and PDF → MusicXML

In the frontend MusicXML module, choose **Image / PDF → MusicXML**, either in
the empty module or its Analysis sidebar. The authenticated backend uploads the
selected file, runs HOMR 0.7 on CPU, and returns MusicXML directly into the
existing score module. The directory handle and other files are not uploaded.
The generated score uses the existing score library and workspace persistence.

Normal backend startup needs no extra command. First OMR use creates a separate
Python 3.12 environment in `.tools/homr` using the launcher's project-local `uv`.
It downloads dependencies and official model weights (internet needed). Models
are stored in `MAB_MODEL_ROOT/homr`; install logs in
`MAB_DATA_ROOT/omr-logs/install.log`; recognition logs and intermediate pages in
`MAB_DATA_ROOT/omr-uploads/<job>/homr.log`. Results are content-addressed in
`MAB_DATA_ROOT/omr-cache`, including the worker adapter version.

HOMR's NumPy 2 environment does not alter the main audio/Essentia environment.
Pinned CPU dependencies have been exercised on Windows x64, with a real JPG
and a two-page PDF. Apple Silicon requires macOS 14+ for the pinned ONNX wheel;
it has not been exercised from this Windows workspace. HOMR 0.7 requires ONNX
Runtime >=1.24.1, which does **not** ship macOS Intel wheels. An Intel Mac can
connect the web frontend to a compatible remote backend, or provide a custom
compatible runtime using `MAB_OMR_PYTHON`; automatic setup reports this limitation
instead of attempting an impossible installation.

Configuration in backend `.env`:

```dotenv
MAB_OMR_AUTO_INSTALL=true
MAB_OMR_TIMEOUT_SECONDS=1800
# Optional alternative, already installed HOMR 0.7 environment:
# MAB_OMR_PYTHON=/absolute/path/to/python
```

PNG, JPG/JPEG, WebP and TIFF images, and 1–20 page PDFs, are accepted up to
50 MiB. PDFs render through PDFium; no external Ghostscript installation is
required. Image size is capped at 40 million pixels, PDF renders at 20 million
pixels per page. Jobs are serialized to avoid concurrent model memory spikes.
Cancel stops the recognition process; failed/cancelled jobs do not produce a
completed cache entry. Longer PDFs should be split before upload.

PDF pages are processed in order, merged by corresponding part, and measure
numbers renumbered. Pages with missing parts or no recognized notes are rejected
instead of silently importing an incomplete score. Verify page joins yourself:
page-local voice, tie, slur and part recognition can be wrong. Staff count must
remain consistent; this is not orchestral score reconstruction across arbitrary
instrumentation changes.

This is **recognition, not ground truth**. Check pitch, rhythm, voices, signatures,
repeats, tuplets, slurs and titles before DTW alignment. In a real HOMR example,
Verovio reported unresolved slurs; we retain the generated music but display a
review reminder rather than falsely claiming a corrected score.

API: `GET /v1/omr/capabilities`; `POST /v1/omr` with raw file bytes and URL-encoded
`X-File-Name`; then poll `/v1/jobs/{jobId}`. The standard job cancel endpoint
also applies. All endpoints use the existing bearer authentication.

Upstream: [HOMR](https://github.com/liebharc/homr),
[PyPI package](https://pypi.org/project/homr/),
[ONNX Runtime wheels](https://pypi.org/project/onnxruntime/1.24.1/#files).
HOMR is AGPL-3.0; review the upstream software and model terms when distributing
or hosting it. OMR is not part of the native Essentia DSP installation.
