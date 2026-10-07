# Signal statistics

The [SInES Tools](https://sinestools.univie.ac.at/index.html) feature inventory
inspired this extension. Implementations are independent NumPy/SciPy/librosa DSP,
not copies of website code and not claims of bit-for-bit compatibility.

Use the existing authenticated `POST /v1/interactive-assets/{id}:librosa` endpoint.
No model downloads, optional Essentia runtime, or additional dependencies are needed.
Restart the backend after upgrading. For example:

```json
{"algorithm":"spectralTilt","options":{"frameLength":2048,"hopLength":512,"silenceThresholdDb":-60,"splitFrequency":1000},"cachePolicy":"use"}
```

## Analysis and conventions

Audio is decoded at its original rate and mixed to mono, as for other librosa
modules. Frames are centered at `i * hopLength` with zero padding at file edges.
The endpoint reports original sample rate/duration; frontends map frames to their
decoded playback timeline. Frame lengths must be even (128–8192), hop lengths
1–8192. Spectral magnitudes use a periodic Hann window and normalization by its
sum. DC is omitted for spectral distribution features. Work is batched in 256
frames to avoid materializing a whole-track complex STFT.
At most five million scalar results are returned; choose a larger hop or fewer
LPCC coefficients for longer files. Unrepresentable/invalid predictor outputs
are excluded rather than becoming infinity in frontend Float32 storage.

| Algorithm | Definition / output |
| --- | --- |
| `peakAmplitude` | Maximum absolute sample amplitude |
| `crestFactor` | Peak / RMS, a linear ratio (not dB) |
| `rmsDb` | 20 log10(RMS), floor −120 dBFS; **not SPL** |
| `dcOffset` | Mean signed sample amplitude |
| `timeSkewness` | Third central sample moment / variance^(3/2) |
| `timeKurtosis` | Fourth central sample moment / variance² − 3 (Fisher excess kurtosis) |
| `temporalCentroid` | Energy-weighted sample position within each frame, scaled to [0,1] |
| `spectralTilt` | 10 log10(power below / power above `splitFrequency`), clipped ±120 dB |
| `spectralSlope` | Least-squares amplitude-dB slope against log2 frequency within `fmin`/`fmax`, dB/octave |
| `spectralDecrease` | Σ(k>0) (A[k]−A[0])/k divided by Σ(k>0) A[k]; A[0] is first non-DC bin |
| `spectralSmoothness` | Mean absolute second difference of amplitude-dB spectrum; higher means less smooth |
| `spectralIrregularityJensen` | Σ adjacent (A[k]−A[k+1])² / Σ A[k]² |
| `spectralIrregularityKrimphoff` | Sum of deviations from a three-bin local amplitude mean; gain dependent |
| `positiveSpectralFlux` | L2 norm of positive differences between unit-L2-normalized spectra; gain invariant |
| `spectralTonality` | clamp(−10 log10(power flatness)/60, 0,1); **a proxy, not a key estimator** |
| `harmonicNoiseRatio` | Normalized autocorrelation peak in `fmin`/`fmax` period range; 10 log10(r/(1−r)), bounded ±60 dB |
| `formants` | Three ascending LPC root resonance candidates F1/F2/F3, each a Hz vector |
| `lpcc` | Unit-gain predictor cepstral coefficients c1…cN, time-major matrix |

## Silence and estimates

`silenceThresholdDb` (default −60 dBFS) gates frames before estimating features.
Gated frames have value 0, except RMS level (−120 dBFS) and HNR (−60 dB).
They are excluded from summary statistics. This gate is a digital threshold,
not calibrated hearing or noise-floor estimation.

HNR is an **estimate for isolated, approximately stationary pitched sounds**.
It must not be interpreted as exact harmonic/noise source energies in mixed
music. Analysis windows must contain at least two periods at `fmin`; otherwise
the endpoint returns 422. A dominant autocorrelation peak can represent octave
or periodicity ambiguities.

LPC candidates are **resonance estimates**, not reliable anatomical formants in
polyphonic music. Remove DC, apply `preEmphasis` (default .97), use a Hamming
window, and estimate Burg LPC (`lpcOrder`, default 16). Positive-angle stable
roots within `fmin`/`fmax` and below `maximumBandwidth` are retained. Missing
candidates are zero, explicitly advertised by `metadata.missingValue=0`, and
excluded from summaries. Candidate indices may switch when resonances vanish;
this is not a temporal formant-tracking model. LPCC defaults to 13 coefficients;
c0 (gain) is omitted. Invalid predictor frames are zero and excluded.

## Statistics and cache

Each returned vector has flat numeric metadata fields:
`statistics.0.count`, `.min`, `.max`, `.mean`, `.median`, `.std`, `.p05`, `.p95`
(replace 0 with the channel index). Standard deviation uses the population
convention (`ddof=0`). All-invalid data has count 0 and no misleading numeric
mean. Summaries describe **feature frames across the whole audio**, not a cursor
selection, acoustic samples, independent statistical observations, or inferential
significance. Existing librosa vector modules also return descriptive summaries.
Constant-amplitude frames have no defined sample skewness/kurtosis and are
excluded from those summaries (displayed as zero).

Results use the existing backend cache; both the original engine file and this
DSP implementation contribute to its version hash. Frontend IndexedDB and folder
`.audio_toolkit` storage preserve vectors/matrices and statistics. Cache menu
indicators and bulk cached-module loading use each module's analysis descriptor;
display color does not invalidate results. Older cached legacy vectors may lack
summary metadata until explicitly reanalyzed.

New DSP results also include `metadata["validity.N"]`: a base64 little-bit-order
bitmap, bit i indicating whether frame i is a valid observation in vector N.
For LPCC, N is the coefficient index. Zero remains a legitimate numeric value;
the bitmap distinguishes it from gated or failed estimates. Existing JSON and
binary workspace caches preserve these primitive metadata strings.
The frontend computes range summaries from original numeric arrays. A positive
selection uses [start,end); otherwise the full audio is used (including an exact
endpoint control point). Statistics are observation-wise, not time-weighted;
arithmetic dB averages must not be interpreted as integrated acoustic power.
Legacy results without a bitmap include finite values only. Reanalyze if missing
estimate filtering matters. Changes to all three DSP files invalidate the backend
cache version.

## Roughness estimates

The [SInES roughness inventory](https://sinestools.univie.ac.at/roughness_estimator.htm)
is covered by six `roughness…` algorithms. Existing RMS and Bark-band modules
remain available; no duplicate RMS module is added. All six are Vector curves.

Spectral variants: `roughnessVassilakis`, `roughnessSethares`,
`roughnessPlompLevelt`. Hann-normalized amplitudes with interpolated peak
frequencies, at most `maxPeaks` (80), above `peakThresholdDb` (−40 relative).
Unordered peak pairs use exp(−a·s·Δf)−exp(−b·s·Δf), summed with an amplitude
weight and scale .28. Sethares uses s=.24/(.0207·min(f)+18.96), a=3.51,b=5.75;
Plomp–Levelt uses .021,19,3.5,5.25. Vassilakis uses
.5·(a1·a2)^.1·(2·min(a1,a2)/(a1+a2))^3.11 instead of amplitude product.
The minimum amplitude, not the lower-frequency amplitude, is used: unequal
partials remain symmetric. FFT normalization/interpolation means values are not
bitwise or scale-compatible with the website's raw FFT magnitudes.

Time variants: `roughnessZwicker`, `roughnessAures`, `roughnessDanielWeber`.
These are **simplified Bark-envelope modulation proxies**, not complete published
auditory-filter/psychoacoustic models. Twenty-five critical bands up to 20 kHz
are sampled internally at ≥800 Hz with ~10 ms windows. Rolling energy mean,
variance and hysteretic crossing rate over `modulationWindowSeconds` (.4 s)
estimate modulation depth and rate (15–350 Hz). Weighting peaks near 70 Hz.
Zwicker combines depth and weighting; Aures adds level weighting; Daniel/Weber
uses squared depth and a different rate/level weighting. Unlike the website,
local rolling baselines and adaptive envelope hops avoid aliasing at lower
audio sample rates; results will differ. `referenceLevelDb` (120) is an explicit
digital-level offset, **not microphone calibration**. Outputs are labeled
**Roughness index**, never certified Asper. Polyphonic, transient or noisy signals
may be unreliable.

Defaults: frameLength=4096, hopLength=512, smoothingFrames=1. Optional centered
smoothing is validity-aware. Spectral processing is batched; modulation buffers
are capped at 12 million band values (~10 min) and processed one band at a time.
Output is capped at 500,000 frames; adjust hopLength or split unusually long input.
These modules use the same authenticated endpoint, caching and dependencies.

Primary background: [Vassilakis dissertation](https://www.acousticslab.org/papers/diss.htm),
[Sethares spectral-pair model](https://sethares.engr.wisc.edu/paperspdf/adaptun2002.pdf).

## Not included in this extension

Existing RMS, ZCR, centroid, bandwidth, rolloff, flatness, pitch, MFCC, chroma,
Essentia Bark/ERB bands, dissonance and other covered features are not duplicated.
Absolute SPL/phon/sone with measurement calibration, room impulse-response
metrics, auditory models, and inferential spreadsheet tools (t/ANOVA, κ, GMM/PCA,
cross-series lag correlation) need separate acquisition or UX and are not
pretended to be ordinary per-frame sound descriptors.

References:

- [SInES Signal Analysis II](https://sinestools.univie.ac.at/jsxtract_analyser.htm)
- [SInES Harmonicity Analyzer](https://sinestools.univie.ac.at/harmonicnoiseratio.htm)
- [SciPy kurtosis definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html)
