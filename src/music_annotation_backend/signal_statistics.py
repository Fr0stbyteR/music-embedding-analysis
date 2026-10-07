"""Frame-wise DSP and descriptive statistics, independently implemented with NumPy.

The SInES inventory is a reference for feature selection, not source code.
Definitions and limitations are documented in docs/signal-statistics.md.
"""
from __future__ import annotations

import math
import base64
from typing import Any

import librosa
import numpy as np
from scipy.signal import find_peaks

SIGNAL_STATISTICS_ALGORITHMS = {
    "peakAmplitude", "crestFactor", "rmsDb", "dcOffset", "timeSkewness",
    "timeKurtosis", "temporalCentroid", "spectralTilt", "spectralSlope",
    "spectralDecrease", "spectralSmoothness", "spectralIrregularityJensen",
    "spectralIrregularityKrimphoff", "positiveSpectralFlux", "spectralTonality",
    "harmonicNoiseRatio", "formants", "lpcc",
}


def option(options: dict, name: str, default: float, low: float, high: float,
           *, integer: bool = False) -> float | int:
    value = options.get(name, default)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if isinstance(value, bool) or not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    if integer and not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(number) if integer else number


def describe(values: np.ndarray, valid: np.ndarray | None = None) -> dict[str, float | int]:
    """Population statistics; only finite, valid observations are included."""
    array = np.asarray(values, dtype=np.float64)
    selected = array[np.isfinite(array) & (valid if valid is not None else True)]
    summary: dict[str, float | int] = {"count": int(selected.size)}
    if selected.size:
        summary.update({
            "min": float(np.min(selected)), "max": float(np.max(selected)),
            "mean": float(np.mean(selected)), "median": float(np.median(selected)),
            "std": float(np.std(selected)), "p05": float(np.percentile(selected, 5)),
            "p95": float(np.percentile(selected, 95)),
        })
    return summary


def _lpc(frame: np.ndarray, order: int, emphasis: float) -> np.ndarray | None:
    centered = frame - np.mean(frame)
    filtered = np.concatenate((centered[:1], centered[1:] - emphasis * centered[:-1]))
    filtered *= np.hamming(len(frame))
    if np.dot(filtered, filtered) < 1e-16:
        return None
    try:
        coefficients = librosa.lpc(filtered, order=order)
    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
        return None
    return coefficients if np.all(np.isfinite(coefficients)) else None


def _lpcc(coefficients: np.ndarray, count: int) -> np.ndarray:
    # Unit-gain predictor cepstrum: c0=0; c1..cN via the standard recursion.
    cepstrum = np.zeros(count + 1)
    order = len(coefficients) - 1
    for n in range(1, count + 1):
        cepstrum[n] = -(coefficients[n] if n <= order else 0)
        for k in range(max(1, n - order), n):
            cepstrum[n] -= k / n * cepstrum[k] * coefficients[n - k]
    return cepstrum[1:]


def _hnr(frame: np.ndarray, sample_rate: int, fmin: float, fmax: float) -> float:
    centered = frame - np.mean(frame)
    n = len(frame)
    spectrum = np.fft.rfft(centered, n=2 * n)
    correlation = np.fft.irfft(np.abs(spectrum) ** 2, n=2 * n)[:n]
    squared = np.concatenate(([0.], np.cumsum(centered ** 2)))
    lags = np.arange(n)
    denominator = np.sqrt(np.maximum((squared[n - lags]) * (squared[n] - squared[lags]), 0))
    normalized = np.divide(correlation, denominator, out=np.zeros(n), where=denominator > 1e-20)
    low, high = max(1, int(sample_rate / fmax)), min(n // 2, int(sample_rate / fmin))
    peaks, _ = find_peaks(normalized)
    peaks = peaks[(peaks >= low) & (peaks <= high)]
    strength = float(np.max(normalized[peaks])) if len(peaks) else 0.
    strength = np.clip(strength, 1e-6, 1 - 1e-6)
    return float(10 * np.log10(strength / (1 - strength)))


def analyze_signal(y: np.ndarray, sample_rate: int, algorithm: str, options: dict[str, Any]) -> dict:
    if algorithm not in SIGNAL_STATISTICS_ALGORITHMS:
        raise ValueError(f"unsupported signal statistic: {algorithm}")
    frame_length = int(option(options, "frameLength", 4096 if algorithm == "harmonicNoiseRatio" else 2048, 128, 8192, integer=True))
    if frame_length % 2:
        raise ValueError("frameLength must be even for centered analysis")
    hop = int(option(options, "hopLength", 512, 1, 8192, integer=True))
    threshold = float(option(options, "silenceThresholdDb", -60, -120, 0))
    if len(y) == 0 or not np.all(np.isfinite(y)):
        raise ValueError("audio must contain finite samples")
    count = 1 + len(y) // hop
    if count > 2_000_000:
        raise ValueError("too many analysis frames; increase hopLength")
    padded = np.pad(np.asarray(y, dtype=np.float64), (frame_length // 2, frame_length // 2))
    frames = librosa.util.frame(padded, frame_length=frame_length, hop_length=hop).T
    channels = 3 if algorithm == "formants" else 1
    if algorithm == "lpcc":
        channels = int(option(options, "coefficients", 13, 1, 64, integer=True))
    if channels * count > 5_000_000:
        raise ValueError("analysis result is too large; increase hopLength or reduce coefficients")
    values = np.zeros((channels, count), dtype=np.float64)
    validity = np.zeros_like(values, dtype=bool)
    nyquist = sample_rate / 2
    frequencies = np.fft.rfftfreq(frame_length, 1 / sample_rate)
    window = np.hanning(frame_length + 1)[:-1]
    previous_spectrum = np.zeros(len(frequencies) - 1)
    unit = "Ratio"
    metadata: dict[str, Any] = {
        "frameLength": frame_length, "hopLength": hop, "centered": True,
        "silenceThresholdDb": threshold, "method": algorithm,
    }
    if algorithm == "spectralTilt":
        split = float(option(options, "splitFrequency", 1000, 1, nyquist))
        if split >= nyquist:
            raise ValueError("splitFrequency must be below Nyquist")
        metadata["splitFrequency"] = split
    if algorithm in {"spectralSlope", "harmonicNoiseRatio", "formants"}:
        fmin = float(option(options, "fmin", 60 if algorithm == "harmonicNoiseRatio" else 90 if algorithm == "formants" else 20, 1, nyquist))
        fmax = float(option(options, "fmax", min(2000 if algorithm == "harmonicNoiseRatio" else 5000 if algorithm == "formants" else 10000, nyquist * .95), 1, 96000))
        fmax = min(fmax, nyquist * .999)
        if fmax <= fmin:
            raise ValueError("fmax must exceed fmin and fit the audio sample rate")
        if algorithm == "harmonicNoiseRatio" and frame_length < 2 * sample_rate / fmin:
            raise ValueError("frameLength must cover at least two periods at fmin")
        metadata.update({"fmin": fmin, "fmax": fmax})
    if algorithm in {"formants", "lpcc"}:
        order = int(option(options, "lpcOrder", 16, 2, min(64, frame_length - 2), integer=True))
        emphasis = float(option(options, "preEmphasis", .97, 0, .999))
        metadata.update({"lpcOrder": order, "preEmphasis": emphasis, "missingValue": 0})
    if algorithm == "formants":
        max_bandwidth = float(option(options, "maximumBandwidth", 700, 1, 5000))
        metadata["maximumBandwidth"] = max_bandwidth
    if algorithm == "spectralSlope":
        slope_bins = (frequencies >= fmin) & (frequencies <= fmax)
        x = np.log2(frequencies[slope_bins])
        if x.size < 2:
            raise ValueError("slope frequency range must contain at least two FFT bins")
        x -= np.mean(x)

    # Only 256 frames are materialized per batch; no full-track complex STFT copy.
    for start in range(0, count, 256):
        batch = frames[start:start + 256]
        rms = np.sqrt(np.mean(batch ** 2, axis=1))
        active = rms > max(10 ** (threshold / 20), 1e-12)
        if algorithm in {"formants", "lpcc", "harmonicNoiseRatio"}:
            for local, frame in enumerate(batch):
                index = start + local
                if not active[local]:
                    if algorithm == "harmonicNoiseRatio":
                        values[0, index] = -60
                    continue
                if algorithm == "harmonicNoiseRatio":
                    values[0, index] = _hnr(frame, sample_rate, fmin, fmax)
                    validity[0, index] = True
                    unit = "dB"
                    continue
                coefficients = _lpc(frame, order, emphasis)
                if coefficients is None:
                    continue
                if algorithm == "lpcc":
                    values[:, index] = _lpcc(coefficients, channels)
                    validity[:, index] = True
                else:
                    roots = np.roots(coefficients)
                    roots = roots[np.imag(roots) > 0]
                    frequency = np.angle(roots) * sample_rate / (2 * np.pi)
                    bandwidth = -sample_rate / np.pi * np.log(np.maximum(np.abs(roots), 1e-12))
                    candidates = np.sort(frequency[(frequency >= fmin) & (frequency <= fmax) & (bandwidth > 0) & (bandwidth <= max_bandwidth)])[:3]
                    values[:len(candidates), index] = candidates
                    validity[:len(candidates), index] = True
            continue
        if algorithm in {"peakAmplitude", "crestFactor", "rmsDb", "dcOffset", "timeSkewness", "timeKurtosis", "temporalCentroid"}:
            peak = np.max(np.abs(batch), axis=1)
            if algorithm == "peakAmplitude":
                data, unit = peak, "Amplitude"
            elif algorithm == "crestFactor":
                data = np.divide(peak, rms, out=np.zeros_like(rms), where=rms > 1e-12)
            elif algorithm == "rmsDb":
                data, unit = 20 * np.log10(np.maximum(rms, 1e-6)), "dBFS"
            elif algorithm == "dcOffset":
                data, unit = np.mean(batch, axis=1), "Amplitude"
            elif algorithm == "temporalCentroid":
                energy = np.sum(batch ** 2, axis=1)
                data = np.divide(batch ** 2 @ np.linspace(0, 1, frame_length), energy, out=np.zeros_like(energy), where=energy > 1e-20)
            else:
                centered = batch - np.mean(batch, axis=1, keepdims=True)
                variance = np.mean(centered ** 2, axis=1)
                active &= variance > 1e-12
                power = 3 if algorithm == "timeSkewness" else 4
                denominator = variance ** (power / 2)
                data = np.divide(np.mean(centered ** power, axis=1), denominator, out=np.zeros_like(variance), where=denominator > 1e-24)
                if power == 4:
                    data = np.where(variance > 1e-12, data - 3, 0)
                unit = "Skewness" if power == 3 else "Excess kurtosis"
        else:
            amplitude = np.abs(np.fft.rfft(batch * window, axis=1)) / np.sum(window)
            spectrum = amplitude[:, 1:]
            energy = spectrum ** 2
            if algorithm == "spectralTilt":
                low = np.sum(energy[:, frequencies[1:] < split], axis=1)
                high = np.sum(energy[:, frequencies[1:] >= split], axis=1)
                data, unit = np.clip(10 * np.log10(np.maximum(low, 1e-20) / np.maximum(high, 1e-20)), -120, 120), "dB"
            elif algorithm == "spectralSlope":
                db = 20 * np.log10(np.maximum(amplitude[:, slope_bins], 1e-12))
                data, unit = db @ x / np.sum(x ** 2), "dB/oct"
            elif algorithm == "spectralDecrease":
                numerator = np.sum((spectrum[:, 1:] - spectrum[:, :1]) / np.arange(1, spectrum.shape[1]), axis=1)
                denominator = np.sum(spectrum[:, 1:], axis=1)
                data = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-20)
            elif algorithm == "spectralSmoothness":
                data, unit = np.mean(np.abs(np.diff(20 * np.log10(np.maximum(spectrum, 1e-12)), n=2, axis=1)), axis=1), "dB"
            elif algorithm == "spectralIrregularityJensen":
                numerator, denominator = np.sum(np.diff(spectrum, axis=1) ** 2, axis=1), np.sum(energy, axis=1)
                data = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-20)
            elif algorithm == "spectralIrregularityKrimphoff":
                data = np.sum(np.abs(spectrum[:, 1:-1] - (spectrum[:, :-2] + spectrum[:, 1:-1] + spectrum[:, 2:]) / 3), axis=1)
                unit = "Amplitude"
            elif algorithm == "positiveSpectralFlux":
                norm = np.linalg.norm(spectrum, axis=1, keepdims=True)
                normalized = np.divide(spectrum, norm, out=np.zeros_like(spectrum), where=norm > 1e-20)
                normalized[~active] = 0
                difference = normalized - np.vstack((previous_spectrum, normalized[:-1]))
                previous_spectrum = normalized[-1].copy()
                data, unit = np.linalg.norm(np.maximum(difference, 0), axis=1), "Flux"
            else:  # flatness-derived proxy, not a learned tonal/harmonic classifier
                flatness = np.exp(np.mean(np.log(np.maximum(energy, 1e-20)), axis=1)) / np.maximum(np.mean(energy, axis=1), 1e-20)
                data = np.clip(-10 * np.log10(np.maximum(flatness, 1e-6)) / 60, 0, 1)
        values[0, start:start + len(batch)] = np.where(active, data, -120 if algorithm == "rmsDb" else 0)
        validity[0, start:start + len(batch)] = active

    if algorithm == "formants":
        unit = "Hz"
        metadata.update({"channel0": "F1", "channel1": "F2", "channel2": "F3"})
    if algorithm == "lpcc":
        unit = "Coefficient"
        metadata["firstCoefficient"] = 1
    if algorithm == "harmonicNoiseRatio":
        unit = "dB"
    metadata["unit"] = unit
    # The frontend persists Float32 data: do not emit values that overflow that
    # format, even if a pathological high-order predictor stays finite in float64.
    representable = np.isfinite(values) & (np.abs(values) <= np.finfo(np.float32).max)
    validity &= representable
    values[~representable] = 0
    for index, vector in enumerate(values):
        metadata[f"validity.{index}"] = base64.b64encode(np.packbits(validity[index], bitorder="little").tobytes()).decode("ascii")
        for key, value in describe(vector, validity[index]).items():
            metadata[f"statistics.{index}.{key}"] = value
    result: dict[str, Any] = {"metadata": metadata}
    if algorithm == "lpcc":
        result["matrix"] = values.T.tolist()
        metadata.update({"bins": channels, "minValue": float(np.min(values)), "maxValue": float(np.max(values))})
    else:
        result["vectors"] = values.tolist()
    return result
