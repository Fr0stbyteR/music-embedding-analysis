"""Independent spectral-pair and Bark modulation roughness estimates.

Relative digital-audio indices, not calibrated Asper measurements.
See docs/signal-statistics.md for equations and differences from SInES.
"""
from __future__ import annotations

import base64
from typing import Any

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from .signal_statistics import describe, option

ROUGHNESS_ALGORITHMS = {
    "roughnessVassilakis", "roughnessSethares", "roughnessPlompLevelt",
    "roughnessZwicker", "roughnessAures", "roughnessDanielWeber",
}
_EDGES = np.array([0, 100, 200, 300, 400, 510, 630, 770, 920, 1080,
                   1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700,
                   4400, 5300, 6400, 7700, 9500, 12000, 15500, 20000])


def pair_roughness(frequencies: np.ndarray, amplitudes: np.ndarray, algorithm: str) -> float:
    """Sum unordered partial pairs, with Hann-normalized peak amplitudes."""
    i, j = np.triu_indices(len(frequencies), 1)
    low = np.minimum(frequencies[i], frequencies[j])
    difference = np.abs(frequencies[i] - frequencies[j])
    product = amplitudes[i] * amplitudes[j]
    if algorithm == "roughnessPlompLevelt":
        scale = .24 / (.021 * low + 19)
        curve = np.exp(-3.5 * scale * difference) - np.exp(-5.25 * scale * difference)
    else:
        scale = .24 / (.0207 * low + 18.96)
        curve = np.exp(-3.51 * scale * difference) - np.exp(-5.75 * scale * difference)
    if algorithm == "roughnessVassilakis":
        ratio = 2 * np.minimum(amplitudes[i], amplitudes[j]) / np.maximum(amplitudes[i] + amplitudes[j], 1e-20)
        product = .5 * product ** .1 * ratio ** 3.11
    return float(.28 * np.sum(product * curve))


def _modulation_values(energy: np.ndarray, frame_rate: float, history: int,
                       algorithm: str, reference_db: float) -> np.ndarray:
    # Process one band at a time: bounded memory, no history-window tensor.
    result = np.zeros(len(energy))
    indices = np.arange(len(energy))
    for band in energy.T:
        mean = uniform_filter1d(band, history, mode="nearest")
        square = uniform_filter1d(band ** 2, history, mode="nearest")
        depth = np.clip(np.sqrt(2 * np.maximum(square - mean ** 2, 0)) / np.maximum(mean, 1e-20), 0, 1)
        centered = band - mean
        state = np.where(centered > .05 * mean, 1, np.where(centered < -.05 * mean, -1, 0))
        last_active = np.maximum.accumulate(np.where(state != 0, indices, 0))
        held = state[last_active]
        previous = np.concatenate(([0], held[:-1]))
        crossing = ((held != previous) & (previous != 0)).astype(float)
        frequency = uniform_filter1d(crossing, history, mode="constant") * frame_rate / 2
        usable = (mean > 1e-16) & (depth >= .05) & (frequency >= 15) & (frequency <= 350)
        ratio = np.maximum(frequency, 1e-12) / 70
        weight = 1 / (1 + (ratio - 1 / ratio) ** 2)
        level = 10 * np.log10(np.maximum(mean, 1e-20)) + reference_db
        if algorithm == "roughnessZwicker":
            values = .0425 * depth * weight
        elif algorithm == "roughnessAures":
            usable &= level > 20
            values = .057557 * depth * weight * np.minimum(2, (np.maximum(level, 0) / 70) ** 1.5)
        else:
            weight = 4 / (ratio + 1 / ratio) ** 2
            factor = np.where(level < 40, .5, (np.maximum(level, 0) / 70) ** 1.2)
            values = .05633 * depth ** 2 * weight * factor
        result += np.where(usable, values, 0)
    return result


def analyze_roughness(y: np.ndarray, sample_rate: int, algorithm: str, options: dict[str, Any]) -> dict:
    if algorithm not in ROUGHNESS_ALGORITHMS:
        raise ValueError(f"unsupported roughness model: {algorithm}")
    frame_length = int(option(options, "frameLength", 4096, 128, 8192, integer=True))
    hop = int(option(options, "hopLength", 512, 1, 8192, integer=True))
    threshold = float(option(options, "silenceThresholdDb", -60, -120, 0))
    smoothing = int(option(options, "smoothingFrames", 1, 1, 101, integer=True))
    if frame_length % 2 or len(y) == 0 or not np.all(np.isfinite(y)):
        raise ValueError("audio must be finite and nonempty; frameLength must be even")
    count = 1 + len(y) // hop
    if count > 500_000:
        raise ValueError("too many roughness frames; increase hopLength")
    metadata: dict[str, Any] = {"frameLength": frame_length, "hopLength": hop,
                               "centered": True, "unit": "Roughness index", "method": algorithm,
                               "calibrated": False, "silenceThresholdDb": threshold,
                               "smoothingFrames": smoothing}
    frames = librosa.util.frame(np.pad(y, (frame_length // 2, frame_length // 2)),
                                frame_length=frame_length, hop_length=hop).T
    valid = np.zeros(count, dtype=bool)
    values = np.zeros(count)
    for start in range(0, count, 256):
        block = frames[start:start + 256]
        valid[start:start + len(block)] = np.sqrt(np.mean(block.astype(float) ** 2, axis=1)) > max(1e-12, 10 ** (threshold / 20))
    if algorithm in {"roughnessVassilakis", "roughnessSethares", "roughnessPlompLevelt"}:
        maximum = int(option(options, "maxPeaks", 80, 2, 128, integer=True))
        peak_db = float(option(options, "peakThresholdDb", -40, -80, -6))
        metadata.update({"maxPeaks": maximum, "peakThresholdDb": peak_db})
        window = np.hanning(frame_length + 1)[:-1]
        for start in range(0, count, 256):
            spectra = 2 * np.abs(np.fft.rfft(frames[start:start + 256] * window, axis=1)) / np.sum(window)
            for local, spectrum in enumerate(spectra):
                index = start + local
                if not valid[index]:
                    continue
                peaks, _ = find_peaks(spectrum, height=np.max(spectrum) * 10 ** (peak_db / 20))
                if len(peaks) > maximum:
                    peaks = peaks[np.argsort(spectrum[peaks])[-maximum:]]
                # Log-magnitude interpolation reduces FFT-bin quantization.
                left, center, right = np.log(np.maximum(spectrum[peaks - 1], 1e-20)), np.log(np.maximum(spectrum[peaks], 1e-20)), np.log(np.maximum(spectrum[peaks + 1], 1e-20))
                denominator = left - 2 * center + right
                delta = np.divide(.5 * (left - right), denominator, out=np.zeros_like(center), where=np.abs(denominator) > 1e-12)
                delta = np.clip(delta, -.5, .5)
                frequency = (peaks + delta) * sample_rate / frame_length
                amplitude = np.exp(center - .25 * (left - right) * delta)
                values[index] = pair_roughness(frequency, amplitude, algorithm)
    else:
        seconds = float(option(options, "modulationWindowSeconds", .4, .1, 2))
        reference = float(option(options, "referenceLevelDb", 120, 0, 140))
        # Internal envelope sampling >= 800 Hz resolves 15–350 Hz even with a
        # coarse output hop. A ~10 ms window limits carrier leakage.
        envelope_hop = max(1, sample_rate // 800)
        short_length = max(128, int(2 ** round(np.log2(sample_rate * .01))))
        envelope_count = 1 + len(y) // envelope_hop
        if envelope_count * 25 > 12_000_000:
            raise ValueError("modulation roughness supports up to about 10 minutes per request")
        short_frames = librosa.util.frame(np.pad(y, (short_length // 2, short_length // 2)), frame_length=short_length, hop_length=envelope_hop).T
        energy = np.zeros((envelope_count, 25))
        window = np.hanning(short_length + 1)[:-1]
        frequencies = np.fft.rfftfreq(short_length, 1 / sample_rate)
        bands = [(frequencies > 0) & (frequencies >= lo) & (frequencies < hi) for lo, hi in zip(_EDGES[:-1], _EDGES[1:])]
        for start in range(0, envelope_count, 256):
            power = (2 * np.abs(np.fft.rfft(short_frames[start:start + 256] * window, axis=1)) / np.sum(window)) ** 2
            for band, mask in enumerate(bands):
                energy[start:start + len(power), band] = np.sum(power[:, mask], axis=1)
        rate = sample_rate / envelope_hop
        history = max(3, round(seconds * rate))
        dense = _modulation_values(energy, rate, history, algorithm, reference)
        values = np.interp(np.arange(count) * hop, np.arange(envelope_count) * envelope_hop, dense)
        metadata.update({"modulationWindowSeconds": seconds, "referenceLevelDb": reference,
                         "envelopeHopLength": envelope_hop, "shortFrameLength": short_length,
                         "estimated": True})
    if smoothing > 1:
        weights = uniform_filter1d(valid.astype(float), smoothing, mode="nearest")
        values = np.divide(uniform_filter1d(np.where(valid, values, 0), smoothing, mode="nearest"), weights, out=np.zeros_like(values), where=weights > 0)
    values[~valid] = 0
    valid &= np.isfinite(values) & (np.abs(values) <= np.finfo(np.float32).max)
    values[~valid] = 0
    metadata["validity.0"] = base64.b64encode(np.packbits(valid, bitorder="little").tobytes()).decode("ascii")
    for key, value in describe(values, valid).items():
        metadata[f"statistics.0.{key}"] = value
    return {"vectors": [values.tolist()], "metadata": metadata}
