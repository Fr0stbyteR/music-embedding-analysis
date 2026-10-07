"""Legacy Audio Toolkit feature algorithms, hosted by the music backend."""

from __future__ import annotations

from typing import Any


try:
    import librosa
    import numpy as np
except ImportError as exc:
    raise RuntimeError(
        "Librosa is unavailable in music-embedding-analysis. Install the backend "
        f"Python package and its dependencies. ({exc})"
    )

from .signal_statistics import SIGNAL_STATISTICS_ALGORITHMS, analyze_signal, describe
from .roughness import ROUGHNESS_ALGORITHMS, analyze_roughness


def number(value: Any) -> float:
    array = np.asarray(value).reshape(-1)
    return float(array[0]) if array.size else 0.0


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    path = payload["path"]
    algorithm = payload["algorithm"]
    options = payload.get("options") or {}
    y, sample_rate = librosa.load(path, sr=None, mono=True)
    duration = float(len(y) / sample_rate)
    result: dict[str, Any] = {
        "algorithm": algorithm,
        "sampleRate": int(sample_rate),
        "duration": duration,
    }

    if algorithm in ROUGHNESS_ALGORITHMS:
        result.update(analyze_roughness(y, int(sample_rate), algorithm, options))
    elif algorithm in SIGNAL_STATISTICS_ALGORITHMS:
        result.update(analyze_signal(y, int(sample_rate), algorithm, options))
    elif algorithm == "beats":
        hop_length = int(options.get("hopLength", 512))
        start_bpm = float(options.get("startBpm", 120))
        tempo, frames = librosa.beat.beat_track(
            y=y, sr=sample_rate, hop_length=hop_length, start_bpm=start_bpm
        )
        result["values"] = librosa.frames_to_time(
            frames, sr=sample_rate, hop_length=hop_length
        ).tolist()
        result["metadata"] = {"tempo": number(tempo)}
    elif algorithm == "onsets":
        hop_length = int(options.get("hopLength", 512))
        backtrack = bool(options.get("backtrack", False))
        frames = librosa.onset.onset_detect(
            y=y, sr=sample_rate, hop_length=hop_length, backtrack=backtrack
        )
        result["values"] = librosa.frames_to_time(
            frames, sr=sample_rate, hop_length=hop_length
        ).tolist()
    elif algorithm == "nonSilent":
        top_db = float(options.get("topDb", 60))
        frame_length = int(options.get("frameLength", 2048))
        hop_length = int(options.get("hopLength", 512))
        intervals = librosa.effects.split(
            y, top_db=top_db, frame_length=frame_length, hop_length=hop_length
        )
        result["intervals"] = (intervals / float(sample_rate)).tolist()
    elif algorithm == "rms":
        frame_length = int(options.get("frameLength", 2048))
        hop_length = int(options.get("hopLength", 512))
        rms = librosa.feature.rms(
            y=y, frame_length=frame_length, hop_length=hop_length
        )[0]
        result["vectors"] = [rms.tolist()]
        result["metadata"] = {
            "frameLength": frame_length,
            "hopLength": hop_length,
            "unit": "RMS",
        }
    elif algorithm == "spectralCentroid":
        frame_length = int(options.get("frameLength", 2048))
        hop_length = int(options.get("hopLength", 512))
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=sample_rate, n_fft=frame_length, hop_length=hop_length
        )[0]
        result["vectors"] = [centroid.tolist()]
        result["metadata"] = {
            "frameLength": frame_length,
            "hopLength": hop_length,
            "unit": "Hz",
        }
    elif algorithm == "spectralBandwidth":
        frame_length = int(options.get("frameLength", 2048))
        hop_length = int(options.get("hopLength", 512))
        power = float(options.get("power", 2.0))
        bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=sample_rate, n_fft=frame_length,
            hop_length=hop_length, p=power
        )[0]
        result["vectors"] = [bandwidth.tolist()]
        result["metadata"] = {
            "frameLength": frame_length, "hopLength": hop_length,
            "power": power, "unit": "Hz"
        }
    elif algorithm == "spectralRolloff":
        frame_length = int(options.get("frameLength", 2048))
        hop_length = int(options.get("hopLength", 512))
        roll_percent = float(options.get("rollPercent", 0.85))
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sample_rate, n_fft=frame_length,
            hop_length=hop_length, roll_percent=roll_percent
        )[0]
        result["vectors"] = [rolloff.tolist()]
        result["metadata"] = {
            "frameLength": frame_length, "hopLength": hop_length,
            "rollPercent": roll_percent, "unit": "Hz"
        }
    elif algorithm == "spectralFlatness":
        frame_length = int(options.get("frameLength", 2048))
        hop_length = int(options.get("hopLength", 512))
        flatness = librosa.feature.spectral_flatness(
            y=y, n_fft=frame_length, hop_length=hop_length
        )[0]
        result["vectors"] = [flatness.tolist()]
        result["metadata"] = {
            "frameLength": frame_length, "hopLength": hop_length,
            "unit": "ratio"
        }
    elif algorithm == "zeroCrossingRate":
        frame_length = int(options.get("frameLength", 2048))
        hop_length = int(options.get("hopLength", 512))
        rate = librosa.feature.zero_crossing_rate(
            y, frame_length=frame_length, hop_length=hop_length
        )[0]
        result["vectors"] = [rate.tolist()]
        result["metadata"] = {
            "frameLength": frame_length, "hopLength": hop_length,
            "unit": "ratio"
        }
    elif algorithm == "onsetStrength":
        hop_length = int(options.get("hopLength", 512))
        strength = librosa.onset.onset_strength(
            y=y, sr=sample_rate, hop_length=hop_length
        )
        result["vectors"] = [strength.tolist()]
        result["metadata"] = {"hopLength": hop_length, "unit": "strength"}
    elif algorithm == "pitch":
        frame_length = int(options.get("frameLength", 2048))
        hop_length = int(options.get("hopLength", 512))
        fmin = float(options.get("fmin", librosa.note_to_hz("C2")))
        fmax = float(options.get("fmax", librosa.note_to_hz("C7")))
        pitch = librosa.yin(
            y,
            sr=sample_rate,
            fmin=fmin,
            fmax=fmax,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        result["vectors"] = [np.nan_to_num(pitch, nan=0.0).tolist()]
        result["metadata"] = {
            "frameLength": frame_length,
            "hopLength": hop_length,
            "unit": "Hz",
            "method": "YIN",
        }
    elif algorithm == "melSpectrogram":
        frame_length = int(options.get("frameLength", 2048))
        hop_length = int(options.get("hopLength", 512))
        mel_bins = int(options.get("melBins", 128))
        power = librosa.feature.melspectrogram(
            y=y,
            sr=sample_rate,
            n_fft=frame_length,
            hop_length=hop_length,
            n_mels=mel_bins,
        )
        decibels = librosa.power_to_db(power, ref=np.max)
        result["matrix"] = decibels.T.tolist()
        result["metadata"] = {
            "frameLength": frame_length,
            "hopLength": hop_length,
            "bins": mel_bins,
            "minValue": float(np.min(decibels)),
            "maxValue": float(np.max(decibels)),
            "unit": "dB",
        }
    elif algorithm == "chroma":
        frame_length = int(options.get("frameLength", 2048))
        hop_length = int(options.get("hopLength", 512))
        # librosa's automatic tuning estimator currently reaches a Numba gufunc
        # that can access-violate on Python 3.13/Windows. A fixed default is both
        # deterministic and safe; callers can still supply an explicit tuning.
        tuning = float(options.get("tuning", 0.0))
        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=sample_rate,
            n_fft=frame_length,
            hop_length=hop_length,
            tuning=tuning,
        )
        result["matrix"] = chroma.T.tolist()
        result["metadata"] = {
            "frameLength": frame_length,
            "hopLength": hop_length,
            "bins": 12,
            "tuning": tuning,
            "minValue": 0.0,
            "maxValue": 1.0,
            "unit": "strength",
        }
    elif algorithm == "mfcc":
        frame_length = int(options.get("frameLength", 2048))
        hop_length = int(options.get("hopLength", 512))
        coefficients = int(options.get("coefficients", 20))
        mfcc = librosa.feature.mfcc(
            y=y, sr=sample_rate, n_mfcc=coefficients,
            n_fft=frame_length, hop_length=hop_length
        )
        result["matrix"] = mfcc.T.tolist()
        result["metadata"] = {
            "frameLength": frame_length, "hopLength": hop_length,
            "bins": coefficients,
            "minValue": float(np.min(mfcc)),
            "maxValue": float(np.max(mfcc)),
            "unit": "coefficient"
        }
    else:
        raise ValueError(f"Unsupported analysis algorithm: {algorithm}")
    if "vectors" in result and algorithm not in SIGNAL_STATISTICS_ALGORITHMS | ROUGHNESS_ALGORITHMS:
        metadata = result.setdefault("metadata", {})
        for index, vector in enumerate(result["vectors"]):
            for key, value in describe(np.asarray(vector)).items():
                metadata[f"statistics.{index}.{key}"] = value
    return result
