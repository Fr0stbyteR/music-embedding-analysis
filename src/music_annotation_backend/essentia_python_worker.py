"""Isolated adapter to the official native wheel; no librosa substitute for Essentia."""
from contextlib import redirect_stdout
import json
import math
import sys

import numpy as np

SCALARS = {
    "rms": "RMS", "energy": "Energy", "loudness": "Loudness", "zeroCrossingRate": "ZeroCrossingRate",
    "spectralCentroid": "Centroid", "spectralRolloff": "RollOff", "spectralFlatness": "Flatness",
    "spectralCrest": "Crest", "spectralFlux": "Flux", "spectralEntropy": "Entropy",
    "spectralComplexity": "SpectralComplexity", "hfc": "HFC", "dissonance": "Dissonance",
}
BANDS = {"melBands": "MelBands", "barkBands": "BarkBands", "erbBands": "ERBBands", "mfcc": "MFCC", "gfcc": "GFCC"}
REQUIRED = set(SCALARS.values()) | set(BANDS.values()) | {"FrameCutter", "Windowing", "Spectrum", "SpectralPeaks", "PitchYinFFT", "HPCP", "Key", "CentralMoments", "DistributionShape", "OnsetDetection", "Onsets"}


def probe():
    print("Essentia probe: importing native package and TensorFlow algorithms...", file=sys.stderr, flush=True)
    import essentia
    import essentia.standard as es
    print("Essentia probe: native imports ready; checking RMS and algorithm availability...", file=sys.stderr, flush=True)
    from .essentia_api import ALGORITHMS
    missing = sorted(name for name in REQUIRED if not hasattr(es, name))
    if missing:
        raise RuntimeError(f"Essentia wheel is missing algorithms: {', '.join(missing)}")
    tone = np.sin(2 * np.pi * 16 * np.arange(512) / 512).astype(np.float32)
    if abs(es.RMS()(tone) - np.sqrt(.5)) > .00001:
        raise RuntimeError("Essentia RMS self-test failed")
    return {"protocol": 1, "runtime": "essentia-python", "essentiaVersion": essentia.__version__,
        "features": sorted(ALGORITHMS), "tensorflow": hasattr(es, "TensorflowPredictMusiCNN") and hasattr(es, "TensorflowPredict2D"),
        "tensorflowFeatures": int(all(hasattr(es, name) for name in ("TensorflowInputMusiCNN", "TensorflowInputTempoCNN", "TensorflowPredict")))}


def regions(classes, duration, hop_seconds, minimum_duration, name):
    intervals, labels = [], []
    previous, first = -1, 0
    for index, current in enumerate([*classes, -1]):
        if current == previous:
            continue
        left, right = first * hop_seconds, min(duration, index * hop_seconds)
        if previous >= 0 and right - left >= minimum_duration:
            intervals.append([left, right]); labels.append(name(previous))
        previous, first = current, index
    return intervals, labels


def analyze(audio, algorithm, o):
    import essentia.standard as es
    from .essentia_api import ALGORITHMS, EssentiaOptions
    if algorithm not in ALGORITHMS:
        raise ValueError("Unknown Essentia algorithm")
    o = EssentiaOptions.model_validate(o)
    sr, frame, hop = o.sample_rate, o.frame_length, o.hop_length
    duration = len(audio) / sr
    window = es.Windowing(type="hann", normalized=False)
    fft = es.Spectrum(size=frame)
    descriptor = None
    if algorithm in SCALARS:
        kwargs = {"range": sr / 2} if algorithm == "spectralCentroid" else {"sampleRate": sr, "cutoff": o.roll_percent} if algorithm == "spectralRolloff" else {"sampleRate": sr} if algorithm in {"hfc", "spectralComplexity"} else {}
        descriptor = getattr(es, SCALARS[algorithm])(**kwargs)
    elif algorithm in BANDS:
        kwargs = {"sampleRate": sr, "numberBands": min(28, o.bands)} if algorithm == "barkBands" else {"inputSize": frame // 2 + 1, "sampleRate": sr, "numberBands": o.bands, "highFrequencyBound": sr / 2}
        if algorithm in {"mfcc", "gfcc"}:
            kwargs["numberCoefficients"] = o.coefficients
        descriptor = getattr(es, BANDS[algorithm])(**kwargs)
    peaks = es.SpectralPeaks(sampleRate=sr, maxFrequency=min(5000, sr / 2), maxPeaks=100, orderBy="frequency") if algorithm in {"hpcp", "keyRegions", "dissonance"} else None
    pcp = es.HPCP(size=12, sampleRate=sr, referenceFrequency=o.tuning) if algorithm in {"hpcp", "keyRegions"} else None
    pitch = es.PitchYinFFT(frameSize=frame, sampleRate=sr, maxFrequency=min(5000, sr / 2)) if algorithm in {"pitch", "pitchConfidence", "pitchNotes"} else None
    moments = es.CentralMoments(range=sr / 2) if algorithm in {"spectralSpread", "spectralSkewness", "spectralKurtosis"} else None
    shape = es.DistributionShape() if moments else None
    onset = es.OnsetDetection(method="flux", sampleRate=sr) if algorithm in {"onsets", "onsetStrength"} else None
    rms = es.RMS()
    primary, rows, pitches, confidences = [], [], [], []
    for raw in es.FrameGenerator(audio, frameSize=frame, hopSize=hop, startFromZero=False, validFrameThresholdRatio=0):
        spectrum = fft(window(raw))
        value = 0.
        if pitch:
            frequency, confidence = pitch(spectrum)
            pitches.append(frequency); confidences.append(confidence)
            value = confidence if algorithm == "pitchConfidence" else frequency if confidence >= o.confidence else 0.
        if peaks:
            frequencies, magnitudes = peaks(spectrum)
            if pcp:
                rows.append(pcp(frequencies, magnitudes))
            else:
                value = descriptor(frequencies, magnitudes)
        elif descriptor:
            output = descriptor(raw if algorithm in {"rms", "energy", "loudness", "zeroCrossingRate"} else spectrum)
            if algorithm in {"mfcc", "gfcc"}:
                rows.append(output[1])
            elif algorithm in BANDS:
                rows.append(output)
            else:
                value = output
        if moments:
            value = shape(moments(spectrum))[["spectralSpread", "spectralSkewness", "spectralKurtosis"].index(algorithm)]
        if onset:
            value = onset(spectrum, np.empty(0, dtype=np.float32))
        if algorithm == "silenceRegions":
            value = rms(raw)
        primary.append(float(value) if math.isfinite(value) else 0.)
        if len(primary) > 500000 or (rows and len(rows) * len(rows[0]) > 2000000):
            raise ValueError("Feature result exceeds point budget; increase hop length")
    result = {"protocol": 1, "algorithm": algorithm, "sampleRate": sr, "duration": duration}
    minimum, maximum = 0., 1.
    if algorithm == "onsets":
        times = es.Onsets(frameRate=sr / hop)(np.asarray([primary], dtype=np.float32), np.ones(1, dtype=np.float32))
        result["values"] = [float(t) for t in times if 0 <= t <= duration]
    elif algorithm in {"silenceRegions", "pitchNotes"}:
        if algorithm == "silenceRegions":
            classes = [0 if value <= 10 ** (o.threshold_db / 20) else -1 for value in primary]
            label = lambda _: "Silence"
        else:
            classes = []
            for f, c in zip(pitches, confidences):
                midi = math.floor(69 + 12 * math.log2(f / o.tuning) + .5) if f > 0 and c >= o.confidence else -1
                classes.append(midi if 0 <= midi <= 127 else -1)
            names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            label = lambda midi: f"{names[midi % 12]}{midi // 12 - 1}"
        result["intervals"], result["labels"] = regions(classes, duration, hop / sr, o.minimum_duration, label)
    elif algorithm == "keyRegions":
        key = es.Key(profileType="temperley", pcpSize=12)
        intervals, labels = [], []
        step = max(1, math.floor(o.key_window * sr / hop + .5))
        for start in range(0, len(rows), step):
            end = min(len(rows), start + step)
            mean = np.mean(rows[start:end], axis=0).astype(np.float32)
            if mean.sum() <= 0:
                continue
            tonic, mode, strength, _ = key(mean)
            if not math.isfinite(strength) or strength < o.confidence:
                continue
            left, right = start * hop / sr, min(duration, end * hop / sr)
            label = f"{tonic} {mode}"
            if labels and labels[-1] == label and abs(intervals[-1][1] - left) < .0001:
                intervals[-1][1] = right
            elif right > left:
                intervals.append([left, right]); labels.append(label)
        result.update(intervals=intervals, labels=labels)
    elif rows:
        matrix = np.asarray(rows, dtype=np.float32)
        matrix[~np.isfinite(matrix)] = 0
        if algorithm in {"melBands", "barkBands", "erbBands"}:
            matrix = np.maximum(-100, 10 * np.log10(np.maximum(matrix, 1e-10) / max(1e-10, float(matrix.max()))))
            minimum, maximum = -100., 0.
        else:
            minimum, maximum = float(matrix.min()), float(matrix.max())
            if maximum <= minimum:
                maximum = minimum + 1
        result["matrix"] = matrix.tolist()
    else:
        result["vectors"] = [primary]
    result["metadata"] = {"frameLength": frame, "hopLength": hop, "minValue": minimum, "maxValue": maximum, "engine": "essentia-python"}
    return result


def main():
    # Any upstream Python chatter must not corrupt the worker's JSON protocol.
    with redirect_stdout(sys.stderr):
        if sys.argv[1:] in (["--capabilities"], ["--self-test"]):
            result = probe()
        elif len(sys.argv) in (5, 7) and sys.argv[1] in {"--tf-backbone", "--tf-head"}:
            from .essentia_tf_worker import backbone, head
            data = sys.stdin.buffer.read(10800 * 16000 * 4 + 1)
            if not data or len(data) % 4 or len(data) > 10800 * 16000 * 4: raise ValueError("Invalid TF PCM size")
            values = np.frombuffer(data, dtype="<f4")
            if not np.isfinite(values).all(): raise ValueError("Non-finite TF input")
            if sys.argv[1] == "--tf-backbone" and len(sys.argv) == 5:
                result = backbone(values, sys.argv[2], sys.argv[3], int(sys.argv[4]))
            elif sys.argv[1] == "--tf-head" and len(sys.argv) == 7:
                result = head(values, sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]))
            else: raise ValueError("Invalid TF command")
        elif len(sys.argv) == 14 and sys.argv[1] == "--features":
            fields = ["sampleRate", "frameLength", "hopLength", "bands", "coefficients", "rollPercent", "thresholdDb", "minimumDuration", "confidence", "keyWindow", "tuning"]
            options = {key: int(value) if i < 5 else float(value) for i, (key, value) in enumerate(zip(fields, sys.argv[3:]))}
            data = sys.stdin.buffer.read(10800 * 96000 * 4 + 1)
            if not data or len(data) % 4 or len(data) > 10800 * options["sampleRate"] * 4:
                raise ValueError("Invalid PCM size")
            audio = np.frombuffer(data, dtype="<f4")
            if not np.isfinite(audio).all():
                raise ValueError("PCM must contain finite samples")
            result = analyze(audio, sys.argv[2], options)
        else:
            raise ValueError("Usage: --capabilities | --features algorithm sr frame hop bands coefficients roll db minimumDuration confidence keyWindow tuning")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
