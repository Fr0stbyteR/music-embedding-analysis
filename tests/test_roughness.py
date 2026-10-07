from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from music_annotation_backend.roughness import ROUGHNESS_ALGORITHMS, analyze_roughness, pair_roughness

RATE = 16000


def signal(modulation=0, depth=.8, duration=1.5):
    time = np.arange(round(RATE * duration)) / RATE
    return .3 * (1 + depth * np.sin(2 * np.pi * modulation * time)) * np.sin(2 * np.pi * 1000 * time)


@pytest.mark.parametrize("algorithm", sorted(ROUGHNESS_ALGORITHMS))
def test_roughness_finite_timing_validity_and_short_silence(algorithm):
    result = analyze_roughness(signal(70), RATE, algorithm, {"hopLength": 256})
    assert len(result["vectors"][0]) == 1 + len(signal(70)) // 256
    assert result["metadata"]["hopLength"] == 256
    assert result["metadata"]["calibrated"] is False
    assert np.isfinite(result["vectors"]).all()
    json.dumps(result, allow_nan=False)
    silence = analyze_roughness(np.zeros(73), RATE, algorithm, {})
    assert silence["vectors"] == [[0]]
    assert silence["metadata"]["statistics.0.count"] == 0
    assert base64.b64decode(silence["metadata"]["validity.0"]) == b"\0"


@pytest.mark.parametrize("algorithm", ["roughnessVassilakis", "roughnessSethares", "roughnessPlompLevelt"])
def test_pair_symmetry_close_beats_and_gain(algorithm):
    frequencies, amplitudes = np.array([440., 470.]), np.array([.1, .8])
    value = pair_roughness(frequencies, amplitudes, algorithm)
    assert value > pair_roughness(np.array([440., 880.]), amplitudes, algorithm)
    assert value > pair_roughness(np.array([440., 440.]), amplitudes, algorithm)
    assert value == pytest.approx(pair_roughness(frequencies[::-1], amplitudes[::-1], algorithm))
    exponent = .2 if algorithm == "roughnessVassilakis" else 2
    assert pair_roughness(frequencies, amplitudes * .5, algorithm) == pytest.approx(value * .5 ** exponent)


@pytest.mark.parametrize("algorithm", ["roughnessZwicker", "roughnessAures", "roughnessDanielWeber"])
def test_modulation_prefers_70hz_over_unmodulated_or_slow_carrier(algorithm):
    def middle(modulation, depth=.8):
        result = analyze_roughness(signal(modulation, depth), RATE, algorithm, {})
        return np.median(result["vectors"][0][12:-12])
    assert middle(70) > .001
    assert middle(70) > middle(0) * 5
    assert middle(70) > middle(5) * 2
    assert middle(70) > middle(70, .2)


@pytest.mark.parametrize("options", [{"hopLength": 0}, {"frameLength": 129}, {"maxPeaks": 200}, {"smoothingFrames": float("inf")}, {"peakThresholdDb": 1}])
def test_invalid_spectral_options(options):
    with pytest.raises(ValueError):
        analyze_roughness(signal(), RATE, "roughnessVassilakis", options)


@pytest.mark.parametrize("options", [{"modulationWindowSeconds": 0}, {"referenceLevelDb": 200}])
def test_invalid_modulation_options(options):
    with pytest.raises(ValueError):
        analyze_roughness(signal(), RATE, "roughnessZwicker", options)


def test_normalized_spectral_peaks_are_consistent_across_fft_sizes():
    time = np.arange(RATE * 2) / RATE
    samples = .3 * np.sin(2 * np.pi * 500 * time) + .3 * np.sin(2 * np.pi * 540 * time)
    estimates = [np.median(analyze_roughness(samples, RATE, "roughnessSethares", {"frameLength": length})["vectors"][0][10:-10]) for length in [2048, 4096]]
    assert estimates[0] == pytest.approx(estimates[1], rel=.08)
