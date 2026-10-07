from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.signal import lfilter

from music_annotation_backend.signal_statistics import (
    SIGNAL_STATISTICS_ALGORITHMS, _lpcc, analyze_signal, describe,
)

RATE = 16000


def tone(frequency=440, duration=1):
    return .5 * np.sin(2 * np.pi * frequency * np.arange(int(RATE * duration)) / RATE)


def request(algorithm, samples=None, **options):
    return analyze_signal(tone() if samples is None else samples, RATE, algorithm, options)


@pytest.mark.parametrize("algorithm", sorted(SIGNAL_STATISTICS_ALGORITHMS))
def test_all_statistics_are_finite_and_keep_frame_timing(algorithm):
    result = request(algorithm, hopLength=128)
    array = np.asarray(result.get("vectors", result.get("matrix")))
    assert np.isfinite(array).all()
    assert (array.shape[0] if algorithm == "lpcc" else array.shape[1]) == 1 + RATE // 128
    assert result["metadata"]["hopLength"] == 128
    assert result["metadata"]["statistics.0.count"] >= 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("algorithm", sorted(SIGNAL_STATISTICS_ALGORITHMS))
def test_silence_is_finite_and_not_counted_as_an_observation(algorithm):
    result = request(algorithm, np.zeros(73))
    array = np.asarray(result.get("vectors", result.get("matrix")))
    assert np.isfinite(array).all()
    assert array.size > 0
    assert result["metadata"]["statistics.0.count"] == 0
    assert "statistics.0.mean" not in result["metadata"]
    json.dumps(result, allow_nan=False)


def test_peak_rms_and_crest_have_analytical_sine_values():
    assert np.median(request("peakAmplitude")["vectors"][0]) == pytest.approx(.5, abs=.002)
    assert np.median(request("crestFactor")["vectors"][0][3:-3]) == pytest.approx(np.sqrt(2), abs=.01)
    assert np.median(request("rmsDb")["vectors"][0][3:-3]) == pytest.approx(20 * np.log10(.5 / np.sqrt(2)), abs=.05)


def test_time_moments_centroid_and_dc_are_defined():
    assert np.median(request("timeSkewness")["vectors"][0][3:-3]) == pytest.approx(0, abs=.02)
    assert np.median(request("timeKurtosis")["vectors"][0][3:-3]) == pytest.approx(-1.5, abs=.02)
    assert np.median(request("temporalCentroid")["vectors"][0][3:-3]) == pytest.approx(.5, abs=.01)
    assert np.median(request("dcOffset", tone() + .2)["vectors"][0][3:-3]) == pytest.approx(.2, abs=.005)
    # Constant non-silent samples have no variance; do not produce NaN kurtosis.
    assert np.allclose(request("timeKurtosis", np.ones(RATE))["vectors"][0][3:-3], 0)
    assert request("timeKurtosis", np.ones(RATE))["metadata"]["statistics.0.count"] < 10


def test_spectral_tilt_distinguishes_low_and_high_tones():
    assert np.median(request("spectralTilt", tone(250))["vectors"][0][3:-3]) > 40
    assert np.median(request("spectralTilt", tone(3000))["vectors"][0][3:-3]) < -40


def test_slope_is_invariant_to_overall_gain():
    rng = np.random.default_rng(42)
    audio = rng.normal(0, .1, RATE)
    original = np.asarray(request("spectralSlope", audio)["vectors"][0])
    quieter = np.asarray(request("spectralSlope", audio * .1)["vectors"][0])
    np.testing.assert_allclose(original, quieter, atol=1e-10)


def test_tonality_and_hnr_do_not_treat_noise_as_periodic_tone():
    noise = np.random.default_rng(42).normal(0, .2, RATE)
    tonal = np.median(request("spectralTonality")["vectors"][0][3:-3])
    noisy = np.median(request("spectralTonality", noise)["vectors"][0][3:-3])
    assert tonal > .8 and noisy < .2
    assert np.median(request("harmonicNoiseRatio")["vectors"][0][3:-3]) > 30
    assert np.median(request("harmonicNoiseRatio", noise)["vectors"][0][3:-3]) < -8


def test_flux_is_continuous_across_batches_and_gain_invariant():
    long = tone(duration=10)
    result = np.asarray(request("positiveSpectralFlux", long, hopLength=128)["vectors"][0])
    assert result[256] < .02 and result[512] < .02
    quieter = request("positiveSpectralFlux", long * .1, hopLength=128)["vectors"][0]
    np.testing.assert_allclose(result, quieter, atol=1e-10)


def test_formants_find_synthetic_all_pole_resonances():
    frequencies = np.array([500., 1500., 2500.])
    bandwidth = 100
    roots = []
    for frequency in frequencies:
        root = np.exp(-np.pi * bandwidth / RATE + 2j * np.pi * frequency / RATE)
        roots.extend([root, root.conjugate()])
    coefficients = np.poly(roots).real
    excitation = np.random.default_rng(2).normal(0, .03, RATE * 2)
    audio = lfilter([1.], coefficients, excitation)
    audio /= np.max(np.abs(audio))
    result = request("formants", audio, lpcOrder=6, preEmphasis=0)
    estimated = np.median(np.asarray(result["vectors"])[:, 3:-3], axis=1)
    np.testing.assert_allclose(estimated, frequencies, atol=70)
    assert result["metadata"]["unit"] == "Hz"


def test_lpcc_recursion_and_matrix_shape():
    np.testing.assert_allclose(_lpcc(np.array([1., -.5]), 3), [.5, .125, .125 / 3])
    result = request("lpcc", coefficients=5)
    assert len(result["matrix"][0]) == 5
    assert result["metadata"]["firstCoefficient"] == 1


def test_statistics_exclude_invalid_observations_and_use_population_std():
    result = describe(np.array([1., 2., 3., np.nan, 999]), np.array([True, True, True, True, False]))
    assert result["count"] == 3
    assert result["mean"] == result["median"] == 2
    assert result["std"] == pytest.approx(np.sqrt(2 / 3))
    assert result["p05"] == pytest.approx(1.1)


@pytest.mark.parametrize("options", [
    {"frameLength": 127}, {"frameLength": 257}, {"frameLength": 1024.5},
    {"hopLength": 0}, {"hopLength": -1}, {"hopLength": True},
    {"hopLength": "NaN"}, {"silenceThresholdDb": float("inf")},
])
def test_invalid_parameters_are_rejected(options):
    with pytest.raises(ValueError):
        request("peakAmplitude", **options)


@pytest.mark.parametrize("algorithm,options", [
    ("harmonicNoiseRatio", {"frameLength": 128, "fmin": 60}),
    ("spectralTilt", {"splitFrequency": 8000}),
    ("spectralSlope", {"fmin": 5000, "fmax": 1000}),
    ("formants", {"lpcOrder": 0}), ("lpcc", {"coefficients": 1000}),
])
def test_algorithm_parameter_constraints(algorithm, options):
    with pytest.raises(ValueError):
        request(algorithm, **options)


def test_empty_or_nonfinite_audio_is_rejected():
    for audio in (np.array([]), np.array([np.nan])):
        with pytest.raises(ValueError):
            request("peakAmplitude", audio)
