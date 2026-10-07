"""DSP integration uses the installed worker; never downloads or builds dependencies."""
import io
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from music_annotation_backend.app import create_app
from music_annotation_backend.config import Settings
from music_annotation_backend.essentia_api import ALGORITHMS, MATRIX_ALGORITHMS, VECTOR_ALGORITHMS, EssentiaAnalyzer, EssentiaOptions
from music_annotation_backend.essentia_runtime import feature_runtime


@pytest.mark.parametrize("options", [
    {"frameLength": 1000}, {"hopLength": 4096, "frameLength": 2048},
    {"sampleRate": 44100.5}, {"bands": 12, "coefficients": 20},
    {"confidence": float("nan")}, {"typoOption": 1},
])
def test_invalid_options_rejected(options):
    with pytest.raises(ValidationError):
        EssentiaOptions.model_validate(options)


def test_missing_runtime_and_auth(tmp_path):
    settings = Settings(data_root=tmp_path / "data", model_root=tmp_path / "models", vendor_root=tmp_path / "vendor", essentia_native_executable=tmp_path / "missing.exe", session_token="test", auto_load_provider="")
    with TestClient(create_app(settings)) as api:
        assert api.get("/v1/essentia/capabilities").status_code == 401
        capability = api.get("/v1/essentia/capabilities", headers={"Authorization": "Bearer test"}).json()
        assert capability["available"] is False
        assert capability["algorithms"] == []


@pytest.mark.parametrize("algorithm,data", [
    ("rms", {"vectors": [[float("nan")]]}),
    ("rms", {"vectors": [[1], [2]]}),
    ("mfcc", {"matrix": [[1, 2]]}),
    ("mfcc", {"matrix": [[1], [1, 2]]}),
    ("pitchNotes", {"intervals": [None], "labels": ["A4"]}),
    ("pitchNotes", {"intervals": [[1, 0]], "labels": ["A4"]}),
    ("onsets", {"values": [1, 0]}),
])
def test_malformed_worker_results(algorithm, data):
    with pytest.raises(RuntimeError):
        EssentiaAnalyzer.validate_result({"protocol": 1, "algorithm": algorithm, "sampleRate": 44100, "duration": 3, **data}, algorithm, EssentiaOptions(), 3)


def test_all_native_features_cache_parameters_and_markers(tmp_path):
    backend = Path(__file__).resolve().parents[1]
    runtime_settings = Settings(vendor_root=backend / "vendor", essentia_native_executable=None)
    try:
        native_info = feature_runtime(runtime_settings).probe()
    except RuntimeError as error:
        if os.environ.get("MAB_TEST_REQUIRE_ESSENTIA") == "1":
            pytest.fail(f"Required native Essentia runtime unavailable: {error}")
        pytest.skip("Optional native Essentia runtime not installed")
    import soundfile as sf
    rate = 44100
    audio = np.sin(2 * np.pi * 440 * np.arange(3 * rate) / rate).astype(np.float32) * .2
    audio[rate:2 * rate] = 0
    wav = io.BytesIO()
    sf.write(wav, audio, rate, format="WAV")
    settings = Settings(data_root=tmp_path / "data", model_root=tmp_path / "models", vendor_root=backend / "vendor", essentia_native_executable=None, session_token="test", auto_load_provider="")
    headers = {"Authorization": "Bearer test"}
    with TestClient(create_app(settings)) as api:
        capability = api.get("/v1/essentia/capabilities", headers=headers).json()
        assert capability["available"], capability
        assert set(capability["algorithms"]) == ALGORITHMS
        upload = api.post("/v1/interactive-assets", content=wav.getvalue(), headers={**headers, "X-File-Name": "features.wav"})
        assert upload.status_code == 201, upload.text
        url = f'/v1/interactive-assets/{upload.json()["id"]}:essentia'
        results = {}
        for algorithm in sorted(ALGORITHMS):
            options = {"frameLength": 4096} if algorithm in {"pitch", "pitchConfidence", "pitchNotes", "hpcp", "keyRegions"} else {}
            if algorithm == "melBands":
                options["bands"] = 64
            elif algorithm == "barkBands":
                options["bands"] = 27
            elif algorithm in {"mfcc", "gfcc"}:
                options["coefficients"] = 20
            response = api.post(url, json={"algorithm": algorithm, "options": options}, headers=headers)
            assert response.status_code == 200, (algorithm, response.text)
            result = results[algorithm] = response.json()
            assert result["metadata"]["engine"] == native_info["runtime"]
            assert result["cache"]["status"] == "miss"
            if algorithm in VECTOR_ALGORITHMS:
                assert len(result["vectors"]) == 1
                assert len(result["vectors"][0]) > 200
                assert result["metadata"]["statistics.0.count"] == len(result["vectors"][0])
                assert result["metadata"]["statistics.0.rms"] == pytest.approx(np.sqrt(np.mean(np.square(result["vectors"][0]))))
            elif algorithm in MATRIX_ALGORITHMS:
                bins = 12 if algorithm == "hpcp" else 27 if algorithm == "barkBands" else 20 if algorithm in {"mfcc", "gfcc"} else 64 if algorithm == "melBands" else 40
                assert len(result["matrix"][0]) == bins
                assert result["metadata"]["statistics.matrix.0.count"] == len(result["matrix"]) * bins
                assert result["metadata"]["statistics.matrix.0.bin.0.count"] == len(result["matrix"])
        pitch = np.array(results["pitch"]["vectors"][0])
        assert np.median(pitch[pitch > 0]) == pytest.approx(440, abs=5)
        rms = results["rms"]["vectors"][0]
        assert np.median(rms[20:60]) == pytest.approx(.2 / np.sqrt(2), abs=.01)
        assert results["silenceRegions"]["labels"] == ["Silence"]
        start, end = results["silenceRegions"]["intervals"][0]
        assert start == pytest.approx(1, abs=.08)
        assert end == pytest.approx(2, abs=.08)
        assert "A4" in results["pitchNotes"]["labels"]
        assert all(-100 <= value <= 0 for row in results["melBands"]["matrix"] for value in row)
        cached = api.post(url, json={"algorithm": "rms"}, headers=headers).json()
        assert cached["cache"]["status"] == "hit"
        assert cached["vectors"] == results["rms"]["vectors"]
        assert cached["metadata"] == results["rms"]["metadata"]
        assert api.post(url, json={"algorithm": "rms", "cachePolicy": "refresh"}, headers=headers).json()["cache"]["status"] == "refresh"
        for request in ({"algorithm": "nonexistent"}, {"algorithm": "rms", "options": {"frameLength": 1000}}, {"algorithm": "rms", "options": {"typo": 1}}):
            assert api.post(url, json=request, headers=headers).status_code == 422
        assert api.post("/v1/interactive-assets/00000000-0000-0000-0000-000000000000:essentia", json={"algorithm": "rms"}, headers=headers).status_code == 404
