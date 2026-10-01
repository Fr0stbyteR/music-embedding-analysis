import io
import os
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
from fastapi.testclient import TestClient

from music_annotation_backend.app import create_app
from music_annotation_backend.config import Settings
from music_annotation_backend.mood import MoodAnalyzer
from music_annotation_backend.native_essentia import NativeEssentia


def test_native_process_contract_and_validation(monkeypatch, tmp_path):
    worker = NativeEssentia(tmp_path / "worker.exe", 45)
    calls = []
    payload = {"protocol": 1, "points": [{"timeSeconds": t, "valence": .2, "arousal": -.4} for t in (0, 1, 2)]}
    def run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        samples = np.frombuffer(kwargs["stdin"].read(), dtype="<f4")
        np.testing.assert_array_equal(samples, np.arange(10, dtype=np.float32))
        return subprocess.CompletedProcess(arguments, 0, json.dumps(payload).encode(), b"")
    monkeypatch.setattr(subprocess, "run", run)
    assert len(worker.curve(np.arange(10), tmp_path / "a.pb", tmp_path / "b.pb", 3, 1, 2)) == 3
    assert calls[0][1]["timeout"] == 45
    assert calls[0][1]["check"] is False
    assert "shell" not in calls[0][1]
    payload["points"][1]["timeSeconds"] = 100
    with pytest.raises(RuntimeError, match="out-of-range"):
        worker.curve(np.arange(10), tmp_path / "a.pb", tmp_path / "b.pb", 3, 1, 2)
    payload["points"][1]["valence"] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        worker.curve(np.arange(10), tmp_path / "a.pb", tmp_path / "b.pb", 3, 1, 2)


def test_native_runtime_errors_and_missing_explicit_worker(monkeypatch, tmp_path):
    worker = NativeEssentia(tmp_path / "worker.exe")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 5, b"", b"bad graph"))
    with pytest.raises(RuntimeError, match="bad graph"):
        worker.probe()
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, b"not-json", b""))
    with pytest.raises(RuntimeError, match="invalid JSON"):
        worker.probe()
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 30)
    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="could not run"):
        worker.probe()
    analyzer = MoodAnalyzer(Settings(essentia_native_executable=tmp_path / "missing.exe", data_root=tmp_path / "data", model_root=tmp_path / "models", vendor_root=tmp_path / "vendor", auto_load_provider=""))
    assert analyzer.capabilities()["available"] is False
    assert analyzer.capabilities()["runtime"] == "essentia-cpp"


def test_probe_has_independent_cold_import_deadline_and_caches_success(monkeypatch, tmp_path):
    worker = NativeEssentia(tmp_path / "worker.exe", 45, probe_timeout_seconds=180)
    calls = []
    def run(arguments, **kwargs):
        calls.append(kwargs["timeout"])
        return subprocess.CompletedProcess(arguments, 0, b'{"protocol": 1, "runtime": "essentia-cpp"}', b"")
    monkeypatch.setattr(subprocess, "run", run)
    assert worker.probe()["runtime"] == "essentia-cpp"
    assert worker.probe()["runtime"] == "essentia-cpp"
    assert calls == [180]
    assert worker.timeout_seconds == 45


@pytest.mark.parametrize("stderr", [b"Importing TensorFlow...", "Importing TensorFlow..."])
def test_probe_timeout_preserves_worker_diagnostics_and_is_not_cached(monkeypatch, tmp_path, stderr):
    worker = NativeEssentia(tmp_path / "worker.exe", probe_timeout_seconds=180)
    def timeout(arguments, **kwargs):
        raise subprocess.TimeoutExpired(arguments, kwargs["timeout"], stderr=stderr)
    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="timed out after 180 seconds.*Importing TensorFlow"):
        worker.probe()
    assert worker.info is None
    assert worker.signature is None


def test_real_native_mood_api_and_cache_when_installed(tmp_path):
    # Optional integration: never download models or compile during pytest.
    backend = Path(__file__).resolve().parents[1]
    settings = Settings(data_root=tmp_path / "data", model_root=backend / "models", vendor_root=backend / "vendor", essentia_native_executable=None, session_token="test", auto_load_provider="")
    available = MoodAnalyzer(settings).capabilities()
    if not available["available"]:
        if os.environ.get("MAB_TEST_REQUIRE_ESSENTIA") == "1":
            pytest.fail(f"Required Essentia VA unavailable: {available}")
        pytest.skip("Optional native runtime/weights not installed")
    import soundfile as sf
    # Change from 440 to 220 Hz to exercise different windows, not model accuracy.
    audio = np.sin(2 * np.pi * np.arange(16000 * 8) * np.repeat([440, 220], 16000 * 4) / 16000).astype(np.float32) * .1
    wav = io.BytesIO()
    sf.write(wav, audio, 16000, format="WAV")
    headers = {"Authorization": "Bearer test"}
    with TestClient(create_app(settings)) as api:
        capability = api.get("/v1/mood/capabilities", headers=headers).json()
        assert capability["available"] is True, capability
        assert capability["runtime"] == available["runtime"]
        upload = api.post("/v1/interactive-assets", content=wav.getvalue(), headers={**headers, "X-File-Name": "native-tone.wav"})
        assert upload.status_code == 201, upload.text
        url = f'/v1/interactive-assets/{upload.json()["id"]}:mood-curve'
        request = {"windowSeconds": 3, "hopSeconds": 2, "timelineDurationSeconds": 8}
        response = api.post(url, json=request, headers=headers)
        assert response.status_code == 200, response.text
        result = response.json()
        assert [point["timeSeconds"] for point in result["points"]] == [0, 2, 4, 6, 8]
        assert all(-1 <= point[axis] <= 1 for point in result["points"] for axis in ("valence", "arousal"))
        assert result["cached"] is False
        cached = api.post(url, json=request, headers=headers).json()
        assert cached["cached"] is True
        assert cached["points"] == result["points"]
        assert api.post(url, json={**request, "cachePolicy": "refresh"}, headers=headers).json()["cached"] is False
