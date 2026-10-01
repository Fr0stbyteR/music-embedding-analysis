from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from music_annotation_backend.app import create_app
from music_annotation_backend.config import Settings
from music_annotation_backend.mood import MoodAnalyzer, MoodUnavailable
from test_librosa_api import tone_wav


def settings(tmp_path: Path) -> Settings:
    return Settings(data_root=tmp_path / "data", model_root=tmp_path / "models", vendor_root=tmp_path / "vendor", session_token="test", auto_load_provider="")


def test_deam_order_normalization_padding_and_invalid_output(tmp_path: Path):
    analyzer = MoodAnalyzer(settings(tmp_path))
    lengths = []
    def embedding(audio):
        lengths.append(len(audio))
        return np.zeros((1, 200))
    analyzer.models = (embedding, lambda _: np.array([[9, 1], [5, 5]]))
    assert analyzer.predict(np.ones(10)) == (.5, -.5)
    assert lengths == [48000]
    analyzer.models = (embedding, lambda _: np.array([[np.nan, 1]]))
    with pytest.raises(RuntimeError, match="invalid"):
        analyzer.predict(np.ones(48000))


def test_optional_windows_runtime_is_explicit(monkeypatch, tmp_path: Path):
    analyzer = MoodAnalyzer(settings(tmp_path))
    monkeypatch.setattr("music_annotation_backend.mood.platform.system", lambda: "Windows")
    capability = analyzer.capabilities()
    assert capability["available"] is False
    assert "start.cmd" in capability["reason"]


def test_mood_api_auth_validation_missing_runtime_cache_and_timeline(monkeypatch, tmp_path: Path):
    app = create_app(settings(tmp_path))
    analyzer = app.state.mood
    headers = {"Authorization": "Bearer test"}
    with TestClient(app) as api:
        assert api.get("/v1/mood/capabilities").status_code == 401
        assert api.get("/v1/mood/capabilities", headers=headers).status_code == 200
        upload = api.post("/v1/interactive-assets", content=tone_wav(), headers={**headers, "X-File-Name": "tone.wav"})
        assert upload.status_code == 201
        url = f'/v1/interactive-assets/{upload.json()["id"]}:mood-curve'
        assert api.post(url, json={"windowSeconds": 0}, headers=headers).status_code == 422
        def unavailable(*_):
            raise MoodUnavailable("Missing test models")
        original = analyzer.analyze
        monkeypatch.setattr(analyzer, "analyze", unavailable)
        assert api.post(url, json={}, headers=headers).status_code == 503
        monkeypatch.setattr(analyzer, "analyze", original)
        # Test the actual API/cache/timeline plumbing, not a claimed model accuracy test.
        monkeypatch.setattr(analyzer, "capabilities", lambda: {"available": True, "model": "deam-msd-musicnn-2", "scale": "normalized-minus-one-to-one"})
        def load():
            analyzer.identity = (None, ["fake-backbone-hash", "fake-head-hash"])
            analyzer.models = (lambda audio: np.zeros((1, 200)), lambda _: np.array([[9, 1]]))
        monkeypatch.setattr(analyzer, "_load", load)
        request = {"timelineDurationSeconds": 2, "windowSeconds": 3, "hopSeconds": 1}
        first = api.post(url, json=request, headers=headers)
        assert first.status_code == 200, first.text
        result = first.json()
        assert result["cached"] is False
        assert [point["timeSeconds"] for point in result["points"]] == [0, 1, 2]
        assert all(point["valence"] == 1 and point["arousal"] == -1 for point in result["points"])
        assert api.post(url, json=request, headers=headers).json()["cached"] is True
        assert api.post(url, json={**request, "cachePolicy": "refresh"}, headers=headers).json()["cached"] is False
        assert api.post(url, json={**request, "timelineDurationSeconds": 12000}, headers=headers).status_code == 422
