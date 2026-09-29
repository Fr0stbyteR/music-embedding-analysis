from __future__ import annotations

import io
import math
import struct
import wave
from pathlib import Path

from fastapi.testclient import TestClient

from music_annotation_backend.app import create_app
from music_annotation_backend.config import Settings


def tone_wav() -> bytes:
    stream = io.BytesIO()
    sample_rate = 22050
    samples = [round(10000 * math.sin(2 * math.pi * 440 * index / sample_rate)) for index in range(sample_rate)]
    with wave.open(stream, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return stream.getvalue()


def test_interactive_librosa_upload_cache_and_auth(tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path / "data", model_root=tmp_path / "models", vendor_root=tmp_path / "vendor", session_token="test", auto_load_provider="")
    with TestClient(create_app(settings)) as api:
        assert len(api.get("/v1/health").json()["librosaEngineVersion"]) == 64
        missing_token = api.post("/v1/interactive-assets", content=tone_wav(), headers={"X-File-Name": "tone.wav"})
        assert missing_token.status_code == 401
        headers = {"Authorization": "Bearer test", "X-File-Name": "tone.wav", "Content-Type": "audio/wav"}
        upload = api.post("/v1/interactive-assets", content=tone_wav(), headers=headers)
        assert upload.status_code == 201, upload.text
        asset_id = upload.json()["id"]
        url = f"/v1/interactive-assets/{asset_id}:librosa"
        request = {"algorithm": "rms", "options": {"frameLength": 1024, "hopLength": 256}}
        analysis_headers = {"Authorization": "Bearer test"}
        first = api.post(url, json=request, headers=analysis_headers)
        assert first.status_code == 200, first.text
        assert first.json()["cache"]["status"] == "miss"
        assert first.json()["vectors"][0]
        second = api.post(url, json=request, headers=analysis_headers)
        assert second.status_code == 200
        assert second.json()["cache"]["status"] == "hit"
        refreshed = api.post(url, json={**request, "cachePolicy": "refresh"}, headers=analysis_headers)
        assert refreshed.status_code == 200
        assert refreshed.json()["cache"]["status"] == "refresh"
        unsupported = api.post(url, json={"algorithm": "not-a-feature"}, headers=analysis_headers)
        assert unsupported.status_code == 422
