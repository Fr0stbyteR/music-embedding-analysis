import hashlib
import io
import json
import os
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from music_annotation_backend.app import create_app
from music_annotation_backend.config import Settings
from music_annotation_backend.essentia_tf import ALGORITHMS, MODELS, BACKBONES, MANIFEST, TensorflowAnalyzer, TFOptions, candidate_regions, matrix
from music_annotation_backend.schemas import InteractiveLibrosaRequest

def test_catalog_is_unique_whitelisted_and_checksum_pinned():
    assert len(ALGORITHMS) == 21
    assert len(MANIFEST) == 34
    assert len({entry["name"] for entry in MANIFEST}) == 34
    assert all(entry["url"].startswith("https://essentia.upf.edu/models/") and len(entry["sha256"]) == 64 for entry in MANIFEST)
    names = {entry["name"] for entry in MANIFEST}
    assert all(f"{model.stem}.pb" in names and f"{BACKBONES[model.family]}.pb" in names for model in MODELS.values())

@pytest.mark.parametrize("options", [{"hopSeconds": 0}, {"hopSeconds": float("inf")}, {"threshold": float("nan")}, {"threshold": 2}, {"minimumDuration": 0}, {"graphFilename": "arbitrary.pb"}])
def test_bad_options(options):
    with pytest.raises(ValidationError): TFOptions.model_validate(options)

def test_hysteresis_candidates_are_bounded_and_have_minimum_duration():
    intervals, labels = candidate_regions([.8, .5, .8, .3, .9], 2, 9, .6, 2, "piano")
    assert intervals == [[0., 6.]]
    assert labels == ["piano · candidate"]
    assert candidate_regions([.8, .5, .8, .3, .9], 2, 9, .6, .5, "piano")[0] == [[0., 6.], [8., 9.]]
    assert candidate_regions([0, 0], 2, 3, .6, .1, "piano") == ([], [])
    assert candidate_regions([0, 0], 2, 3, 0, .1, "piano")[0] == [[0., 3]]

@pytest.mark.parametrize("value", [[[float("nan")]], [[2]], [[-.1]], [[.1, .2]], [True]])
def test_bad_score_results(value):
    with pytest.raises(RuntimeError): matrix(value, 1, 1, scores=True)

def test_missing_runtime_auth_and_invalid_requests(tmp_path):
    settings = Settings(data_root=tmp_path / "data", vendor_root=tmp_path / "vendor", model_root=tmp_path / "models", essentia_native_executable=tmp_path / "missing", session_token="test", auto_load_provider="")
    with TestClient(create_app(settings)) as api:
        assert api.get("/v1/essentia-tf/capabilities").status_code == 401
        capability = api.get("/v1/essentia-tf/capabilities", headers={"Authorization": "Bearer test"}).json()
        assert not capability["available"] and capability["algorithms"] == []

def test_weight_mismatch_is_not_overwritten(tmp_path):
    analyzer = TensorflowAnalyzer(Settings(model_root=tmp_path, vendor_root=tmp_path / "vendor"))
    analyzer.root.mkdir(); path = analyzer.root / "msd-musicnn-1.pb"
    path.write_bytes(b"incorrect weights")
    with pytest.raises(RuntimeError, match="Checksum mismatch"): analyzer.weight(path.name)
    assert path.read_bytes() == b"incorrect weights"

def test_native_models_api_and_shared_caches(tmp_path):
    backend = Path(__file__).resolve().parents[1]
    settings = Settings(data_root=tmp_path / "data", vendor_root=backend / "vendor", model_root=backend / "models", session_token="test", auto_load_provider="")
    app = create_app(settings); analyzer = app.state.essentia_tf
    capability = analyzer.capabilities()
    if set(capability["algorithms"]) != ALGORITHMS:
        if os.environ.get("MAB_TEST_REQUIRE_ESSENTIA_TF") == "1": pytest.fail(str(capability))
        pytest.skip("Optional native TF runtime/models not prepared")
    import soundfile as sf
    wav = io.BytesIO(); sf.write(wav, .2 * np.sin(2 * np.pi * 440 * np.arange(6 * 16000) / 16000), 16000, format="WAV")
    headers = {"Authorization": "Bearer test"}
    with TestClient(app) as api:
        upload = api.post("/v1/interactive-assets", content=wav.getvalue(), headers={**headers, "X-File-Name": "tf.wav"})
        assert upload.status_code == 201
        url = f'/v1/interactive-assets/{upload.json()["id"]}:essentia-tf'
        assert api.post(url, json={"algorithm": "unknown"}, headers=headers).status_code == 422
        assert api.post(url, json={"algorithm": "tfInstrument", "options": {"hopSeconds": 0}}, headers=headers).status_code == 422
        assert api.post(url, json={"algorithm": "tfInstrumentCurve", "options": {"label": "../arbitrary"}}, headers=headers).status_code == 422
        results = {}
        for algorithm in sorted(ALGORITHMS):
            response = api.post(url, json={"algorithm": algorithm}, headers=headers)
            assert response.status_code == 200, (algorithm, response.text)
            result = response.json(); results[algorithm] = result
            assert result["algorithm"] == algorithm and abs(result["duration"] - 6) < .001
            assert result["metadata"]["scoreKind"] == "model-score-not-calibrated-confidence"
            for value in result.get("matrix", []) + result.get("vectors", []): assert np.isfinite(value).all()
        assert len(results["tfInstrument"]["labels"]) == 40
        assert len(results["tfMoodTheme"]["labels"]) == 56
        assert len(results["tfGenre"]["labels"]) == 87
        assert len(results["tfTags"]["labels"]) == 50
        assert results["tfInstrumentCurve"]["cache"]["status"] == "hit"
        assert results["tfInstrumentRegions"]["metadata"]["embeddingCache"] == "hit"
        assert results["tfTempoCandidates"]["labels"][0] == "30" and results["tfTempoCandidates"]["labels"][-1] == "285"
        for algorithm in ("tfVoice", "tfTonal", "tfHappy"):
            result = results[algorithm]
            labels = json.loads(result["metadata"]["classLabels"])
            index = labels.index(MODELS[algorithm].target)
            # Do not assume the positive label is column 0; original models differ.
            assert index in (0, 1)
        changed = api.post(url, json={"algorithm": "tfInstrumentCurve", "options": {"label": "violin"}}, headers=headers).json()
        assert changed["cache"]["status"] == "hit"
        col = results["tfInstrument"]["labels"].index("violin")
        assert changed["vectors"][0] == [row[col] for row in results["tfInstrument"]["matrix"]]
        embedding_files = list(analyzer.cache_root.glob("*-embeddings.npz"))
        assert len(embedding_files) == 3, "one shared cache per backbone, not per module"
        # Corrupt score data must be recomputed, not sent to the renderer.
        for path in analyzer.cache_root.glob("*-scores.npz"): path.write_bytes(b"truncated")
        repaired = api.post(url, json={"algorithm": "tfInstrumentCurve"}, headers=headers).json()
        assert repaired["cache"]["status"] == "miss" and repaired["metadata"]["embeddingCache"] == "hit"
        refreshed = api.post(url, json={"algorithm": "tfInstrument", "cachePolicy": "refresh"}, headers=headers).json()
        assert refreshed["cache"]["status"] == "refresh" and refreshed["metadata"]["embeddingCache"] == "miss"
