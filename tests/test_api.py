from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf

from fastapi.testclient import TestClient

from music_annotation_backend.app import create_app
from music_annotation_backend.config import Settings


def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(Settings(data_root=tmp_path / "data", model_root=tmp_path / "models", vendor_root=tmp_path / "vendor", session_token="test")), headers={"Authorization": "Bearer test"})


def wait_for_job(api: TestClient, job_id: str, timeout: float = 120) -> dict:
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = api.get(f"/v1/jobs/{job_id}").json()
        if job["state"] not in {"queued", "running", "cancel_requested"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish")


def test_health_does_not_require_auth(tmp_path: Path) -> None:
    app = create_app(Settings(data_root=tmp_path / "data", model_root=tmp_path / "models", vendor_root=tmp_path / "vendor", session_token="secret"))
    with TestClient(app) as api:
        response = api.get("/v1/health")
        assert response.status_code == 200
        assert response.json()["protocolVersion"] == "music-annotation/1"


def test_annotation_round_trip_and_conflict(tmp_path: Path) -> None:
    with client(tmp_path) as api:
        project = api.post("/v1/projects", json={"name": "Guqin", "directory": str(tmp_path)}).json()
        project_id, asset_id = project["id"], str(uuid4())
        created_response = api.post(f"/v1/projects/{project_id}/annotations", json={
            "assetId": asset_id, "startSample": 100, "endSample": 500,
            "labelIds": ["guqin.fan-yin"], "state": "confirmed", "provenance": "human"
        })
        assert created_response.status_code == 201
        annotation = created_response.json()
        updated = api.put(
            f"/v1/projects/{project_id}/annotations/{annotation['id']}",
            headers={"If-Match": '"1"'}, json={"note": "泛音"},
        )
        assert updated.status_code == 200
        assert updated.json()["revision"] == 2
        conflict = api.put(
            f"/v1/projects/{project_id}/annotations/{annotation['id']}",
            headers={"If-Match": '"1"'}, json={"note": "stale"},
        )
        assert conflict.status_code == 409
        queried = api.get(f"/v1/projects/{project_id}/annotations", params={"assetId": asset_id, "startSample": 0, "endSample": 1000})
        assert queried.status_code == 200
        assert queried.json()[0]["note"] == "泛音"


def test_mock_provider_load_job(tmp_path: Path) -> None:
    with client(tmp_path) as api:
        accepted = api.post("/v1/providers/mock:load", json={"device": "cpu"})
        assert accepted.status_code == 202
        job_id = accepted.json()["jobId"]
        import time
        for _ in range(50):
            job = api.get(f"/v1/jobs/{job_id}").json()
            if job["state"] not in {"queued", "running"}:
                break
            time.sleep(0.01)
        assert job["state"] == "succeeded"
        caps = api.get("/v1/capabilities").json()
        mock = next(p for p in caps["providers"] if p["providerId"] == "mock")
        assert mock["loaded"] is True
        probe = api.post("/v1/providers/mock:probe").json()
        for _ in range(50):
            job = api.get(f"/v1/jobs/{probe['jobId']}").json()
            if job["state"] not in {"queued", "running"}:
                break
            time.sleep(0.01)
        assert job["state"] == "succeeded"
        assert job["result"]["audioShape"] == [1, 32]


def test_musicology_request_compiles_to_safe_multifamily_plan(tmp_path: Path) -> None:
    with client(tmp_path) as api:
        project = api.post("/v1/projects", json={"name": "Analysis", "directory": str(tmp_path)}).json()
        response = api.post(
            f"/v1/projects/{project['id']}/detection-plans:compile",
            json={"instruction": "标注古琴的泛音和滑音，并分析音高、曲式、调性与和弦"},
        )
        assert response.status_code == 202
        job = wait_for_job(api, response.json()["jobId"])
        assert job["state"] == "succeeded", job.get("error")
        plan = job["result"]["plan"]
        families = {label["family"] for label in plan["labels"]}
        assert {"instrument", "technique", "pitch", "form", "harmony"} <= families
        assert plan["policy"]["protectConfirmed"] is True
        assert plan["policy"]["persistPreview"] is False
        assert all(label["thresholds"]["autoAccept"] is None for label in plan["labels"])
        validation = api.post(f"/v1/projects/{project['id']}/detection-plans/{plan['id']}:validate")
        assert validation.status_code == 200
        assert validation.json()["valid"] is True
        assert "pitch.notes" in validation.json()["runnableLabels"]

        unknown = api.post(
            f"/v1/projects/{project['id']}/detection-plans:compile",
            json={"instruction": "标注 sul ponticello 弓法"},
        )
        unknown_plan = wait_for_job(api, unknown.json()["jobId"])["result"]["plan"]
        assert unknown_plan["labels"][0]["labelId"].startswith("custom.request.")
        assert "sul ponticello" in unknown_plan["labels"][0]["prompts"]["positive"][0]


def test_asset_preview_and_run_create_only_suggested_annotations(tmp_path: Path) -> None:
    sample_rate = 22050
    seconds = 2.0
    timeline = np.arange(round(sample_rate * seconds), dtype=np.float32) / sample_rate
    audio = 0.2 * np.sin(2 * np.pi * 440 * timeline)
    audio[timeline > 1] += 0.12 * np.sin(2 * np.pi * 554.37 * timeline[timeline > 1])
    audio_path = tmp_path / "tone.wav"
    sf.write(audio_path, audio, sample_rate)

    with client(tmp_path) as api:
        project = api.post("/v1/projects", json={"name": "Measured", "directory": str(tmp_path)}).json()
        imported = api.post(f"/v1/projects/{project['id']}/assets", json={"path": str(audio_path)})
        import_job = wait_for_job(api, imported.json()["jobId"])
        assert import_job["state"] == "succeeded", import_job.get("error")
        asset = import_job["result"]
        assert asset["sampleRate"] == sample_rate

        compiled = api.post(
            f"/v1/projects/{project['id']}/detection-plans:compile",
            json={"instruction": "分析音高、调性、和弦、力度和曲式"},
        )
        plan_job = wait_for_job(api, compiled.json()["jobId"])
        plan = plan_job["result"]["plan"]
        request = {"planRevision": plan["revision"], "assetIds": [asset["id"]], "maximumCandidates": 200}

        preview = api.post(f"/v1/projects/{project['id']}/detection-plans/{plan['id']}:preview", json=request)
        preview_job = wait_for_job(api, preview.json()["jobId"])
        assert preview_job["state"] == "succeeded", preview_job.get("error")
        preview_run = preview_job["result"]
        assert preview_run["preview"] is True
        candidates = api.get(f"/v1/projects/{project['id']}/detection-runs/{preview_run['id']}/candidates").json()
        assert candidates
        assert all(candidate["evidence"]["scoreKind"] == "heuristic-not-calibrated" for candidate in candidates)
        empty = api.get(
            f"/v1/projects/{project['id']}/annotations",
            params={"assetId": asset["id"], "startSample": 0, "endSample": asset["samples"]},
        ).json()
        assert empty == []

        run = api.post(f"/v1/projects/{project['id']}/detection-plans/{plan['id']}:run", json=request)
        run_job = wait_for_job(api, run.json()["jobId"])
        assert run_job["state"] == "succeeded", run_job.get("error")
        annotations = api.get(
            f"/v1/projects/{project['id']}/annotations",
            params={"assetId": asset["id"], "startSample": 0, "endSample": asset["samples"]},
        ).json()
        assert annotations
        assert {annotation["state"] for annotation in annotations} == {"suggested"}
        assert {annotation["provenance"] for annotation in annotations} == {"model"}


def test_loaded_text_provider_returns_reviewable_instrument_and_technique_evidence(tmp_path: Path) -> None:
    sample_rate = 16000
    audio_path = tmp_path / "semantic.wav"
    sf.write(audio_path, 0.15 * np.sin(2 * np.pi * 330 * np.arange(sample_rate) / sample_rate), sample_rate)
    with client(tmp_path) as api:
        project = api.post("/v1/projects", json={"name": "Semantic", "directory": str(tmp_path)}).json()
        load_job = wait_for_job(api, api.post("/v1/providers/mock:load", json={"device": "cpu"}).json()["jobId"])
        assert load_job["state"] == "succeeded"
        asset_job = wait_for_job(api, api.post(f"/v1/projects/{project['id']}/assets", json={"path": str(audio_path)}).json()["jobId"])
        asset = asset_job["result"]
        plan_job = wait_for_job(api, api.post(
            f"/v1/projects/{project['id']}/detection-plans:compile",
            json={"instruction": "识别古琴和泛音", "availableProviderIds": ["mock"]},
        ).json()["jobId"])
        plan = plan_job["result"]["plan"]
        assert {label["detectors"][0]["providerId"] for label in plan["labels"]} == {"mock"}
        preview_job = wait_for_job(api, api.post(
            f"/v1/projects/{project['id']}/detection-plans/{plan['id']}:preview",
            json={"planRevision": 1, "assetIds": [asset["id"]]},
        ).json()["jobId"])
        assert preview_job["state"] == "succeeded", preview_job.get("error")
        candidates = api.get(f"/v1/projects/{project['id']}/detection-runs/{preview_job['result']['id']}/candidates").json()
        assert {candidate["labelId"] for candidate in candidates} == {"instrument.guqin", "technique.harmonic"}
        assert all(candidate["evidence"]["scoreKind"] == "prompt-margin-not-calibrated" for candidate in candidates)
