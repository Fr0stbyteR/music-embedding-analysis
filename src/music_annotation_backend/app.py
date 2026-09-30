from __future__ import annotations

import json
import asyncio
import hashlib
from collections import Counter
from datetime import datetime, timezone
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import unquote
from uuid import NAMESPACE_URL, UUID, uuid5

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import __version__
from .analysis import analyze_range, inspect_asset
from .config import Settings
from .jobs import JobManager
from .interactive import describe_asset, relevance_curve
from .librosa_api import analyze_interactive_asset, engine_version
from .providers import ProviderRegistry
from .schemas import (
    Annotation, AnnotationCreate, AnnotationPatch, Asset, AssetImport, Capabilities,
    DetectionCandidate, DetectionPlan, DetectionPlanCompileRequest, DetectionPlanRunRequest,
    DetectionPlanValidation, DetectionRun, Health, Job, JobAccepted, Project,
    InteractiveDescribeRequest, InteractiveDescriptionResult, InteractiveLibrosaRequest, ProjectCreate, SemanticCurveRequest, SemanticCurveResult,
    ProviderCapability, ProviderLoadRequest,
)
from .planner import compile_plan, validate_plan
from .store import ConflictError, NotFoundError, ProjectStore


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or Settings()
    cfg.prepare()
    store, jobs, providers = ProjectStore(cfg.data_root), JobManager(), ProviderRegistry(cfg)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if cfg.auto_load_provider:
            try:
                provider = providers.get(cfg.auto_load_provider)
            except KeyError as exc:
                raise RuntimeError(f"Cannot auto-load unknown provider {cfg.auto_load_provider}") from exc
            await provider.load(cfg.auto_load_device, cfg.auto_load_checkpoint_path, cfg.auto_load_allow_download)
        yield
        for provider in providers.providers.values():
            provider.unload()

    app = FastAPI(title="Local Music Annotation Service", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=cfg.cors_origin_regex,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-File-Name"],
    )
    app.state.settings, app.state.store, app.state.jobs, app.state.providers = cfg, store, jobs, providers
    app.state.interactive_assets = {}

    bearer = HTTPBearer(auto_error=False)

    def authorize(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
        if cfg.session_token is None:
            return
        if credentials is None or credentials.credentials != cfg.session_token:
            raise HTTPException(status_code=401, detail="invalid session token")

    protected = [Depends(authorize)]

    @app.get("/v1/health", response_model=Health)
    def health() -> Health:
        return Health(service_version=__version__, librosa_engine_version=engine_version())

    @app.get("/v1/capabilities", response_model=Capabilities, dependencies=protected)
    def capabilities() -> Capabilities:
        devices = ["cpu"]
        try:
            import torch
            if torch.cuda.is_available():
                devices.append("cuda")
        except ImportError:
            pass
        return Capabilities(devices=devices, providers=providers.capabilities())

    @app.post("/v1/interactive-assets", response_model=Asset, status_code=201, dependencies=protected)
    async def upload_interactive_asset(request: Request, x_file_name: str | None = Header(default=None)) -> Asset:
        upload_root = cfg.data_root / "interactive-assets"
        upload_root.mkdir(parents=True, exist_ok=True)
        temporary = upload_root / f"upload-{__import__('uuid').uuid4().hex}.tmp"
        digest, size = hashlib.sha256(), 0
        try:
            with temporary.open("wb") as stream:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > cfg.maximum_upload_bytes:
                        raise HTTPException(status_code=413, detail="audio upload exceeds the configured size limit")
                    digest.update(chunk)
                    stream.write(chunk)
            if size == 0:
                raise HTTPException(status_code=422, detail="audio upload is empty")
            original_name = Path(unquote(x_file_name or "audio")).name
            suffix = Path(original_name).suffix.lower()
            if not suffix or len(suffix) > 12:
                suffix = ".audio"
            content_hash = digest.hexdigest()
            destination = upload_root / f"{content_hash}{suffix}"
            if destination.exists():
                temporary.unlink()
            else:
                temporary.replace(destination)
            asset = inspect_asset(str(destination), original_name)
            asset = asset.model_copy(update={"id": uuid5(NAMESPACE_URL, f"interactive:{content_hash}")})
            app.state.interactive_assets[asset.id] = asset
            return asset
        except HTTPException:
            if temporary.exists():
                temporary.unlink()
            raise
        except Exception as exc:
            if temporary.exists():
                temporary.unlink()
            raise HTTPException(status_code=422, detail=f"could not decode audio: {exc}") from exc

    @app.post("/v1/interactive-assets/{asset_id}:describe", response_model=InteractiveDescriptionResult, dependencies=protected)
    async def describe_interactive_asset(asset_id: UUID, value: InteractiveDescribeRequest) -> InteractiveDescriptionResult:
        asset = app.state.interactive_assets.get(asset_id)
        if asset is None:
            raise HTTPException(status_code=404, detail="interactive asset is not available; upload it again")
        try:
            return await asyncio.to_thread(describe_asset, asset, value, providers)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/interactive-assets/{asset_id}:relevance-curve", response_model=SemanticCurveResult, dependencies=protected)
    async def interactive_relevance_curve(asset_id: UUID, value: SemanticCurveRequest) -> SemanticCurveResult:
        asset = app.state.interactive_assets.get(asset_id)
        if asset is None:
            raise HTTPException(status_code=404, detail="interactive asset is not available; upload it again")
        try:
            return await asyncio.to_thread(relevance_curve, asset, value, providers, cfg.data_root / "relevance-cache")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/interactive-assets/{asset_id}:librosa", dependencies=protected)
    async def interactive_librosa(asset_id: UUID, value: InteractiveLibrosaRequest) -> dict:
        asset = app.state.interactive_assets.get(asset_id)
        if asset is None:
            raise HTTPException(status_code=404, detail="interactive asset is not available; upload it again")
        try:
            return await asyncio.to_thread(analyze_interactive_asset, asset, value, cfg.data_root / "librosa-cache")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/providers/{provider_id}:load", response_model=JobAccepted, status_code=202, dependencies=protected)
    async def load_provider(provider_id: str, request: ProviderLoadRequest) -> JobAccepted:
        try:
            provider = providers.get(provider_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        async def runner(progress):
            progress("loading", 0.1, f"Loading {provider.display_name}")
            return await provider.load(request.device, request.checkpoint_path, request.allow_download)

        job = jobs.submit("model-load", runner)
        return JobAccepted(job_id=job.id, state=job.state)

    @app.post("/v1/providers/{provider_id}:unload", response_model=ProviderCapability, dependencies=protected)
    def unload_provider(provider_id: str) -> ProviderCapability:
        try:
            provider = providers.get(provider_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        provider.unload()
        return provider.capability()

    @app.post("/v1/providers/{provider_id}:probe", response_model=JobAccepted, status_code=202, dependencies=protected)
    async def probe_provider(provider_id: str) -> JobAccepted:
        try:
            provider = providers.get(provider_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        async def runner(progress):
            progress("probing", 0.1, f"Probing {provider.display_name}")
            return await provider.probe()

        job = jobs.submit("model-probe", runner)
        return JobAccepted(job_id=job.id, state=job.state)

    @app.post("/v1/projects", response_model=Project, status_code=201, dependencies=protected)
    def create_project(value: ProjectCreate, response: Response) -> Project:
        project = store.create_project(value)
        response.headers["ETag"] = f'"{project.revision}"'
        return project

    @app.get("/v1/projects/{project_id}", response_model=Project, dependencies=protected)
    def get_project(project_id: UUID, response: Response) -> Project:
        try:
            project = store.get_project(project_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        response.headers["ETag"] = f'"{project.revision}"'
        return project

    @app.get("/v1/projects/{project_id}/assets", response_model=list[Asset], dependencies=protected)
    def list_assets(project_id: UUID) -> list[Asset]:
        try:
            return store.list_assets(project_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc

    async def submit_asset_import(project_id: UUID, value: AssetImport) -> JobAccepted:
        try:
            store.get_project(project_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc

        async def runner(progress):
            progress("inspecting", 0.1, "Inspecting and hashing audio")
            asset = await asyncio.to_thread(inspect_asset, value.path, value.name)
            progress("persisting", 0.8, "Registering asset")
            asset = store.save_asset(project_id, asset)
            return asset.model_dump(mode="json", by_alias=True)

        job = jobs.submit("asset-import", runner, project_id)
        return JobAccepted(job_id=job.id, state=job.state)

    @app.post("/v1/projects/{project_id}/assets", response_model=JobAccepted, status_code=202, dependencies=protected)
    async def import_asset(project_id: UUID, value: AssetImport) -> JobAccepted:
        return await submit_asset_import(project_id, value)

    @app.post("/v1/projects/{project_id}/assets:import", response_model=JobAccepted, status_code=202, dependencies=protected, include_in_schema=False)
    async def import_asset_alias(project_id: UUID, value: AssetImport) -> JobAccepted:
        return await submit_asset_import(project_id, value)

    @app.get("/v1/projects/{project_id}/annotations", response_model=list[Annotation], dependencies=protected)
    def query_annotations(
        project_id: UUID,
        asset_id: UUID = Query(alias="assetId"),
        start_sample: int = Query(ge=0, alias="startSample"),
        end_sample: int = Query(ge=1, alias="endSample"),
    ) -> list[Annotation]:
        if end_sample <= start_sample:
            raise HTTPException(status_code=422, detail="endSample must be greater than startSample")
        try:
            return store.query_annotations(project_id, asset_id, start_sample, end_sample)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc

    @app.post("/v1/projects/{project_id}/annotations", response_model=Annotation, status_code=201, dependencies=protected)
    def create_annotation(project_id: UUID, value: AnnotationCreate, response: Response) -> Annotation:
        try:
            annotation = store.create_annotation(project_id, value)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        response.headers["ETag"] = f'"{annotation.revision}"'
        return annotation

    @app.put("/v1/projects/{project_id}/annotations/{annotation_id}", response_model=Annotation, dependencies=protected)
    def update_annotation(project_id: UUID, annotation_id: UUID, value: AnnotationPatch, response: Response, if_match: str | None = Header(default=None)) -> Annotation:
        expected = _parse_etag(if_match)
        try:
            annotation = store.update_annotation(project_id, annotation_id, value, expected)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail="annotation not found") from exc
        except ConflictError as exc:
            raise HTTPException(status_code=409, detail={"currentRevision": exc.current_revision}) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        response.headers["ETag"] = f'"{annotation.revision}"'
        return annotation

    @app.delete("/v1/projects/{project_id}/annotations/{annotation_id}", status_code=204, dependencies=protected)
    def delete_annotation(project_id: UUID, annotation_id: UUID, if_match: str | None = Header(default=None)) -> Response:
        expected = _parse_etag(if_match)
        try:
            store.delete_annotation(project_id, annotation_id, expected)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail="annotation not found") from exc
        except ConflictError as exc:
            raise HTTPException(status_code=409, detail={"currentRevision": exc.current_revision}) from exc
        return Response(status_code=204)

    @app.post("/v1/projects/{project_id}/detection-plans:compile", response_model=JobAccepted, status_code=202, dependencies=protected)
    async def compile_detection_plan(project_id: UUID, value: DetectionPlanCompileRequest) -> JobAccepted:
        try:
            store.get_project(project_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc

        async def runner(progress):
            progress("interpreting", 0.2, "Mapping the request to music-analysis families")
            plan = compile_plan(value, providers.capabilities())
            progress("persisting", 0.8, "Saving reproducible detection plan")
            store.save_detection_plan(project_id, plan)
            validation = validate_plan(plan, providers.capabilities())
            return {
                "plan": plan.model_dump(mode="json", by_alias=True),
                "validation": validation.model_dump(mode="json", by_alias=True),
            }

        job = jobs.submit("detection-plan-compile", runner, project_id)
        return JobAccepted(job_id=job.id, state=job.state)

    @app.get("/v1/projects/{project_id}/detection-plans/{plan_id}", response_model=DetectionPlan, dependencies=protected)
    def get_detection_plan(project_id: UUID, plan_id: UUID, response: Response) -> DetectionPlan:
        try:
            plan = store.get_detection_plan(project_id, plan_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail="detection plan not found") from exc
        response.headers["ETag"] = f'"{plan.revision}"'
        return plan

    @app.put("/v1/projects/{project_id}/detection-plans/{plan_id}", response_model=DetectionPlan, dependencies=protected)
    def update_detection_plan(project_id: UUID, plan_id: UUID, value: DetectionPlan, response: Response, if_match: str | None = Header(default=None)) -> DetectionPlan:
        expected = _parse_etag(if_match)
        try:
            plan = store.update_detection_plan(project_id, plan_id, value, expected)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail="detection plan not found") from exc
        except ConflictError as exc:
            raise HTTPException(status_code=409, detail={"currentRevision": exc.current_revision}) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        response.headers["ETag"] = f'"{plan.revision}"'
        return plan

    @app.post("/v1/projects/{project_id}/detection-plans/{plan_id}:validate", response_model=DetectionPlanValidation, dependencies=protected)
    def validate_detection_plan(project_id: UUID, plan_id: UUID) -> DetectionPlanValidation:
        try:
            plan = store.get_detection_plan(project_id, plan_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail="detection plan not found") from exc
        return validate_plan(plan, providers.capabilities())

    async def submit_detection_run(project_id: UUID, plan_id: UUID, request: DetectionPlanRunRequest, preview: bool) -> JobAccepted:
        try:
            plan = store.get_detection_plan(project_id, plan_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail="detection plan not found") from exc
        if plan.revision != request.plan_revision:
            raise HTTPException(status_code=409, detail={"currentRevision": plan.revision})
        validation = validate_plan(plan, providers.capabilities())
        if validation.errors:
            raise HTTPException(status_code=422, detail=validation.model_dump(mode="json", by_alias=True))

        async def runner(progress):
            run_id = __import__("uuid").uuid4()
            all_candidates: list[DetectionCandidate] = []
            warnings = [finding.message for finding in validation.warnings]
            tasks: list[tuple[Asset, int, int]] = []
            for asset_id in request.asset_ids:
                asset = store.get_asset(project_id, asset_id)
                selected_ranges = [item for item in request.ranges if item.asset_id == asset_id]
                tasks.extend((asset, item.start_sample, item.end_sample) for item in selected_ranges)
                if not selected_ranges:
                    tasks.append((asset, 0, asset.samples))
            total = len(tasks)
            for index, (asset, start_sample, end_sample) in enumerate(tasks):
                progress("analyzing", index / total, f"Analyzing range {index + 1} of {total}")
                remaining = max(1, request.maximum_candidates - len(all_candidates))
                candidates, asset_warnings = await asyncio.to_thread(
                    analyze_range, asset, plan, run_id, start_sample, end_sample, providers, remaining,
                )
                all_candidates.extend(candidates)
                warnings.extend(asset_warnings)
                if len(all_candidates) >= request.maximum_candidates:
                    break
            decision_counts = dict(Counter(candidate.decision for candidate in all_candidates))
            run = DetectionRun(
                id=run_id, plan_id=plan.id, plan_revision=plan.revision, state="succeeded", preview=preview,
                candidate_count=len(all_candidates), decision_counts=decision_counts,
                warnings=warnings, created_at=datetime.now(timezone.utc),
            )
            progress("persisting", 0.9, "Saving evidence and review candidates")
            store.save_detection_run(project_id, run, all_candidates)
            if not preview:
                for candidate in all_candidates:
                    if candidate.decision not in {"review", "auto_accept"}:
                        continue
                    if store.has_equivalent_suggestion(project_id, candidate.asset_id, candidate.start_sample, candidate.end_sample, candidate.label_id):
                        continue
                    store.create_annotation(project_id, AnnotationCreate(
                        asset_id=candidate.asset_id, start_sample=candidate.start_sample, end_sample=candidate.end_sample,
                        label_ids=[candidate.label_id], state="suggested", provenance="model", confidence=candidate.score,
                        candidate_scores={candidate.label_id: candidate.score},
                        note=f"Detection run {run.id}; plan {plan.name} r{plan.revision}; review required",
                    ))
            return run.model_dump(mode="json", by_alias=True)

        job = jobs.submit("detection-preview" if preview else "detection-run", runner, project_id)
        return JobAccepted(job_id=job.id, state=job.state)

    @app.post("/v1/projects/{project_id}/detection-plans/{plan_id}:preview", response_model=JobAccepted, status_code=202, dependencies=protected)
    async def preview_detection_plan(project_id: UUID, plan_id: UUID, value: DetectionPlanRunRequest) -> JobAccepted:
        return await submit_detection_run(project_id, plan_id, value, True)

    @app.post("/v1/projects/{project_id}/detection-plans/{plan_id}:run", response_model=JobAccepted, status_code=202, dependencies=protected)
    async def run_detection_plan(project_id: UUID, plan_id: UUID, value: DetectionPlanRunRequest) -> JobAccepted:
        return await submit_detection_run(project_id, plan_id, value, False)

    @app.get("/v1/projects/{project_id}/detection-runs/{run_id}", response_model=DetectionRun, dependencies=protected)
    def get_detection_run(project_id: UUID, run_id: UUID) -> DetectionRun:
        try:
            return store.get_detection_run(project_id, run_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail="detection run not found") from exc

    @app.get("/v1/projects/{project_id}/detection-runs/{run_id}/candidates", response_model=list[DetectionCandidate], dependencies=protected)
    def list_detection_candidates(
        project_id: UUID, run_id: UUID, asset_id: UUID | None = Query(default=None, alias="assetId"),
        start_sample: int | None = Query(default=None, ge=0, alias="startSample"),
        end_sample: int | None = Query(default=None, ge=1, alias="endSample"),
    ) -> list[DetectionCandidate]:
        try:
            store.get_detection_run(project_id, run_id)
            return store.list_detection_candidates(project_id, run_id, asset_id, start_sample, end_sample)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail="detection run not found") from exc

    @app.get("/v1/jobs/{job_id}", response_model=Job, dependencies=protected)
    def get_job(job_id: UUID) -> Job:
        try:
            return jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.post("/v1/jobs/{job_id}:cancel", response_model=Job, dependencies=protected)
    def cancel_job(job_id: UUID) -> Job:
        try:
            return jobs.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.get("/v1/events", dependencies=protected)
    async def events() -> StreamingResponse:
        queue = __import__("asyncio").Queue(maxsize=100)
        jobs.events.add(queue)

        async def stream() -> AsyncIterator[str]:
            try:
                while True:
                    job = await queue.get()
                    yield f"event: job.updated\ndata: {json.dumps(job.model_dump(mode='json', by_alias=True))}\n\n"
            finally:
                jobs.events.discard(queue)
        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def _parse_etag(value: str | None) -> int:
    if value is None:
        raise HTTPException(status_code=status.HTTP_428_PRECONDITION_REQUIRED, detail="If-Match is required")
    try:
        return int(value.strip('W/"'))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid If-Match") from exc
