from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from .schemas import Job


Runner = Callable[[Callable[[str, float | None, str], None]], Awaitable[dict[str, Any]]]


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[UUID, Job] = {}
        self.tasks: dict[UUID, asyncio.Task[None]] = {}
        self.events: set[asyncio.Queue[Job]] = set()

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def submit(self, kind: str, runner: Runner, project_id: UUID | None = None) -> Job:
        now, job_id = self._now(), uuid4()
        job = Job(id=job_id, project_id=project_id, kind=kind, state="queued", phase="queued", created_at=now, updated_at=now)
        self.jobs[job_id] = job
        self.tasks[job_id] = asyncio.create_task(self._run(job_id, runner))
        return job

    async def _run(self, job_id: UUID, runner: Runner) -> None:
        self._update(job_id, state="running", phase="starting")

        def progress(phase: str, value: float | None, message: str = "") -> None:
            self._update(job_id, phase=phase, progress=value, message=message)

        try:
            result = await runner(progress)
            self._update(job_id, state="succeeded", phase="complete", progress=1.0, result=result)
        except asyncio.CancelledError:
            self._update(job_id, state="cancelled", phase="cancelled")
        except Exception as exc:  # job boundary converts errors to stable API data
            self._update(job_id, state="failed", phase="failed", error={"type": "provider-error", "title": type(exc).__name__, "status": 500, "detail": str(exc)})

    def _update(self, job_id: UUID, **changes: Any) -> None:
        job = self.jobs[job_id].model_copy(update={**changes, "updated_at": self._now()})
        self.jobs[job_id] = job
        for queue in tuple(self.events):
            try:
                queue.put_nowait(job)
            except asyncio.QueueFull:
                pass

    def get(self, job_id: UUID) -> Job:
        return self.jobs[job_id]

    def cancel(self, job_id: UUID) -> Job:
        job = self.jobs[job_id]
        if job.state in {"queued", "running"}:
            self._update(job_id, state="cancel_requested", phase=job.phase)
            self.tasks[job_id].cancel()
        return self.jobs[job_id]

