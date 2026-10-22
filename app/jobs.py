from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

MAX_JOBS_IN_MEMORY = 500


@dataclass(slots=True)
class Job:
    id: str
    kind: str
    status: str = "queued"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class JobManager:
    """Fila de jobs assíncronos dentro do processo da API.

    ponytail: jobs em andamento se perdem se o processo reiniciar (o histórico fica
    em `job_records`); para durabilidade real, troque por uma fila externa.
    """

    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.tasks: dict[str, asyncio.Task[Any]] = {}

    async def submit(self, kind: str, fn: Callable[[], Awaitable[dict[str, Any]]]) -> Job:
        job = Job(id=uuid.uuid4().hex, kind=kind)
        self.jobs[job.id] = job
        while len(self.jobs) > MAX_JOBS_IN_MEMORY:
            self.jobs.pop(next(iter(self.jobs)))
        task = asyncio.create_task(self._run(job, fn))
        self.tasks[job.id] = task
        task.add_done_callback(lambda _: self.tasks.pop(job.id, None))
        return job

    async def _run(self, job: Job, fn: Callable[[], Awaitable[dict[str, Any]]]) -> None:
        job.status = "running"
        job.started_at = datetime.now(UTC)
        try:
            job.result = await fn()
            job.status = "completed" if job.result.get("status", "completed") == "completed" else "failed"
        except Exception as exc:
            job.error = str(exc)
            job.status = "failed"
        finally:
            job.finished_at = datetime.now(UTC)


job_manager = JobManager()
