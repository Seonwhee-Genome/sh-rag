"""Minimal background jobs (model pulls, index builds) with progress for the console."""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobCancelled(Exception):
    """Raised inside a job's thread (from report()) once cancel() was requested."""


@dataclass
class Job:
    id: str
    kind: str
    key: str
    title: str
    status: str = "running"  # running | done | error | cancelled
    message: str = ""
    progress: float | None = None
    detail: dict = field(default_factory=dict)  # kind-specific extras, e.g. bytes for pulls
    cancel_requested: bool = False
    created_at: str = field(default_factory=_now)
    finished_at: str | None = None
    result: Any = None
    error: str | None = None

    def report(self, message: str | None = None, progress: float | None = None, **detail: Any) -> None:
        """Update progress. Raises JobCancelled if cancel() was called, so the
        worker unwinds at its next progress event (closing streams on the way)."""
        if message is not None:
            self.message = message
        self.progress = progress
        if detail:
            self.detail.update(detail)
        if self.cancel_requested:
            raise JobCancelled(self.id)

    def cancel(self) -> bool:
        if self.status != "running":
            return False
        self.cancel_requested = True
        self.message = "Cancelling…"
        return True


class JobManager:
    def __init__(self, keep: int = 30):
        self.keep = keep
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, kind: str, key: str, title: str, fn: Callable[[Job], Any], detail: dict | None = None) -> Job:
        """Run fn(job) in a thread. A running job with the same key is reused, not duplicated."""
        with self._lock:
            for job in self._jobs.values():
                if job.key == key and job.status == "running":
                    return job
            job = Job(uuid.uuid4().hex[:12], kind, key, title, detail=dict(detail or {}))
            self._jobs[job.id] = job
            finished = [j for j in self._jobs.values() if j.status != "running"]
            for old in finished[: max(0, len(self._jobs) - self.keep)]:
                del self._jobs[old.id]
        threading.Thread(target=self._run, args=(job, fn), name=f"job-{kind}", daemon=True).start()
        return job

    def _run(self, job: Job, fn: Callable[[Job], Any]) -> None:
        try:
            job.result = fn(job)
            job.status, job.progress = "done", 1.0
        except JobCancelled:
            job.status, job.message = "cancelled", "Cancelled"
        except Exception as exc:
            if job.cancel_requested:  # e.g. the closed stream raised before report() did
                job.status, job.message = "cancelled", "Cancelled"
            else:
                log.exception("job %s (%s) failed", job.id, job.title)
                job.status, job.error = "error", str(exc)
        finally:
            job.finished_at = _now()

    def cancel(self, job_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        if job is not None:
            job.cancel()
        return job

    def running(self, kind: str) -> Job | None:
        with self._lock:
            return next((j for j in self._jobs.values() if j.kind == kind and j.status == "running"), None)

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[dict]:
        with self._lock:
            return [asdict(j) for j in reversed(self._jobs.values())]
