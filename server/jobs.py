"""In-memory job registry. One process, one GPU worker: a dict guarded by a lock is all the persistence needed."""
import threading
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Job:
    id: str
    kind: str                      # "floor" | "object"
    spec: dict                     # the validated request body
    out_dir: object                # pathlib.Path
    status: str = "queued"         # queued | running | done | failed
    stage: str | None = None
    stages_done: list = field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None

    @property
    def elapsed_s(self):
        t0 = self.started or self.created
        return round((self.finished or time.time()) - t0, 1)


class JobRegistry:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, kind, spec, out_root):
        jid = uuid.uuid4().hex[:12]
        job = Job(jid, kind, spec, out_root / jid)
        job.out_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._jobs[jid] = job
        return job

    def get(self, jid):
        with self._lock:
            return self._jobs.get(jid)

    def position(self, job):
        """0 while running / next up; otherwise the number of queued jobs created before it."""
        with self._lock:
            if job.status != "queued":
                return 0
            return sum(1 for j in self._jobs.values() if j.status in ("queued", "running") and j.created < job.created)

    def fail_unfinished(self, reason):
        with self._lock:
            for j in self._jobs.values():
                if j.status in ("queued", "running"):
                    j.status, j.error, j.finished = "failed", reason, time.time()
