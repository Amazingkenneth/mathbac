"""Compute queue for heavy tool runs.

Agents never run heavy jobs inline. They submit a job (tool + args + resources)
and get a job id back. Results are cached by a hash of the tool, its arguments
and the contents of any input files the arguments name, so a retry does not
recompute 464 complexes.

Backends:
- ``local``: runs jobs in the run's sandbox on a thread pool (default).
- ``slurm``: renders an sbatch script that runs the command under Apptainer and
  submits it. Not exercised by the tests; check the template against your cluster.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout  # distinct from builtin TimeoutError on Python 3.10
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..ledger import Ledger, sha256_file
from ..sandbox import Sandbox
from ..schemas import JobSpec


@dataclass
class JobRecord:
    job_id: str
    tool: str
    command: str
    cache_key: str
    state: str = "queued"  # queued | running | done | failed | cached
    exit_code: int | None = None
    output_tail: str = ""
    seconds: float = 0.0
    node_id: str | None = None
    submitted_at: float = field(default_factory=time.time)


class JobQueue:
    def __init__(
        self,
        sandbox: Sandbox,
        ledger: Ledger,
        cache_dir: Path,
        backend: str = "local",
        max_workers: int = 4,
        slurm: dict[str, Any] | None = None,
    ):
        self.sandbox = sandbox
        self.ledger = ledger
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.slurm = slurm or {}
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="job")
        self._jobs: dict[str, JobRecord] = {}
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def cache_key(self, spec: JobSpec) -> str:
        h = hashlib.sha256()
        h.update(spec.tool.encode())
        h.update(json.dumps(spec.args, sort_keys=True, default=str).encode())
        # Only declared inputs count: hashing outputs would make every rerun a cache miss.
        for rel in sorted(spec.input_paths):
            p = (self.sandbox.workspace / rel).resolve()
            if p.is_file() and p.is_relative_to(self.sandbox.workspace.resolve()):
                h.update(rel.encode())
                h.update(sha256_file(p).encode())
            else:
                h.update(f"{rel}:missing".encode())
        return h.hexdigest()[:24]

    def submit(self, spec: JobSpec, command: str) -> JobRecord:
        key = self.cache_key(spec)
        cached = self.cache_dir / f"{key}.json"
        job_id = f"job-{uuid.uuid4().hex[:8]}"
        if cached.exists():
            prev = json.loads(cached.read_text())
            rec = JobRecord(job_id=job_id, tool=spec.tool, command=command, cache_key=key, state="cached",
                            exit_code=0, output_tail=prev.get("output_tail", ""), node_id=spec.node_id)
            with self._lock:
                self._jobs[job_id] = rec
            self.ledger.append("job", **asdict(rec))
            return rec

        rec = JobRecord(job_id=job_id, tool=spec.tool, command=command, cache_key=key, node_id=spec.node_id)
        with self._lock:
            self._jobs[job_id] = rec
        self.ledger.append("job", **asdict(rec))
        runner = self._run_slurm if self.backend == "slurm" else self._run_local
        self._futures[job_id] = self._pool.submit(runner, rec, spec)
        return rec

    def status(self, job_id: str) -> JobRecord:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"unknown job {job_id}")
            return self._jobs[job_id]

    def wait(self, job_id: str, timeout_s: float | None = None) -> JobRecord:
        fut = self._futures.get(job_id)
        if fut is not None:
            try:
                fut.result(timeout=timeout_s)
            except FutureTimeout:
                pass  # still running; the caller gets the current state
        return self.status(job_id)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------
    def _finish(self, rec: JobRecord, exit_code: int, output: str, seconds: float) -> None:
        rec.exit_code = exit_code
        rec.output_tail = output[-4000:]
        rec.seconds = seconds
        rec.state = "done" if exit_code == 0 else "failed"
        if exit_code == 0:
            (self.cache_dir / f"{rec.cache_key}.json").write_text(json.dumps(asdict(rec)))
        self.ledger.append("job", **asdict(rec))

    def _run_local(self, rec: JobRecord, spec: JobSpec) -> None:
        rec.state = "running"
        res = self.sandbox.exec(rec.command, timeout_s=spec.time_limit_min * 60)
        self._finish(rec, res.exit_code, res.output, res.seconds)

    def _run_slurm(self, rec: JobRecord, spec: JobSpec) -> None:
        if shutil.which("sbatch") is None:
            self._finish(rec, 127, "sbatch not found: this machine is not a Slurm submit host", 0.0)
            return
        script = self.render_sbatch(rec, spec)
        path = self.sandbox.workspace / ".jobs" / f"{rec.job_id}.sbatch"
        path.parent.mkdir(exist_ok=True)
        path.write_text(script)
        t0 = time.time()
        p = subprocess.run(["sbatch", "--parsable", "--wait", str(path)], capture_output=True, text=True)
        log = path.with_suffix(".out")
        out = log.read_text() if log.exists() else p.stdout + p.stderr
        self._finish(rec, p.returncode, out, time.time() - t0)

    def render_sbatch(self, rec: JobRecord, spec: JobSpec) -> str:
        image = self.slurm.get("apptainer_image", "mathbac-cha2022.sif")
        partition = self.slurm.get("partition")
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name=mathbac-{rec.tool}",
            f"#SBATCH --cpus-per-task={spec.cpus}",
            f"#SBATCH --mem={int(spec.mem_gb)}G",
            f"#SBATCH --time={int(spec.time_limit_min)}",
            f"#SBATCH --output={self.sandbox.workspace}/.jobs/{rec.job_id}.out",
        ]
        if spec.gpus:
            lines.append(f"#SBATCH --gres=gpu:{spec.gpus}")
        if partition:
            lines.append(f"#SBATCH --partition={partition}")
        binds = [f"{self.sandbox.workspace}:/work", *self.slurm.get("binds", [])]  # e.g. "/data/cha2022/paper:/materials/paper:ro"
        lines += [
            "set -euo pipefail",
            f"apptainer exec --bind {','.join(binds)} --pwd /work {image} bash -lc {json.dumps(rec.command)}",
        ]
        return "\n".join(lines) + "\n"
