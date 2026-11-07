"""Where agent commands run.

Agents use the command line instead of a VM desktop. Commands are untrusted
model output, so the container is the security boundary: it sees only the run's
workspace (read-write, at /work) and the materials allowed at this access level
(read-only, at /materials). API keys are never passed into it.

``LocalSandbox`` runs commands directly on the host and exists for development
and tests only; it must be enabled explicitly.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

MAX_OUTPUT_CHARS = 30_000

# Environment variables never forwarded to agent commands.
_SECRET_ENV = ("API_KEY", "AUTH_TOKEN", "SECRET", "PASSWORD", "TOKEN")


@dataclass
class ExecResult:
    exit_code: int
    output: str
    timed_out: bool
    seconds: float


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... [{len(text) - limit} characters omitted] ...\n{text[-half:]}"


def scrubbed_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not any(s in k.upper() for s in _SECRET_ENV)}
    env.update(extra or {})
    return env


class Sandbox(ABC):
    workspace: Path
    #: Path of the workspace as seen by commands inside the sandbox.
    inner_workspace: str
    inner_materials: str

    @abstractmethod
    def exec(self, command: str, timeout_s: float = 600) -> ExecResult: ...

    def close(self) -> None:  # pragma: no cover - default no-op
        pass


class LocalSandbox(Sandbox):
    def __init__(self, workspace: Path, materials: dict[str, Path] | None = None, allow_unsafe_local: bool = False):
        if not allow_unsafe_local:
            raise PermissionError(
                "LocalSandbox runs model-written commands on this machine. Use the docker sandbox, "
                "or pass allow_unsafe_local=True (sandbox.kind: local_unsafe) for development only."
            )
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.inner_workspace = str(self.workspace)
        mdir = self.workspace / "materials"
        mdir.mkdir(exist_ok=True)
        for name, src in (materials or {}).items():
            link = mdir / name
            if not link.exists():
                link.symlink_to(Path(src).resolve(), target_is_directory=True)
        self.inner_materials = str(mdir)

    def exec(self, command: str, timeout_s: float = 600) -> ExecResult:
        t0 = time.time()
        try:
            p = subprocess.run(
                ["bash", "-lc", command],
                cwd=self.workspace,
                env=scrubbed_env({"WORK": self.inner_workspace, "MATERIALS": self.inner_materials}),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            out = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr else "")
            return ExecResult(p.returncode, truncate(out), False, time.time() - t0)
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            return ExecResult(124, truncate(out + f"\n[timed out after {timeout_s:.0f}s]"), True, time.time() - t0)


class DockerSandbox(Sandbox):
    """One long-lived container per run. Commands run through ``docker exec``."""

    def __init__(
        self,
        workspace: Path,
        image: str,
        materials: dict[str, Path] | None = None,
        network: str = "bridge",
        cpus: float = 4,
        memory: str = "16g",
        name: str | None = None,
    ):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.inner_workspace = "/work"
        self.inner_materials = "/materials"
        self.name = name or f"mathbac-{uuid.uuid4().hex[:10]}"
        cmd = [
            "docker", "run", "-d", "--rm", "--name", self.name,
            "--network", network, "--cpus", str(cpus), "--memory", memory,
            "--security-opt", "no-new-privileges",
            "-v", f"{self.workspace}:/work",
            "-w", "/work",
            "-e", "WORK=/work", "-e", "MATERIALS=/materials",
        ]
        for mname, src in (materials or {}).items():
            cmd += ["-v", f"{Path(src).resolve()}:/materials/{mname}:ro"]
        cmd += [image, "sleep", "infinity"]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    def exec(self, command: str, timeout_s: float = 600) -> ExecResult:
        t0 = time.time()
        # `timeout` inside the container stops the command itself, not just our wait on it.
        inner = f"timeout --kill-after=10 {int(timeout_s)} bash -lc {shlex.quote(command)}"
        try:
            p = subprocess.run(
                ["docker", "exec", "-w", "/work", self.name, "bash", "-c", inner],
                capture_output=True, text=True, timeout=timeout_s + 30,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(124, f"[timed out after {timeout_s:.0f}s]", True, time.time() - t0)
        out = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr else "")
        timed_out = p.returncode == 124
        if timed_out:
            out += f"\n[timed out after {timeout_s:.0f}s]"
        return ExecResult(p.returncode, truncate(out), timed_out, time.time() - t0)

    def close(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)


def make_sandbox(cfg: dict, workspace: Path, materials: dict[str, Path]) -> Sandbox:
    kind = cfg.get("kind", "docker")
    if kind == "docker":
        return DockerSandbox(
            workspace,
            image=cfg.get("image", "mathbac-cha2022:latest"),
            materials=materials,
            network=cfg.get("network", "bridge"),
            cpus=cfg.get("cpus", 4),
            memory=cfg.get("memory", "16g"),
        )
    if kind == "local_unsafe":
        return LocalSandbox(workspace, materials, allow_unsafe_local=True)
    raise ValueError(f"unknown sandbox kind {kind!r}")
