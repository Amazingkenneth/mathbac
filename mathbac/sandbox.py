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

    #: Materials as seen by the harness (name -> host path). File tools read through these.
    materials: dict[str, Path]

    @abstractmethod
    def exec(self, command: str, timeout_s: float = 600) -> ExecResult: ...

    def adopt(self, path: Path) -> None:
        """Hand a file the harness wrote on the agent's behalf over to the sandbox's owner (no-op by default)."""

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
        self.materials = dict(materials or {})
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
        self.materials = dict(materials or {})
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


class UserSandbox(Sandbox):
    """Runs commands as an unprivileged local user, for hosts without Docker (HPC nodes, cloud containers).

    The harness runs as root. At start-up it:
    - creates the agent user if needed,
    - removes all access for other users from each `protect` path (the repository: gold data, tests),
    - copies the materials read-only next to the workspace and links them into it,
    - checks that the agent user can traverse to the workspace and cannot read the protected paths.
    Commands get a scrubbed environment: no API keys, a fixed PATH, HOME in the agent's home.
    """

    def __init__(self, workspace: Path, materials: dict[str, Path] | None = None, user: str = "mathbac-agent",
                 protect: list[Path] = (), path: str = "/opt/mathbac/env/bin:/usr/local/bin:/usr/bin:/bin"):
        import grp
        import pwd
        import shutil

        if os.geteuid() != 0:
            raise PermissionError("the user sandbox needs the harness to run as root so it can switch users")
        self.workspace = Path(workspace).resolve()
        for p in protect:
            if self.workspace.is_relative_to(Path(p).resolve()):
                raise ValueError(f"workspace {self.workspace} is inside protected path {p}; set runs_dir outside it")
        try:
            pwd.getpwnam(user)
        except KeyError:
            subprocess.run(["useradd", "--system", "--create-home", "--shell", "/bin/bash", user], check=True, capture_output=True)
        pw = pwd.getpwnam(user)
        self.user, self.uid, self.gid, self.home = user, pw.pw_uid, pw.pw_gid, pw.pw_dir
        self.path = path
        for p in protect:
            p = Path(p).resolve()
            os.chmod(p, os.stat(p).st_mode & ~0o007)

        self.workspace.mkdir(parents=True, exist_ok=True)
        self.inner_workspace = str(self.workspace)
        ro = self.workspace.parent / "materials"
        ro.mkdir(exist_ok=True)
        self.materials = {}
        for name, src in (materials or {}).items():
            dest = ro / name
            if not dest.exists():
                shutil.copytree(src, dest, symlinks=False)
            subprocess.run(["chmod", "-R", "a+rX,a-w", str(dest)], check=True)
            self.materials[name] = dest
        os.chmod(ro, 0o755)
        link_dir = self.workspace / "materials"
        link_dir.mkdir(exist_ok=True)
        for name, dest in self.materials.items():
            link = link_dir / name
            if not link.exists():
                link.symlink_to(dest, target_is_directory=True)
        self.inner_materials = str(link_dir)
        subprocess.run(["chown", "-R", f"{self.uid}:{self.gid}", str(self.workspace)], check=True)

        # Self-checks: the agent can work in the workspace and cannot read what is protected.
        probe = self._run(f"test -w {shlex.quote(str(self.workspace))} && echo ok", 30)
        if probe.output.strip() != "ok":
            raise PermissionError(f"agent user {user} cannot write the workspace (check o+x on its parent folders): {probe.output}")
        for p in protect:
            leak = self._run(f"ls {shlex.quote(str(p))} >/dev/null 2>&1 && echo LEAK || echo ok", 30)
            if leak.output.strip() != "ok":
                raise PermissionError(f"agent user can still read protected path {p}")

    def _run(self, command: str, timeout_s: float) -> ExecResult:
        env = ["env", "-i", f"HOME={self.home}", f"PATH={self.path}", "LANG=C.UTF-8", f"USER={self.user}",
               f"WORK={self.workspace}", f"MATERIALS={self.inner_materials}",
               f"HTTPS_PROXY={os.environ.get('HTTPS_PROXY', '')}", f"HTTP_PROXY={os.environ.get('HTTP_PROXY', '')}",
               f"NO_PROXY={os.environ.get('NO_PROXY', '')}", f"SSL_CERT_FILE={os.environ.get('SSL_CERT_FILE', '')}",
               f"REQUESTS_CA_BUNDLE={os.environ.get('REQUESTS_CA_BUNDLE', '')}"]
        argv = ["runuser", "-u", self.user, "--", *env, "timeout", "--kill-after=10", str(int(timeout_s)), "bash", "-c", command]
        t0 = time.time()
        try:
            p = subprocess.run(argv, cwd=self.workspace, capture_output=True, text=True, timeout=timeout_s + 30)
        except subprocess.TimeoutExpired:
            return ExecResult(124, f"[timed out after {timeout_s:.0f}s]", True, time.time() - t0)
        out = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr else "")
        timed_out = p.returncode == 124
        if timed_out:
            out += f"\n[timed out after {timeout_s:.0f}s]"
        return ExecResult(p.returncode, truncate(out), timed_out, time.time() - t0)

    def exec(self, command: str, timeout_s: float = 600) -> ExecResult:
        return self._run(command, timeout_s)

    def adopt(self, path: Path) -> None:
        path = Path(path)
        for p in [path, *path.parents]:
            if p == self.workspace or not p.is_relative_to(self.workspace):
                break
            os.chown(p, self.uid, self.gid)


def make_sandbox(cfg: dict, workspace: Path, materials: dict[str, Path], protect: list[Path] = ()) -> Sandbox:
    kind = cfg.get("kind", "docker")
    if kind == "user":
        return UserSandbox(workspace, materials, user=cfg.get("user", "mathbac-agent"), protect=list(protect),
                           path=cfg.get("path", "/opt/mathbac/env/bin:/usr/local/bin:/usr/bin:/bin"))
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
