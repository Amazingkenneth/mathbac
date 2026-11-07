"""Isolation checks for the unprivileged-user sandbox. Needs root and runuser; skipped otherwise."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from mathbac.sandbox import UserSandbox

pytestmark = pytest.mark.skipif(os.geteuid() != 0 or shutil.which("runuser") is None, reason="needs root and runuser")


@pytest.fixture
def base():
    # pytest's tmp_path is not traversable by other users, so build the layout somewhere that is.
    d = Path(tempfile.mkdtemp(prefix="mathbac-sbtest-", dir="/srv" if os.path.isdir("/srv") else "/var/tmp"))
    os.chmod(d, 0o755)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_isolation(base, monkeypatch):
    repo = base / "repo"
    (repo / "gold").mkdir(parents=True)
    (repo / "gold" / "answers.yaml").write_text("contact_cutoff_A: 7.0\n")
    os.chmod(repo, 0o755)
    mat = base / "src_materials" / "paper"
    mat.mkdir(parents=True)
    (mat / "paper.txt").write_text("the paper")
    runs = base / "runs" / "r1"
    runs.mkdir(parents=True)
    os.chmod(base / "runs", 0o755)
    os.chmod(runs, 0o755)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-leak-0000000000")

    sb = UserSandbox(runs / "workspace", {"paper": mat}, protect=[repo])

    who = sb.exec("id -un; echo key=${ANTHROPIC_API_KEY:-none}")
    assert who.output.split()[0] == "mathbac-agent"
    assert "key=none" in who.output

    assert "Permission denied" in sb.exec(f"cat {repo}/gold/answers.yaml").output
    assert sb.exec("cat materials/paper/paper.txt").output.strip() == "the paper"
    assert sb.exec("echo x > materials/paper/new.txt").exit_code != 0  # read-only
    assert sb.exec("mkdir -p out && echo hi > out/a.txt && cat out/a.txt").output.strip() == "hi"

    f = sb.workspace / "spec" / "spec.yaml"
    f.parent.mkdir()
    f.write_text("x: 1\n")
    sb.adopt(f)
    assert sb.exec("echo y >> spec/spec.yaml && cat spec/spec.yaml").output.split() == ["x:", "1", "y"]


def test_refuses_workspace_inside_protected_path(base):
    repo = base / "repo"
    repo.mkdir()
    with pytest.raises(ValueError):
        UserSandbox(repo / "runs" / "w", {}, protect=[repo])
