from pathlib import Path

import pytest

from mathbac.compute import JobQueue
from mathbac.ledger import Ledger
from mathbac.sandbox import LocalSandbox
from mathbac.schemas import NodeSpec, ToolCard
from mathbac.tools import PathError, ToolContext, ToolRegistry, ToolRegistryError, run_handler, worker_tools

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def ctx(tmp_path):
    ws = tmp_path / "ws"
    mat = tmp_path / "materials" / "paper"
    mat.mkdir(parents=True)
    (mat / "paper.txt").write_text("contact cutoff 7 Å")
    (tmp_path / "secret.txt").write_text("nope")
    sb = LocalSandbox(ws, {"paper": mat}, allow_unsafe_local=True)
    led = Ledger(tmp_path / "run")
    q = JobQueue(sb, led, tmp_path / "cache")
    card = ToolCard(name="echo", tier=1, title="echo", purpose="echo", installed=True,
                    invoke="echo hello {who} > {out}", args={"who": {"type": "string"}, "out": {"type": "string"}})
    reg = ToolRegistry({"echo": card})
    yield ToolContext(workspace=sb.workspace, materials={"paper": mat}, sandbox=sb, queue=q, registry=reg, ledger=led,
                      node=NodeSpec(id="N1", role="r", goal="g", tools=["echo"]), secrets=["sk-ant-THISISASECRETKEY"])
    q.shutdown()


def tool(name):
    return next(t for t in worker_tools() if t.spec.name == name)


def test_path_confinement(ctx):
    assert ctx.resolve("a/b.txt") == ctx.workspace / "a" / "b.txt"
    assert ctx.resolve("/work/a.txt") == ctx.workspace / "a.txt"
    assert ctx.resolve("/materials/paper/paper.txt").read_text().startswith("contact")
    for bad in ["../secret.txt", "/etc/passwd", "a/../../secret.txt", "/materials/../secret.txt"]:
        with pytest.raises(PathError):
            ctx.resolve(bad)
    with pytest.raises(PathError):
        ctx.resolve("/materials/paper/x.txt", write=True)
    with pytest.raises(PathError):
        ctx.resolve("/materials/code/x.py")  # not available at this access level


def test_symlink_escape_rejected(ctx, tmp_path):
    (ctx.workspace / "link").symlink_to(tmp_path)
    with pytest.raises(PathError):
        ctx.resolve("link/secret.txt")


def test_write_read_and_artifact_recorded(ctx):
    out, err, _ = run_handler(tool("write_file"), ctx, {"path": "spec/x.txt", "content": "hello\nworld"})
    assert not err
    out, err, _ = run_handler(tool("read_file"), ctx, {"path": "spec/x.txt"})
    assert "world" in out and not err
    assert any(e["event"] == "artifact" and e["path"] == "spec/x.txt" for e in ctx.ledger.events())


def test_shell_output_is_redacted_and_env_scrubbed(ctx, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-THISISASECRETKEY")
    out, err, _ = run_handler(tool("run_shell"), ctx, {"command": "echo key=$ANTHROPIC_API_KEY; echo sk-ant-THISISASECRETKEY"})
    assert "THISISASECRETKEY" not in out
    assert "key=\n" in out or "key=" in out.splitlines()[1]


def test_run_tool_through_queue_with_cache(ctx):
    out, err, _ = run_handler(tool("run_tool"), ctx, {"name": "echo", "args": {"who": "world; rm -rf /", "out": "o.txt"}})
    assert not err, out
    assert (ctx.workspace / "o.txt").read_text().strip() == "hello world; rm -rf /"  # quoted, not executed
    out2, _, _ = run_handler(tool("run_tool"), ctx, {"name": "echo", "args": {"who": "world; rm -rf /", "out": "o.txt"}})
    assert '"cached"' in out2


def test_cache_keys_on_input_contents(ctx):
    card = ToolCard(name="cat", tier=1, title="cat", purpose="cat", installed=True, invoke="cat {src} > {out}",
                    args={"src": {"type": "string", "input": True}, "out": {"type": "string"}})
    ctx.registry.cards["cat"] = card
    ctx.node = ctx.node.model_copy(update={"tools": ["cat"]})
    (ctx.workspace / "in.txt").write_text("v1")
    args = {"name": "cat", "args": {"src": "in.txt", "out": "o.txt"}}
    assert '"done"' in run_handler(tool("run_tool"), ctx, args)[0]
    assert '"cached"' in run_handler(tool("run_tool"), ctx, args)[0]
    (ctx.workspace / "in.txt").write_text("v2")
    assert '"done"' in run_handler(tool("run_tool"), ctx, args)[0]
    assert (ctx.workspace / "o.txt").read_text() == "v2"


def test_run_tool_respects_node_tool_list(ctx):
    ctx.node = ctx.node.model_copy(update={"tools": ["other"]})
    out, err, _ = run_handler(tool("run_tool"), ctx, {"name": "echo", "args": {}})
    assert err and "not in this node's tool list" in out


def test_registry_cards_load_and_render():
    reg = ToolRegistry.load(REPO / "toolcards")
    assert len(reg.cards) == 20
    assert "Pitfalls" in reg.render("ghecom")
    with pytest.raises(ToolRegistryError):
        reg.command("alphafold3", {})  # card only
    with pytest.raises(ToolRegistryError):
        reg.command("dnn_train", {"dataset": "d.npz", "subset": "GT", "options": "--ok ; rm -rf /"})
    with pytest.raises(ToolRegistryError):
        reg.command("ghecom", {"pdb": "x.pdb"})  # missing required args
