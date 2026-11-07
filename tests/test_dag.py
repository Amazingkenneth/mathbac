import pytest

from mathbac.dag import DAGError, TaskDAG
from mathbac.schemas import NodeSpec, NodeStatus


def n(i, *deps):
    return NodeSpec(id=i, role="r", goal=f"goal {i}", depends_on=list(deps))


def test_ready_frontier_follows_dependencies():
    dag = TaskDAG([n("a"), n("b"), n("c", "a", "b"), n("d", "c")])
    assert [s.id for s in dag.ready()] == ["a", "b"]
    dag.mark("a", NodeStatus.DONE)
    assert [s.id for s in dag.ready()] == ["b"]
    dag.mark("b", NodeStatus.DONE)
    assert [s.id for s in dag.ready()] == ["c"]


def test_unknown_dependency_and_duplicate_rejected():
    dag = TaskDAG([n("a")])
    with pytest.raises(DAGError):
        dag.add(n("b", "zzz"))
    with pytest.raises(DAGError):
        dag.add(n("a"))


def test_cancel_cascades_downstream():
    dag = TaskDAG([n("a"), n("b", "a"), n("c", "b"), n("x")])
    assert dag.cancel("a") == ["a", "b", "c"]
    assert dag.status("x") == NodeStatus.PENDING


def test_rewrite_rejects_cycles_and_running_nodes():
    dag = TaskDAG([n("a"), n("b", "a")])
    with pytest.raises(DAGError):
        dag.rewrite(n("a", "b"))
    assert dag.spec("a").depends_on == []
    dag.mark("b", NodeStatus.RUNNING)
    with pytest.raises(DAGError):
        dag.rewrite(n("b", "a"))


def test_blocked_after_failure_and_finished():
    dag = TaskDAG([n("a"), n("b", "a")])
    dag.mark("a", NodeStatus.FAILED)
    assert dag.blocked() == ["b"]
    assert dag.is_finished()


def test_subset_drops_dependencies_on_removed_nodes():
    dag = TaskDAG([n("a"), n("b", "a"), n("c", "b")])
    sub = dag.subset(["b", "c"])
    assert sub.spec("b").depends_on == []
    assert sub.spec("c").depends_on == ["b"]


def test_reference_dag_loads_and_is_acyclic():
    from pathlib import Path

    from mathbac.task import Task

    task = Task.load(Path(__file__).resolve().parents[1] / "tasks" / "cha2022")
    dag = task.reference_dag()
    assert dag.order()[0] in ("N1", "N2")
    assert set(dag.ids()) >= {"N1", "N2", "N3", "N4a", "N4b", "N4c", "N5", "N6a", "N6b", "N7", "N8", "N9"}
    # every acceptance check named in the DAG exists
    for i in dag.ids():
        if dag.spec(i).acceptance:
            task.acceptance(dag.spec(i).acceptance)
