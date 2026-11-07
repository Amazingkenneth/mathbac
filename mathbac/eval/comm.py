"""Communication and process metrics computed from the run ledger.

See the design doc, "Evaluating the system". Each metric is defined so it can be
computed from logged events alone.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from ..ledger import read_events


def _norm(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def _same(a: Any, b: Any) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)
    except (TypeError, ValueError):
        pass
    if isinstance(a, list) and isinstance(b, list):
        return sorted(map(str, a)) == sorted(map(str, b))
    return str(a).strip().lower() == str(b).strip().lower()


def _resolve_fact(key: str, facts: dict[str, dict]) -> str | None:
    k = _norm(key)
    for name, f in facts.items():
        if k == _norm(name) or k in {_norm(a) for a in f.get("aliases", [])}:
            return name
    return None


def _evidence_exists(evidence: str | None, workspace: Path) -> bool:
    if not evidence:
        return False
    e = evidence
    if e.startswith("node:"):
        return True  # facts passed by the manager point at the node spec itself
    if e.startswith("artifact:"):
        e = e[len("artifact:"):].split("@", 1)[0]
    e = e.split("#", 1)[0].strip()
    for prefix in ("/work/", "./"):
        if e.startswith(prefix):
            e = e[len(prefix):]
    return bool(e) and (workspace / e).exists()


def compute(run_dir: Path, gold_dir: Path, checkpoint_status: dict[str, str] | None = None,
            checkpoint_nodes: dict[str, list[str]] | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    workspace = run_dir / "workspace"
    events = list(read_events(run_dir))
    facts = (yaml.safe_load((Path(gold_dir) / "facts.yaml").read_text()) or {}).get("facts", {}) \
        if (Path(gold_dir) / "facts.yaml").exists() else {}

    msgs = [e for e in events if e["event"] == "message"]
    specs = [m for m in msgs if m["kind"] == "spec"]
    results = [m for m in msgs if m["kind"] == "result"]
    rejections = [m for m in msgs if m["kind"] == "rejection"]

    # --- spec retention: facts the manager passed to nodes -----------------
    passed: dict[str, list[bool]] = defaultdict(list)
    for m in specs:
        for c in m.get("claims", []):
            name = _resolve_fact(c["key"], facts)
            if name:
                passed[name].append(_same(c["value"], facts[name]["value"]))
    coverage = len(passed) / len(facts) if facts else None
    correct = [v for vs in passed.values() for v in vs]
    fact_accuracy = sum(correct) / len(correct) if correct else None

    # --- grounding: result claims backed by an existing file -------------
    rclaims = [c for m in results for c in m.get("claims", [])]
    grounded = [c for c in rclaims if _evidence_exists(c.get("evidence"), workspace)]
    grounding_ratio = len(grounded) / len(rclaims) if rclaims else None

    # --- contradictions: the same method fact asserted with different values ---
    # Only method facts count; node-specific keys (descriptor_family, models) legitimately differ by node.
    by_key: dict[str, list[Any]] = defaultdict(list)
    for m in specs + results:
        for c in m.get("claims", []):
            name = _resolve_fact(c["key"], facts)
            if name:
                by_key[name].append(c["value"])
    contradicted = sorted(k for k, vs in by_key.items() if any(not _same(vs[0], v) for v in vs[1:]))
    contradiction_rate = len(contradicted) / len(by_key) if by_key else None

    # --- tool use and verification ---------------------------------------
    calls = [e for e in events if e["event"] == "tool_call"]
    tool_error_rate = sum(1 for c in calls if c.get("is_error")) / len(calls) if calls else None
    rej_by_node = Counter(m["node_id"] for m in rejections)

    # --- undetected faults: verifier accepted, evaluator failed ----------
    final_status: dict[str, str] = {}
    for e in events:
        if e["event"] == "node_status":
            final_status[e["node_id"]] = e["status"]
    undetected = []
    if checkpoint_status and checkpoint_nodes:
        for cp, status in checkpoint_status.items():
            if status == "fail":
                for n in checkpoint_nodes.get(cp, []):
                    if final_status.get(n) == "done":
                        undetected.append({"checkpoint": cp, "node": n})

    # --- communication graph ----------------------------------------------
    edges = Counter((m["sender"].split("@")[0], m["recipient"].split("@")[0]) for m in msgs)
    curvature = _edge_curvature(edges)

    # --- cost -------------------------------------------------------------
    turns = [e for e in events if e["event"] == "model_turn"]
    tokens_by_role: dict[str, int] = defaultdict(int)
    cost_by_role: dict[str, float] = defaultdict(float)
    unknown_cost = False
    for t in turns:
        u = t.get("usage", {})
        tokens_by_role[t["role"]] += sum(int(u.get(k, 0)) for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"))
        if t.get("cost_usd") is None:
            unknown_cost = True
        else:
            cost_by_role[t["role"]] += t["cost_usd"]
    ts = [e["ts"] for e in events]
    edits = Counter(e["op"] for e in events if e["event"] == "dag_edit")
    jobs = [e for e in events if e["event"] == "job" and e.get("state") in ("done", "failed")]

    return {
        "spec_retention": {"fact_coverage": coverage, "fact_accuracy": fact_accuracy,
                           "facts_passed": {k: f"{sum(v)}/{len(v)} correct" for k, v in passed.items()},
                           "facts_never_passed": sorted(set(facts) - set(passed))},
        "grounding_ratio": grounding_ratio,
        "contradiction_rate": contradiction_rate,
        "contradicted_keys": contradicted,
        "tool_error_rate": tool_error_rate,
        "verifier_rejections": dict(rej_by_node),
        "undetected_faults": undetected,
        "messages": {"total": len(msgs), "spec": len(specs), "result": len(results), "rejection": len(rejections)},
        "communication_graph": {"edges": [{"from": a, "to": b, "count": n} for (a, b), n in edges.items()],
                                "ollivier_ricci": curvature},
        "dag_edits": dict(edits),
        "cost": {"model_turns": len(turns), "tokens_by_role": dict(tokens_by_role),
                 "usd_by_role": {k: round(v, 4) for k, v in cost_by_role.items()},
                 "usd_total": round(sum(cost_by_role.values()), 4),
                 "usd_incomplete": unknown_cost,
                 "wall_seconds": (max(ts) - min(ts)) if ts else 0.0,
                 "jobs": len(jobs), "job_seconds": round(sum(j.get("seconds", 0.0) for j in jobs), 1)},
    }


def _edge_curvature(edges: Counter) -> dict[str, float] | str:
    """Ollivier-Ricci curvature of the undirected, weighted agent-message graph (hypothesis H: negative edges predict failures)."""
    try:
        import networkx as nx
        from GraphRicciCurvature.OllivierRicci import OllivierRicci
    except ImportError:
        return "not computed: install networkx and GraphRicciCurvature"
    G = nx.Graph()
    for (a, b), n in edges.items():
        if a == b:
            continue
        w = G[a][b]["weight"] + n if G.has_edge(a, b) else n
        G.add_edge(a, b, weight=w)
    if G.number_of_edges() == 0:
        return {}
    orc = OllivierRicci(G, alpha=0.5, weight="weight", verbose="ERROR")
    orc.compute_ricci_curvature()
    return {f"{u}--{v}": round(d["ricciCurvature"], 3) for u, v, d in orc.G.edges(data=True)}
