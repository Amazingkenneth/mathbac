"""Append-only experiment ledger.

Every node, message, model turn, tool call, job and artifact hash is one JSON line
in ``<run_dir>/ledger.jsonl``. The manager plans from it, the evaluator scores from
it, and the communication metrics are computed from it.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json", by_alias=True)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


class Ledger:
    FILENAME = "ledger.jsonl"

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / self.FILENAME
        self._lock = threading.Lock()
        self._seq = sum(1 for _ in self.path.open()) if self.path.exists() else 0

    def append(self, event: str, **payload: Any) -> dict[str, Any]:
        with self._lock:
            record = {"seq": self._seq, "ts": time.time(), "event": event, **_jsonable(payload)}
            with self.path.open("a") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
            self._seq += 1
            return record

    def record_artifact(self, workspace: Path, rel_path: str, node_id: str | None = None) -> str | None:
        p = (Path(workspace) / rel_path).resolve()
        if not p.is_file():
            return None
        digest = sha256_file(p)
        self.append("artifact", path=rel_path, sha256=digest, size=p.stat().st_size, node_id=node_id)
        return digest

    def events(self, event: str | None = None) -> Iterator[dict[str, Any]]:
        yield from read_events(self.path, event)


def read_events(path: Path, event: str | None = None) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if path.is_dir():
        path = path / Ledger.FILENAME
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if event is None or rec.get("event") == event:
                yield rec
