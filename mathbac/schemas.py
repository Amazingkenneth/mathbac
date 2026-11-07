"""Typed records shared by the manager, workers, ledger and evaluator.

Messages between agents carry free text plus typed ``Claim`` objects. Claims are
what make communication measurable: each one names a key, a value and the
evidence (an artifact path or hash) that supports it.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Claim(BaseModel):
    """A single fact an agent asserts, e.g. ``contact_cutoff_A = 7.0``."""

    key: str
    value: Any
    evidence: str | None = Field(
        default=None,
        description="Workspace path, 'path#anchor', or 'artifact:<path>@sha256:<hash>' supporting the claim.",
    )


class OutputSpec(BaseModel):
    path: str = Field(description="Glob relative to the workspace, e.g. 'features/GE/*.csv'.")
    schema_: str | None = Field(default=None, alias="schema", description="Columns or keys the file must contain.")

    model_config = {"populate_by_name": True}


class Budget(BaseModel):
    max_turns: int = 60
    max_tokens: int = 600_000
    wall_minutes: float = 120.0
    gpu_hours: float = 0.0


class NodeSpec(BaseModel):
    """One node of the task DAG, as written by the manager (or the reference DAG)."""

    id: str
    role: str
    goal: str
    depends_on: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[OutputSpec] = Field(default_factory=list)
    acceptance: str | None = Field(default=None, description="Acceptance check name under the task's acceptance/ folder.")
    tools: list[str] = Field(default_factory=list, description="Registered scientific tools this node may call.")
    facts: dict[str, Any] = Field(
        default_factory=dict,
        description="Method facts the manager passes down (e.g. contact_cutoff_A). Used to measure spec retention per hop.",
    )
    checkpoint: str | None = None
    budget: Budget = Field(default_factory=Budget)


class NodeResult(BaseModel):
    node_id: str
    status: NodeStatus
    summary: str = ""
    outputs: list[str] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    error: str | None = None
    attempts: int = 1


class AgentMessage(BaseModel):
    """A message between agents. Logged to the ledger as event type 'message'."""

    sender: str
    recipient: str
    kind: str = Field(description="spec | result | question | answer | rejection | replan")
    text: str = ""
    claims: list[Claim] = Field(default_factory=list)
    node_id: str | None = None


class JobSpec(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    input_paths: list[str] = Field(default_factory=list, description="Workspace files whose contents key the result cache")
    node_id: str | None = None
    cpus: int = 1
    gpus: int = 0
    mem_gb: float = 4.0
    time_limit_min: float = 60.0


class ToolArg(BaseModel):
    type: str = "string"  # string | integer | number | flags (space-separated --switches)
    description: str = ""
    required: bool = True
    default: Any = None
    input: bool = Field(default=False, description="The value is a workspace file read by the tool; its contents key the cache")


class ToolCard(BaseModel):
    """Instruction file an agent reads before calling a scientific tool.

    Cards live in ``toolcards/*.yaml``. The fields mirror the PPI2PNI spreadsheet
    (inputs, outputs, PNI translatability, notes) plus what an agent needs to call
    the tool correctly.
    """

    name: str
    tier: int = Field(description="1 = needed to reproduce Cha 2022, 2 = PNI extension, 3 = structure/baseline")
    title: str
    purpose: str
    source: str | None = None
    doi: str | None = None
    installed: bool = False
    invoke: str | None = Field(default=None, description="Command template run in the sandbox; {arg} placeholders are shell-quoted.")
    args: dict[str, ToolArg] = Field(default_factory=dict)
    inputs: str = ""
    outputs: str = ""
    pni_translatability: str = ""
    cost: str = ""
    resources: dict[str, Any] = Field(default_factory=dict)
    preconditions: list[str] = Field(default_factory=list)
    pitfalls: list[str] = Field(default_factory=list)
    smoke_test: str | None = None
