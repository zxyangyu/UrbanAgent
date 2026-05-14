"""Structured objects used by the UrbanAgent method pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from urbanagent.types import ActionResult, CityState, DispatchPlan


ConstraintKind = Literal["hard", "soft"]
TaskNodeStatus = Literal["pending", "running", "succeeded", "failed", "skipped"]


@dataclass
class UrbanConstraint:
    """A computable user or domain constraint."""

    name: str
    kind: ConstraintKind
    expression: str
    satisfied: bool | None = None


@dataclass
class UrbanTask:
    """Cognition output U = <I, E, C> from the paper."""

    intent: str
    entities: dict[str, Any] = field(default_factory=dict)
    constraints: list[UrbanConstraint] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    rationale: str = ""
    source: str = "rule"


@dataclass
class TaskNode:
    """One atomic task node in the planning DAG."""

    id: str
    description: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    status: TaskNodeStatus = "pending"


@dataclass
class TaskGraph:
    """Planning output G = (V, E), represented by nodes and dependencies."""

    nodes: list[TaskNode] = field(default_factory=list)
    source: str = "rule"
    rationale: str = ""


@dataclass
class ExecutionObservation:
    """One observation returned by a simulated city tool."""

    node_id: str
    tool: str
    status: TaskNodeStatus
    data: Any = None
    error: str | None = None
    retries: int = 0
    repaired_by_llm: bool = False


@dataclass
class DispatchSolution:
    """One feasible urban decision solution R."""

    title: str
    summary: str
    plan: DispatchPlan
    action_results: list[ActionResult]
    hard_constraints_satisfied: bool
    score: float
    notes: list[str] = field(default_factory=list)
    candidate_summary: list[str] = field(default_factory=list)
    sandbox_commands: list[dict[str, Any]] = field(default_factory=list)
    report: str = ""


@dataclass
class UrbanAgentResult:
    """End-to-end output of the UrbanAgent method pipeline."""

    query: str
    task: UrbanTask
    graph: TaskGraph
    observations: list[ExecutionObservation]
    solutions: list[DispatchSolution]
    final_report: str
    initial_state: CityState | None = None
    llm_used: bool = False
