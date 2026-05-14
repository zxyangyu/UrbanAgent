"""UrbanAgent: emergency dispatch agents for CarlaBridge/CARLA city sandbox."""

from __future__ import annotations

from urbanagent.agent import UrbanAgent
from urbanagent.carla_bridge import CarlaBridgeSandboxClient
from urbanagent.dispatch import DispatchPolicy, assignment_to_action
from urbanagent.errors import SandboxWireError, UrbanAgentPipelineError
from urbanagent.multiagent import UrbanMultiAgentResult, UrbanMultiAgentSystem
from urbanagent.routing import (
    ExternalMapRoutePlanner,
    LocalGraphRoutePlanner,
    RoutePlanner,
    StraightLineRoutePlanner,
)
from urbanagent.sandbox import (
    MockSandboxClient,
    SandboxClient,
    build_single_fire_state,
)
from urbanagent.schemas import (
    DispatchSolution,
    ExecutionObservation,
    TaskGraph,
    TaskNode,
    UrbanAgentResult,
    UrbanConstraint,
    UrbanTask,
)
from urbanagent.tooling import (
    ExternalToolFacade,
    build_external_tool_facade,
)
from urbanagent.tools import create_urban_tools
from urbanagent.types import (
    ActionResult,
    CandidateScore,
    CityState,
    Coordinate,
    DispatchAssignment,
    DispatchPlan,
    DroneBase,
    FireStation,
    Incident,
    PoliceStation,
    RouteEstimate,
    TrafficSignal,
    UrbanAction,
    UrbanResource,
)

__all__ = [
    "ExternalToolFacade",
    "SandboxWireError",
    "ActionResult",
    "CandidateScore",
    "CarlaBridgeSandboxClient",
    "CityState",
    "Coordinate",
    "DispatchSolution",
    "DispatchAssignment",
    "DispatchPlan",
    "DispatchPolicy",
    "DroneBase",
    "ExternalMapRoutePlanner",
    "FireStation",
    "Incident",
    "ExecutionObservation",
    "LocalGraphRoutePlanner",
    "MockSandboxClient",
    "PoliceStation",
    "RouteEstimate",
    "RoutePlanner",
    "SandboxClient",
    "StraightLineRoutePlanner",
    "TrafficSignal",
    "TaskGraph",
    "TaskNode",
    "UrbanAction",
    "UrbanAgent",
    "UrbanAgentPipelineError",
    "UrbanAgentResult",
    "UrbanConstraint",
    "UrbanMultiAgentResult",
    "UrbanMultiAgentSystem",
    "UrbanResource",
    "UrbanTask",
    "assignment_to_action",
    "build_external_tool_facade",
    "build_single_fire_state",
    "create_urban_tools",
]
