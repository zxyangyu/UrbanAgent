"""Tool bindings for UrbanAgent (optional ReAct-style harness integration)."""
from __future__ import annotations

from urbanagent.dispatch import DispatchPolicy, assignment_to_action
from urbanagent.sandbox import MockSandboxClient, SandboxClient
from urbanagent.types import Coordinate, UrbanAction, to_json


def create_urban_tools(
    client: SandboxClient | None = None,
    *,
    include_low_level_tools: bool = False,
) -> list:
    """Create Python callables that expose sandbox operations as Agent tools.

    By default the LLM only sees high-level tools. Resource selection, route
    timing, and action execution stay inside deterministic code/API layers.
    Low-level tools are available only for debugging or future MCP/API adapters.
    """

    sandbox = client or MockSandboxClient()
    policy = DispatchPolicy()
    last_plan = None

    async def get_city_state() -> str:
        """Return the latest city sandbox state, including incidents, available resources, traffic signals, and roads. Call this before making emergency-dispatch decisions."""

        return to_json(await sandbox.get_state())

    async def create_dispatch_plan() -> str:
        """Create a deterministic emergency dispatch plan from the current city state. The algorithm scores available fire trucks, police cars, and drones by route time, congestion, capability fit, load, and station reserve constraints."""

        nonlocal last_plan
        state = await sandbox.get_state()
        last_plan = policy.create_plan(state)
        return to_json(last_plan)

    async def apply_dispatch_plan() -> str:
        """Apply the latest deterministic dispatch plan to the sandbox. If no plan exists yet, create one from the current city state first."""

        nonlocal last_plan
        if last_plan is None:
            state = await sandbox.get_state()
            last_plan = policy.create_plan(state)
        results = []
        touched_incidents: set[str] = set()
        for assignment in last_plan.assignments:
            results.append(await sandbox.send_action(assignment_to_action(assignment)))
            touched_incidents.add(assignment.incident_id)
        for incident_id in sorted(touched_incidents):
            results.append(
                await sandbox.send_action(
                    UrbanAction(
                        kind="mark_incident",
                        target_id=incident_id,
                        parameters={"status": "responding"},
                        reason="Initial dispatch plan applied.",
                    )
                )
            )
        return to_json(results)

    async def dispatch_vehicle(
        resource_id: str,
        destination_x: float,
        destination_y: float,
        reason: str = "",
    ) -> str:
        """Low-level API wrapper: dispatch a ground vehicle to a destination coordinate. Prefer create_dispatch_plan/apply_dispatch_plan so resource choice and timing stay in deterministic code."""

        action = UrbanAction(
            kind="dispatch_vehicle",
            target_id=resource_id,
            destination=Coordinate(x=float(destination_x), y=float(destination_y)),
            reason=reason,
        )
        return to_json(await sandbox.send_action(action))

    async def dispatch_drone(
        resource_id: str,
        destination_x: float,
        destination_y: float,
        altitude: float = 30.0,
        reason: str = "",
    ) -> str:
        """Low-level API wrapper: dispatch a drone to observe an emergency site. Prefer create_dispatch_plan/apply_dispatch_plan so resource choice and timing stay in deterministic code."""

        action = UrbanAction(
            kind="dispatch_drone",
            target_id=resource_id,
            destination=Coordinate(
                x=float(destination_x),
                y=float(destination_y),
                z=float(altitude),
            ),
            reason=reason,
        )
        return to_json(await sandbox.send_action(action))

    async def control_traffic_light(
        signal_id: str,
        mode: str = "emergency_preemption",
        reason: str = "",
    ) -> str:
        """Low-level API wrapper: change a traffic signal mode. Prefer a dispatch plan or a future traffic-control MCP/API policy tool."""

        action = UrbanAction(
            kind="control_traffic_light",
            target_id=signal_id,
            parameters={"mode": mode},
            reason=reason,
        )
        return to_json(await sandbox.send_action(action))

    async def mark_incident(
        incident_id: str,
        status: str = "responding",
        reason: str = "",
    ) -> str:
        """Low-level API wrapper: update an incident lifecycle status after dispatch actions have been issued."""

        action = UrbanAction(
            kind="mark_incident",
            target_id=incident_id,
            parameters={"status": status},
            reason=reason,
        )
        return to_json(await sandbox.send_action(action))

    tools = [
        get_city_state,
        create_dispatch_plan,
        apply_dispatch_plan,
    ]
    if include_low_level_tools:
        tools.extend(
            [
                dispatch_vehicle,
                dispatch_drone,
                control_traffic_light,
                mark_incident,
            ]
        )
    return tools
