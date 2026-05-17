"""UrbanAgent multi-agent MVP pipeline (meta + subagents + batch execution)."""
from __future__ import annotations

import asyncio
import copy
import uuid
from pathlib import Path
from typing import cast

from urbanagent.dispatch import DispatchPolicy
from urbanagent.llm.llm import LLM
from urbanagent.multiagent.batch_runner import execute_batch_ordered
from urbanagent.multiagent.gating import run_gate
from urbanagent.multiagent.orchestrator import (
    build_committed_batch,
    deterministic_final_report,
    llm_cognition,
    llm_decompose,
    llm_final_report,
    rule_cognition,
    rule_decompose,
)
from urbanagent.multiagent.registry import SubAgentRegistry, build_default_mvp_registry
from urbanagent.multiagent.schemas import (
    BatchOutcome,
    PatrolFireResponseResult,
    SubAgentRole,
    UrbanMultiAgentResult,
)
from urbanagent.multiagent.toolkit import SubAgentToolkit
from urbanagent.multiagent.world_cache import WorldStateCache
from urbanagent.sandbox import MockSandboxClient, SandboxClient
from urbanagent.types import CityState, Coordinate, Incident, UrbanAction, UrbanResource


class UrbanMultiAgentSystem:
    """Meta + four sub-agents (MVP one each); ordered batch to sandbox + post-batch poll."""

    ROLE_ORDER: tuple[SubAgentRole, ...] = (
        "traffic_signal",
        "police_car",
        "unmanned_vehicle",
        "drone",
    )

    def __init__(
        self,
        sandbox: SandboxClient | None = None,
        *,
        dispatch_policy: DispatchPolicy | None = None,
        registry: SubAgentRegistry | None = None,
        toolkit: SubAgentToolkit | None = None,
        dotenv_path: str | Path | None = None,
        use_llm: bool = True,
        use_llm_batch_rerank: bool = True,
    ) -> None:
        self.sandbox = sandbox or MockSandboxClient()
        self.dispatch_policy = dispatch_policy or DispatchPolicy()
        self.toolkit = toolkit or SubAgentToolkit()
        self.registry = registry or build_default_mvp_registry(self.toolkit)
        self.dotenv_path = str(dotenv_path) if dotenv_path is not None else None
        self.use_llm = use_llm
        self.use_llm_batch_rerank = use_llm_batch_rerank
        self.llm: LLM | None = None
        self.world_cache = WorldStateCache()

    def _ensure_llm(self) -> LLM | None:
        if not self.use_llm:
            return None
        if self.llm is None:
            self.llm = LLM.from_env(dotenv_path=self.dotenv_path)
        return self.llm

    async def run(self, query: str) -> UrbanMultiAgentResult:
        initial = copy.deepcopy(await self.sandbox.get_state())
        self.world_cache.ingest(initial, note="run_start")
        llm = self._ensure_llm()

        gate = await run_gate(query, initial, llm=llm, use_llm=self.use_llm)
        if not gate.should_intervene:
            return UrbanMultiAgentResult(
                query=query,
                gate=gate,
                initial_state=initial,
                skipped_reason=gate.reason or "gate declined",
                llm_used=llm is not None,
            )

        task = await self._cognition(query, initial, llm)
        if task.missing_information:
            return UrbanMultiAgentResult(
                query=query,
                gate=gate,
                urban_task=task,
                initial_state=initial,
                skipped_reason="missing_information:" + ",".join(task.missing_information),
                llm_used=llm is not None,
            )

        subgoals = await self._decompose(task, initial, llm)
        if not subgoals:
            return UrbanMultiAgentResult(
                query=query,
                gate=gate,
                urban_task=task,
                initial_state=initial,
                skipped_reason="decompose produced no subgoals",
                llm_used=llm is not None,
            )

        sub_plans = []
        for role in self.ROLE_ORDER:
            if role not in subgoals:
                continue
            agent = self.registry.primary(cast(SubAgentRole, role))
            sub_plans.append(
                await agent.run(subgoals[role], initial, llm=llm, use_llm=self.use_llm)
            )

        committed = await build_committed_batch(
            sub_plans,
            state=initial,
            task=task,
            dispatch_policy=self.dispatch_policy,
            llm=llm,
            use_llm=self.use_llm,
            use_llm_batch_rerank=self.use_llm_batch_rerank,
        )
        outcome = await execute_batch_ordered(
            self.sandbox,
            committed.batch_id,
            committed.actions,
        )

        if self.use_llm and llm is not None:
            try:
                report = await llm_final_report(
                    query, task, sub_plans, committed, outcome, llm
                )
            except Exception:
                report = deterministic_final_report(
                    query, task, sub_plans, committed, outcome
                )
        else:
            report = deterministic_final_report(
                query, task, sub_plans, committed, outcome
            )

        return UrbanMultiAgentResult(
            query=query,
            gate=gate,
            urban_task=task,
            subgoals=subgoals,
            sub_plans=sub_plans,
            committed=committed,
            batch_outcome=outcome,
            final_report=report,
            initial_state=initial,
            llm_used=llm is not None,
        )

    async def run_patrol_fire_response(
        self,
        *,
        patrol_waypoints: list[Coordinate] | None = None,
        patrol_altitude: float = 60.0,
        patrol_radius: float = 80.0,
        max_patrol_drones: int = 3,
        detection_poll_interval_s: float = 0.5,
        max_detection_rounds: int = 60,
        response_query_template: str = "{incident_id} 高严重度火情，请进行多智能体协同调度。",
        return_after_response: bool = True,
    ) -> PatrolFireResponseResult:
        """Run the full loop: patrol when idle, detect fire from snapshots, dispatch, RTL.

        "Autonomous discovery" is represented by a new/open fire incident appearing
        in the sandbox state while UAV patrol is active. CarlaBridge owns the actual
        perception/simulation; UrbanAgent reacts to the authoritative snapshot.
        """

        notes: list[str] = []
        initial = copy.deepcopy(await self.sandbox.get_state())
        self.world_cache.ingest(initial, note="patrol_start")

        detected = _first_open_fire(initial)
        patrol_outcome: BatchOutcome | None = None
        if detected is None:
            patrol_actions = self._build_patrol_actions(
                initial,
                patrol_waypoints=patrol_waypoints,
                patrol_altitude=patrol_altitude,
                patrol_radius=patrol_radius,
                max_patrol_drones=max_patrol_drones,
            )
            patrol_outcome = await execute_batch_ordered(
                self.sandbox,
                f"patrol-{uuid.uuid4().hex[:8]}",
                patrol_actions,
            )
            if patrol_actions:
                notes.append(f"patrol_started:{len(patrol_actions)}")
            else:
                notes.append("patrol_skipped:no_available_drone")
            detected = await self._wait_for_fire_detection(
                poll_interval_s=detection_poll_interval_s,
                max_rounds=max_detection_rounds,
            )
        else:
            notes.append(f"fire_already_present:{detected.id}")

        if detected is None:
            return PatrolFireResponseResult(
                patrol_outcome=patrol_outcome,
                detection_notes=notes + ["no_fire_detected"],
                final_report="巡逻完成：未发现火情，UrbanAgent 未进入应急调度。",
            )

        notes.append(f"fire_detected:{detected.id}")
        await self._emit_operator_notice(
            f"UAV patrol detected fire incident {detected.id}; UrbanAgent dispatch started."
        )

        response_query = response_query_template.format(incident_id=detected.id)
        response = await self.run(response_query)

        return_outcome: BatchOutcome | None = None
        if return_after_response:
            final_state = (
                response.batch_outcome.final_state
                if response.batch_outcome is not None
                and response.batch_outcome.final_state is not None
                else await self.sandbox.get_state()
            )
            return_actions = self._build_return_actions(response, final_state)
            if return_actions:
                return_outcome = await execute_batch_ordered(
                    self.sandbox,
                    f"return-{uuid.uuid4().hex[:8]}",
                    return_actions,
                )
                notes.append(f"return_started:{len(return_actions)}")
            else:
                notes.append("return_skipped:no_mobile_targets")

        return PatrolFireResponseResult(
            patrol_outcome=patrol_outcome,
            detected_incident_id=detected.id,
            detection_notes=notes,
            response=response,
            return_outcome=return_outcome,
            final_report=self._patrol_response_report(
                detected.id,
                response,
                return_outcome,
            ),
        )

    async def _cognition(self, query: str, state, llm):
        if self.use_llm and llm is not None:
            try:
                return await llm_cognition(query, state, llm)
            except Exception:
                return rule_cognition(query, state)
        return rule_cognition(query, state)

    async def _decompose(self, task, state, llm):
        if self.use_llm and llm is not None:
            try:
                return await llm_decompose(task, state, llm, self.toolkit)
            except Exception:
                return rule_decompose(task, state, self.toolkit)
        return rule_decompose(task, state, self.toolkit)

    def _build_patrol_actions(
        self,
        state: CityState,
        *,
        patrol_waypoints: list[Coordinate] | None,
        patrol_altitude: float,
        patrol_radius: float,
        max_patrol_drones: int,
    ) -> list[UrbanAction]:
        actions: list[UrbanAction] = []
        drones = [
            r
            for r in state.resources
            if r.kind == "drone"
            and r.status == "available"
            and "aerial_recon" in r.capabilities
        ]
        for drone in drones[:max(0, max_patrol_drones)]:
            path = patrol_waypoints or _default_patrol_path(
                drone,
                altitude=patrol_altitude,
                radius=patrol_radius,
            )
            actions.append(
                UrbanAction(
                    kind="patrol_drone",
                    target_id=drone.id,
                    parameters={
                        "path": path,
                        "loop": True,
                        "cruise_speed": 8.0,
                        "role": "fire_watch_patrol",
                    },
                    reason="idle fire-watch patrol",
                )
            )
        return actions

    async def _wait_for_fire_detection(
        self,
        *,
        poll_interval_s: float,
        max_rounds: int,
    ) -> Incident | None:
        for round_idx in range(max(0, max_rounds) + 1):
            state = await self.sandbox.get_state()
            self.world_cache.ingest(state, note="patrol_detection_poll")
            detected = _first_open_fire(state)
            if detected is not None:
                return detected
            if round_idx < max_rounds:
                await asyncio.sleep(max(0.0, poll_interval_s))
        return None

    async def _emit_operator_notice(self, message: str) -> None:
        sender = getattr(self.sandbox, "send_event_log", None)
        if sender is None:
            return
        try:
            await sender(message, severity="info")
        except Exception:
            return

    def _build_return_actions(
        self,
        response: UrbanMultiAgentResult,
        state: CityState,
    ) -> list[UrbanAction]:
        if response.committed is None:
            return []
        target_ids: list[str] = []
        for action in response.committed.actions:
            if action.kind not in {"dispatch_drone", "dispatch_vehicle"}:
                continue
            if action.target_id not in target_ids:
                target_ids.append(action.target_id)

        resources = {r.id: r for r in state.resources}
        return_actions: list[UrbanAction] = []
        for target_id in target_ids:
            resource = resources.get(target_id)
            if resource is None:
                continue
            if resource.kind == "drone":
                return_actions.append(
                    UrbanAction(
                        kind="return_drone",
                        target_id=target_id,
                        parameters={"role": "post_fire_return"},
                        reason="return to launch after fire response",
                    )
                )
            elif resource.kind in {"unmanned_vehicle", "ground_vehicle"}:
                return_actions.append(
                    UrbanAction(
                        kind="return_vehicle",
                        target_id=target_id,
                        parameters={"role": "post_fire_return"},
                        reason="return to base after fire response",
                    )
                )
        return return_actions

    def _patrol_response_report(
        self,
        incident_id: str,
        response: UrbanMultiAgentResult,
        return_outcome: BatchOutcome | None,
    ) -> str:
        response_ok = (
            response.batch_outcome is not None
            and response.batch_outcome.criteria_satisfied
        )
        return_ok = return_outcome is not None and return_outcome.criteria_satisfied
        return (
            f"巡逻发现火情 {incident_id}；"
            f"应急调度{'成功' if response_ok else '未完全成功'}；"
            f"返航{'已下发' if return_ok else '未完成或无可返航资源'}。"
        )


def _first_open_fire(state: CityState) -> Incident | None:
    return next(
        (
            incident
            for incident in state.incidents
            if incident.kind == "fire" and incident.status in {"open", "responding"}
        ),
        None,
    )


def _default_patrol_path(
    drone: UrbanResource,
    *,
    altitude: float,
    radius: float,
) -> list[Coordinate]:
    x = drone.position.x
    y = drone.position.y
    z = max(float(altitude), drone.position.z)
    half = float(radius) / 2.0
    return [
        Coordinate(x - half, y - half, z),
        Coordinate(x + half, y - half, z),
        Coordinate(x + half, y + half, z),
        Coordinate(x - half, y + half, z),
    ]
