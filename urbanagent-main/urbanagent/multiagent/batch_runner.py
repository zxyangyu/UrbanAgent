"""G6–G7: ordered submit to sandbox; poll aggregate state after the full batch."""
from __future__ import annotations

import asyncio

from urbanagent.multiagent.schemas import BatchOutcome
from urbanagent.sandbox import SandboxClient
from urbanagent.types import ActionResult, CityState, UrbanAction


def _find_resource(state: CityState, rid: str):
    return next((r for r in state.resources if r.id == rid), None)


def _find_signal(state: CityState, sid: str):
    return next((s for s in state.traffic_signals if s.id == sid), None)


def batch_criteria_met(
    state: CityState,
    actions: list[UrbanAction],
    results: list[ActionResult],
) -> bool:
    """All steps accepted/applied and world reflects the committed batch."""
    if len(results) != len(actions):
        return False
    if any(r.status not in {"accepted", "applied"} for r in results):
        return False
    for index, action in enumerate(actions):
        if action.kind == "control_traffic_light":
            sig = _find_signal(state, action.target_id)
            if sig is None:
                return False
            want = str(action.parameters.get("mode", "emergency_preemption"))
            if sig.mode != want and not (
                want == "emergency_preemption" and sig.mode in {"green", "emergency_preemption"}
            ):
                return False
        elif action.kind in ("dispatch_vehicle", "dispatch_drone"):
            res = _find_resource(state, action.target_id)
            if res is None or res.status != "dispatched":
                return False
        elif action.kind == "mark_incident":
            if results[index].status == "accepted":
                # CarlaBridge may not expose incident lifecycle in state.snapshot.
                # In that case MARK_EVENT remains an accepted system command.
                continue
            inc = next((i for i in state.incidents if i.id == action.target_id), None)
            if inc is None:
                return False
            want = str(action.parameters.get("status", "responding"))
            if inc.status != want:
                return False
    return True


async def execute_batch_ordered(
    sandbox: SandboxClient,
    batch_id: str,
    actions: list[UrbanAction],
    *,
    max_poll_rounds: int = 60,
    poll_interval_s: float = 0.05,
) -> BatchOutcome:
    per_step: list[ActionResult] = []
    notes: list[str] = []
    for action in actions:
        result = await sandbox.send_action(action)
        per_step.append(result)
        if result.status == "rejected":
            notes.append(f"rejected at {action.kind} target={action.target_id}: {result.message}")
            break

    final_state = await sandbox.get_state()
    if len(per_step) != len(actions) or any(
        r.status not in {"accepted", "applied"} for r in per_step
    ):
        return BatchOutcome(
            batch_id=batch_id,
            per_step_results=per_step,
            polling_iterations=0,
            criteria_satisfied=False,
            final_state=final_state,
            notes=notes,
        )

    if batch_criteria_met(final_state, actions, per_step):
        return BatchOutcome(
            batch_id=batch_id,
            per_step_results=per_step,
            polling_iterations=0,
            criteria_satisfied=True,
            final_state=final_state,
            notes=notes,
        )

    iterations = 0
    for _ in range(max_poll_rounds):
        await asyncio.sleep(poll_interval_s)
        iterations += 1
        final_state = await sandbox.get_state()
        if batch_criteria_met(final_state, actions, per_step):
            return BatchOutcome(
                batch_id=batch_id,
                per_step_results=per_step,
                polling_iterations=iterations,
                criteria_satisfied=True,
                final_state=final_state,
                notes=notes,
            )

    final_state = await sandbox.get_state()
    ok = batch_criteria_met(final_state, actions, per_step)
    if not ok:
        notes.append("polling exhausted without meeting aggregate criteria")
    return BatchOutcome(
        batch_id=batch_id,
        per_step_results=per_step,
        polling_iterations=iterations,
        criteria_satisfied=ok,
        final_state=final_state,
        notes=notes,
    )
