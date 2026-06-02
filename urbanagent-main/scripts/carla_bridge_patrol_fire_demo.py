"""Run UrbanMultiAgentSystem.run_patrol_fire_response against CarlaBridge.

Closed loop: 无火情 -> UAV 沿矩形区域巡逻
(-20,-20) -> (20,-20) -> (20,20) -> (-20,20) 循环，默认高度 10m ->
state_snapshot 出现 fire incident -> 等待巡逻到达着火点上方后 hold ->
event_log 提示 -> UrbanAgent 火情调度 -> 灭火完成后 UAV/UGV 返航.

By default this runs forever: each cycle patrols until a fire appears, dispatches
the response, waits for every unit to return home, then re-enters patrol. Stop it
with Ctrl+C, or cap the number of cycles with --cycles. If a patrol window ends
with no fire, it simply re-patrols and keeps waiting for ignition.

NOTE: Do not supply --fallback-incident-id unless you want to bypass the patrol
phase. The patrol loop only triggers when the initial CarlaBridge snapshot has
no open fire incident.

Example:
    python scripts/carla_bridge_patrol_fire_demo.py --url http://127.0.0.1:5000 --no-llm
    python scripts/carla_bridge_patrol_fire_demo.py --no-llm --cycles 5 --cycle-pause 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from urbanagent import CarlaBridgeSandboxClient, UrbanMultiAgentSystem
from urbanagent.multiagent.pipeline import DEFAULT_FIRE_WATCH_XY
from urbanagent.types import Coordinate, Incident

# 默认矩形巡逻角点（CARLA 地面 xy，z 由 --patrol-altitude 填充）
DEFAULT_PATROL_XY: tuple[tuple[float, float], ...] = (
    (-20.0, -20.0),
    (20.0, -20.0),
    (20.0, 20.0),
    (-20.0, 20.0),
)


def _default_patrol_waypoints(altitude: float) -> list[Coordinate]:
    return [Coordinate(x, y, altitude) for x, y in DEFAULT_PATROL_XY]


def _parse_waypoint(text: str) -> Coordinate:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"waypoint must be x,y,z (got {text!r})"
        )
    try:
        return Coordinate(float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _print_batch_outcome(label: str, outcome) -> None:
    if outcome is None:
        print(f"[{label}] outcome=None")
        return
    print(
        f"[{label}] batch_id={outcome.batch_id} "
        f"criteria_satisfied={outcome.criteria_satisfied} "
        f"steps={len(outcome.per_step_results)}"
    )
    for step in outcome.per_step_results:
        print(
            f"  - {step.action.kind} target={step.action.target_id} "
            f"status={step.status} message={step.message}"
        )
    for note in outcome.notes:
        print(f"  note: {note}")


def _fmt_coord(coord) -> str:
    if coord is None:
        return "None"
    return f"({coord.x:.2f}, {coord.y:.2f}, {coord.z:.2f})"


def _print_thinking(response) -> None:
    """打印 UrbanAgent 火情调度的模型思考过程（认知 G2 -> 分解 G3 -> 子智能体）。"""
    print("-" * 72)
    print("[THINKING] 模型思考过程（火情调度）")
    print(
        f"  llm_used={response.llm_used} "
        f"skipped_reason={response.skipped_reason!r}"
    )

    gate = response.gate
    if gate is not None:
        print(
            f"  [G1 门控] should_intervene={gate.should_intervene} "
            f"priority={gate.priority} trigger={gate.trigger_kind!r}"
        )
        if gate.reason:
            print(f"            reason: {gate.reason}")

    task = response.urban_task
    if task is not None:
        print(f"  [G2 认知] intent={task.intent} source={task.source}")
        print(f"            entities: {task.entities}")
        if task.constraints:
            print("            constraints:")
            for c in task.constraints:
                print(f"              - {c.name}({c.kind}): {c.expression}")
        if task.missing_information:
            print(f"            missing_information: {task.missing_information}")
        if task.rationale:
            print(f"            rationale: {task.rationale}")

    if response.subgoals:
        print("  [G3 分解] 子目标分配:")
        for role, goal in response.subgoals.items():
            print(
                f"            - {role}: {goal.narrative or '(无)'} "
                f"resources={goal.allowed_resource_ids} "
                f"signals={goal.allowed_signal_ids} hints={goal.hard_hints}"
            )

    if response.sub_plans:
        print("  [S* 子智能体推理]:")
        for sp in response.sub_plans:
            print(
                f"            - {sp.role}: status={sp.status} "
                f"drafts={len(sp.action_drafts)}"
            )
            if sp.rationale:
                print(f"              rationale: {sp.rationale}")
            for w in sp.warnings:
                print(f"              warning: {w}")
            if sp.llm_stages:
                print(f"              llm_stages: {list(sp.llm_stages.keys())}")


def _print_dispatch_plan(response) -> None:
    """打印最终输出的调度计划（CommittedBatch 的有序动作）。"""
    print("-" * 72)
    print("[PLAN] 最终调度计划")
    committed = response.committed
    if committed is None:
        print("  (无 committed 批次：未生成可下发的调度计划)")
        return
    print(
        f"  batch_id={committed.batch_id} "
        f"actions={len(committed.actions)}"
    )
    if committed.rationale:
        print(f"  rationale: {committed.rationale}")
    for idx, action in enumerate(committed.actions, start=1):
        print(
            f"  {idx}. {action.kind} target={action.target_id} "
            f"destination={_fmt_coord(action.destination)}"
        )
        if action.parameters:
            print(f"     parameters: {action.parameters}")
        if action.reason:
            print(f"     reason: {action.reason}")


def _build_frontend_payload(result) -> dict:
    """汇总思考过程 + 调度计划为结构化 dict，方便后续前端展示。"""
    response = result.response

    def _coord(coord):
        if coord is None:
            return None
        return {"x": coord.x, "y": coord.y, "z": coord.z}

    payload: dict = {
        "detected_incident_id": result.detected_incident_id,
        "final_report": result.final_report,
        "thinking": None,
        "dispatch_plan": None,
        "response_final_report": None,
    }
    if response is None:
        return payload

    payload["response_final_report"] = response.final_report

    task = response.urban_task
    payload["thinking"] = {
        "llm_used": response.llm_used,
        "skipped_reason": response.skipped_reason,
        "gate": (
            {
                "should_intervene": response.gate.should_intervene,
                "priority": response.gate.priority,
                "reason": response.gate.reason,
                "trigger_kind": response.gate.trigger_kind,
            }
            if response.gate is not None
            else None
        ),
        "cognition": (
            {
                "intent": task.intent,
                "source": task.source,
                "entities": task.entities,
                "constraints": [
                    {"name": c.name, "kind": c.kind, "expression": c.expression}
                    for c in task.constraints
                ],
                "missing_information": task.missing_information,
                "rationale": task.rationale,
            }
            if task is not None
            else None
        ),
        "subgoals": {
            role: {
                "narrative": goal.narrative,
                "allowed_resource_ids": goal.allowed_resource_ids,
                "allowed_signal_ids": goal.allowed_signal_ids,
                "hard_hints": goal.hard_hints,
            }
            for role, goal in response.subgoals.items()
        },
        "sub_plans": [
            {
                "role": sp.role,
                "status": sp.status,
                "rationale": sp.rationale,
                "warnings": sp.warnings,
                "llm_stages": list(sp.llm_stages.keys()),
            }
            for sp in response.sub_plans
        ],
    }

    committed = response.committed
    if committed is not None:
        payload["dispatch_plan"] = {
            "batch_id": committed.batch_id,
            "rationale": committed.rationale,
            "actions": [
                {
                    "kind": a.kind,
                    "target_id": a.target_id,
                    "destination": _coord(a.destination),
                    "parameters": a.parameters,
                    "reason": a.reason,
                }
                for a in committed.actions
            ],
        }
    return payload


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5000", help="CarlaBridge URL.")
    parser.add_argument("--namespace", default="/agent", help="Socket.IO namespace.")
    parser.add_argument("--dotenv-path", default=".env", help="LLM .env path.")
    parser.add_argument("--no-llm", action="store_true", help="Run without LLM.")

    parser.add_argument(
        "--patrol-waypoint",
        type=_parse_waypoint,
        action="append",
        default=None,
        help=(
            "Patrol waypoint in 'x,y,z' (meters, CARLA frame). Pass multiple "
            "times to override the default rectangle "
            "(-20,-20)->(20,-20)->(20,20)->(-20,20)."
        ),
    )
    parser.add_argument(
        "--fire-watch-x",
        type=float,
        default=DEFAULT_FIRE_WATCH_XY[0],
        help="Fire-watch ground x (patrol line origin; default 25.3).",
    )
    parser.add_argument(
        "--fire-watch-y",
        type=float,
        default=DEFAULT_FIRE_WATCH_XY[1],
        help="Fire-watch ground y (patrol line origin; default 24.4).",
    )
    parser.add_argument("--patrol-altitude", type=float, default=10.0)
    parser.add_argument(
        "--patrol-leg-m",
        type=float,
        default=50.0,
        help="Straight patrol leg length in meters (forward then back; default 90).",
    )
    parser.add_argument(
        "--patrol-forward-axis",
        choices=("x", "y"),
        default="x",
        help="CARLA axis for the outbound leg (+x or +y; default +x).",
    )
    parser.add_argument(
        "--max-patrol-drones",
        type=int,
        default=3,
        help="Cap of UAVs to send on the initial patrol batch.",
    )

    parser.add_argument(
        "--detection-poll-interval",
        type=float,
        default=0.1,
        help="Seconds between two state_snapshot polls when waiting for fire.",
    )
    parser.add_argument(
        "--max-detection-rounds",
        type=int,
        default=120,
        help="Max polls before giving up (default: 60s at 0.5s interval).",
    )
    parser.add_argument(
        "--arrival-poll-interval",
        type=float,
        default=0.05,
        help="Poll interval while waiting for patrol UAV at fire (default: detection interval).",
    )
    parser.add_argument(
        "--max-arrival-rounds",
        type=int,
        default=120,
        help="Max polls waiting for patrol to reach fire-watch anchor before hold.",
    )
    parser.add_argument(
        "--no-return",
        action="store_true",
        help="Skip the post-response RTL batch (debug only).",
    )

    parser.add_argument(
        "--fallback-incident-id",
        default=None,
        help=(
            "If set, inject a default fire incident into the CarlaBridge "
            "snapshot when it has no incidents. This BYPASSES the patrol "
            "phase. Only use to debug the dispatch leg in isolation."
        ),
    )
    parser.add_argument("--fallback-x", type=float, default=0.0)
    parser.add_argument("--fallback-y", type=float, default=0.0)
    parser.add_argument("--fallback-z", type=float, default=0.0)

    parser.add_argument("--command-timeout", type=float, default=180.0)
    parser.add_argument("--ack-timeout", type=float, default=2.0)
    parser.add_argument("--state-timeout", type=float, default=30.0)

    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Number of patrol->fire->return cycles to run (0 = run forever).",
    )
    parser.add_argument(
        "--cycle-pause",
        type=float,
        default=0.0,
        help="Seconds to pause between cycles once everyone has returned home.",
    )
    parser.add_argument(
        "--return-poll-interval",
        type=float,
        default=0.5,
        help="Seconds between snapshot polls while waiting for in_flight_commands to drain.",
    )
    parser.add_argument(
        "--max-return-rounds",
        type=int,
        default=600,
        help="Max snapshot polls waiting for in_flight_commands==[] before re-patrolling.",
    )
    parser.add_argument(
        "--emit-frontend-json",
        action="store_true",
        help=(
            "Also print a structured JSON payload (thinking + dispatch plan) "
            "between [FRONTEND_JSON_BEGIN]/[FRONTEND_JSON_END] markers for the "
            "frontend / log scrapers to parse."
        ),
    )

    args = parser.parse_args()

    patrol_waypoints = args.patrol_waypoint
    if patrol_waypoints is None:
        patrol_waypoints = _default_patrol_waypoints(args.patrol_altitude)

    default_incidents: list[Incident] = []
    if args.fallback_incident_id:
        default_incidents.append(
            Incident(
                id=args.fallback_incident_id,
                kind="fire",
                severity="high",
                position=Coordinate(args.fallback_x, args.fallback_y, args.fallback_z),
                description="Fallback incident supplied by patrol-fire demo.",
            )
        )

    sandbox = CarlaBridgeSandboxClient(
        args.url,
        namespace=args.namespace,
        default_incidents=default_incidents,
        command_timeout=args.command_timeout,
        ack_timeout=args.ack_timeout,
        state_timeout=args.state_timeout,
    )
    agent = UrbanMultiAgentSystem(
        sandbox=sandbox,
        dotenv_path=args.dotenv_path,
        use_llm=not args.no_llm,
        use_llm_batch_rerank=not args.no_llm,
    )

    last_response_ok = False
    try:
        cycle = 0
        while args.cycles <= 0 or cycle < args.cycles:
            cycle += 1
            cycle_label = f"{cycle}/{args.cycles}" if args.cycles > 0 else f"{cycle}/∞"
            print("#" * 72)
            print(f"[CYCLE {cycle_label}] 进入巡逻，等待火情…")

            result = await agent.run_patrol_fire_response(
                patrol_waypoints=patrol_waypoints,
                fire_watch_point=Coordinate(
                    args.fire_watch_x, args.fire_watch_y, 0.0
                ),
                patrol_altitude=args.patrol_altitude,
                patrol_leg_m=args.patrol_leg_m,
                patrol_forward_axis=args.patrol_forward_axis,
                max_patrol_drones=args.max_patrol_drones,
                detection_poll_interval_s=args.detection_poll_interval,
                max_detection_rounds=args.max_detection_rounds,
                arrival_poll_interval_s=args.arrival_poll_interval,
                max_arrival_rounds=args.max_arrival_rounds,
                return_after_response=not args.no_return,
            )

            last_response_ok = _print_result(
                result, emit_frontend_json=args.emit_frontend_json
            )

            if result.detected_incident_id is None:
                # 没点火：本轮巡逻窗口内未发现火情，立刻重新巡逻继续等待。
                print(f"[CYCLE {cycle_label}] 未发现火情，重新巡逻继续等待点火…")
                continue

            # 完成一次调度后，以 snapshot 的 in_flight_commands==[] 作为
            # “全部归位”的判据：轮询直到没有在飞命令再进入下一轮巡逻。
            print(f"[CYCLE {cycle_label}] 调度完成，等待 in_flight_commands 清空（全部归位）…")
            drained = await _wait_until_all_returned(
                sandbox,
                poll_interval_s=args.return_poll_interval,
                max_rounds=args.max_return_rounds,
            )
            if drained:
                print(f"[CYCLE {cycle_label}] in_flight_commands 已清空，单位全部归位。")
            else:
                print(
                    f"[CYCLE {cycle_label}] 等待归位超时，仍有在飞命令："
                    f"{[c.get('command_id') or c.get('id') for c in sandbox.in_flight_commands_view]}"
                )
            if args.cycle_pause > 0 and (args.cycles <= 0 or cycle < args.cycles):
                print(f"[CYCLE {cycle_label}] 暂停 {args.cycle_pause:.1f}s 后再次巡逻…")
                await asyncio.sleep(args.cycle_pause)
    except KeyboardInterrupt:
        print("\n[LOOP] 收到中断信号，停止循环。")
    finally:
        await sandbox.close()

    return 0 if last_response_ok else 1


async def _wait_until_all_returned(
    sandbox,
    *,
    poll_interval_s: float,
    max_rounds: int,
) -> bool:
    """Poll the latest state_snapshot until ``in_flight_commands`` is empty.

    Returns True once no command is in flight (all units returned / idle), or
    False if ``max_rounds`` polls elapse first. Uses the cached snapshot the
    Socket.IO client keeps in ``in_flight_commands_view``; ``get_state`` ensures
    the connection is alive and a snapshot has been received.
    """
    for _ in range(max(1, max_rounds)):
        try:
            await sandbox.get_state()
        except Exception:
            pass
        if not sandbox.in_flight_commands_view:
            return True
        await asyncio.sleep(poll_interval_s)
    return not sandbox.in_flight_commands_view


def _print_result(result, *, emit_frontend_json: bool = False) -> bool:
    """Print one cycle's outcome and return whether the dispatch succeeded."""
    print("=" * 72)
    print("final_report:", result.final_report)
    print("detected_incident_id:", result.detected_incident_id)
    print("detection_notes:")
    for note in result.detection_notes:
        print(f"  - {note}")

    _print_batch_outcome("PATROL", result.patrol_outcome)
    _print_batch_outcome("HOLD", result.hold_outcome)

    if result.response is not None:
        # 模型思考过程 + 最终调度计划（后续给前端展示用）。
        _print_thinking(result.response)
        _print_dispatch_plan(result.response)

        _print_batch_outcome("RESPONSE", result.response.batch_outcome)
        if result.response.final_report:
            print("[RESPONSE] final_report:", result.response.final_report)

    _print_batch_outcome("RETURN", result.return_outcome)

    if emit_frontend_json:
        payload = _build_frontend_payload(result)
        print("[FRONTEND_JSON_BEGIN]")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print("[FRONTEND_JSON_END]")

    return (
        result.response is not None
        and result.response.batch_outcome is not None
        and result.response.batch_outcome.criteria_satisfied
    )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
