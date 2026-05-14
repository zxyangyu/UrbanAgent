"""Run UrbanMultiAgentSystem against CarlaBridge Socket.IO `/agent`.

Example:
    python scripts/carla_bridge_multiagent_demo.py --url http://127.0.0.1:5000 --no-llm
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from urbanagent import CarlaBridgeSandboxClient, UrbanMultiAgentSystem
from urbanagent.types import Coordinate, Incident


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5000", help="CarlaBridge URL.")
    parser.add_argument("--namespace", default="/agent", help="Socket.IO namespace.")
    parser.add_argument("--dotenv-path", default=".env", help="LLM .env path.")
    parser.add_argument("--no-llm", action="store_true", help="Run without LLM.")
    parser.add_argument("--task", default="incident-fire-001 高严重度火情，请进行多智能体协同调度。")
    parser.add_argument(
        "--incident-id",
        default="incident-fire-001",
        help="Fallback incident id when CarlaBridge snapshot has no incidents.",
    )
    parser.add_argument("--incident-x", type=float, default=0.0)
    parser.add_argument("--incident-y", type=float, default=0.0)
    parser.add_argument("--incident-z", type=float, default=0.0)
    args = parser.parse_args()

    fallback_incident = Incident(
        id=args.incident_id,
        kind="fire",
        severity="high",
        position=Coordinate(args.incident_x, args.incident_y, args.incident_z),
        description="Fallback incident supplied by UrbanAgent demo arguments.",
    )
    sandbox = CarlaBridgeSandboxClient(
        args.url,
        namespace=args.namespace,
        default_incidents=[fallback_incident],
    )
    agent = UrbanMultiAgentSystem(
        sandbox=sandbox,
        dotenv_path=args.dotenv_path,
        use_llm=not args.no_llm,
        use_llm_batch_rerank=not args.no_llm,
    )
    try:
        result = await agent.run(args.task)
        print(result.final_report or result.skipped_reason)
        if result.committed is not None:
            print(f"batch_id={result.committed.batch_id} actions={len(result.committed.actions)}")
        if result.batch_outcome is not None:
            print(
                "criteria_satisfied="
                f"{result.batch_outcome.criteria_satisfied} "
                f"polls={result.batch_outcome.polling_iterations}"
            )
            for note in result.batch_outcome.notes:
                print(f"note: {note}")
    finally:
        await sandbox.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
