"""Method-style UrbanAgent demo for a single fire incident.

Full pipeline (default): LLM cognition + LLM planning + sandbox execution + LLM synthesis
report. Requires a valid `.env` (see `.env.example`).

Usage from the project root (repository root, e.g. `urbanagent-main/`):
    pip install -e .
    python scripts/urban_agent_fire_demo.py --dotenv-path .env

Offline only (rule-based cognition/planning, no LLM — not the full paper-style path):
    python scripts/urban_agent_fire_demo.py --no-llm

Or without pip install:
    python scripts/urban_agent_fire_demo.py --dotenv-path .env
    (script adds the project root to sys.path)

Optional (OpenAI-compatible APIs e.g. DeepSeek): set in the environment
    AGENT_LAB_CHAT_JSON_OBJECT=1
to request response_format json_object and reduce malformed JSON from the model.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from urbanagent.llm.errors import LLMError
from urbanagent import (
    CarlaBridgeSandboxClient,
    MockSandboxClient,
    UrbanAgent,
    UrbanAgentPipelineError,
    build_external_tool_facade,
)
from urbanagent.types import Coordinate, Incident, to_json


DEFAULT_TASK = (
    "城市沙盘报告 incident-fire-001 附近发生高严重度火情。"
    "请调度最近可用消防资源、安排空中侦察、控制附近路口，"
    "并输出可执行的调度方案。"
)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK, help="Emergency task text.")
    parser.add_argument(
        "--dotenv-path",
        default=".env",
        help="Path to .env for the LLM provider config.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help=(
            "离线调试：使用内置规则做认知与规划（不调用 LLM）。"
            "完整 UrbanAgent 请省略此参数并配置好 .env。"
        ),
    )
    parser.add_argument(
        "--show-json",
        action="store_true",
        help="Print the full mock city state and sandbox action JSON.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum closed-loop retry count for each simulated tool node.",
    )
    parser.add_argument(
        "--http-tools",
        default=None,
        help=(
            "JSON file of HTTP API tools (see examples/http_tools.example.json). "
            "Overrides env URBANAGENT_HTTP_TOOLS when set."
        ),
    )
    parser.add_argument(
        "--mcp-servers",
        default=None,
        help=(
            "JSON file listing MCP stdio servers (see examples/mcp_servers.example.json). "
            "Overrides env URBANAGENT_MCP_SERVERS when set."
        ),
    )
    parser.add_argument(
        "--no-llm-dispatch-ranking",
        action="store_true",
        help="Do not call LLM to reorder dispatch candidates (use code sort only).",
    )
    parser.add_argument(
        "--llm-dispatch-ranking-strict",
        action="store_true",
        help="If LLM candidate reorder fails, abort instead of falling back to code order.",
    )
    parser.add_argument(
        "--carla-bridge-url",
        default=None,
        help=(
            "CarlaBridge Socket.IO URL, e.g. http://127.0.0.1:5000. "
            "Overrides env URBANAGENT_CARLA_BRIDGE_URL when set."
        ),
    )
    parser.add_argument(
        "--carla-namespace",
        default="/agent",
        help="CarlaBridge Socket.IO namespace (default: /agent).",
    )
    parser.add_argument(
        "--incident-id",
        default="incident-fire-001",
        help="Fallback incident id when CarlaBridge snapshots do not contain incidents.",
    )
    parser.add_argument("--incident-x", type=float, default=0.0)
    parser.add_argument("--incident-y", type=float, default=0.0)
    parser.add_argument("--incident-z", type=float, default=0.0)
    args = parser.parse_args()

    bridge_url = (
        args.carla_bridge_url or os.environ.get("URBANAGENT_CARLA_BRIDGE_URL") or ""
    ).strip()
    sandbox = None
    try:
        if bridge_url:
            fallback_incident = Incident(
                id=args.incident_id,
                kind="fire",
                severity="high",
                position=Coordinate(args.incident_x, args.incident_y, args.incident_z),
                description="Fallback incident supplied by UrbanAgent demo arguments.",
            )
            sandbox = CarlaBridgeSandboxClient(
                bridge_url,
                namespace=args.carla_namespace,
                default_incidents=[fallback_incident],
            )
            await sandbox.connect()
        else:
            sandbox = MockSandboxClient()
        initial_state = await sandbox.get_state()
        print("[urban-demo] initial city state summary:")
        print(_state_summary(initial_state))
        if args.show_json:
            print("\n[urban-demo] initial city state JSON:")
            print(to_json(initial_state))

        http_tools = args.http_tools or os.environ.get("URBANAGENT_HTTP_TOOLS")
        mcp_servers = args.mcp_servers or os.environ.get("URBANAGENT_MCP_SERVERS")
        external = build_external_tool_facade(
            http_tools_path=http_tools,
            mcp_servers_path=mcp_servers,
        )

        agent = UrbanAgent(
            sandbox,
            dotenv_path=args.dotenv_path,
            use_llm=not args.no_llm,
            max_retries=args.max_retries,
            external_tools=external,
            llm_dispatch_ranking=not args.no_llm_dispatch_ranking,
            llm_dispatch_ranking_strict=args.llm_dispatch_ranking_strict,
        )
        try:
            result = await agent.run(args.task)
        except UrbanAgentPipelineError as exc:
            print("\n[urban-demo] UrbanAgent 管线失败（完整 LLM 模式下认知、规划或综合阶段失败）。")
            print(f"[urban-demo] {exc}")
            print("[urban-demo] 请检查模型输出是否为合法 JSON，或改用 --no-llm 使用规则认知/规划与确定性综合。")
            return 3
        except LLMError as exc:
            print("\n[urban-demo] LLM 配置或调用失败。")
            print(f"[urban-demo] {exc}")
            print("[urban-demo] 请检查 --dotenv-path 指向的 .env，或使用 --no-llm 运行离线规则模式。")
            return 2

        print("\n[urban-demo] method pipeline output:")
        print(result.final_report)
        print("\n[urban-demo] applied sandbox actions:")
        for item in getattr(sandbox, "applied_results", []):
            if args.show_json:
                print(to_json(item))
            else:
                print(f"- {item.status}: {item.message}")
        print(f"\n[urban-demo] llm_used={result.llm_used}")
        return 0
    finally:
        if bridge_url and sandbox is not None:
            await sandbox.close()


def _state_summary(state) -> str:
    incidents = ", ".join(
        f"{item.id}({item.kind}/{item.severity}/{item.status})"
        for item in state.incidents
    )
    available = [item for item in state.resources if item.status == "available"]
    resources = ", ".join(
        f"{item.kind}:{item.id}" for item in available
    )
    signals = ", ".join(
        f"{item.id}:{item.mode}" for item in state.traffic_signals
    )
    return (
        f"- incidents: {incidents or 'none'}\n"
        f"- available resources: {resources or 'none'}\n"
        f"- traffic signals: {signals or 'none'}"
    )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
