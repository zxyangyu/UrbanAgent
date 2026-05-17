# UrbanAgent（CarlaBridge 接入版）

城市应急调度智能体：面向 **CarlaBridge / CARLA 城市沙盘** 的多智能体编排系统。当前主流程是：

`3D 状态快照 → G1 LLM 介入门控 → 元智能体认知/分解 → 子 Agent 局部工具规划 → 元智能体整合有序批次 → CarlaBridge 下发 → 批末状态轮询 → LLM 报告`

## CarlaBridge 接入协议

本项目按照 **Bridge × Agent Protocol v1.0**（`bridge-agent-protocol-v1.md`）接入中间件，不直接连接 CARLA。

- 协议：Socket.IO v4.x（基于 WebSocket）
- URL：`http://<bridge_ip>:<port>`，默认 `http://localhost:5000`
- Namespace：`/agent`
- 协议版本字段：`version = "1.0"`
- 坐标：CARLA 左手系笛卡尔坐标，单位米。**注意**协议对坐标用两种形式：
  - `vehicles[].pose / uavs[].pose / traffic_lights[].pose` 是 `[x, y, z]` 数组
  - `incidents[].position` 和命令参数（`waypoint` / `dest` / `position`）是 `{x, y, z}` 字典

Agent 侧客户端是：

```python
from urbanagent import CarlaBridgeSandboxClient
```

它的行为：

- 连接 `/agent` 后通过 `sio.call("hello", {agent_id, version})` 完成握手；保存 `bridge_session_id`（用于检测 Bridge 重启）
- 监听 Bridge 下行：
  - `state_snapshot`（10 Hz；包含 `run_id` / `bridge_session_id` / `in_flight_commands`）
  - `command_status`（命令生命周期：`accepted / ongoing / completed / failed / cancelled`）
  - `scenario_event`（v1.0 仅 `event="reset"`，会取消所有未决命令）
  - `event_log`（仅记录，不进入决策）
- 上行通过 `sio.call("agent.command", envelope)` 同步等待 `accepted/rejected`，再异步等待终态 `command_status` 才把结果返回给 batch_runner
- `send_action()` 把 `UrbanAction` 翻译为 v1.0 的 8 类命令之一：
  - `dispatch_drone` → `UAV_GOTO`（参数 `waypoint`）
  - `dispatch_vehicle` → `UGV_GOTO`（参数 `dest`），当 `parameters["intent"]="extinguish"` 且目标距离某 `fire` 类 incident ≤ 5 m 时升级为 `UGV_EXTINGUISH`
  - `control_traffic_light` / `mark_incident` 不在 v1.0 协议内，适配层立即返回 `rejected` 并写 warning 级 `event_log`

> ℹ **目标实体校验**：`UrbanAction.target_id` 会由适配层与最近一次 `state_snapshot.vehicles / uavs` 的 ID 集合比对，未匹配（如 LLM 幻觉出 `drone-01`）会在 RPC 发出前本地驳回（`rejected: unknown_target ...`）并写 warning 级 `event_log`，避免到 Bridge 才返回 `unknown_target`。子 Agent 通过 `state.resources` 选 target 时已经会得到合规 ID，校验仅作防御。

所有消息使用 Envelope：`version/msg_id/type/timestamp/frame/sim_time/sender/payload`。

## 安装

```bash
pip install -e .
```

需要 Socket.IO 客户端依赖：`python-socketio[client]`，已在 `pyproject.toml` 中声明。

建议（OpenAI 兼容 API，减轻非法 JSON）：

```bash
export AGENT_LAB_CHAT_JSON_OBJECT=1
```

Windows PowerShell：

```powershell
$env:AGENT_LAB_CHAT_JSON_OBJECT="1"
```

## 配置 LLM

复制 `.env.example` 为 `.env`，至少配置：

- `AGENT_LAB_LLM_PROVIDER`（如 `deepseek`、`openai`、`anthropic`）
- 对应厂商的 `*_API_KEY`、必要时 `*_BASE_URL`、`*_MODEL`

离线调试可以使用 `--no-llm`，此时门控/认知/分解/报告使用规则或确定性回退。

## 运行 MultiAgent 离线 Demo

先不连接 CarlaBridge，只验证多智能体管线：

```bash
python scripts/multiagent_smoke.py
python -m unittest tests.test_multiagent_mvp tests.test_carla_bridge_protocol_v1 -v
```

预期：测试全部通过，`multiagent_smoke.py` 输出 `ok ... actions`。

## 连接 CarlaBridge 运行

确保 CarlaBridge 已启动，并在配置中开启：

运行：

```bash
python scripts/carla_bridge_multiagent_demo.py \
  --url http://127.0.0.1:5000 \
  --namespace /agent \
  --no-llm
```

如果 Bridge 的 `state.snapshot` 暂时不包含 `incidents`，可以通过参数给 UrbanAgent 一个 fallback 事件：

```bash
python scripts/carla_bridge_multiagent_demo.py \
  --url http://127.0.0.1:5000 \
  --no-llm \
  --incident-id incident-fire-001 \
  --incident-x 50 \
  --incident-y -100 \
  --incident-z 0
```

启用 LLM：

```bash
python scripts/carla_bridge_multiagent_demo.py \
  --url http://127.0.0.1:5000 \
  --dotenv-path .env
```

也可以设置环境变量：

```bash
export URBANAGENT_CARLA_BRIDGE_URL=http://127.0.0.1:5000
```

## 代码中接入 CarlaBridge

```python
import asyncio
from urbanagent import CarlaBridgeSandboxClient, UrbanMultiAgentSystem
from urbanagent.types import Coordinate, Incident

async def main():
    sandbox = CarlaBridgeSandboxClient(
        "http://127.0.0.1:5000",
        namespace="/agent",
        default_incidents=[
            Incident(
                id="incident-fire-001",
                kind="fire",
                severity="high",
                position=Coordinate(50, -100, 0),
            )
        ],
    )
    agent = UrbanMultiAgentSystem(
        sandbox=sandbox,
        dotenv_path=".env",
        use_llm=True,
    )
    try:
        result = await agent.run("incident-fire-001 高严重度火情，请进行多智能体协同调度。")
        print(result.final_report)
    finally:
        await sandbox.close()

asyncio.run(main())
```

## MultiAgent 设计

当前是最小 MVP：每种类型默认一个子 Agent，后续可通过 `SubAgentRegistry` 扩展为多实例。

- 元智能体：G1/G2/G3/G5/G8 使用 LLM 或规则回退
- 子 Agent：无人机、无人车、警车、信号灯
- 子 Agent 通信：只向元智能体汇报 `SubPlan`，子 Agent 之间不直接 P2P 通信
- S0-S2：可用 LLM 做子目标解析、能力对齐、观测摘要
- S3：只用 tools（路由、资源筛选、信号灯选择），不调用 LLM
- 执行：元智能体整合为有序 `agent_command` 批次，逐条发给 CarlaBridge
- 闭环：收到 `agent_ack` / `agent_reject` 后，整批结束再通过 `state.snapshot` 判据轮询确认

## LangGraph + DefenseAgent 独立项目

LangGraph + DefenseAgent 版本已经从本仓库完全拆出，位于：

```text
D:\论文代码\urbanagent_langgraph_defenseagent
```

当前仓库继续保留手写 `urbanagent.multiagent` 作为 CarlaBridge baseline。独立项目拥有自己的 `pyproject.toml`、README、源码包、profiles、demo 脚本和测试，可单独安装、运行和演进。

## CarlaBridge 动作映射

`UrbanAction` 会转换为 CarlaBridge `agent_command`：

- `dispatch_vehicle`
  - `UGV-*` → `UGV_DISPATCH`
  - `POLICE*` / `POL*` → `POLICE_DISPATCH`
  - 其他 → `VEHICLE_DISPATCH`
- `dispatch_drone` → `UAV_DISPATCH`
- `control_traffic_light` → `TL_SET_STATE`
- `mark_incident` → `MARK_EVENT`

若 CarlaBridge 实际枚举不同，可以构造客户端时覆盖：

```python
CarlaBridgeSandboxClient(
    "http://127.0.0.1:5000",
    action_map={
        "dispatch_drone": "YOUR_UAV_ENUM",
        "control_traffic_light": "YOUR_SIGNAL_ENUM",
    },
)
```

## 外部工具（MCP / HTTP API）

MCP / HTTP API 是论文中的外部工具库 **T**，与 CarlaBridge 环境 **W** 分离。

- 单智能体 `UrbanAgent` 仍支持 `build_external_tool_facade(http_tools_path=..., mcp_servers_path=...)`
- 多智能体 MVP 目前尚未把 MCP / HTTP 接入子 Agent 工具层，后续应作为 `SubAgentToolkit` 或元智能体工具扩展

## 包结构

- `urbanagent.carla_bridge`：CarlaBridge Socket.IO v1.1 适配器
- `urbanagent.multiagent`：元智能体、子 Agent、批执行与状态轮询
- `urbanagent.llm`：LLM 门面（OpenAI 兼容 / Anthropic）
- `urbanagent.tooling`：外部工具门面（HTTP JSON、MCP stdio）
- `urbanagent.sandbox`：离线 `MockSandboxClient` 与抽象 `SandboxClient`
