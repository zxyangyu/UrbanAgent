"""Method-style UrbanAgent for executable city emergency decisions."""
from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from urbanagent.llm.llm import LLM
from urbanagent.llm.types import Message
from urbanagent.dispatch import DispatchPolicy, assignment_to_action
from urbanagent.errors import UrbanAgentPipelineError
from urbanagent.sandbox import MockSandboxClient, SandboxClient
from urbanagent.schemas import (
    DispatchSolution,
    ExecutionObservation,
    TaskGraph,
    TaskNode,
    UrbanAgentResult,
    UrbanConstraint,
    UrbanTask,
)
from urbanagent.tooling.facade import ExternalToolFacade
from urbanagent.types import (
    ActionResult,
    CandidateScore,
    CityState,
    Coordinate,
    DispatchPlan,
    Incident,
    TrafficSignal,
    UrbanAction,
)


class UrbanAgent:
    """A minimal City Life Agent for fire dispatch.

    The public contract follows the paper's method section: natural language query
    plus current city state goes through cognition, planning, execution, and
    synthesis. With ``use_llm=True``, synthesis runs a deterministic core (plan,
    scores, constraint flags) then an LLM pass that rewrites narrative fields only.
    """

    def __init__(
        self,
        sandbox: SandboxClient | None = None,
        *,
        dispatch_policy: DispatchPolicy | None = None,
        llm: LLM | None = None,
        dotenv_path: str | Path | None = None,
        use_llm: bool = True,
        max_retries: int = 3,
        external_tools: ExternalToolFacade | None = None,
        llm_dispatch_ranking: bool = True,
        llm_dispatch_ranking_strict: bool = False,
    ) -> None:
        self.sandbox = sandbox or MockSandboxClient()
        self.dispatch_policy = dispatch_policy or DispatchPolicy()
        self.max_retries = max(1, max_retries)
        self.llm = llm
        self.dotenv_path = str(dotenv_path) if dotenv_path is not None else None
        self.use_llm = use_llm
        self.external_tools = external_tools
        self._external_facade_active: ExternalToolFacade | None = None
        self.llm_dispatch_ranking = llm_dispatch_ranking
        self.llm_dispatch_ranking_strict = llm_dispatch_ranking_strict
        self._runtime_task: UrbanTask | None = None

    async def _coerce_llm_json(
        self,
        llm: LLM,
        raw_text: str,
        *,
        stage: str,
        schema_hint: str,
    ) -> dict[str, Any]:
        """Parse model output as JSON; on failure ask the model once to emit valid JSON."""
        try:
            return _try_loads_json_object(raw_text)
        except ValueError:
            pass
        repair = await llm.chat(
            [
                Message(
                    role="user",
                    content=(
                        "下面内容本应是一个 JSON 对象但语法无效。"
                        "请只输出一个合法 JSON 对象，不要 Markdown，不要解释。\n\n"
                        f"必须包含的字段: {schema_hint}\n\n"
                        f"原始输出:\n{raw_text[:32000]}"
                    ),
                )
            ],
            temperature=0.0,
            max_tokens=8192,
            system="You repair JSON. Output only a single valid JSON object.",
        )
        try:
            return _try_loads_json_object(repair.content)
        except ValueError as exc:
            raise UrbanAgentPipelineError(
                stage,
                f"模型 JSON 无法解析，经一次修复后仍失败: {exc}",
            ) from exc

    async def run(self, query: str) -> UrbanAgentResult:
        """Run Q -> U -> G -> O -> R for one fire-dispatch request."""
        facade = self.external_tools
        if facade is None or not facade.enabled:
            return await self._run_with_external_tools(query, None)
        async with facade:
            return await self._run_with_external_tools(query, facade)

    async def _run_with_external_tools(
        self,
        query: str,
        facade: ExternalToolFacade | None,
    ) -> UrbanAgentResult:
        self._external_facade_active = facade
        try:
            # Snapshots: sandbox mutates CityState in place (dispatch, mark_incident).
            # Keep a deep copy so cognition/facts/result.initial_state stay pre-run truth.
            initial_state = copy.deepcopy(await self.sandbox.get_state())
            llm = self._ensure_llm() if self.use_llm else None
            task = await self._cognition(query, initial_state, llm)
            graph = await self._planning(task, initial_state, llm)
            self._runtime_task = task
            try:
                observations = await self._execution(graph, llm)
            finally:
                self._runtime_task = None
            solutions = await self._synthesis(task, observations, llm)
            final_report = await self._render_report(
                query,
                task,
                graph,
                observations,
                solutions,
                llm,
                initial_state,
            )
            return UrbanAgentResult(
                query=query,
                task=task,
                graph=graph,
                observations=observations,
                solutions=solutions,
                final_report=final_report,
                initial_state=initial_state,
                llm_used=llm is not None,
            )
        finally:
            self._external_facade_active = None

    def _all_tool_metadata(self) -> list[dict[str, Any]]:
        """Built-in environment operations (W) plus optional external toolset (T: MCP, HTTP)."""
        rows = list(_builtin_env_operation_metadata())
        if self._external_facade_active is not None:
            rows.extend(self._external_facade_active.planner_metadata())
        return rows

    def _ensure_llm(self) -> LLM | None:
        if self.llm is not None:
            return self.llm
        if not self.use_llm:
            return None
        self.llm = LLM.from_env(dotenv_path=self.dotenv_path)
        return self.llm

    async def _cognition(
        self,
        query: str,
        state: CityState,
        llm: LLM | None,
    ) -> UrbanTask:
        if self.use_llm:
            if llm is None:
                raise UrbanAgentPipelineError(
                    "cognition",
                    "LLM 未就绪：use_llm=True 但未能构造 LLM。请检查 .env。",
                )
            try:
                return await self._llm_cognition(query, state, llm)
            except UrbanAgentPipelineError:
                raise
            except Exception as exc:
                raise UrbanAgentPipelineError(
                    "cognition",
                    "LLM 认知失败：模型返回无效 JSON、缺字段或 API 错误。"
                    f" 详情: {exc}",
                ) from exc
        return self._rule_cognition(query, state)

    async def _llm_cognition(
        self,
        query: str,
        state: CityState,
        llm: LLM,
    ) -> UrbanTask:
        state_summary = _city_state_brief_for_cognition(state)
        response = await llm.chat(
            [
                Message(
                    role="user",
                    content=(
                        "用户火灾调度请求：\n"
                        f"{query}\n\n"
                        "当前模拟 3D 城市沙盘状态摘要：\n"
                        f"{json.dumps(state_summary, ensure_ascii=False, indent=2)}\n\n"
                        "请只输出 JSON，不要 Markdown。字段必须是："
                        "intent(string), entities(object), constraints(array), "
                        "missing_information(array), rationale(string)。"
                        "constraints 中每项包含 name, kind(hard或soft), expression。"
                        "不要编造不存在的 incident_id 或资源 id。"
                    ),
                )
            ],
            temperature=0.2,
            max_tokens=4096,
            system=(
                "You are the cognition module of UrbanAgent. "
                "Transform a city emergency request into U=<I,E,C>. "
                "Return strict JSON only."
            ),
        )
        data = await self._coerce_llm_json(
            llm,
            response.content,
            stage="cognition",
            schema_hint=(
                "intent (string), entities (object), constraints (array of "
                "{name, kind, expression}), missing_information (array), "
                "rationale (string)"
            ),
        )
        constraints = [
            UrbanConstraint(
                name=str(item.get("name", "")).strip() or "unnamed_constraint",
                kind="soft" if str(item.get("kind", "")).strip() == "soft" else "hard",
                expression=str(item.get("expression", "")).strip(),
            )
            for item in data.get("constraints", [])
            if isinstance(item, dict)
        ]
        if not constraints:
            raise UrbanAgentPipelineError(
                "cognition",
                "LLM 输出缺少有效的非空 constraints 数组。",
            )

        task = UrbanTask(
            intent=str(data.get("intent", "fire_emergency_dispatch")).strip()
            or "fire_emergency_dispatch",
            entities=dict(data.get("entities", {})),
            constraints=constraints,
            missing_information=[
                str(item)
                for item in data.get("missing_information", [])
                if str(item).strip()
            ],
            rationale=str(data.get("rationale", "")).strip(),
            source="llm",
        )
        return self._ground_llm_task(task, state)

    def _rule_cognition(self, query: str, state: CityState) -> UrbanTask:
        incident_id = _extract_incident_id(query)
        fire_incidents = [
            incident
            for incident in state.incidents
            if incident.kind == "fire" and incident.status in {"open", "responding"}
        ]
        if incident_id is None and fire_incidents:
            incident_id = fire_incidents[0].id

        incident = _find_incident(state, incident_id) if incident_id else None
        severity = incident.severity if incident is not None else _extract_severity(query)
        constraints = [
            UrbanConstraint(
                name="fire_suppression_required",
                kind="hard",
                expression="at least one available fire_suppression resource is dispatched",
            ),
            UrbanConstraint(
                name="aerial_recon_required",
                kind="hard",
                expression="at least one aerial_recon resource is dispatched",
            ),
            UrbanConstraint(
                name="reserve_ratio",
                kind="hard",
                expression="dispatch must not violate station reserve ratios",
            ),
            UrbanConstraint(
                name="min_response_time",
                kind="soft",
                expression="prefer lower travel time and lower congestion",
            ),
        ]
        if severity in {"high", "critical"}:
            constraints.append(
                UrbanConstraint(
                    name="police_control_required",
                    kind="hard",
                    expression="at least one traffic_control police resource is dispatched",
                )
            )
            constraints.append(
                UrbanConstraint(
                    name="traffic_control_required",
                    kind="hard",
                    expression="control a nearby traffic signal for emergency passage",
                )
            )

        missing = []
        if incident_id is None:
            missing.append("incident_id")
        elif incident is None:
            missing.append(f"known incident {incident_id}")

        return UrbanTask(
            intent="fire_emergency_dispatch",
            entities={
                "incident_id": incident_id,
                "incident_kind": "fire",
                "severity": severity,
                "location": incident.position if incident is not None else None,
            },
            constraints=constraints,
            missing_information=missing,
            rationale="Rule parser matched the active fire incident and required emergency roles.",
            source="rule",
        )

    def _ground_llm_task(self, task: UrbanTask, state: CityState) -> UrbanTask:
        """Align LLM entities with CityState without any rule-based cognition."""
        if not task.constraints:
            raise UrbanAgentPipelineError(
                "cognition",
                "grounding 要求非空 constraints。",
            )
        if task.missing_information:
            return task
        incident_id = task.entities.get("incident_id")
        if not incident_id:
            raise UrbanAgentPipelineError(
                "cognition",
                "LLM 未提供 entities.incident_id，且 missing_information 为空。"
                "若信息不足请填充 missing_information，仅规划 ask_user。",
            )
        incident = _find_incident(state, str(incident_id))
        if incident is None:
            raise UrbanAgentPipelineError(
                "cognition",
                f"entities.incident_id={incident_id!r} 在当前 CityState 中不存在。",
            )
        entities = dict(task.entities)
        entities["incident_id"] = str(incident_id)
        entities.setdefault("incident_kind", "fire")
        if not entities.get("severity"):
            entities["severity"] = incident.severity
        entities["location"] = incident.position
        return UrbanTask(
            intent=task.intent or "fire_emergency_dispatch",
            entities=entities,
            constraints=task.constraints,
            missing_information=[],
            rationale=task.rationale,
            source="llm",
        )

    async def _planning(
        self,
        task: UrbanTask,
        state: CityState,
        llm: LLM | None,
    ) -> TaskGraph:
        if self.use_llm:
            if llm is None:
                raise UrbanAgentPipelineError(
                    "planning",
                    "LLM 未就绪：use_llm=True 但未能构造 LLM。请检查 .env。",
                )
            try:
                return await self._llm_planning(task, state, llm)
            except UrbanAgentPipelineError:
                raise
            except Exception as exc:
                raise UrbanAgentPipelineError(
                    "planning",
                    "LLM 规划失败：模型返回无效 DAG、非法 tool 名、依赖错误或 API 错误。"
                    f" 详情: {exc}",
                ) from exc
        return self._rule_planning(task)

    async def _llm_planning(
        self,
        task: UrbanTask,
        state: CityState,
        llm: LLM,
    ) -> TaskGraph:
        payload = {
            "task": _task_to_data(task),
            "city_state": _city_state_summary(state),
            "available_tools": self._all_tool_metadata(),
            "requirements": {
                "output": "strict JSON object",
                "schema": {
                    "rationale": "string",
                    "nodes": [
                        {
                            "id": "string",
                            "description": "string",
                            "tool": "one of available_tools.name",
                            "args": "object",
                            "depends_on": ["node ids"],
                        }
                    ],
                },
                "rules": [
                    "Use only available tools.",
                    "Create an executable DAG, not prose.",
                    "For fire dispatch, observe city state before building or applying a plan.",
                    "Do not directly invent 3D sandbox state; use tools.",
                    "If incident information is missing, plan ask_user only.",
                ],
            },
        }
        response = await llm.chat(
            [
                Message(
                    role="user",
                    content=(
                        "请作为 UrbanAgent planning module，根据任务 U、城市状态 W 和工具"
                        " metadata 生成可执行 DAG。只输出 JSON，不要 Markdown。\n\n"
                        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
                    ),
                )
            ],
            temperature=0.2,
            max_tokens=1500,
            system=(
                "You are the planning module of UrbanAgent. "
                "Generate a schema-constrained executable DAG using only provided tools."
            ),
        )
        data = await self._coerce_llm_json(
            llm,
            response.content,
            stage="planning",
            schema_hint='rationale (string), nodes (array of {id, description, tool, args, depends_on})',
        )
        nodes = self._parse_llm_nodes(data)
        if not nodes:
            raise ValueError("LLM planning returned no executable nodes")
        return TaskGraph(
            nodes=nodes,
            source="llm",
            rationale=str(data.get("rationale", "")).strip(),
        )

    def _parse_llm_nodes(self, data: dict[str, Any]) -> list[TaskNode]:
        allowed = {item["name"] for item in self._all_tool_metadata()}
        raw_nodes = data.get("nodes", [])
        if not isinstance(raw_nodes, list):
            raise ValueError("LLM planning nodes must be a list")
        nodes: list[TaskNode] = []
        seen: set[str] = set()
        for index, raw in enumerate(raw_nodes, start=1):
            if not isinstance(raw, dict):
                continue
            tool = str(raw.get("tool", "")).strip()
            if tool not in allowed:
                raise ValueError(f"LLM planning used unknown tool: {tool}")
            node_id = str(raw.get("id") or f"node_{index}").strip()
            if not node_id:
                node_id = f"node_{index}"
            if node_id in seen:
                raise ValueError(f"duplicate node id: {node_id}")
            depends_on = [
                str(item)
                for item in raw.get("depends_on", [])
                if str(item).strip()
            ]
            unknown_deps = [item for item in depends_on if item not in seen]
            if unknown_deps:
                raise ValueError(
                    f"node {node_id} depends on unknown or later nodes: {unknown_deps}"
                )
            nodes.append(
                TaskNode(
                    id=node_id,
                    description=str(raw.get("description", "")).strip(),
                    tool=tool,
                    args=dict(raw.get("args", {})),
                    depends_on=depends_on,
                )
            )
            seen.add(node_id)
        return nodes

    def _rule_planning(self, task: UrbanTask) -> TaskGraph:
        if task.missing_information:
            return TaskGraph(
                nodes=[
                    TaskNode(
                        id="clarify_missing_information",
                        description="Stop execution and request missing fire incident information.",
                        tool="ask_user",
                        args={"missing": task.missing_information},
                    )
                ],
                source="rule",
                rationale="Missing information prevents executable planning.",
            )

        incident_id = str(task.entities["incident_id"])
        nodes = [
            TaskNode(
                id="observe_city_state",
                description="Read the current mock city state W.",
                tool="get_city_state",
            ),
            TaskNode(
                id="build_dispatch_plan",
                description="Score available resources and build a dispatch DAG node plan.",
                tool="create_dispatch_plan",
                args={"incident_id": incident_id},
                depends_on=["observe_city_state"],
            ),
            TaskNode(
                id="apply_dispatch_plan",
                description="Dispatch selected fire truck, drone, and police resources.",
                tool="apply_dispatch_plan",
                depends_on=["build_dispatch_plan"],
            ),
            TaskNode(
                id="control_emergency_signal",
                description="Switch the nearest traffic signal to emergency preemption.",
                tool="control_nearest_traffic_signal",
                args={"incident_id": incident_id},
                depends_on=["apply_dispatch_plan"],
            ),
            TaskNode(
                id="mark_incident_responding",
                description="Mark the incident as responding after actions are accepted.",
                tool="mark_incident",
                args={"incident_id": incident_id, "status": "responding"},
                depends_on=["apply_dispatch_plan", "control_emergency_signal"],
            ),
        ]
        return TaskGraph(
            nodes=nodes,
            source="rule",
            rationale="Rule planner selected the standard fire-dispatch workflow.",
        )

    async def _execution(
        self,
        graph: TaskGraph,
        llm: LLM | None,
    ) -> list[ExecutionObservation]:
        observations: list[ExecutionObservation] = []
        context: dict[str, Any] = {}
        handlers: dict[str, Callable[[TaskNode], Awaitable[Any]]] = {
            "ask_user": self._execute_ask_user,
            "get_city_state": self._execute_get_city_state,
            "create_dispatch_plan": lambda node: self._execute_create_dispatch_plan(
                node, context, llm
            ),
            "apply_dispatch_plan": lambda node: self._execute_apply_dispatch_plan(node, context),
            "control_nearest_traffic_signal": self._execute_control_signal,
            "mark_incident": self._execute_mark_incident,
        }

        completed: set[str] = set()
        for node in graph.nodes:
            if any(dependency not in completed for dependency in node.depends_on):
                node.status = "skipped"
                observations.append(
                    ExecutionObservation(
                        node_id=node.id,
                        tool=node.tool,
                        status="skipped",
                        error="dependency not completed",
                    )
                )
                continue

            handler = handlers.get(node.tool)
            if handler is None:
                if self._external_facade_active is None:
                    raise UrbanAgentPipelineError(
                        "execution",
                        f"未知工具 {node.tool!r}，且未配置外部工具（MCP / HTTP）。",
                    )
                handler = self._execute_external_tool
            observation = await self._execute_with_retry(node, handler, llm)
            observations.append(observation)
            if observation.status == "succeeded":
                completed.add(node.id)
                context[node.id] = observation.data
                context[node.tool] = observation.data
            else:
                node.status = observation.status
        return observations

    async def _execute_with_retry(
        self,
        node: TaskNode,
        handler: Callable[[TaskNode], Awaitable[Any]],
        llm: LLM | None,
    ) -> ExecutionObservation:
        last_error: str | None = None
        repaired_by_llm = False
        for attempt in range(self.max_retries):
            try:
                node.status = "running"
                data = await handler(node)
                node.status = "succeeded"
                return ExecutionObservation(
                    node_id=node.id,
                    tool=node.tool,
                    status="succeeded",
                    data=data,
                    retries=attempt,
                    repaired_by_llm=repaired_by_llm,
                )
            except Exception as exc:  # noqa: BLE001 - converted to method observation.
                last_error = str(exc)
                if llm is not None and attempt < self.max_retries - 1:
                    repaired = await self._repair_node_args(node, last_error, llm)
                    if repaired:
                        repaired_by_llm = True
        node.status = "failed"
        return ExecutionObservation(
            node_id=node.id,
            tool=node.tool,
            status="failed",
            error=last_error,
            retries=self.max_retries - 1,
            repaired_by_llm=repaired_by_llm,
        )

    async def _repair_node_args(
        self,
        node: TaskNode,
        error: str,
        llm: LLM,
    ) -> bool:
        response = await llm.chat(
            [
                Message(
                    role="user",
                    content=(
                        "UrbanAgent 执行 DAG 节点失败。请根据工具 schema 修正当前节点 args。"
                        "只输出 JSON：{\"args\": {...}}，不要 Markdown。\n\n"
                        f"node={json.dumps(_node_to_data(node), ensure_ascii=False)}\n"
                        f"error={error}\n"
                        f"available_tools={json.dumps(self._all_tool_metadata(), ensure_ascii=False)}"
                    ),
                )
            ],
            temperature=0.2,
            max_tokens=600,
            system=(
                "You are UrbanAgent's local retry policy. "
                "Repair tool arguments only; do not change the tool."
            ),
        )
        try:
            data = _try_loads_json_object(response.content)
        except ValueError:
            return False
        args = data.get("args")
        if not isinstance(args, dict):
            return False
        node.args = args
        return True

    async def _execute_external_tool(self, node: TaskNode) -> Any:
        if self._external_facade_active is None:
            raise RuntimeError("external tool facade is not active")
        return await self._external_facade_active.invoke(node.tool, dict(node.args))

    async def _execute_ask_user(self, node: TaskNode) -> dict[str, Any]:
        return {
            "message": "missing information prevents execution",
            "missing": list(node.args.get("missing", [])),
        }

    async def _execute_get_city_state(self, node: TaskNode) -> CityState:
        # Do not return the live sandbox object: later dispatch mutates it and would
        # corrupt this observation when serialized for synthesis.
        return copy.deepcopy(await self.sandbox.get_state())

    async def _execute_create_dispatch_plan(
        self,
        node: TaskNode,
        context: dict[str, Any],
        llm: LLM | None,
    ) -> DispatchPlan:
        state = context.get("observe_city_state") or context.get("get_city_state")
        if not isinstance(state, CityState):
            state = copy.deepcopy(await self.sandbox.get_state())
        candidates = self.dispatch_policy.score_dispatch_candidates(state)
        notes_prefix: list[str] = []
        if (
            self.use_llm
            and llm is not None
            and self.llm_dispatch_ranking
            and candidates
        ):
            try:
                candidates = await self._llm_reorder_dispatch_candidates(
                    state, candidates, llm
                )
            except Exception as exc:
                if self.llm_dispatch_ranking_strict:
                    raise UrbanAgentPipelineError(
                        "execution",
                        f"LLM 调度候选排序失败: {exc}",
                    ) from exc
                notes_prefix.append(
                    f"LLM 调度候选排序失败 ({exc!s})；已回退为代码默认分数排序。"
                )
        plan = self.dispatch_policy.build_plan_from_ordered_candidates(state, candidates)
        if notes_prefix:
            plan.notes = notes_prefix + plan.notes
        return plan

    async def _llm_reorder_dispatch_candidates(
        self,
        state: CityState,
        candidates: list[CandidateScore],
        llm: LLM,
    ) -> list[CandidateScore]:
        rows = _candidate_rows_for_llm(state, candidates)
        task_payload = (
            _task_to_data(self._runtime_task)
            if self._runtime_task is not None
            else {}
        )
        payload = {
            "candidates": rows,
            "task": task_payload,
            "greedy_rules": (
                "后续用确定性贪心：按列表从前到后，依次为每个 (incident_id, role) 选取第一个尚未被占用的 "
                "resource_id；同一 resource 不能派往两个角色。请输出 ranked_candidate_indices："
                "0 到 n-1 的一个排列，越靠前表示贪心时越早尝试该候选。"
            ),
        }
        response = await llm.chat(
            [
                Message(
                    role="user",
                    content=(
                        "你是 UrbanAgent 的调度排序模块。根据候选车辆/无人机与任务 U，"
                        "只输出 JSON，不要 Markdown。字段：ranked_candidate_indices（整数数组，"
                        f"须为 0..{len(candidates) - 1} 的一个全排列）。\n"
                        "可结合 task.constraints 中的软约束调整顺序；不得引入列表外的 resource_id。\n\n"
                        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
                    ),
                )
            ],
            temperature=0.2,
            max_tokens=2048,
            system="Output only strict JSON with ranked_candidate_indices (a permutation).",
        )
        data = await self._coerce_llm_json(
            llm,
            response.content,
            stage="dispatch_ranking",
            schema_hint="ranked_candidate_indices (array of int, permutation of 0..n-1)",
        )
        raw_indices = data.get("ranked_candidate_indices")
        if not isinstance(raw_indices, list):
            raise ValueError("ranked_candidate_indices must be a list")
        reordered = _apply_candidate_index_order(candidates, raw_indices)
        if len(reordered) != len(candidates):
            raise ValueError("reorder lost candidates")
        return reordered

    async def _execute_apply_dispatch_plan(
        self,
        node: TaskNode,
        context: dict[str, Any],
    ) -> list[ActionResult]:
        plan = context.get("build_dispatch_plan") or context.get("create_dispatch_plan")
        if not isinstance(plan, DispatchPlan):
            raise ValueError("dispatch plan is not available")
        results = []
        for assignment in plan.assignments:
            results.append(await self.sandbox.send_action(assignment_to_action(assignment)))
        return results

    async def _execute_control_signal(self, node: TaskNode) -> ActionResult:
        state = await self.sandbox.get_state()
        incident_id = node.args.get("incident_id") or _first_open_incident_id(state)
        incident = _find_incident(state, str(incident_id))
        if incident is None:
            raise ValueError(f"incident not found: {incident_id}")
        signal = _nearest_signal(state.traffic_signals, incident.position)
        if signal is None:
            raise ValueError("no traffic signal available")
        action = UrbanAction(
            kind="control_traffic_light",
            target_id=signal.id,
            parameters={
                "mode": "emergency_preemption",
                "incident_id": incident.id,
            },
            reason=f"Open emergency corridor for {incident.id}.",
        )
        return await self.sandbox.send_action(action)

    async def _execute_mark_incident(self, node: TaskNode) -> ActionResult:
        state = await self.sandbox.get_state()
        incident_id = node.args.get("incident_id") or _first_open_incident_id(state)
        action = UrbanAction(
            kind="mark_incident",
            target_id=str(incident_id),
            parameters={"status": str(node.args.get("status", "responding"))},
            reason="Method pipeline completed initial dispatch actions.",
        )
        return await self.sandbox.send_action(action)

    async def _synthesis(
        self,
        task: UrbanTask,
        observations: list[ExecutionObservation],
        llm: LLM | None,
    ) -> list[DispatchSolution]:
        solutions = self._synthesis_deterministic(task, observations)
        if not self.use_llm or llm is None:
            return solutions
        try:
            return await self._llm_synthesis(task, observations, solutions, llm)
        except UrbanAgentPipelineError:
            raise
        except Exception as exc:
            raise UrbanAgentPipelineError(
                "synthesis",
                "LLM 综合失败：模型返回无效 JSON、缺字段或 API 错误。"
                f" 详情: {exc}",
            ) from exc

    def _synthesis_deterministic(
        self,
        task: UrbanTask,
        observations: list[ExecutionObservation],
    ) -> list[DispatchSolution]:
        plan = _observation_data(
            observations, "build_dispatch_plan", DispatchPlan,
        ) or _observation_tool_data(observations, "create_dispatch_plan", DispatchPlan)
        applied = (
            _observation_data(observations, "apply_dispatch_plan", list)
            or _observation_tool_data(observations, "apply_dispatch_plan", list)
            or []
        )
        signal = _observation_data(
            observations, "control_emergency_signal", ActionResult,
        ) or _observation_tool_data(
            observations, "control_nearest_traffic_signal", ActionResult,
        )
        mark = _observation_data(
            observations, "mark_incident_responding", ActionResult,
        ) or _observation_tool_data(observations, "mark_incident", ActionResult)
        action_results: list[ActionResult] = [
            result for result in applied if isinstance(result, ActionResult)
        ]
        if isinstance(signal, ActionResult):
            action_results.append(signal)
        if isinstance(mark, ActionResult):
            action_results.append(mark)

        if plan is None:
            return [
                DispatchSolution(
                    title="无法生成火灾调度方案",
                    summary="缺少必要事件信息，未进入调度执行。",
                    plan=DispatchPlan(notes=["missing information"]),
                    action_results=action_results,
                    hard_constraints_satisfied=False,
                    score=math.inf,
                    notes=task.missing_information,
                )
            ]

        verified_constraints = self._verify_constraints(task, plan, action_results)
        hard_ok = all(
            constraint.satisfied is not False
            for constraint in verified_constraints
            if constraint.kind == "hard"
        )
        score = _solution_score(plan)
        summary = _solution_summary(plan, action_results, hard_ok)
        candidate_summary = _candidate_summary(plan)
        sandbox_commands = [
            _action_command(result.action)
            for result in action_results
            if result.status == "applied"
        ]
        task.constraints = verified_constraints
        return [
            DispatchSolution(
                title="高严重度火灾初始调度方案",
                summary=summary,
                plan=plan,
                action_results=action_results,
                hard_constraints_satisfied=hard_ok,
                score=score,
                notes=list(plan.notes),
                candidate_summary=candidate_summary,
                sandbox_commands=sandbox_commands,
            )
        ]

    async def _llm_synthesis(
        self,
        task: UrbanTask,
        observations: list[ExecutionObservation],
        solutions: list[DispatchSolution],
        llm: LLM,
    ) -> list[DispatchSolution]:
        """LLM rewrites narrative fields; plan, score, hard flags stay deterministic."""
        if not solutions:
            return solutions
        det = solutions[0]
        payload = {
            "task": _task_to_data(task),
            "observations": [_observation_to_data(o) for o in observations],
            "deterministic_solution": _solution_to_data(det),
            "immutable_fields": (
                "hard_constraints_satisfied, score, plan/assignments, action_results, "
                "sandbox_commands are computed in code; do not contradict them."
            ),
            "output_schema": {
                "title": "string",
                "summary": "string",
                "notes": ["string"],
                "candidate_summary": ["string"],
            },
        }
        response = await llm.chat(
            [
                Message(
                    role="user",
                    content=(
                        "你是 UrbanAgent 的 synthesis（综合）模块。根据 task、observations 与 "
                        "deterministic_solution（已由代码从执行观测算出，视为事实）。\n"
                        "请只输出一个 JSON 对象，不要 Markdown。字段：title, summary, notes(字符串数组), "
                        "candidate_summary(字符串数组)。\n"
                        "硬性规则：\n"
                        "1) 叙述必须与 deterministic_solution 中的 assignments、action_results、"
                        "sandbox_commands、hard_constraints_satisfied、score 一致，不得编造未出现的资源 id、"
                        "路线或状态。\n"
                        "2) candidate_summary 应概括候选/指派要点，可与 deterministic 列表语义等价但允许更清晰；"
                        "条目数量可与原列表相近。\n"
                        "3) notes 可含执行要点与约束说明；不要添加与 facts 矛盾的硬约束判定。\n\n"
                        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
                    ),
                )
            ],
            temperature=0.2,
            max_tokens=4096,
            system=(
                "You are the synthesis module. Output only strict JSON with "
                "title, summary, notes, candidate_summary. Ground every claim in the payload."
            ),
        )
        data = await self._coerce_llm_json(
            llm,
            response.content,
            stage="synthesis",
            schema_hint="title (string), summary (string), notes (array of string), "
            "candidate_summary (array of string)",
        )
        title = str(data.get("title", "")).strip() or det.title
        summary = str(data.get("summary", "")).strip() or det.summary
        raw_notes = data.get("notes", [])
        notes = (
            [str(x).strip() for x in raw_notes if str(x).strip()]
            if isinstance(raw_notes, list)
            else list(det.notes)
        )
        if not notes:
            notes = list(det.notes)
        raw_cand = data.get("candidate_summary", [])
        candidate_summary = (
            [str(x).strip() for x in raw_cand if str(x).strip()]
            if isinstance(raw_cand, list)
            else list(det.candidate_summary)
        )
        if not candidate_summary:
            candidate_summary = list(det.candidate_summary)

        merged = replace(
            det,
            title=title,
            summary=summary,
            notes=notes,
            candidate_summary=candidate_summary,
        )
        return [merged]

    def _verify_constraints(
        self,
        task: UrbanTask,
        plan: DispatchPlan,
        action_results: list[ActionResult],
    ) -> list[UrbanConstraint]:
        roles = {assignment.role for assignment in plan.assignments}
        signal_applied = any(
            result.status == "applied" and result.action.kind == "control_traffic_light"
            for result in action_results
        )
        checks = {
            "fire_suppression_required": "fire_suppression" in roles,
            "aerial_recon_required": "aerial_recon" in roles,
            "police_control_required": "police_control" in roles,
            "reserve_ratio": not any("violate reserve ratio" in note for note in plan.notes),
            "traffic_control_required": signal_applied,
        }
        verified = []
        for constraint in task.constraints:
            verified.append(
                replace(
                    constraint,
                    satisfied=checks.get(constraint.name, constraint.satisfied),
                )
            )
        return verified

    async def _render_report(
        self,
        query: str,
        task: UrbanTask,
        graph: TaskGraph,
        observations: list[ExecutionObservation],
        solutions: list[DispatchSolution],
        llm: LLM | None,
        initial_state: CityState,
    ) -> str:
        deterministic_report = self._render_deterministic_report(
            task, graph, observations, solutions,
        )
        if llm is None:
            return deterministic_report
        try:
            llm_report = await self._llm_report(
                query,
                task,
                graph,
                observations,
                solutions,
                deterministic_report,
                llm,
                initial_state,
            )
        except Exception:
            return deterministic_report
        if solutions:
            solutions[0].report = llm_report
        return llm_report

    def _render_deterministic_report(
        self,
        task: UrbanTask,
        graph: TaskGraph,
        observations: list[ExecutionObservation],
        solutions: list[DispatchSolution],
    ) -> str:
        lines = [
            "UrbanAgent 火灾调度结果",
            "",
            "1. Cognition / 认知解析",
            f"- Intent: {task.intent}",
            f"- Source: {task.source}",
            f"- Rationale: {task.rationale or 'n/a'}",
            f"- Entities: {task.entities}",
            "- Constraints: "
            + "; ".join(
                f"{constraint.name}={constraint.satisfied}"
                for constraint in task.constraints
            ),
            "",
            "2. Planning / DAG 任务图",
            f"- Source: {graph.source}",
            f"- Rationale: {graph.rationale or 'n/a'}",
        ]
        for node in graph.nodes:
            deps = ",".join(node.depends_on) if node.depends_on else "none"
            lines.append(f"- {node.id}: {node.tool}, depends_on={deps}, status={node.status}")

        lines.extend(["", "3. Execution / 工具观测"])
        for observation in observations:
            lines.append(
                f"- {observation.node_id}: {observation.status}, "
                f"tool={observation.tool}, retries={observation.retries}, "
                f"repaired_by_llm={observation.repaired_by_llm}"
            )
            if observation.error:
                lines.append(f"  error: {observation.error}")

        lines.extend(["", "4. Synthesis / 决策方案"])
        for solution in solutions:
            lines.append(f"- {solution.title}: {solution.summary}")
            lines.append(
                f"  hard_constraints_satisfied={solution.hard_constraints_satisfied}, "
                f"score={solution.score:.2f}"
            )
            for assignment in solution.plan.assignments:
                route = assignment.score.route
                path = "->".join(route.path) if route.path else "direct"
                lines.append(
                    "  "
                    f"{assignment.role}: {assignment.resource_id} -> "
                    f"{assignment.incident_id}, ETA={route.travel_time:.2f}, "
                    f"distance={route.distance:.2f}, route={path}"
                )
            for result in solution.action_results:
                lines.append(f"  action: {result.status} - {result.message}")
            if solution.sandbox_commands:
                lines.append("  sandbox_commands:")
                for command in solution.sandbox_commands:
                    lines.append(f"    - {command}")
        return "\n".join(lines)

    async def _llm_report(
        self,
        query: str,
        task: UrbanTask,
        graph: TaskGraph,
        observations: list[ExecutionObservation],
        solutions: list[DispatchSolution],
        deterministic_report: str,
        llm: LLM,
        initial_state: CityState,
    ) -> str:
        facts = {
            "query": query,
            "initial_state_summary": _city_state_brief_for_cognition(initial_state),
            "task": _task_to_data(task),
            "dag": [_node_to_data(node) for node in graph.nodes],
            "observations": [_observation_to_data(item) for item in observations],
            "solutions": [_solution_to_data(item) for item in solutions],
            "provenance": {
                "cognition_source": task.source,
                "planning_source": graph.source,
            },
            "algorithm_sandbox_relation": {
                "algorithm_layer": "UrbanAgent cognition/planning/dispatch/synthesis",
                "adapter_layer": "SandboxClient",
                "sandbox_state_input": "CityState",
                "sandbox_command_output": "UrbanAction",
                "sandbox_feedback": "ActionResult",
            },
        }
        response = await llm.chat(
            [
                Message(
                    role="user",
                    content=(
                        "请基于下面 JSON 「facts」与随后的「确定性报告」生成中文火灾调度决策报告。\n"
                        "硬性写作规则（违反视为错误输出）：\n"
                        "1) 只使用 facts 与确定性报告中已出现的信息；禁止推测、补充或编造事件位置详情、"
                        "资源履历、未给出的 ETA/路线、工具名、DAG 节点 id、约束名称与条款。\n"
                        "2) 认知段的约束列表必须与 facts.task.constraints 一致（含 name/kind/expression/satisfied），"
                        "不得新增硬/软约束；满足状态已给出时禁止用「待校验」代替 facts 中的 satisfied。\n"
                        "3) DAG 段：节点、依赖与工具名以 facts.dag 为准；执行结果以 facts.observations 为准。\n"
                        "4) 资源选择与数值：以 facts.solutions（及确定性报告中的 assignments / action_results）"
                        "为准；必要时注明 cognition/planning 来源 facts.provenance。\n"
                        "5) initial_state_summary 是 run 起步时的沙盘快照；get_city_state 的 observation.data "
                        "是该工具执行时刻的快照（通常与前者一致）。执行后实体状态只能依据 solutions、"
                        "action_results 及后续 observation，不得混用。认知解析中事件的「初始」状态必须来自 "
                        "initial_state_summary.incidents，不得写成执行后状态。\n"
                        "6) 资源在调度前的 status 以 initial_state_summary.resources 与 get_city_state 快照为准；"
                        "勿根据后续 apply 推断「get_state 时已是 dispatched」。\n"
                        "7) 算法层与 3D 沙盘的数据交换只描述 facts.algorithm_sandbox_relation 中的对象，"
                        "勿发明额外协议字段。\n"
                        "结构建议：认知解析、算法-沙盘交互、DAG 执行、资源选择理由、约束校验、沙盘动作、结论。\n\n"
                        f"{json.dumps(facts, ensure_ascii=False, indent=2)}\n\n"
                        "确定性报告（与 facts 一致时可直接引用其中数值）：\n"
                        f"{deterministic_report}"
                    ),
                )
            ],
            temperature=0.2,
            max_tokens=3500,
            system=(
                "You are the synthesis module of UrbanAgent. "
                "Write an operational emergency-dispatch report in Chinese. "
                "Ground every claim in the provided JSON facts or deterministic report; "
                "if something is not stated there, say it is not in the record—do not guess."
            ),
        )
        return response.content.strip() or deterministic_report


def _extract_incident_id(query: str) -> str | None:
    match = re.search(r"incident-[a-zA-Z]+-\d+", query)
    return match.group(0) if match else None


def _extract_severity(query: str) -> str:
    if any(token in query for token in ["critical", "重大", "极高", "严重"]):
        return "critical"
    if any(token in query for token in ["high", "高"]):
        return "high"
    if any(token in query for token in ["low", "低"]):
        return "low"
    return "medium"


def _find_incident(state: CityState, incident_id: str | None) -> Incident | None:
    if incident_id is None:
        return None
    return next((incident for incident in state.incidents if incident.id == incident_id), None)


def _first_open_incident_id(state: CityState) -> str | None:
    incident = next(
        (
            item
            for item in state.incidents
            if item.status in {"open", "responding"}
        ),
        None,
    )
    return incident.id if incident is not None else None


def _nearest_signal(
    signals: list[TrafficSignal],
    position: Coordinate,
) -> TrafficSignal | None:
    if not signals:
        return None
    return min(signals, key=lambda signal: _distance(signal.position, position))


def _distance(left: Coordinate, right: Coordinate) -> float:
    return math.sqrt(
        (left.x - right.x) ** 2
        + (left.y - right.y) ** 2
        + (left.z - right.z) ** 2
    )


def _apply_candidate_index_order(
    candidates: list[CandidateScore],
    ranked_indices: list[Any],
) -> list[CandidateScore]:
    n = len(candidates)
    seen: set[int] = set()
    out: list[CandidateScore] = []
    for raw in ranked_indices:
        try:
            i = int(raw)
        except (TypeError, ValueError):
            continue
        if 0 <= i < n and i not in seen:
            out.append(candidates[i])
            seen.add(i)
    for i, c in enumerate(candidates):
        if i not in seen:
            out.append(c)
    return out


def _candidate_rows_for_llm(
    state: CityState,
    candidates: list[CandidateScore],
) -> list[dict[str, Any]]:
    by_id = {r.id: r for r in state.resources}
    rows: list[dict[str, Any]] = []
    for i, c in enumerate(candidates):
        r = by_id.get(c.resource_id)
        rows.append(
            {
                "index": i,
                "incident_id": c.incident_id,
                "role": c.role,
                "resource_id": c.resource_id,
                "code_score": round(c.score, 4),
                "eta": round(c.response_time, 4),
                "route_source": c.route.source,
                "capabilities": list(r.capabilities) if r else [],
                "resource_kind": r.kind if r else None,
            }
        )
    return rows


def _observation_data(
    observations: list[ExecutionObservation],
    node_id: str,
    expected_type: type,
) -> Any:
    for observation in observations:
        if observation.node_id == node_id and isinstance(observation.data, expected_type):
            return observation.data
    return None


def _observation_tool_data(
    observations: list[ExecutionObservation],
    tool: str,
    expected_type: type,
) -> Any:
    for observation in observations:
        if observation.tool == tool and isinstance(observation.data, expected_type):
            return observation.data
    return None


def _solution_score(plan: DispatchPlan) -> float:
    if not plan.assignments:
        return math.inf
    return sum(assignment.score.score for assignment in plan.assignments)


def _solution_summary(
    plan: DispatchPlan,
    action_results: list[ActionResult],
    hard_ok: bool,
) -> str:
    dispatched = [
        result.action.target_id
        for result in action_results
        if result.status == "applied"
        and result.action.kind in {"dispatch_vehicle", "dispatch_drone"}
    ]
    signal = next(
        (
            result.action.target_id
            for result in action_results
            if result.status == "applied"
            and result.action.kind == "control_traffic_light"
        ),
        "未控制",
    )
    status = "满足硬约束" if hard_ok else "存在硬约束未满足"
    return (
        f"调派 {', '.join(dispatched) if dispatched else '无资源'}，"
        f"交通信号 {signal} 已进入应急优先；{status}。"
    )


def _try_loads_json_object(text: str) -> dict[str, Any]:
    """Extract and parse a JSON object; retry with common LLM syntax fixes."""
    last_err: str | None = None
    for variant in _iter_json_object_candidates(text):
        try:
            data = json.loads(variant)
        except json.JSONDecodeError as exc:
            last_err = str(exc)
            continue
        if isinstance(data, dict):
            return data
        last_err = "top-level JSON is not an object"
    raise ValueError(last_err or "no JSON object found")


def _loads_json_object(text: str) -> dict[str, Any]:
    """Backwards-compatible name for strict JSON object parsing."""
    return _try_loads_json_object(text)


def _iter_json_object_candidates(text: str) -> list[str]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    out: list[str] = []
    seen: set[str] = set()

    def add(s: str) -> None:
        if s not in seen:
            seen.add(s)
            out.append(s)

    if start >= 0 and end >= start:
        core = cleaned[start : end + 1]
        add(core)
        loosened = core
        for _ in range(16):
            nxt = re.sub(r",(\s*[}\]])", r"\1", loosened)
            if nxt == loosened:
                break
            loosened = nxt
            add(loosened)
    add(cleaned)
    return out


def _city_state_brief_for_cognition(state: CityState) -> dict[str, Any]:
    """Smaller prompt for cognition to reduce copy-paste errors in model JSON output."""
    return {
        "timestamp": state.timestamp,
        "incidents": [
            {
                "id": incident.id,
                "kind": incident.kind,
                "severity": incident.severity,
                "status": incident.status,
                "position": _coord_data(incident.position),
                "description": incident.description,
            }
            for incident in state.incidents
        ],
        "resources": [
            {
                "id": resource.id,
                "kind": resource.kind,
                "status": resource.status,
                "capabilities": resource.capabilities,
            }
            for resource in state.resources
        ],
        "traffic_signal_ids": [signal.id for signal in state.traffic_signals],
        "roads_count": len(state.roads),
        "note": "Full CityState including roads is obtained via get_city_state during execution.",
    }


def _city_state_summary(state: CityState) -> dict[str, Any]:
    return {
        "timestamp": state.timestamp,
        "incidents": [
            {
                "id": incident.id,
                "kind": incident.kind,
                "severity": incident.severity,
                "status": incident.status,
                "position": _coord_data(incident.position),
                "description": incident.description,
            }
            for incident in state.incidents
        ],
        "resources": [
            {
                "id": resource.id,
                "kind": resource.kind,
                "status": resource.status,
                "position": _coord_data(resource.position),
                "home_base_id": resource.home_base_id,
                "speed": resource.speed,
                "water_remaining": resource.water_remaining,
                "battery_remaining": resource.battery_remaining,
                "capabilities": resource.capabilities,
            }
            for resource in state.resources
        ],
        "traffic_signals": [
            {
                "id": signal.id,
                "mode": signal.mode,
                "status": signal.status,
                "position": _coord_data(signal.position),
            }
            for signal in state.traffic_signals
        ],
        "roads": [
            {
                "id": road.id,
                "from_node": road.from_node,
                "to_node": road.to_node,
                "congestion": road.congestion,
                "blocked": road.blocked,
                "allowed_resource_kinds": road.allowed_resource_kinds,
            }
            for road in state.roads
        ],
    }


def _task_to_data(task: UrbanTask) -> dict[str, Any]:
    return {
        "intent": task.intent,
        "entities": _plain(task.entities),
        "constraints": [
            {
                "name": constraint.name,
                "kind": constraint.kind,
                "expression": constraint.expression,
                "satisfied": constraint.satisfied,
            }
            for constraint in task.constraints
        ],
        "missing_information": task.missing_information,
        "rationale": task.rationale,
        "source": task.source,
    }


def _node_to_data(node: TaskNode) -> dict[str, Any]:
    return {
        "id": node.id,
        "description": node.description,
        "tool": node.tool,
        "args": _plain(node.args),
        "depends_on": node.depends_on,
        "status": node.status,
    }


def _observation_to_data(observation: ExecutionObservation) -> dict[str, Any]:
    return {
        "node_id": observation.node_id,
        "tool": observation.tool,
        "status": observation.status,
        "error": observation.error,
        "retries": observation.retries,
        "repaired_by_llm": observation.repaired_by_llm,
        "data": _plain(observation.data),
    }


def _solution_to_data(solution: DispatchSolution) -> dict[str, Any]:
    return {
        "title": solution.title,
        "summary": solution.summary,
        "hard_constraints_satisfied": solution.hard_constraints_satisfied,
        "score": solution.score,
        "notes": solution.notes,
        "candidate_summary": solution.candidate_summary,
        "assignments": [
            {
                "incident_id": assignment.incident_id,
                "resource_id": assignment.resource_id,
                "role": assignment.role,
                "action_kind": assignment.action_kind,
                "score": assignment.score.score,
                "eta": assignment.score.route.travel_time,
                "distance": assignment.score.route.distance,
                "route": assignment.score.route.path,
                "reason": assignment.reason,
            }
            for assignment in solution.plan.assignments
        ],
        "sandbox_commands": solution.sandbox_commands,
        "action_results": [
            {
                "status": result.status,
                "message": result.message,
                "action": _action_command(result.action),
            }
            for result in solution.action_results
        ],
    }


def _candidate_summary(plan: DispatchPlan) -> list[str]:
    rows = []
    for candidate in plan.candidate_scores[:8]:
        rows.append(
            f"{candidate.role}:{candidate.resource_id} "
            f"score={candidate.score:.2f} eta={candidate.route.travel_time:.2f} "
            f"route={candidate.route.source}"
        )
    return rows


def _action_command(action: UrbanAction) -> dict[str, Any]:
    return {
        "kind": action.kind,
        "target_id": action.target_id,
        "destination": _coord_data(action.destination) if action.destination else None,
        "parameters": _plain(action.parameters),
        "reason": action.reason,
    }


def _coord_data(coord: Coordinate) -> dict[str, float]:
    return {"x": coord.x, "y": coord.y, "z": coord.z}


def _builtin_env_operation_metadata() -> list[dict[str, Any]]:
    return [
        {
            "name": "ask_user",
            "description": "Stop execution and request missing information from the user.",
            "args_schema": {"missing": ["string"]},
            "returns": "clarification request observation",
        },
        {
            "name": "get_city_state",
            "description": "Read current 3D sandbox city state W.",
            "args_schema": {},
            "returns": "CityState",
        },
        {
            "name": "create_dispatch_plan",
            "description": (
                "Use deterministic dispatch policy to score available fire trucks, "
                "police cars, and drones. Returns DispatchPlan."
            ),
            "args_schema": {"incident_id": "string"},
            "returns": "DispatchPlan",
        },
        {
            "name": "apply_dispatch_plan",
            "description": (
                "Convert DispatchPlan assignments into UrbanAction commands and send "
                "them through SandboxClient."
            ),
            "args_schema": {},
            "returns": "list[ActionResult]",
        },
        {
            "name": "control_nearest_traffic_signal",
            "description": (
                "Find nearest traffic signal for an incident and set emergency "
                "preemption through SandboxClient."
            ),
            "args_schema": {"incident_id": "string"},
            "returns": "ActionResult",
        },
        {
            "name": "mark_incident",
            "description": "Update incident lifecycle status in the sandbox.",
            "args_schema": {"incident_id": "string", "status": "string"},
            "returns": "ActionResult",
        },
    ]


def _plain(value: Any) -> Any:
    if isinstance(value, Coordinate):
        return _coord_data(value)
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: _plain(getattr(value, key))
            for key in getattr(value, "__dataclass_fields__")
        }
    return value
