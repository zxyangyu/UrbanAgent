"""UrbanAgent multi-agent MVP pipeline (meta + subagents + batch execution)."""
from __future__ import annotations

import copy
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
from urbanagent.multiagent.schemas import SubAgentRole, UrbanMultiAgentResult
from urbanagent.multiagent.toolkit import SubAgentToolkit
from urbanagent.multiagent.world_cache import WorldStateCache
from urbanagent.sandbox import MockSandboxClient, SandboxClient


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
