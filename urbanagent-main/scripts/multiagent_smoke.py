"""Smoke test: UrbanMultiAgentSystem without LLM."""
from __future__ import annotations

import asyncio

from urbanagent import UrbanMultiAgentSystem


async def main() -> None:
    sys = UrbanMultiAgentSystem(use_llm=False, use_llm_batch_rerank=False)
    r = await sys.run("incident-fire-001 高严重度火情")
    assert r.gate.should_intervene
    assert r.committed is not None
    assert r.batch_outcome is not None
    assert r.batch_outcome.criteria_satisfied, r.batch_outcome.notes
    print("ok", len(r.committed.actions), "actions")


if __name__ == "__main__":
    asyncio.run(main())
