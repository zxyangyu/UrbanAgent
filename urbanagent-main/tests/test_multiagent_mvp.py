"""Multi-agent MVP tests (no LLM / no network)."""
from __future__ import annotations

import unittest

from urbanagent import UrbanMultiAgentSystem


class MultiAgentMVPTest(unittest.IsolatedAsyncioTestCase):
    async def test_run_no_llm_succeeds(self) -> None:
        sys = UrbanMultiAgentSystem(use_llm=False, use_llm_batch_rerank=False)
        r = await sys.run("incident-fire-001 高严重度火情")
        self.assertTrue(r.gate.should_intervene)
        self.assertIsNotNone(r.committed)
        self.assertIsNotNone(r.batch_outcome)
        self.assertTrue(r.batch_outcome.criteria_satisfied, msg=str(r.batch_outcome.notes))
        self.assertGreater(len(r.committed.actions), 0)

    async def test_gate_skips_when_no_trigger(self) -> None:
        from urbanagent.sandbox import MockSandboxClient
        from urbanagent.types import CityState

        empty = CityState(incidents=[], resources=[], timestamp="t0")
        sys = UrbanMultiAgentSystem(sandbox=MockSandboxClient(empty), use_llm=False)
        r = await sys.run("hello world no emergency")
        self.assertFalse(r.gate.should_intervene)


if __name__ == "__main__":
    unittest.main()
