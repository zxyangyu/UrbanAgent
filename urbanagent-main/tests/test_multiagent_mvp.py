"""Multi-agent MVP tests (no LLM / no network)."""
from __future__ import annotations

import unittest

from urbanagent import UrbanMultiAgentSystem
from urbanagent.dispatch import DispatchPolicy
from urbanagent.multiagent.subagents.default_agents import integrate_actions_deterministic
from urbanagent.sandbox import MockSandboxClient
from urbanagent.types import CityState, Coordinate, Incident, TrafficSignal, UrbanResource


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
        empty = CityState(incidents=[], resources=[], timestamp="t0")
        sys = UrbanMultiAgentSystem(sandbox=MockSandboxClient(empty), use_llm=False)
        r = await sys.run("hello world no emergency")
        self.assertFalse(r.gate.should_intervene)

    async def test_no_dispatchable_resources_degrades_hard(self) -> None:
        state = CityState(
            timestamp="t0",
            incidents=[
                Incident(
                    id="incident-fire-001",
                    kind="fire",
                    severity="high",
                    position=Coordinate(10, 10, 0),
                )
            ],
            resources=[
                UrbanResource(
                    id="fire-low",
                    kind="fire_truck",
                    position=Coordinate(0, 0, 0),
                    battery_remaining=1.0,
                    water_remaining=100.0,
                    capabilities=["fire_suppression"],
                ),
                UrbanResource(
                    id="drone-low",
                    kind="drone",
                    position=Coordinate(0, 0, 10),
                    battery_remaining=1.0,
                    capabilities=["aerial_recon"],
                ),
                UrbanResource(
                    id="ugv-low",
                    kind="unmanned_vehicle",
                    position=Coordinate(0, 0, 0),
                    battery_remaining=1.0,
                    capabilities=["logistics_support"],
                ),
                UrbanResource(
                    id="police-low",
                    kind="police_car",
                    position=Coordinate(0, 0, 0),
                    battery_remaining=1.0,
                    capabilities=["traffic_control"],
                ),
            ],
            traffic_signals=[
                TrafficSignal(id="signal-01", position=Coordinate(9, 9, 0)),
            ],
        )
        sys = UrbanMultiAgentSystem(
            sandbox=MockSandboxClient(state),
            use_llm=False,
            use_llm_batch_rerank=False,
        )
        r = await sys.run("incident-fire-001 高严重度火情")
        self.assertIsNotNone(r.committed)
        self.assertEqual(r.committed.actions, [])
        self.assertIn("no dispatchable mobile resources", r.committed.rationale)
        self.assertIsNotNone(r.batch_outcome)
        self.assertFalse(r.batch_outcome.criteria_satisfied)
        self.assertIn("no executable actions", " ".join(r.batch_outcome.notes))

    async def test_carla_bridge_fire_ugv_gets_goto_then_extinguish(self) -> None:
        state = CityState(
            timestamp="t0",
            incidents=[
                Incident(
                    id="incident-fire-001",
                    kind="fire",
                    severity="high",
                    position=Coordinate(10, 10, 0),
                )
            ],
            resources=[
                UrbanResource(
                    id="UGV-01",
                    kind="unmanned_vehicle",
                    position=Coordinate(0, 0, 0),
                    capabilities=["fire_suppression", "logistics_support"],
                ),
                UrbanResource(
                    id="UAV-01",
                    kind="drone",
                    position=Coordinate(0, 0, 10),
                    battery_remaining=100.0,
                    capabilities=["aerial_recon"],
                ),
            ],
        )
        actions, rationale = integrate_actions_deterministic(
            [],
            state=state,
            dispatch_policy=DispatchPolicy(),
        )
        self.assertNotIn("no executable", rationale)
        ugv_actions = [a for a in actions if a.target_id == "UGV-01"]
        self.assertEqual(len(ugv_actions), 2)
        self.assertTrue(
            all(a.parameters.get("intent") == "extinguish" for a in ugv_actions)
        )
        self.assertFalse(ugv_actions[0].parameters.get("force_extinguish", False))
        self.assertTrue(ugv_actions[1].parameters.get("force_extinguish"))


if __name__ == "__main__":
    unittest.main()
