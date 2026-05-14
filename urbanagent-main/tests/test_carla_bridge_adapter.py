"""Tests for CarlaBridge protocol adapter helpers."""
from __future__ import annotations

import unittest

from urbanagent.carla_bridge import carla_snapshot_to_city_state
from urbanagent.multiagent.batch_runner import batch_criteria_met
from urbanagent.types import ActionResult, Coordinate, Incident, UrbanAction


class CarlaBridgeAdapterTest(unittest.TestCase):
    def test_snapshot_maps_bridge_entities(self) -> None:
        state = carla_snapshot_to_city_state(
            {
                "traffic_lights": [
                    {"id": "TL-01", "state": "green", "position": {"x": 1, "y": 2, "z": 0}},
                ],
                "vehicles": [
                    {
                        "id": "UGV-01",
                        "role": "dispatchable",
                        "position": {"x": 3, "y": 4, "z": 0},
                        "state": "moving",
                        "speed": 2.0,
                        "battery": 88,
                    }
                ],
                "uavs": [
                    {
                        "id": "UAV-01",
                        "position": {"x": 5, "y": 6, "z": 10},
                        "state": "hover",
                        "battery": 70,
                    }
                ],
            },
            default_incidents=[
                Incident(
                    id="incident-fire-001",
                    kind="fire",
                    position=Coordinate(10, 10, 0),
                    severity="high",
                )
            ],
        )

        self.assertEqual(state.resources[0].kind, "unmanned_vehicle")
        self.assertEqual(state.resources[0].status, "dispatched")
        self.assertEqual(state.resources[1].kind, "drone")
        self.assertEqual(state.traffic_signals[0].mode, "green")
        self.assertEqual(state.incidents[0].id, "incident-fire-001")

    def test_batch_criteria_accepts_carla_ack_plus_snapshot(self) -> None:
        action = UrbanAction(
            kind="dispatch_vehicle",
            target_id="UGV-01",
            destination=Coordinate(10, 10, 0),
            parameters={"incident_id": "incident-fire-001"},
        )
        state = carla_snapshot_to_city_state(
            {
                "vehicles": [
                    {
                        "id": "UGV-01",
                        "position": {"x": 8, "y": 8, "z": 0},
                        "state": "moving",
                    }
                ]
            }
        )
        result = ActionResult(status="accepted", action=action, message="queued")
        self.assertTrue(batch_criteria_met(state, [action], [result]))


if __name__ == "__main__":
    unittest.main()
