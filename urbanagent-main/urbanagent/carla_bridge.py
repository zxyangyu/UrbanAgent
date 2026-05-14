"""Socket.IO adapter for CarlaBridge Urban Agent protocol v1.1.

This client connects UrbanAgent to the middleware, not directly to CARLA.
It implements :class:`urbanagent.sandbox.SandboxClient` so existing single-agent
and multi-agent pipelines can use it as their sandbox adapter.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Mapping
from typing import Any

from urbanagent.errors import SandboxWireError
from urbanagent.sandbox import SandboxClient
from urbanagent.types import (
    ActionResult,
    CityState,
    Coordinate,
    Incident,
    TrafficSignal,
    UrbanAction,
    UrbanResource,
)


DEFAULT_ACTION_MAP = {
    "dispatch_drone": "UAV_DISPATCH",
    "control_traffic_light": "TL_SET_STATE",
    "mark_incident": "MARK_EVENT",
}


class CarlaBridgeSandboxClient(SandboxClient):
    """CarlaBridge Socket.IO v4 `/agent` client.

    The bridge pushes `state.snapshot`; `get_state()` returns the latest cached
    snapshot. `send_action()` emits `agent_command` and waits for `agent_ack` or
    `agent_reject`; physical completion is confirmed later by polling snapshots.
    """

    def __init__(
        self,
        url: str,
        *,
        namespace: str = "/agent",
        agent_id: str = "urban_agent_v1",
        capabilities: list[str] | None = None,
        action_map: Mapping[str, str] | None = None,
        connect_timeout: float = 30.0,
        ack_timeout: float = 10.0,
        state_timeout: float = 30.0,
        heartbeat_interval: float = 1.0,
        default_incidents: list[Incident] | None = None,
    ) -> None:
        self.url = url
        self.namespace = namespace
        self.agent_id = agent_id
        self.capabilities = capabilities or [
            "ugv_control",
            "uav_control",
            "traffic_light_control",
        ]
        self.action_map = dict(DEFAULT_ACTION_MAP)
        if action_map:
            self.action_map.update(action_map)
        self.connect_timeout = connect_timeout
        self.ack_timeout = ack_timeout
        self.state_timeout = state_timeout
        self.heartbeat_interval = heartbeat_interval
        self.default_incidents = list(default_incidents or [])

        self._sio: Any | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._connected = asyncio.Event()
        self._state_event = asyncio.Event()
        self._latest_state: CityState | None = None
        self._last_frame: int | None = None
        self._last_sim_time: float | None = None
        self._pending: dict[str, asyncio.Future[ActionResult]] = {}
        self._pending_actions: dict[str, UrbanAction] = {}
        self._events: list[dict[str, Any]] = []
        self._suggestions: list[dict[str, Any]] = []

    @property
    def event_logs(self) -> list[dict[str, Any]]:
        return list(self._events)

    @property
    def suggestions(self) -> list[dict[str, Any]]:
        return list(self._suggestions)

    async def connect(self) -> None:
        if self._sio is not None and self._sio.connected:
            return
        try:
            import socketio
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "CarlaBridgeSandboxClient requires python-socketio[client]. "
                "Run `pip install -e .` after updating dependencies.",
            ) from exc

        self._connected.clear()
        self._sio = socketio.AsyncClient(reconnection=True)
        self._register_handlers()
        await self._sio.connect(
            self.url,
            namespaces=[self.namespace],
            wait_timeout=self.connect_timeout,
        )
        await asyncio.wait_for(self._connected.wait(), timeout=self.connect_timeout)
        await self._emit_envelope(
            "hello",
            {
                "agent_id": self.agent_id,
                "capabilities": list(self.capabilities),
            },
        )
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def close(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        if self._sio is not None and self._sio.connected:
            await self._sio.disconnect()
        self._sio = None
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(SandboxWireError("CarlaBridge connection closed"))
        self._pending.clear()
        self._pending_actions.clear()

    async def __aenter__(self) -> CarlaBridgeSandboxClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def get_state(self) -> CityState:
        await self.connect()
        if self._latest_state is not None:
            return self._latest_state
        await asyncio.wait_for(self._state_event.wait(), timeout=self.state_timeout)
        if self._latest_state is None:
            raise SandboxWireError("CarlaBridge did not provide a state snapshot")
        return self._latest_state

    async def send_action(self, action: UrbanAction) -> ActionResult:
        await self.connect()
        msg_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[ActionResult] = loop.create_future()
        self._pending[msg_id] = fut
        self._pending_actions[msg_id] = action
        payload = self._action_to_agent_command(action)
        try:
            await self._emit_envelope("agent_command", payload, msg_id=msg_id)
            return await asyncio.wait_for(fut, timeout=self.ack_timeout)
        except asyncio.TimeoutError as exc:
            raise SandboxWireError(
                f"CarlaBridge did not ack command within {self.ack_timeout}s "
                f"(cmd_id={msg_id})",
            ) from exc
        finally:
            self._pending.pop(msg_id, None)
            self._pending_actions.pop(msg_id, None)

    async def send_event_log(self, message: str, *, severity: str = "info") -> None:
        await self.connect()
        await self._emit_envelope("event_log", {"severity": severity, "message": message})

    def _register_handlers(self) -> None:
        assert self._sio is not None

        @self._sio.event(namespace=self.namespace)
        async def connect() -> None:  # type: ignore[no-redef]
            self._connected.set()

        @self._sio.event(namespace=self.namespace)
        async def disconnect() -> None:  # type: ignore[no-redef]
            self._connected.clear()

        @self._sio.on("state.snapshot", namespace=self.namespace)
        async def state_snapshot(data: Any) -> None:
            env = _as_envelope("state.snapshot", data)
            self._last_frame = _maybe_int(env.get("frame"))
            self._last_sim_time = _maybe_float(env.get("sim_time"))
            self._latest_state = carla_snapshot_to_city_state(
                dict(env.get("payload") or {}),
                timestamp=str(env.get("timestamp", time.time())),
                default_incidents=self.default_incidents,
            )
            self._state_event.set()

        @self._sio.on("suggestion", namespace=self.namespace)
        async def suggestion(data: Any) -> None:
            env = _as_envelope("suggestion", data)
            self._suggestions.append(dict(env.get("payload") or {}))

        @self._sio.on("agent_ack", namespace=self.namespace)
        async def agent_ack(data: Any) -> None:
            env = _as_envelope("agent_ack", data)
            payload = dict(env.get("payload") or {})
            cmd_id = str(payload.get("cmd_id", ""))
            fut = self._pending.get(cmd_id)
            if fut is not None and not fut.done():
                fut.set_result(
                    ActionResult(
                        status="accepted",
                        action=self._pending_actions.get(cmd_id, _pending_action(payload)),
                        message=str(payload.get("comment", "queued")),
                    )
                )

        @self._sio.on("agent_reject", namespace=self.namespace)
        async def agent_reject(data: Any) -> None:
            env = _as_envelope("agent_reject", data)
            payload = dict(env.get("payload") or {})
            cmd_id = str(payload.get("cmd_id", ""))
            err = payload.get("error") if isinstance(payload.get("error"), dict) else {}
            fut = self._pending.get(cmd_id)
            if fut is not None and not fut.done():
                fut.set_result(
                    ActionResult(
                        status="rejected",
                        action=self._pending_actions.get(cmd_id, _pending_action(payload)),
                        message=str(err.get("message") or payload.get("status") or "rejected"),
                    )
                )

        @self._sio.on("event_log", namespace=self.namespace)
        async def event_log(data: Any) -> None:
            env = _as_envelope("event_log", data)
            self._events.append(dict(env.get("payload") or {}))

        @self._sio.on("pong", namespace=self.namespace)
        async def pong(data: Any) -> None:
            return None

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            if self._sio is not None and self._sio.connected:
                await self._emit_envelope("ping", {})

    async def _emit_envelope(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        msg_id: str | None = None,
    ) -> str:
        if self._sio is None:
            raise SandboxWireError("CarlaBridge Socket.IO client is not connected")
        mid = msg_id or str(uuid.uuid4())
        env = {
            "version": "1.0",
            "msg_id": mid,
            "type": event_type,
            "timestamp": time.time(),
            "frame": self._last_frame,
            "sim_time": self._last_sim_time,
            "sender": "agent",
            "payload": payload,
        }
        await self._sio.emit(event_type, env, namespace=self.namespace)
        return mid

    def _action_to_agent_command(self, action: UrbanAction) -> dict[str, Any]:
        bridge_action = self._bridge_action_name(action)
        params: dict[str, Any] = dict(action.parameters)
        if action.destination is not None:
            params.setdefault("position", _coord_data(action.destination))
        if action.kind == "control_traffic_light":
            mode = str(params.get("mode", "emergency_preemption"))
            params["state"] = "green" if mode == "emergency_preemption" else mode
        if action.kind == "mark_incident":
            params.setdefault("status", str(action.parameters.get("status", "responding")))
        return {
            "target": "" if action.kind == "mark_incident" else action.target_id,
            "action": bridge_action,
            "priority": str(action.parameters.get("priority", "normal")),
            "related_suggestion": action.parameters.get("related_suggestion"),
            "params": params,
        }

    def _bridge_action_name(self, action: UrbanAction) -> str:
        if action.kind == "dispatch_vehicle":
            target = action.target_id.upper()
            if target.startswith("UGV"):
                return "UGV_DISPATCH"
            if "POLICE" in target or target.startswith("POL"):
                return "POLICE_DISPATCH"
            return "VEHICLE_DISPATCH"
        return self.action_map.get(action.kind, action.kind.upper())


def carla_snapshot_to_city_state(
    payload: dict[str, Any],
    *,
    timestamp: str = "",
    default_incidents: list[Incident] | None = None,
) -> CityState:
    resources: list[UrbanResource] = []
    for raw in payload.get("vehicles") or []:
        if isinstance(raw, dict):
            resources.append(_vehicle_resource(raw))
    for raw in payload.get("uavs") or []:
        if isinstance(raw, dict):
            resources.append(_uav_resource(raw))
    signals = [
        _traffic_signal(raw)
        for raw in (payload.get("traffic_lights") or [])
        if isinstance(raw, dict)
    ]
    incidents = [
        _incident(raw)
        for raw in (payload.get("incidents") or [])
        if isinstance(raw, dict)
    ]
    if not incidents and default_incidents:
        incidents = list(default_incidents)
    return CityState(
        timestamp=timestamp,
        incidents=incidents,
        resources=resources,
        traffic_signals=signals,
    )


def _vehicle_resource(raw: dict[str, Any]) -> UrbanResource:
    rid = str(raw.get("id", ""))
    role = str(raw.get("role", "")).lower()
    upper = rid.upper()
    if "POLICE" in role or "POLICE" in upper or upper.startswith("POL"):
        kind = "police_car"
        caps = ["traffic_control", "perimeter_control"]
    elif upper.startswith("UGV"):
        kind = "unmanned_vehicle"
        caps = ["logistics_support", "perimeter_support"]
    else:
        kind = "ground_vehicle"
        caps = ["ground_mobility"]
    return UrbanResource(
        id=rid,
        kind=kind,  # type: ignore[arg-type]
        position=_coord_required(raw.get("position")),
        status=_resource_status(str(raw.get("state", "idle"))),
        speed=float(raw.get("speed", 1.0) or 1.0),
        battery_remaining=_maybe_float(raw.get("battery")),
        payload_remaining=_maybe_float(raw.get("payload")),
        capabilities=caps,
        label=rid,
    )


def _uav_resource(raw: dict[str, Any]) -> UrbanResource:
    rid = str(raw.get("id", ""))
    return UrbanResource(
        id=rid,
        kind="drone",
        position=_coord_required(raw.get("position")),
        status=_resource_status(str(raw.get("state", "hover"))),
        speed=float(raw.get("speed", 2.0) or 2.0),
        battery_remaining=_maybe_float(raw.get("battery")),
        capabilities=["aerial_recon", "thermal_imaging"],
        label=rid,
    )


def _traffic_signal(raw: dict[str, Any]) -> TrafficSignal:
    return TrafficSignal(
        id=str(raw.get("id", "")),
        position=_coord_required(raw.get("position")),
        mode=str(raw.get("state", raw.get("mode", "normal"))),
        status="available",
    )


def _incident(raw: dict[str, Any]) -> Incident:
    return Incident(
        id=str(raw.get("id", "")),
        kind=str(raw.get("kind", "fire")),  # type: ignore[arg-type]
        position=_coord_required(raw.get("position")),
        severity=str(raw.get("severity", "high")),  # type: ignore[arg-type]
        status=str(raw.get("status", "open")),  # type: ignore[arg-type]
        description=str(raw.get("description", "")),
    )


def _resource_status(state: str):
    s = state.strip().lower()
    if s in {"moving", "taking_off", "landing"}:
        return "dispatched"
    if s in {"error", "offline"}:
        return "offline"
    return "available"


def _coord_required(raw: Any) -> Coordinate:
    d = raw if isinstance(raw, dict) else {}
    return Coordinate(
        x=float(d.get("x", 0.0) or 0.0),
        y=float(d.get("y", 0.0) or 0.0),
        z=float(d.get("z", 0.0) or 0.0),
    )


def _coord_data(coord: Coordinate) -> dict[str, float]:
    return {"x": coord.x, "y": coord.y, "z": coord.z}


def _maybe_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _as_envelope(event_type: str, data: Any) -> dict[str, Any]:
    if isinstance(data, dict) and "payload" in data:
        return data
    return {
        "version": "1.0",
        "msg_id": "",
        "type": event_type,
        "timestamp": time.time(),
        "frame": None,
        "sim_time": None,
        "sender": "middleware",
        "payload": data if isinstance(data, dict) else {},
    }


def _pending_action(payload: dict[str, Any]) -> UrbanAction:
    return UrbanAction(
        kind="dispatch_vehicle",
        target_id=str(payload.get("target", "")),
        parameters={"bridge_payload": payload},
        reason="CarlaBridge command ack/reject",
    )
