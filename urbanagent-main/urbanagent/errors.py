"""Errors raised by the UrbanAgent method pipeline."""


class UrbanAgentPipelineError(RuntimeError):
    """Cognition or planning failed and no rule-based fallback is allowed."""

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        super().__init__(f"[UrbanAgent:{stage}] {message}")


class SandboxWireError(RuntimeError):
    """3D sandbox WebSocket wire protocol or payload decoding failed."""

    def __init__(self, message: str) -> None:
        super().__init__(f"[SandboxWire] {message}")
