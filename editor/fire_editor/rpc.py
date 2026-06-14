"""JSON-RPC 2.0 dispatch for the Fire Editor control channel (EDITOR_PRD §4).

Transport-agnostic on purpose: ``Dispatcher`` turns a decoded JSON-RPC message
dict into a response dict (or ``None`` for notifications), so it is unit-testable
without a socket. ``server.py`` owns the WebSocket and feeds frames here.

Handlers are ``async`` callables taking ``(params: dict) -> result`` and may
raise :class:`RpcError` to return a structured JSON-RPC error. Any other
exception becomes an ``INTERNAL_ERROR`` with the message attached.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from ._generated import ErrorCode

Handler = Callable[[dict], Awaitable[object]]


class RpcError(Exception):
    """A structured JSON-RPC error to return to the client.

    Args:
        code: One of ``ErrorCode.*``.
        message: Human-readable summary.
        data: Optional JSON-serialisable detail (traceback string, field name…).
    """

    def __init__(self, code: int, message: str, data: object | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


class Dispatcher:
    """Registry + executor for JSON-RPC methods.

    Example::

        d = Dispatcher()
        d.register("ping", lambda params: {"pong": True})
        resp = await d.dispatch({"jsonrpc": "2.0", "id": 1, "method": "ping"})
        # resp == {"jsonrpc": "2.0", "id": 1, "result": {"pong": True}}
    """

    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {}

    def register(self, name: str, handler: Handler) -> None:
        """Register ``handler`` for method ``name`` (overwrites any existing)."""
        self._handlers[name] = handler

    def method(self, name: str) -> Callable[[Handler], Handler]:
        """Decorator form of :meth:`register`."""

        def deco(fn: Handler) -> Handler:
            self.register(name, fn)
            return fn

        return deco

    @staticmethod
    def _error(msg_id: object, code: int, message: str, data: object | None = None) -> dict:
        err: dict = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        return {"jsonrpc": "2.0", "id": msg_id, "error": err}

    async def dispatch(self, message: dict) -> dict | None:
        """Execute one decoded JSON-RPC message.

        Returns the response dict, or ``None`` for a notification (no ``id``).
        Never raises for protocol-level problems — they become error responses.
        """
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return self._error(
                message.get("id") if isinstance(message, dict) else None,
                ErrorCode.INVALID_REQUEST,
                "not a JSON-RPC 2.0 message",
            )

        msg_id = message.get("id")
        is_notification = "id" not in message
        method = message.get("method")
        params = message.get("params", {})
        if not isinstance(params, dict):
            params = {}

        handler = self._handlers.get(method) if isinstance(method, str) else None
        if handler is None:
            if is_notification:
                return None
            return self._error(msg_id, ErrorCode.METHOD_NOT_FOUND, f"unknown method: {method!r}")

        try:
            result = await handler(params)
        except RpcError as e:
            if is_notification:
                return None
            return self._error(msg_id, e.code, e.message, e.data)
        except Exception as e:  # noqa: BLE001 — boundary: never let a handler kill the daemon
            if is_notification:
                return None
            return self._error(msg_id, ErrorCode.INTERNAL_ERROR, f"{type(e).__name__}: {e}")

        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}
