"""WebSocket transport for the Fire Editor daemon (EDITOR_PRD §4).

Wraps the ``websockets`` library. Text frames are JSON-RPC control messages fed
to a :class:`~fire_editor.rpc.Dispatcher`; binary frames are protocol payloads
(meshes/textures) the daemon pushes to clients via :meth:`broadcast_binary`.

The server tracks connected clients so services can broadcast notifications and
binary payloads. Localhost only — this is a developer tool bound to ``127.0.0.1``.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Iterable

import websockets
from websockets.legacy.server import WebSocketServerProtocol

from .rpc import Dispatcher

log = logging.getLogger("fire_editor.server")


class EditorServer:
    """Async WebSocket server bridging the socket and the JSON-RPC dispatcher.

    Args:
        dispatcher: Configured method dispatcher (see :mod:`fire_editor.daemon`).
        host: Bind address; defaults to localhost.
    """

    def __init__(self, dispatcher: Dispatcher, host: str = "127.0.0.1") -> None:
        self._dispatcher = dispatcher
        self._host = host
        self._clients: set[WebSocketServerProtocol] = set()
        self._server: websockets.WebSocketServer | None = None

    @property
    def client_count(self) -> int:
        """Number of currently connected clients."""
        return len(self._clients)

    async def start(self, port: int) -> int:
        """Begin listening on ``port`` (0 = OS-assigned). Returns the bound port."""
        self._server = await websockets.serve(self._handle, self._host, port, max_size=None)
        bound = self._server.sockets[0].getsockname()[1]
        log.info("listening on %s:%d", self._host, bound)
        return bound

    async def serve_forever(self, port: int) -> int:
        """Start and block until cancelled. Returns the bound port via the task's start."""
        bound = await self.start(port)
        await self._server.wait_closed()
        return bound

    async def close(self) -> None:
        """Stop accepting connections and drop existing clients."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, ws: WebSocketServerProtocol) -> None:
        self._clients.add(ws)
        peer = getattr(ws, "remote_address", None)
        log.info("client connected: %s (total %d)", peer, len(self._clients))
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    # Clients do not push bulk payloads in v1; ignore but stay alive.
                    log.debug("ignoring inbound binary frame (%d bytes)", len(raw))
                    continue
                await self._on_text(ws, raw)
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(ws)
            log.info("client disconnected: %s (total %d)", peer, len(self._clients))

    async def _on_text(self, ws: WebSocketServerProtocol, raw: str) -> None:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            await ws.send(json.dumps({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "parse error"},
            }))
            return
        response = await self._dispatcher.dispatch(message)
        if response is not None:
            await ws.send(json.dumps(response))

    async def broadcast_binary(self, frame: bytes) -> None:
        """Send a pre-encoded binary frame to every connected client."""
        await self._broadcast([c.send(frame) for c in tuple(self._clients)])

    async def broadcast_notification(self, method: str, params: dict) -> None:
        """Send a JSON-RPC notification (no ``id``) to every connected client."""
        payload = json.dumps({"jsonrpc": "2.0", "method": method, "params": params})
        await self._broadcast([c.send(payload) for c in tuple(self._clients)])

    @staticmethod
    async def _broadcast(sends: Iterable[asyncio.Future]) -> None:
        results = await asyncio.gather(*sends, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception) and not isinstance(r, websockets.ConnectionClosed):
                log.warning("broadcast send failed: %r", r)
