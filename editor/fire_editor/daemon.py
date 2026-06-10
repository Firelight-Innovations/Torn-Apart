"""Fire Editor daemon — headless engine host behind a WebSocket protocol.

The daemon imports the Torn Apart engine (``torn_apart.*``) through documented
public APIs only and **never imports panda3d** (EDITOR_PRD hard rule 1, enforced
by ``tests/editor/test_no_panda3d.py``). It runs with the game closed and serves
the VS Code / Cursor extension.

Phase E0 wires the transport and handshake. Later phases register service
methods (chunks, scene, edit, texture, model) onto the same dispatcher.
"""
from __future__ import annotations

import logging

import torn_apart

from ._generated import DAEMON_VERSION, PROTOCOL_VERSION, ErrorCode, Method
from .rpc import Dispatcher, RpcError
from .server import EditorServer

log = logging.getLogger("fire_editor.daemon")


class Daemon:
    """Owns the dispatcher + server and the lifetime of an editor session.

    Args:
        host: Bind address (localhost by default).
    """

    def __init__(self, host: str = "127.0.0.1") -> None:
        self.dispatcher = Dispatcher()
        self.server = EditorServer(self.dispatcher, host=host)
        self._register_core_methods()

    def _register_core_methods(self) -> None:
        self.dispatcher.register(Method.HELLO, self._hello)
        self.dispatcher.register(Method.PING, self._ping)

    async def _hello(self, params: dict) -> dict:
        """Handshake: verify protocol compatibility, report versions."""
        client_version = params.get("protocol_version")
        if client_version != PROTOCOL_VERSION:
            raise RpcError(
                ErrorCode.VERSION_MISMATCH,
                f"protocol mismatch: client={client_version} daemon={PROTOCOL_VERSION}. "
                f"Rebuild the Fire Editor daemon/extension from the same commit.",
                data={"client": client_version, "daemon": PROTOCOL_VERSION},
            )
        return {
            "ok": True,
            "protocol_version": PROTOCOL_VERSION,
            "engine_version": torn_apart.__version__,
            "daemon_version": DAEMON_VERSION,
        }

    async def _ping(self, params: dict) -> dict:
        return {"pong": True}

    async def run(self, port: int) -> None:
        """Start the server and block until cancelled.

        Prints a single ``{"event":"listening","port":N}`` line to stdout so the
        extension learns the bound port (when launched with ``--port 0``).
        """
        bound = await self.server.start(port)
        # Machine-readable readiness line on stdout (the extension parses this).
        print(f'{{"event":"listening","port":{bound}}}', flush=True)
        log.info("fire_editor daemon ready (engine=%s, protocol=%d)",
                 torn_apart.__version__, PROTOCOL_VERSION)
        await self.server._server.wait_closed()  # type: ignore[union-attr]
