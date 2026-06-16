"""EditorClient — an async WebSocket client for the Fire Editor daemon.

This is the Python twin of the extension's ``FireEditorClient`` (TypeScript):
it owns one connection, sends JSON-RPC requests, and routes the daemon's
responses, notifications and binary frames. It exists so an *agent* (or any
script/test) can drive the editor headlessly — open a world, stream chunks,
carve terrain, place objects, save scenes — exactly as the VS Code extension
does, without a human clicking in the IDE (EDITOR_PRD agent-harness goal).

The companion CLI is ``tools/editor_client.py``; the browser viewport harness
(``editor/extension/harness/``) is the visual half of the same idea.

Example::

    async with spawn_daemon() as (proc, port):
        client = EditorClient()
        await client.connect(port)
        await client.hello("my-agent")
        await client.request("world.open", {"seed": 1337})
        frames = await client.drain_until_stream_done(
            lambda: client.request("chunks.set_center",
                                   {"x": 0, "y": 0, "z": 24, "resend": True})
        )
        print(f"streamed {len(frames)} chunk meshes")
        await client.close()

No panda3d import here (hard rule 1) — the client only speaks the wire
protocol, so it stays in the headless test suite.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
from collections.abc import Awaitable, Callable

import websockets
from websockets.asyncio.client import ClientConnection

from ._generated import PROTOCOL_VERSION
from .binary import decode_frame


class RpcRemoteError(RuntimeError):
    """A JSON-RPC error response from the daemon (mirrors ``RpcRemoteError`` TS)."""

    def __init__(self, code: int, message: str, data: object = None) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.rpc_message = message
        self.data = data


class BinaryFrame:
    """A decoded binary frame the daemon pushed (mesh / texture payload)."""

    __slots__ = ("payload", "payload_id", "schema_id")

    def __init__(self, schema_id: int, payload_id: int, payload: bytes) -> None:
        self.schema_id = schema_id
        self.payload_id = payload_id
        self.payload = payload


class EditorClient:
    """One async connection to a running Fire Editor daemon.

    The client starts a background reader task on :meth:`connect` that
    dispatches every inbound frame. Requests return their JSON-RPC ``result``
    (or raise :class:`RpcRemoteError`); notifications and binary frames are both
    logged and delivered to any waiters registered via :meth:`wait_notification`
    / :meth:`drain_until_stream_done`.
    """

    def __init__(self) -> None:
        self._ws: ClientConnection | None = None
        self._reader: asyncio.Task | None = None
        self._next_id = 1
        self._pending: dict[int, asyncio.Future] = {}
        # Full logs (useful for assertions in tests / `watch` in the CLI).
        self.notifications: list[tuple[str, dict]] = []
        self.binary_frames: list[BinaryFrame] = []
        # Predicate waiters resolved by the reader task.
        self._notif_waiters: list[tuple[Callable[[str, dict], bool], asyncio.Future]] = []
        # Optional fan-out hooks (the CLI's `watch`/`serve` subscribe here).
        self.on_notification: Callable[[str, dict], None] | None = None
        self.on_binary: Callable[[BinaryFrame], None] | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def connect(self, port: int, host: str = "127.0.0.1") -> None:
        """Open the WebSocket and start the background reader task."""
        self._ws = await websockets.connect(f"ws://{host}:{port}", max_size=None)
        self._reader = asyncio.create_task(self._read_loop())

    async def hello(self, client_name: str = "editor-client") -> dict:
        """Handshake: announce our protocol version (daemon rejects mismatch)."""
        return await self.request(
            "hello", {"protocol_version": PROTOCOL_VERSION, "client": client_name}
        )

    async def close(self) -> None:
        """Cancel the reader, fail pending requests, and close the socket."""
        if self._reader is not None:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader
            self._reader = None
        self._fail_all_pending(ConnectionError("client closed"))
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    # ------------------------------------------------------------------ #
    # Requests / notifications
    # ------------------------------------------------------------------ #
    async def request(self, method: str, params: dict | None = None) -> dict:
        """Send a JSON-RPC request and await its result (or raise on error)."""
        if self._ws is None:
            raise ConnectionError("not connected; call connect() first")
        msg_id = self._next_id
        self._next_id += 1
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[msg_id] = fut
        await self._ws.send(
            json.dumps({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params or {}})
        )
        return await fut

    async def notify(self, method: str, params: dict | None = None) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        if self._ws is None:
            raise ConnectionError("not connected; call connect() first")
        await self._ws.send(
            json.dumps({"jsonrpc": "2.0", "method": method, "params": params or {}})
        )

    # ------------------------------------------------------------------ #
    # Waiting on async daemon output
    # ------------------------------------------------------------------ #
    async def wait_notification(self, method: str, timeout: float = 10.0) -> dict:
        """Block until a notification with ``method`` arrives; return its params."""
        return await self._wait(lambda m, _p: m == method, timeout)

    async def drain_until_stream_done(
        self,
        trigger: Callable[[], Awaitable[object]] | None = None,
        timeout: float = 30.0,
    ) -> list[BinaryFrame]:
        """Run ``trigger`` (e.g. a ``chunks.set_center`` call) and collect every
        MESH frame the daemon streams until the ``stream.done`` notification.

        Returns the binary frames received between the call and ``stream.done``
        (in arrival order). With no ``trigger`` it just waits for the next
        ``stream.done`` from a stream already in flight.
        """
        start = len(self.binary_frames)
        done = self._wait(lambda m, _p: m == "stream.done", timeout)
        done_task = asyncio.ensure_future(done)
        if trigger is not None:
            await trigger()
        await done_task
        return self.binary_frames[start:]

    # ------------------------------------------------------------------ #
    # Reader task internals
    # ------------------------------------------------------------------ #
    async def _read_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if isinstance(raw, (bytes, bytearray)):
                    self._on_binary(bytes(raw))
                else:
                    self._on_text(raw)
        except asyncio.CancelledError:
            raise
        except websockets.ConnectionClosed:
            self._fail_all_pending(ConnectionError("connection closed"))

    def _on_text(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        msg_id = msg.get("id")
        if msg_id is not None:
            fut = self._pending.pop(msg_id, None)
            if fut is None or fut.done():
                return
            err = msg.get("error")
            if err is not None:
                fut.set_exception(
                    RpcRemoteError(err.get("code", 0), err.get("message", ""), err.get("data"))
                )
            else:
                fut.set_result(msg.get("result"))
        elif msg.get("method"):
            method = msg["method"]
            params = msg.get("params") or {}
            self.notifications.append((method, params))
            if self.on_notification is not None:
                self.on_notification(method, params)
            self._resolve_waiters(method, params)

    def _on_binary(self, data: bytes) -> None:
        try:
            schema_id, payload_id, payload = decode_frame(data)
        except ValueError:
            return
        frame = BinaryFrame(schema_id, payload_id, payload)
        self.binary_frames.append(frame)
        if self.on_binary is not None:
            self.on_binary(frame)

    def _wait(self, predicate: Callable[[str, dict], bool], timeout: float) -> Awaitable[dict]:
        # Wait for the NEXT matching notification, not a historical one: the
        # waiter is registered synchronously here (before any trigger fires), so
        # callers that register-then-trigger never miss the result, and stale
        # notifications from an earlier drain never resolve a new wait.
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._notif_waiters.append((predicate, fut))
        return asyncio.wait_for(fut, timeout)

    def _resolve_waiters(self, method: str, params: dict) -> None:
        still: list[tuple[Callable[[str, dict], bool], asyncio.Future]] = []
        for predicate, fut in self._notif_waiters:
            if not fut.done() and predicate(method, params):
                fut.set_result(params)
            elif not fut.done():
                still.append((predicate, fut))
        self._notif_waiters = still

    def _fail_all_pending(self, err: Exception) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(err)
        self._pending.clear()
        for _pred, fut in self._notif_waiters:
            if not fut.done():
                fut.set_exception(err)
        self._notif_waiters.clear()


@contextlib.asynccontextmanager
async def spawn_daemon(python: str | None = None, host: str = "127.0.0.1"):
    """Spawn ``python -m fire_editor --port 0`` and yield ``(proc, port)``.

    Parses the daemon's ``{"event":"listening","port":N}`` stdout line to learn
    the OS-assigned port, then hands back the live subprocess and that port.
    Terminates the daemon on exit. Use this when you want a throwaway daemon for
    a single agent session or test; long-lived servers run the daemon directly.

    The child's ``cwd``/``PYTHONPATH`` are set exactly as the VS Code extension
    sets them (repo root for ``fire_engine``, ``editor/`` for ``fire_editor``),
    so ``python -m fire_editor`` resolves regardless of the caller's cwd.
    """
    editor_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_root = os.path.dirname(editor_dir)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (repo_root, editor_dir, env.get("PYTHONPATH")) if p
    )
    proc = await asyncio.create_subprocess_exec(
        python or sys.executable,
        "-m",
        "fire_editor",
        "--port",
        "0",
        "--host",
        host,
        cwd=repo_root,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        port = await _read_listening_port(proc)
        yield proc, port
    finally:
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=5.0)


async def _read_listening_port(proc: asyncio.subprocess.Process, timeout: float = 30.0) -> int:
    """Read daemon stdout until the listening line, return the bound port."""
    assert proc.stdout is not None

    async def scan() -> int:
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                raise RuntimeError("daemon exited before announcing a port")
            try:
                event = json.loads(line.decode().strip())
            except json.JSONDecodeError:
                continue
            if event.get("event") == "listening":
                return int(event["port"])

    return await asyncio.wait_for(scan(), timeout)
