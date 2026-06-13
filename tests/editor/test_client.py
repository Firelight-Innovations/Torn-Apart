"""EditorClient + harness tests (EDITOR_PRD agent access).

Exercises the Python client against a real in-process daemon over a real
WebSocket: handshake, world.open, chunk streaming, scene ops, the resend
semantics that let a *second* client attach to a running daemon, and the
``spawn_daemon`` subprocess round-trip.
"""
from __future__ import annotations

import asyncio

import pytest

from fire_engine.core import load_config

from fire_editor import Daemon, EditorClient, RpcRemoteError, spawn_daemon
from fire_editor._generated import SchemaId


def _run(coro):
    return asyncio.run(coro)


def _seed() -> int:
    return load_config().world_seed


class TestHandshakeAndStreaming:
    def test_hello_open_and_stream(self):
        async def scenario():
            daemon = Daemon()
            port = await daemon.server.start(0)
            try:
                c = EditorClient()
                await c.connect(port)
                hello = await c.hello("test-agent")
                assert hello["ok"] and hello["protocol_version"] >= 5

                opened = await c.request("world.open", {"seed": _seed()})
                assert opened["seed"] == _seed()
                assert opened["config"]["ground_seed"] > 0

                frames = await c.drain_until_stream_done(
                    lambda: c.request(
                        "chunks.set_center",
                        {"x": 0, "y": 0, "z": 24, "radius": 2, "resend": True},
                    )
                )
                meshes = [f for f in frames if f.schema_id == SchemaId.MESH]
                assert len(meshes) >= 1, "expected at least one streamed chunk mesh"
                await c.close()
            finally:
                await daemon.server.close()

        _run(scenario())

    def test_ground_lut_texture_frame(self):
        async def scenario():
            daemon = Daemon()
            port = await daemon.server.start(0)
            try:
                c = EditorClient()
                await c.connect(port)
                await c.hello("t")
                await c.request("world.open", {"seed": _seed()})
                res = await c.request("world.ground_lut", {})
                assert res["ok"] and res["width"] == 256 and res["height"] >= 1
                tex = [f for f in c.binary_frames if f.schema_id == SchemaId.TEXTURE]
                assert len(tex) == 1, "ground_lut must emit exactly one TEXTURE frame"
                assert tex[0].payload_id == res["payload_id"]
                await c.close()
            finally:
                await daemon.server.close()

        _run(scenario())

    def test_remote_error_propagates(self):
        async def scenario():
            daemon = Daemon()
            port = await daemon.server.start(0)
            try:
                c = EditorClient()
                await c.connect(port)
                await c.hello("t")
                # No world open yet: scene.tree must raise a structured RPC error.
                with pytest.raises(RpcRemoteError):
                    await c.request("scene.tree", {})
                await c.close()
            finally:
                await daemon.server.close()

        _run(scenario())


class TestSceneOps:
    def test_create_and_tree(self):
        async def scenario():
            daemon = Daemon()
            port = await daemon.server.start(0)
            try:
                c = EditorClient()
                await c.connect(port)
                await c.hello("t")
                await c.request("world.open", {"seed": _seed()})
                created = await c.request(
                    "scene.create", {"kind": "cube", "x": 2, "y": 0, "z": 8}
                )
                assert created["object"]["kind"] == "cube"
                tree = await c.request("scene.tree", {})
                assert any(o["kind"] == "cube" for o in tree["objects"])
                await c.close()
            finally:
                await daemon.server.close()

        _run(scenario())


class TestResendSemantics:
    """A second client attaching to a running daemon gets the full chunk set
    only with ``resend=True`` (the latent _client_chunks bug fix)."""

    def test_second_client_needs_resend(self):
        async def scenario():
            daemon = Daemon()
            port = await daemon.server.start(0)
            try:
                a = EditorClient()
                await a.connect(port)
                await a.hello("a")
                await a.request("world.open", {"seed": _seed()})
                first = await a.drain_until_stream_done(
                    lambda: a.request(
                        "chunks.set_center", {"x": 0, "y": 0, "z": 24, "radius": 2}
                    )
                )
                assert sum(1 for f in first if f.schema_id == SchemaId.MESH) >= 1

                # Second client, no resend: the daemon already streamed those
                # chunks, so it sends nothing new.
                b = EditorClient()
                await b.connect(port)
                await b.hello("b")
                no_resend = await b.drain_until_stream_done(
                    lambda: b.request(
                        "chunks.set_center", {"x": 0, "y": 0, "z": 24, "radius": 2}
                    )
                )
                assert sum(1 for f in no_resend if f.schema_id == SchemaId.MESH) == 0

                # With resend: the full in-range set streams again to everyone.
                with_resend = await b.drain_until_stream_done(
                    lambda: b.request(
                        "chunks.set_center",
                        {"x": 0, "y": 0, "z": 24, "radius": 2, "resend": True},
                    )
                )
                assert sum(1 for f in with_resend if f.schema_id == SchemaId.MESH) >= 1
                await a.close()
                await b.close()
            finally:
                await daemon.server.close()

        _run(scenario())


class TestSpawnDaemon:
    def test_spawn_daemon_roundtrip(self):
        async def scenario():
            async with spawn_daemon() as (proc, port):
                assert port > 0
                c = EditorClient()
                await c.connect(port)
                hello = await c.hello("spawned")
                assert hello["ok"]
                await c.request("world.open", {"seed": _seed()})
                await c.close()
            # Context exit terminates the daemon.
            assert proc.returncode is not None

        _run(scenario())
