"""Phase E0 acceptance tests — protocol round-trips and handshake.

Covers EDITOR_PRD Phase E0 acceptance:
- JSON-RPC round-trip (dispatcher + live WebSocket),
- binary frame encode/decode round-trip,
- version-mismatch rejection,
- single-source codegen consistency (generated == schema.json).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import websockets

from fire_editor import Daemon, decode_frame, encode_frame
from fire_editor._generated import PROTOCOL_VERSION, ErrorCode, SchemaId
from fire_editor.binary import BinaryFrameError
from fire_editor.rpc import Dispatcher, RpcError

_SCHEMA = Path(__file__).resolve().parents[2] / "editor" / "protocol" / "schema.json"


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# Binary framing
# --------------------------------------------------------------------------- #
class TestBinaryFrames:
    def test_round_trip(self):
        payload = bytes(range(256)) * 4
        frame = encode_frame(SchemaId.MESH, 42, payload)
        schema_id, payload_id, out = decode_frame(frame)
        assert schema_id == SchemaId.MESH
        assert payload_id == 42
        assert out == payload

    def test_empty_payload_round_trip(self):
        frame = encode_frame(SchemaId.TEXTURE, 0, b"")
        schema_id, payload_id, out = decode_frame(frame)
        assert (schema_id, payload_id, out) == (SchemaId.TEXTURE, 0, b"")

    def test_bad_magic_rejected(self):
        frame = bytearray(encode_frame(SchemaId.MESH, 1, b"abc"))
        frame[0] ^= 0xFF  # corrupt the magic
        with pytest.raises(BinaryFrameError):
            decode_frame(bytes(frame))

    def test_truncated_header_rejected(self):
        with pytest.raises(BinaryFrameError):
            decode_frame(b"\x00\x01")


# --------------------------------------------------------------------------- #
# JSON-RPC dispatch (transport-agnostic)
# --------------------------------------------------------------------------- #
class TestDispatcher:
    def test_request_response(self):
        d = Dispatcher()

        async def echo(params):
            return {"echo": params.get("v")}

        d.register("echo", echo)
        resp = _run(d.dispatch({"jsonrpc": "2.0", "id": 7, "method": "echo", "params": {"v": 5}}))
        assert resp == {"jsonrpc": "2.0", "id": 7, "result": {"echo": 5}}

    def test_unknown_method(self):
        resp = _run(Dispatcher().dispatch({"jsonrpc": "2.0", "id": 1, "method": "nope"}))
        assert resp["error"]["code"] == ErrorCode.METHOD_NOT_FOUND

    def test_notification_returns_none(self):
        d = Dispatcher()
        d.register("noted", lambda p: None)
        # No "id" => notification => no response.
        assert _run(d.dispatch({"jsonrpc": "2.0", "method": "noted"})) is None

    def test_invalid_request(self):
        resp = _run(Dispatcher().dispatch({"method": "x", "id": 1}))
        assert resp["error"]["code"] == ErrorCode.INVALID_REQUEST

    def test_handler_exception_becomes_internal_error(self):
        d = Dispatcher()

        async def boom(params):
            raise ValueError("kaboom")

        d.register("boom", boom)
        resp = _run(d.dispatch({"jsonrpc": "2.0", "id": 1, "method": "boom"}))
        assert resp["error"]["code"] == ErrorCode.INTERNAL_ERROR
        assert "kaboom" in resp["error"]["message"]

    def test_rpc_error_passthrough(self):
        d = Dispatcher()

        async def bad(params):
            raise RpcError(ErrorCode.INVALID_PARAMS, "nope", data={"field": "x"})

        d.register("bad", bad)
        resp = _run(d.dispatch({"jsonrpc": "2.0", "id": 1, "method": "bad"}))
        assert resp["error"]["code"] == ErrorCode.INVALID_PARAMS
        assert resp["error"]["data"] == {"field": "x"}


# --------------------------------------------------------------------------- #
# Handshake (via the daemon's registered methods)
# --------------------------------------------------------------------------- #
class TestHandshake:
    def test_hello_ok(self):
        d = Daemon()
        resp = _run(d.dispatcher.dispatch({
            "jsonrpc": "2.0", "id": 1, "method": "hello",
            "params": {"protocol_version": PROTOCOL_VERSION, "client": "pytest"},
        }))
        assert resp["result"]["ok"] is True
        assert resp["result"]["protocol_version"] == PROTOCOL_VERSION
        assert resp["result"]["engine_version"]  # non-empty

    def test_hello_version_mismatch(self):
        d = Daemon()
        resp = _run(d.dispatcher.dispatch({
            "jsonrpc": "2.0", "id": 1, "method": "hello",
            "params": {"protocol_version": PROTOCOL_VERSION + 999, "client": "pytest"},
        }))
        assert resp["error"]["code"] == ErrorCode.VERSION_MISMATCH


# --------------------------------------------------------------------------- #
# Live WebSocket round-trip
# --------------------------------------------------------------------------- #
class TestLiveSocket:
    def test_full_session(self):
        async def scenario():
            daemon = Daemon()
            port = await daemon.server.start(0)
            try:
                async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                    # hello
                    await ws.send(json.dumps({
                        "jsonrpc": "2.0", "id": 1, "method": "hello",
                        "params": {"protocol_version": PROTOCOL_VERSION, "client": "test"},
                    }))
                    hello = json.loads(await ws.recv())
                    assert hello["result"]["ok"] is True

                    # ping
                    await ws.send(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"}))
                    pong = json.loads(await ws.recv())
                    assert pong["result"] == {"pong": True}

                    # daemon -> client binary broadcast
                    await daemon.server.broadcast_binary(encode_frame(SchemaId.MESH, 99, b"meshbytes"))
                    frame = await ws.recv()
                    assert isinstance(frame, bytes)
                    schema_id, payload_id, payload = decode_frame(frame)
                    assert (schema_id, payload_id, payload) == (SchemaId.MESH, 99, b"meshbytes")
            finally:
                await daemon.server.close()

        _run(scenario())


# --------------------------------------------------------------------------- #
# Codegen single-source consistency
# --------------------------------------------------------------------------- #
class TestCodegenConsistency:
    def test_protocol_version_matches_schema(self):
        schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
        assert schema["protocol_version"] == PROTOCOL_VERSION

    def test_generated_is_current(self):
        """The committed _generated.py must match a fresh codegen run."""
        import importlib.util

        cg_path = _SCHEMA.parent / "codegen.py"
        spec = importlib.util.spec_from_file_location("_codegen", cg_path)
        cg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cg)  # type: ignore[union-attr]
        schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
        fresh = cg.gen_python(schema)
        committed = (cg.PY_OUT).read_text(encoding="utf-8")
        assert fresh == committed, "run: python editor/protocol/codegen.py (schema.json changed)"
