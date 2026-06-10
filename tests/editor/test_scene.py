"""Phase E1 acceptance tests — scene view daemon side.

Covers EDITOR_PRD Phase E1 acceptance (the headless half):
- mesh payload binary round-trip,
- region == engine desired_set (drift guard),
- determinism: mesh frame bytes for seed S are reproducible,
- save -> reopen shows edits (craters visible in the data),
- live socket: world.open(seed) -> set_center -> MESH binary + notifications.
"""
from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
import websockets

from torn_apart.core.math3d import Vec3
from torn_apart.terrain import BrushMode, SphereBrush, apply_brush

from fire_editor import (
    Daemon,
    EditorSession,
    decode_frame,
    decode_mesh_payload,
    encode_frame,
    encode_mesh_payload,
)
from fire_editor._generated import PROTOCOL_VERSION, SchemaId

SEED = 1337
SURFACE = (0, 0, 0)   # spans z [0,16) m; flat ground at 8 m -> non-empty mesh
HIGH_AIR = (0, 0, 3)  # spans z [48,64) m -> all air -> empty mesh


def _run(coro):
    return asyncio.run(coro)


def _session():
    s = EditorSession.from_seed(SEED)
    coords = s.region_coords(Vec3(0, 0, 12), s.config.view_distance_chunks)
    s.ensure_loaded(coords)
    s.relight()
    return s


# --------------------------------------------------------------------------- #
class TestMeshCodec:
    def test_round_trip(self):
        s = _session()
        mesh = s.mesh(SURFACE)
        payload = encode_mesh_payload(SURFACE, mesh)
        out = decode_mesh_payload(payload)
        assert out["coord"] == SURFACE
        assert out["vertex_count"] == mesh.vertex_count
        assert out["index_count"] == mesh.indices.shape[0]
        assert np.array_equal(out["positions"], mesh.positions.astype(np.float32))
        assert np.array_equal(out["indices"], mesh.indices.astype(np.uint32))

    def test_full_frame_round_trip(self):
        s = _session()
        payload = encode_mesh_payload(SURFACE, s.mesh(SURFACE))
        frame = encode_frame(SchemaId.MESH, 1, payload)
        schema_id, pid, body = decode_frame(frame)
        assert schema_id == SchemaId.MESH and pid == 1
        assert decode_mesh_payload(body)["coord"] == SURFACE


# --------------------------------------------------------------------------- #
class TestRegion:
    def test_region_matches_engine_desired_set(self):
        """region_coords at the default radius must equal ChunkManager.desired_set."""
        s = EditorSession.from_seed(SEED)
        center = Vec3(3.0, -5.0, 12.0)
        ours = set(s.region_coords(center, s.config.view_distance_chunks))
        engine = s.cm.desired_set(center)
        assert ours == engine

    def test_surface_nonempty_air_empty(self):
        s = _session()
        assert not s.mesh(SURFACE).is_empty
        assert s.mesh(HIGH_AIR).is_empty


# --------------------------------------------------------------------------- #
class TestDeterminism:
    def test_same_seed_same_mesh_bytes(self):
        a = _session()
        b = _session()
        fa = encode_frame(SchemaId.MESH, 1, encode_mesh_payload(SURFACE, a.mesh(SURFACE)))
        fb = encode_frame(SchemaId.MESH, 1, encode_mesh_payload(SURFACE, b.mesh(SURFACE)))
        assert fa == fb


# --------------------------------------------------------------------------- #
class TestSaveRoundTrip:
    def test_edit_save_reopen_shows_crater(self, tmp_path):
        s = _session()
        baseline = s.cm.get_or_create(SURFACE).materials.copy()
        touched = apply_brush(
            SphereBrush(radius_m=2.0),
            Vec3(0.0, 0.0, 7.5),  # just under the 8 m flat surface
            BrushMode.REMOVE,
            material=0,
            chunk_provider=s.cm,
            bus=s.bus,
        )
        assert touched, "brush should modify at least one chunk"
        assert s.edited_chunk_count() >= 1

        save_path = tmp_path / "crater.ta"
        s.save(save_path)

        reopened = EditorSession.from_save(str(save_path))
        assert reopened.edited_chunk_count() >= 1
        reopened_mats = reopened.cm.get_or_create(SURFACE).materials
        # The crater (REMOVE) deviates from baseline and survived the round-trip.
        assert not np.array_equal(reopened_mats, baseline)


# --------------------------------------------------------------------------- #
class TestLiveSocket:
    def test_open_seed_and_stream(self):
        async def scenario():
            daemon = Daemon()
            port = await daemon.server.start(0)
            try:
                async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                    await ws.send(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "hello",
                                              "params": {"protocol_version": PROTOCOL_VERSION,
                                                         "client": "t"}}))
                    assert json.loads(await ws.recv())["result"]["ok"] is True

                    await ws.send(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "world.open",
                                              "params": {"seed": SEED}}))
                    opened = json.loads(await ws.recv())["result"]
                    assert opened["ok"] is True and opened["seed"] == SEED

                    await ws.send(json.dumps({"jsonrpc": "2.0", "id": 3, "method": "chunks.set_center",
                                              "params": {"x": 0, "y": 0, "z": 12, "radius": 1}}))
                    # Collect frames until stream.done. Expect >=1 binary MESH frame.
                    got_binary = False
                    got_ready = False
                    stream_done = False
                    set_center_acked = False
                    deadline = asyncio.get_event_loop().time() + 10
                    while not stream_done and asyncio.get_event_loop().time() < deadline:
                        msg = await asyncio.wait_for(ws.recv(), timeout=10)
                        if isinstance(msg, bytes):
                            schema_id, _pid, body = decode_frame(msg)
                            assert schema_id == SchemaId.MESH
                            assert decode_mesh_payload(body)["vertex_count"] > 0
                            got_binary = True
                        else:
                            obj = json.loads(msg)
                            if obj.get("id") == 3:
                                set_center_acked = obj["result"]["ok"]
                            elif obj.get("method") == "chunk.ready":
                                got_ready = True
                            elif obj.get("method") == "stream.done":
                                stream_done = True
                    assert set_center_acked and got_binary and got_ready and stream_done
            finally:
                await daemon.server.close()

        _run(scenario())
