"""Phase E3 acceptance tests — brush editing + undo/redo + the crater round-trip.

The headline integration (EDITOR_PRD E3): carve in the editor -> save -> the
*engine* (the game's own load path) sees the crater. Plus undo restores exact
voxel content (byte-compare) and redo re-applies it.
"""
from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
import websockets

from fire_engine.core import Clock, EventBus, load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.save import SaveManager
from fire_engine.terrain import ChunkManager, generate_chunk

from fire_editor import Daemon, EditorSession, decode_frame, decode_mesh_payload
from fire_editor._generated import PROTOCOL_VERSION, SchemaId

SURFACE = (0, 0, 0)
BRUSH = {"shape": "sphere", "x": 0.0, "y": 0.0, "z": 7.5, "mode": "remove", "radius": 2.0}


def _run(coro):
    return asyncio.run(coro)


def _daemon_with_world():
    cfg = load_config()
    daemon = Daemon()
    daemon.session = EditorSession(cfg)
    return daemon, cfg


class TestUndoRedo:
    def test_brush_undo_redo_exact_restore(self):
        daemon, _cfg = _daemon_with_world()
        s = daemon.session
        baseline = s.cm.get_or_create(SURFACE).materials.copy()

        res = _run(daemon.chunks.brush(BRUSH))
        assert res["ok"] and res["touched"] >= 1 and res["can_undo"] is True
        after_edit = s.cm.chunks[SURFACE].materials.copy()
        assert not np.array_equal(after_edit, baseline), "brush should change voxels"

        u = _run(daemon.chunks.undo({}))
        assert u["ok"] is True
        assert np.array_equal(s.cm.chunks[SURFACE].materials, baseline), "undo must restore exact voxels"

        r = _run(daemon.chunks.redo({}))
        assert r["ok"] is True
        assert np.array_equal(s.cm.chunks[SURFACE].materials, after_edit), "redo must re-apply exact voxels"

    def test_undo_empty_stack(self):
        daemon, _ = _daemon_with_world()
        u = _run(daemon.chunks.undo({}))
        assert u["ok"] is False and u["can_undo"] is False

    def test_brush_clears_redo(self):
        daemon, _ = _daemon_with_world()
        _run(daemon.chunks.brush(BRUSH))
        _run(daemon.chunks.undo({}))
        assert daemon.chunks.history.can_redo is True
        _run(daemon.chunks.brush({**BRUSH, "x": 4.0}))  # a new edit clears redo
        assert daemon.chunks.history.can_redo is False


class TestCraterRoundTrip:
    def test_editor_save_engine_load_shows_crater(self, tmp_path):
        # 1. Carve in the editor and save.
        daemon, cfg = _daemon_with_world()
        s = daemon.session
        _run(daemon.chunks.brush(BRUSH))
        assert s.edited_chunk_count() >= 1
        editor_mats = s.cm.chunks[SURFACE].materials.copy()
        save_path = tmp_path / "crater.ta"
        s.save(str(save_path))

        # 2. Load through the ENGINE's own path (what `python main.py --load` does).
        set_world_seed(cfg.world_seed)
        bus = EventBus()
        clock = Clock(fixed_dt=cfg.fixed_dt, bus=bus)
        cm = ChunkManager(cfg, bus)
        sm = SaveManager(cfg, clock)
        sm.register(cm)
        sm.load(str(save_path))

        # 3. The crater is present and matches the editor, and deviates from baseline.
        assert SURFACE in cm.chunks
        assert np.array_equal(cm.chunks[SURFACE].materials, editor_mats)
        assert not np.array_equal(cm.chunks[SURFACE].materials, generate_chunk(SURFACE, cfg))


class TestLiveBrush:
    def test_brush_streams_remesh(self):
        async def scenario():
            daemon = Daemon()
            port = await daemon.server.start(0)
            try:
                async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                    async def call(mid, method, params):
                        await ws.send(json.dumps({"jsonrpc": "2.0", "id": mid,
                                                  "method": method, "params": params}))

                    await call(1, "hello", {"protocol_version": PROTOCOL_VERSION, "client": "t"})
                    await call(2, "world.open", {"seed": load_config().world_seed})
                    # drain responses for ids 1,2
                    await _await_result(ws, 1)
                    await _await_result(ws, 2)

                    await call(3, "terrain.brush", BRUSH)
                    got_mesh = False
                    got_edit_state = False
                    brush_ok = False
                    deadline = asyncio.get_event_loop().time() + 10
                    while asyncio.get_event_loop().time() < deadline and not (got_mesh and brush_ok and got_edit_state):
                        msg = await asyncio.wait_for(ws.recv(), timeout=10)
                        if isinstance(msg, bytes):
                            sid, _pid, body = decode_frame(msg)
                            if sid == SchemaId.MESH and decode_mesh_payload(body)["vertex_count"] > 0:
                                got_mesh = True
                        else:
                            obj = json.loads(msg)
                            if obj.get("id") == 3:
                                brush_ok = obj["result"]["ok"]
                            elif obj.get("method") == "edit.state":
                                got_edit_state = True
                    assert brush_ok and got_mesh and got_edit_state
            finally:
                await daemon.server.close()

        _run(scenario())


async def _await_result(ws, mid):
    """Read until the response to ``mid`` arrives (ignoring binary/notifications)."""
    while True:
        msg = await asyncio.wait_for(ws.recv(), timeout=10)
        if isinstance(msg, bytes):
            continue
        obj = json.loads(msg)
        if obj.get("id") == mid:
            return obj
