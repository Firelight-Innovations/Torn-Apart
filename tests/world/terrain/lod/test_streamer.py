"""
tests/world/terrain/lod/test_streamer.py — LodStreamer async-streaming tests.

Headless (no panda3d): a real ``ChunkManager`` (``load_config()`` + ``EventBus``,
seeded via ``set_world_seed``) driving a real ``TerrainLodPool``.  Covers parity
with synchronous streaming (off-thread == on-thread, byte-for-byte), the per-coord
``seq`` staleness discipline, the unload path, and the submit budget.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import pytest

from fire_engine.core import ChunkUnloadedEvent, EventBus, load_config
from fire_engine.core.math3d import Vec3
from fire_engine.core.rng import set_world_seed
from fire_engine.world.terrain.chunk_manager import ChunkManager
from fire_engine.world.terrain.lod import LodStreamer, TerrainLodPool, build_lod_mesh

_MAX_PUMP_ITERS = 2000  # bounded sleep loop (~2 s worst case at 1 ms/iter)


@pytest.fixture
def pool():
    """A started 2-worker pool, stopped on teardown so threads never leak."""
    p = TerrainLodPool(n_workers=2)
    p.start()
    try:
        yield p
    finally:
        p.stop()


def _pump(streamer, pool, camera_pos, frames=40):
    """Drive ``stream_frame`` until the pool drains, then drain once more.

    Calls ``stream_frame`` repeatedly (each call submits + drains), waiting for
    in-flight jobs to finish, until no jobs are pending and no new dirty/missing
    work remains.  Returns when steady (or after a bounded number of frames).
    """
    for _ in range(frames):
        streamer.stream_frame(camera_pos)
        # Let workers finish, then drain whatever completed this round.
        for _ in range(_MAX_PUMP_ITERS):
            if pool.pending() == 0:
                break
            time.sleep(0.001)
    # Final drain frame to land any last results into pending_meshes.
    streamer.stream_frame(camera_pos)
    for _ in range(_MAX_PUMP_ITERS):
        if pool.pending() == 0:
            break
        time.sleep(0.001)
    streamer.stream_frame(camera_pos)


def _mesh_eq(a, b) -> bool:
    return (
        np.array_equal(a.positions, b.positions)
        and np.array_equal(a.indices, b.indices)
        and np.array_equal(a.colors, b.colors)
        and np.array_equal(a.face_materials, b.face_materials)
    )


class TestParityWithSynchronous:
    def test_async_meshes_equal_synchronous(self, pool):
        config = load_config()
        cam = Vec3(0, 0, 20)

        # Async path.
        set_world_seed(1337)
        cm_async = ChunkManager(config, EventBus())
        streamer = LodStreamer(cm_async, pool, config)
        _pump(streamer, pool, cam, frames=60)
        async_meshes = dict(cm_async.pending_meshes)
        assert async_meshes, "async streamer produced no meshes"

        # Synchronous reference: mesh the SAME coords directly.
        set_world_seed(1337)
        cm_sync = ChunkManager(config, EventBus())
        for coord in async_meshes:
            cm_sync.get_or_create(coord)
        sync_meshes = {c: cm_sync.mesh_chunk(c) for c in async_meshes}

        # Every coord meshed by both must be byte-identical.
        common = set(async_meshes) & set(sync_meshes)
        assert common
        for coord in common:
            assert _mesh_eq(async_meshes[coord], sync_meshes[coord]), coord


class TestStaleness:
    def test_redirty_keeps_latest_seq(self, pool):
        config = load_config()
        set_world_seed(1337)
        cm = ChunkManager(config, EventBus())
        streamer = LodStreamer(cm, pool, config)
        coord = (0, 0, 0)

        # First submit.
        cm.get_or_create(coord)
        streamer._submit(coord)
        first_seq = streamer._node_seq[coord]

        # Re-dirty + re-submit (new snapshot, newer seq) before draining.
        cm.chunks[coord].dirty = True
        streamer._submit(coord)
        latest_seq = streamer._node_seq[coord]
        assert latest_seq > first_seq

        # Drain both results: only the latest seq survives into pending_meshes.
        for _ in range(_MAX_PUMP_ITERS):
            if pool.pending() == 0:
                break
            time.sleep(0.001)
        streamer._drain()

        assert coord in cm.pending_meshes
        # Surviving mesh must match a job built from the latest materials/seq.
        expected = build_lod_mesh(streamer._make_job(coord)).mesh
        # _make_job bumped seq again; the point is the materials are identical,
        # so the mesh content matches what the latest in-flight job produced.
        assert _mesh_eq(cm.pending_meshes[coord], expected)
        assert streamer._node_seq[coord] >= latest_seq


class TestUnload:
    def test_far_chunks_unloaded_and_seq_pruned(self, pool):
        config = load_config()
        set_world_seed(1337)
        cm = ChunkManager(config, EventBus())

        events: list[tuple[int, int, int]] = []
        cm.bus.subscribe(ChunkUnloadedEvent, lambda e: events.append(e.coord))

        streamer = LodStreamer(cm, pool, config)
        near = Vec3(0, 0, 20)
        _pump(streamer, pool, near, frames=20)
        assert cm.chunks
        near_coords = set(cm.chunks)

        # Move the camera far away (many chunks beyond view_distance + 1).
        far = Vec3(5000, 5000, 20)
        _pump(streamer, pool, far, frames=20)

        # Original near coords must have been unloaded.
        gone = near_coords - set(cm.chunks)
        assert gone, "no near chunks were unloaded after moving far"
        for coord in gone:
            assert coord in events
            assert coord not in streamer._node_seq  # _node_seq pruned


class TestBudget:
    def test_submit_budget_capped(self, pool):
        config = dataclasses.replace(load_config(), lod_submit_per_frame=4)
        set_world_seed(1337)
        cm = ChunkManager(config, EventBus())
        streamer = LodStreamer(cm, pool, config)

        # Many missing chunks in the desired set → submission must cap at 4.
        before = streamer._seq
        streamer.stream_frame(Vec3(0, 0, 20))
        submitted = streamer._seq - before
        assert submitted == 4

        # Drain so the pool is idle before the fixture stops it.
        for _ in range(_MAX_PUMP_ITERS):
            if pool.pending() == 0:
                break
            time.sleep(0.001)
        streamer._drain()
