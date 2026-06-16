"""
tests/world/terrain/lod/test_pool.py — TerrainLodPool round-trip + error sentinel.

Headless: no panda3d imports.  Verifies threaded results equal the synchronous
build, seq is preserved, and a raised job still produces a (sentinel) result so
the consumer never starves.
"""

from __future__ import annotations

import time

import numpy as np

from fire_engine.core import EventBus, load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.world.terrain.chunk_manager import ChunkManager
from fire_engine.world.terrain.lod.job import LodJob, build_lod_mesh
from fire_engine.world.terrain.lod.pool import TerrainLodPool

_MAX_DRAIN_ITERS = 2000  # bounded sleep loop (≈2 s worst case at 1 ms/iter)


def _copy_neighbors(raw):
    return {k: (v if isinstance(v, str) else np.asarray(v).copy()) for k, v in raw.items()}


def _make_job(cm, config, coord, seq):
    chunk = cm.get_or_create(coord)
    return LodJob(
        coord=coord,
        materials=chunk.materials.copy(),
        neighbors=_copy_neighbors(cm._neighbor_materials(coord)),
        chunk_size=int(config.chunk_size),
        voxel_size=float(config.voxel_size),
        shade_strength=float(config.facet_shade_strength),
        mesh_style="faceted",
        seq=seq,
    )


def _drain_until(pool, expected_count):
    """Drain in a bounded sleep loop until *expected_count* results arrive."""
    results = []
    for _ in range(_MAX_DRAIN_ITERS):
        results.extend(pool.drain_results())
        if len(results) >= expected_count:
            break
        time.sleep(0.001)
    return results


class TestPoolRoundTrip:
    def test_submit_drain_matches_synchronous(self):
        config = load_config()
        set_world_seed(1337)
        cm = ChunkManager(config, EventBus())
        coords = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1)]
        jobs = {c: _make_job(cm, config, c, seq=i) for i, c in enumerate(coords)}

        pool = TerrainLodPool(n_workers=2)
        pool.start()
        try:
            for job in jobs.values():
                pool.submit(job)
            results = _drain_until(pool, len(coords))
        finally:
            pool.stop()

        assert len(results) == len(coords)
        by_coord = {r.coord: r for r in results}  # drain order is not submit order
        assert set(by_coord) == set(coords)
        for coord, job in jobs.items():
            r = by_coord[coord]
            assert r.seq == job.seq  # seq preserved
            expected = build_lod_mesh(job).mesh
            assert np.array_equal(r.mesh.positions, expected.positions)
            assert np.array_equal(r.mesh.normals, expected.normals)
            assert np.array_equal(r.mesh.uvs, expected.uvs)
            assert np.array_equal(r.mesh.colors, expected.colors)
            assert np.array_equal(r.mesh.indices, expected.indices)
            assert np.array_equal(r.mesh.face_materials, expected.face_materials)


class TestErrorSentinel:
    def test_bad_job_returns_sentinel_and_good_jobs_complete(self):
        config = load_config()
        set_world_seed(1337)
        cm = ChunkManager(config, EventBus())

        good = _make_job(cm, config, (0, 0, 0), seq=10)
        # Wrong-shape materials → Chunk(...) raises ValueError inside the worker.
        bad = LodJob(
            coord=(5, 5, 5),
            materials=np.zeros((4, 4, 4), dtype=np.uint8),
            neighbors={},
            chunk_size=int(config.chunk_size),
            voxel_size=float(config.voxel_size),
            shade_strength=float(config.facet_shade_strength),
            mesh_style="faceted",
            seq=11,
        )
        good2 = _make_job(cm, config, (1, 0, 0), seq=12)

        pool = TerrainLodPool(n_workers=2)
        pool.start()
        try:
            pool.submit(good)
            pool.submit(bad)
            pool.submit(good2)
            results = _drain_until(pool, 3)
        finally:
            pool.stop()

        by_coord = {r.coord: r for r in results}
        assert set(by_coord) == {(0, 0, 0), (5, 5, 5), (1, 0, 0)}
        # Sentinel for the bad job: empty mesh, seq preserved.
        sentinel = by_coord[(5, 5, 5)]
        assert sentinel.seq == 11
        assert sentinel.mesh.is_empty
        # Good jobs still completed correctly.
        assert by_coord[(0, 0, 0)].seq == 10
        assert not by_coord[(0, 0, 0)].mesh.is_empty
        assert by_coord[(1, 0, 0)].seq == 12
