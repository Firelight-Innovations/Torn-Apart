"""
tests/world/terrain/lod/test_coarse_streamer.py — CoarseLodStreamer async driver.

Headless, no panda3d.  Most tests drive the streamer against a FAKE pool that
records submitted jobs and lets the test inject controlled results, so they run
in milliseconds and exercise the plan/submit/drain/retire/staleness logic
deterministically.  One small end-to-end test uses a real ``TerrainLodPool`` with
a tiny far-radius to prove a coarse node genuinely meshes off-thread into
``pending_coarse_meshes``.

Covered: channels are created on the manager; rank>0 coarse jobs are submitted
(coarsest-rank-first) up to the budget; the per-node ``seq`` staleness map keeps
only the newest result and drops stale/undesired ones; a node that leaves the
desired set is retired into ``unloaded_coarse_this_frame``; ``max_rank=0`` submits
nothing and retires everything; the real pool produces a non-empty coarse mesh.
"""

from __future__ import annotations

import dataclasses
import time

from fire_engine.core import EventBus, load_config
from fire_engine.core.math3d import Vec3
from fire_engine.core.rng import set_world_seed
from fire_engine.world.terrain.chunk_manager import ChunkManager
from fire_engine.world.terrain.lod import CoarseLodStreamer, TerrainLodPool, build_lod_mesh
from fire_engine.world.terrain.lod.types import LodJob, LodResult

_MAX_PUMP_ITERS = 2000  # bounded sleep loop (~2 s worst case at 1 ms/iter)


class _FakePool:
    """Records submitted jobs; returns whatever the test stages as results.

    Mirrors the ``TerrainLodPool`` surface the streamer uses (``submit`` /
    ``drain_results``) without spawning threads, so streamer logic is tested
    synchronously and fast.
    """

    def __init__(self) -> None:
        self.submitted: list[LodJob] = []
        self._staged: list[LodResult] = []

    def submit(self, job: LodJob) -> None:
        self.submitted.append(job)

    def stage(self, result: LodResult) -> None:
        self._staged.append(result)

    def drain_results(self) -> list[LodResult]:
        out = self._staged
        self._staged = []
        return out


def _coarse_config(**overrides):
    """A config with coarse ranks on and a SMALL far radius (cheap plans)."""
    base = load_config()
    defaults = dict(lod_max_rank=2, lod_far_radius_chunks=12, lod_near_radius_chunks=4)
    defaults.update(overrides)
    return dataclasses.replace(base, **defaults)


def _make_cm(config) -> ChunkManager:
    set_world_seed(1337)
    return ChunkManager(config, EventBus())


class TestChannels:
    def test_channels_present_after_construct(self) -> None:
        cm = _make_cm(_coarse_config())
        CoarseLodStreamer(cm, _FakePool(), cm.config)
        assert hasattr(cm, "pending_coarse_meshes")
        assert hasattr(cm, "unloaded_coarse_this_frame")


class TestSubmit:
    def test_submits_coarse_rank_jobs_within_budget(self) -> None:
        cfg = _coarse_config(lod_coarse_submit_per_frame=3)
        cm = _make_cm(cfg)
        pool = _FakePool()
        cs = CoarseLodStreamer(cm, pool, cfg)

        cs.stream_frame(Vec3(0, 0, 20))

        assert len(pool.submitted) == 3  # capped at the budget
        for job in pool.submitted:
            assert job.rank >= 1  # coarse only
            assert job.materials.shape == (32, 32, 32)
            assert job.voxel_size == cfg.voxel_size * (1 << job.rank)
            assert job.neighbors == {}  # open coarse borders (P2)

    def test_coarsest_rank_submitted_first(self) -> None:
        cfg = _coarse_config(lod_max_rank=2, lod_coarse_submit_per_frame=2)
        cm = _make_cm(cfg)
        pool = _FakePool()
        cs = CoarseLodStreamer(cm, pool, cfg)

        cs.stream_frame(Vec3(0, 0, 20))
        # Coarsest-far-first: the very first submitted job is the max rank.
        assert pool.submitted, "no coarse jobs submitted"
        assert pool.submitted[0].rank == cfg.lod_max_rank

    def test_max_rank_zero_submits_nothing(self) -> None:
        cfg = _coarse_config(lod_max_rank=0)
        cm = _make_cm(cfg)
        pool = _FakePool()
        cs = CoarseLodStreamer(cm, pool, cfg)
        cs.stream_frame(Vec3(0, 0, 20))
        assert pool.submitted == []
        assert cm.pending_coarse_meshes == {}


class TestStaleness:
    def test_newest_seq_wins_stale_dropped(self) -> None:
        cfg = _coarse_config(lod_coarse_submit_per_frame=1)
        cm = _make_cm(cfg)
        pool = _FakePool()
        cs = CoarseLodStreamer(cm, pool, cfg)

        cs.stream_frame(Vec3(0, 0, 20))
        assert len(pool.submitted) == 1
        job = pool.submitted[0]
        key = (job.rank, *job.coord)
        live_seq = cs._node_seq[key]

        # A stale result (older seq) for the same node must be dropped.
        stale = LodResult(job.coord, build_lod_mesh(job).mesh, live_seq - 1, job.rank)
        pool.stage(stale)
        cs._drain()
        assert key not in cm.pending_coarse_meshes

        # The live-seq result lands and clears the staleness guard.
        fresh = LodResult(job.coord, build_lod_mesh(job).mesh, live_seq, job.rank)
        pool.stage(fresh)
        cs._drain()
        assert key in cm.pending_coarse_meshes
        assert key not in cs._node_seq  # guard released once delivered

    def test_rank_zero_result_ignored(self) -> None:
        # The coarse drain must never claim a rank-0 (near) result.
        cfg = _coarse_config()
        cm = _make_cm(cfg)
        pool = _FakePool()
        cs = CoarseLodStreamer(cm, pool, cfg)
        mesh = build_lod_mesh(
            LodJob(
                (0, 0, 0),
                cm.get_or_create((0, 0, 0)).materials.copy(),
                {},
                32,
                cfg.voxel_size,
                cfg.facet_shade_strength,
                "faceted",
                1,
            )
        ).mesh
        pool.stage(LodResult((0, 0, 0), mesh, 1, rank=0))
        cs._drain()
        assert cm.pending_coarse_meshes == {}


class TestRetire:
    def test_node_leaving_desired_is_retired(self) -> None:
        cfg = _coarse_config(lod_coarse_submit_per_frame=2)
        cm = _make_cm(cfg)
        pool = _FakePool()
        cs = CoarseLodStreamer(cm, pool, cfg)

        # Frame 1 near the origin: submit + deliver one node's mesh.
        cs.stream_frame(Vec3(0, 0, 20))
        job = pool.submitted[0]
        key = (job.rank, *job.coord)
        assert key in cs._node_seq  # tracked
        pool.stage(LodResult(job.coord, build_lod_mesh(job).mesh, cs._node_seq[key], job.rank))
        cs._drain()
        assert key in cm.pending_coarse_meshes

        # Frame 2 far away: that node leaves the desired set -> retired.
        cs.stream_frame(Vec3(50_000, 50_000, 20))
        assert key in cm.unloaded_coarse_this_frame
        assert key not in cm.pending_coarse_meshes
        assert key not in cs._node_seq

    def test_max_rank_zero_retires_everything(self) -> None:
        cfg = _coarse_config(lod_coarse_submit_per_frame=2)
        cm = _make_cm(cfg)
        pool = _FakePool()
        cs = CoarseLodStreamer(cm, pool, cfg)
        cs.stream_frame(Vec3(0, 0, 20))
        tracked = set(cs._node_seq)
        assert tracked

        # Flip coarse off mid-session: every tracked node must be retired.
        cs._config = dataclasses.replace(cfg, lod_max_rank=0)
        cs.stream_frame(Vec3(0, 0, 20))
        for key in tracked:
            assert key in cm.unloaded_coarse_this_frame
        assert cs._node_seq == {}

    def test_retire_is_idempotent_not_every_frame(self) -> None:
        # Regression: a node must be retired EXACTLY ONCE, not re-reported in
        # unloaded_coarse_this_frame on every subsequent frame (which would make
        # the render layer re-detach phantom nodes forever).
        cfg = _coarse_config(lod_coarse_submit_per_frame=2)
        cm = _make_cm(cfg)
        cs = CoarseLodStreamer(cm, _FakePool(), cfg)
        cs.stream_frame(Vec3(0, 0, 20))
        tracked = set(cs._node_seq)
        assert tracked

        cs._config = dataclasses.replace(cfg, lod_max_rank=0)
        cs.stream_frame(Vec3(0, 0, 20))  # frame 2: retire the tracked nodes once
        assert set(cm.unloaded_coarse_this_frame) == tracked
        cs.stream_frame(Vec3(0, 0, 20))  # frame 3: nothing left — must be empty
        assert cm.unloaded_coarse_this_frame == []


class TestRealPoolEndToEnd:
    def test_real_pool_produces_coarse_mesh(self) -> None:
        # A genuine off-thread coarse mesh lands in pending_coarse_meshes.
        cfg = _coarse_config(lod_max_rank=1, lod_far_radius_chunks=8, lod_coarse_submit_per_frame=4)
        cm = _make_cm(cfg)
        pool = TerrainLodPool(n_workers=2)
        pool.start()
        try:
            cs = CoarseLodStreamer(cm, pool, cfg)
            cam = Vec3(0, 0, 20)
            for _ in range(30):
                cs.stream_frame(cam)
                for _ in range(_MAX_PUMP_ITERS):
                    if pool.pending() == 0:
                        break
                    time.sleep(0.001)
                if cm.pending_coarse_meshes:
                    break
            assert cm.pending_coarse_meshes, "real pool produced no coarse meshes"
            # Every delivered key is a coarse 4-tuple at rank>=1.
            for key, mesh in cm.pending_coarse_meshes.items():
                assert len(key) == 4 and key[0] >= 1
                assert mesh is not None
        finally:
            pool.stop()
