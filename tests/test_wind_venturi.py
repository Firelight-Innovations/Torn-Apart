"""
tests/test_wind_venturi.py — Headless tests for the wind venturi (WP2).

Covers the terrain-funneling half of the wind system:

- ``venturi.solve_venturi`` correctness: a wall-with-a-gap synthetic terrain
  funnels wind (gap ``speedup > 1.3``), open field is ≈ 1, everything stays in
  ``[1, venturi_max]`` and finite; a no-obstacle job is the identity;
- ``VenturiWorker`` lifecycle (mirror of the assembly-worker tests): on-thread
  ``solve_venturi`` equals the worker submit/drain path byte-for-byte;
  idempotent start, clean drain, ``stop`` joins within its timeout, and a job
  engineered to raise posts a valid IDENTITY result while the thread survives
  to process the next job;
- ``WindField`` integration: a field driven with a started worker + synthetic
  chunks shows a higher sampled wind speed in the gap column than over open
  ground at the same y, and a positive ``vz`` over the constriction edge.

Headless: no window, no GPU, no sky package (weather duck-typed).
"""

from __future__ import annotations

import time

import numpy as np

from fire_engine.core.config import Config
from fire_engine.core.rng import set_world_seed
from fire_engine.world.wind import (
    VenturiJob,
    VenturiResult,
    VenturiWorker,
    WindField,
    solve_venturi,
)

VOXEL = 0.5
CHUNK = 32
SEED = 1337


# ---------------------------------------------------------------------------
# Synthetic terrain helpers
# ---------------------------------------------------------------------------


def _chunks_from_region_solid(
    solid_vox: np.ndarray,
    *,
    vz_lo: int,
    vz_hi: int,
    origin_cell: tuple[int, int],
    cells: int,
    vpc: int,
) -> dict:
    """
    Build a ``coord -> _Chunk`` dict realising a per-(x,y) solid column.

    ``solid_vox`` is a ``(region_vx, region_vy)`` bool array (region voxel
    footprint); each True column is filled solid over voxel-z ``[vz_lo, vz_hi)``.
    The region's voxel-(0,0) corner sits at world voxel
    ``(origin_cell * vpc)`` on each axis (matching ``column_solid_fraction``).
    """
    region_vx, region_vy = solid_vox.shape
    vx0 = origin_cell[0] * vpc
    vy0 = origin_cell[1] * vpc
    chunks: dict = {}

    def _chunk(coord):
        ch = chunks.get(coord)
        if ch is None:
            ch = _Chunk()
            chunks[coord] = ch
        return ch

    xs, ys = np.nonzero(solid_vox)
    for lx, ly in zip(xs.tolist(), ys.tolist()):
        gx = vx0 + lx  # global voxel x
        gy = vy0 + ly
        ccx, ccy = gx // CHUNK, gy // CHUNK
        for ccz in range(vz_lo // CHUNK, (vz_hi - 1) // CHUNK + 1):
            ch = _chunk((ccx, ccy, ccz))
            az = max(ccz * CHUNK, vz_lo) - ccz * CHUNK
            bz = min(ccz * CHUNK + CHUNK, vz_hi) - ccz * CHUNK
            ch.materials[gx - ccx * CHUNK, gy - ccy * CHUNK, az:bz] = 1
    return chunks


class _Chunk:
    """Minimal chunk stand-in: just a 32³ materials array."""

    def __init__(self) -> None:
        self.materials = np.zeros((CHUNK, CHUNK, CHUNK), dtype=np.uint8)


def _wall_with_gap_job(cfg: Config, seq: int = 1) -> tuple[VenturiJob, int, int]:
    """
    A small venturi job: a wall (perpendicular to X) with a 2-cell gap in Y.

    Returns ``(job, wall_cell_x, gap_cell_y)`` where the wall sits at wind-cell
    x index ``wall_cell_x`` (solid for all y except the gap), and the gap is at
    wind-cell y indices ``gap_cell_y`` and ``gap_cell_y + 1``.
    """
    cells = 16
    cell_m = float(cfg.wind_cell_m)  # 4.0
    vpc = int(round(cell_m / VOXEL))  # 8
    region_v = cells * vpc  # 128 voxels per axis
    ground = float(cfg.ground_height_m)
    vz_lo = int(np.floor(ground / VOXEL))
    vz_hi = int(np.ceil((ground + float(cfg.wind_layer_m)) / VOXEL))

    wall_cell_x = 8
    gap_cell_y = 7

    solid = np.zeros((region_v, region_v), dtype=bool)
    # Wall: all of the wall cell's voxel columns, every y.
    solid[wall_cell_x * vpc : (wall_cell_x + 1) * vpc, :] = True
    # Punch a 2-cell-wide gap in Y.
    solid[
        wall_cell_x * vpc : (wall_cell_x + 1) * vpc, gap_cell_y * vpc : (gap_cell_y + 2) * vpc
    ] = False

    chunks = _chunks_from_region_solid(
        solid, vz_lo=vz_lo, vz_hi=vz_hi, origin_cell=(0, 0), cells=cells, vpc=vpc
    )
    materials = {c: ch.materials for c, ch in chunks.items()}

    job = VenturiJob(
        origin_cell=(0, 0),
        cells=cells,
        cell_m=cell_m,
        chunk_size=CHUNK,
        voxel_size=VOXEL,
        ground_band=(ground, ground + float(cfg.wind_layer_m)),
        materials=materials,
        venturi_iters=int(cfg.wind_venturi_iters),
        venturi_max=float(cfg.wind_venturi_max),
        deflect_gain=float(cfg.wind_deflect_gain),
        seq=seq,
    )
    return job, wall_cell_x, gap_cell_y


def _empty_job(cfg: Config, seq: int = 1) -> VenturiJob:
    """A job with no loaded terrain — must produce the identity correction."""
    cells = 16
    ground = float(cfg.ground_height_m)
    return VenturiJob(
        origin_cell=(0, 0),
        cells=cells,
        cell_m=float(cfg.wind_cell_m),
        chunk_size=CHUNK,
        voxel_size=VOXEL,
        ground_band=(ground, ground + float(cfg.wind_layer_m)),
        materials={},
        venturi_iters=int(cfg.wind_venturi_iters),
        venturi_max=float(cfg.wind_venturi_max),
        deflect_gain=float(cfg.wind_deflect_gain),
        seq=seq,
    )


def _drain_until(worker, want: int, timeout_s: float = 5.0):
    """Poll the worker until ``want`` results arrive (or timeout)."""
    out = []
    deadline = time.monotonic() + timeout_s
    while len(out) < want and time.monotonic() < deadline:
        out += worker.drain_results()
        if len(out) < want:
            time.sleep(0.001)
    return out


# ---------------------------------------------------------------------------
# solve_venturi correctness
# ---------------------------------------------------------------------------


class TestSolveVenturi:
    def test_gap_speedup_open_identity_and_bounds(self):
        cfg = Config()
        job, wall_x, gap_y = _wall_with_gap_job(cfg)
        res = solve_venturi(job)
        sp = res.speedup

        # Everything is finite and within [1, max].
        assert np.isfinite(sp).all()
        assert np.isfinite(res.deflect).all()
        assert sp.min() >= 1.0 - 1e-5
        assert sp.max() <= float(cfg.wind_venturi_max) + 1e-5

        # The gap (open cells flanked by wall) funnels: speedup > 1.3 there.
        gap = sp[wall_x, gap_y : gap_y + 2]
        assert gap.max() > 1.3, f"gap speedup too low: {gap}"

        # Open field far from the wall is ≈ 1 (no funneling).
        open_region = sp[0:4, :]
        assert open_region.max() < 1.05, f"open field not flat: {open_region.max()}"

    def test_no_obstacle_is_identity(self):
        cfg = Config()
        res = solve_venturi(_empty_job(cfg))
        assert np.array_equal(res.speedup, np.ones((16, 16), dtype=np.float32))
        assert np.array_equal(res.deflect, np.zeros((16, 16, 2), dtype=np.float32))

    def test_pure_function_repeatable(self):
        cfg = Config()
        job, _, _ = _wall_with_gap_job(cfg)
        a = solve_venturi(job)
        b = solve_venturi(job)
        assert np.array_equal(a.speedup, b.speedup)
        assert np.array_equal(a.deflect, b.deflect)


# ---------------------------------------------------------------------------
# VenturiWorker thread
# ---------------------------------------------------------------------------


class TestVenturiWorker:
    def test_worker_matches_on_thread(self):
        cfg = Config()
        job, _, _ = _wall_with_gap_job(cfg, seq=7)
        inline = solve_venturi(job)

        worker = VenturiWorker()
        worker.start()
        try:
            worker.submit(job)
            results = _drain_until(worker, 1)
            assert len(results) == 1
            res = results[0]
            assert res.seq == 7
            assert res.origin_cell == (0, 0)
            assert np.array_equal(res.speedup, inline.speedup)
            assert np.array_equal(res.deflect, inline.deflect)
            assert worker.pending() == 0
        finally:
            worker.stop()

    def test_start_is_idempotent(self):
        worker = VenturiWorker()
        worker.start()
        worker.start()  # second start is a no-op
        worker.stop()

    def test_stop_is_clean_and_idempotent(self):
        worker = VenturiWorker()
        worker.start()
        worker.stop()
        worker.stop()  # second stop must not raise

    def test_stop_joins_within_timeout(self):
        worker = VenturiWorker()
        worker.start()
        t = worker._thread
        worker.stop(join=True, timeout=2.0)
        assert t is not None and not t.is_alive()

    def test_pending_tracks_inflight(self):
        cfg = Config()
        worker = VenturiWorker()
        worker.start()
        try:
            for i in range(3):
                job, _, _ = _wall_with_gap_job(cfg, seq=i)
                worker.submit(job)
            drained = _drain_until(worker, 3)
            assert len(drained) == 3
            assert worker.pending() == 0
        finally:
            worker.stop()

    def test_raising_job_posts_identity_and_thread_survives(self):
        cfg = Config()
        worker = VenturiWorker()
        worker.start()
        try:
            # A job whose materials dict raises when iterated → solve raises.
            class _Boom(dict):
                def items(self):
                    raise RuntimeError("boom")

            bad = VenturiJob(
                origin_cell=(0, 0),
                cells=16,
                cell_m=float(cfg.wind_cell_m),
                chunk_size=CHUNK,
                voxel_size=VOXEL,
                ground_band=(8.0, 16.0),
                materials=_Boom({(0, 0, 0): 1}),
                venturi_iters=int(cfg.wind_venturi_iters),
                venturi_max=float(cfg.wind_venturi_max),
                deflect_gain=float(cfg.wind_deflect_gain),
                seq=99,
            )
            worker.submit(bad)
            results = _drain_until(worker, 1)
            assert len(results) == 1
            # Identity result posted (worker did not die silently).
            assert results[0].seq == 99
            assert np.array_equal(results[0].speedup, np.ones((16, 16), dtype=np.float32))

            # The thread survived: a normal job after the raise still solves.
            good, _, _ = _wall_with_gap_job(cfg, seq=100)
            worker.submit(good)
            more = _drain_until(worker, 1)
            assert len(more) == 1
            assert more[0].seq == 100
            assert more[0].speedup.max() > 1.3
        finally:
            worker.stop()


# ---------------------------------------------------------------------------
# WindField integration
# ---------------------------------------------------------------------------


def _sky(wind_dir=(1.0, 0.0), wind_speed=6.0):
    from types import SimpleNamespace

    return SimpleNamespace(
        wind_dir=wind_dir,
        wind_speed=wind_speed,
        rain_intensity=0.0,
        cloud_coverage=0.0,
        cloud_density=0.0,
    )


class TestWindFieldIntegration:
    def test_gap_samples_faster_and_updraft_positive(self):
        cfg = Config()
        set_world_seed(SEED)
        worker = VenturiWorker()
        worker.start()
        try:
            field = WindField(cfg, worker)
            sky = _sky()
            # Region origin lands at (0,0) when the player sits at the region
            # centre; player at +X half the tile keeps origin (0,0)-ish — but
            # we drive update() with the player at the centre of a (0,0) tile.
            cells = int(cfg.wind_cells)
            cell_m = float(cfg.wind_cell_m)
            centre = (cells * 0.5) * cell_m
            player = (centre, centre, cfg.ground_height_m)

            # Build a full-region wall-with-gap matched to the field's region.
            field.update(0.016, 5.0, sky, player)  # places the region
            origin = field._region.origin_cell
            vpc = int(round(cell_m / VOXEL))
            region_v = cells * vpc
            ground = float(cfg.ground_height_m)
            vz_lo = int(np.floor(ground / VOXEL))
            vz_hi = int(np.ceil((ground + float(cfg.wind_layer_m)) / VOXEL))

            wall_x = cells // 2
            gap_y = cells // 2
            solid = np.zeros((region_v, region_v), dtype=bool)
            solid[wall_x * vpc : (wall_x + 1) * vpc, :] = True
            solid[wall_x * vpc : (wall_x + 1) * vpc, gap_y * vpc : (gap_y + 2) * vpc] = False
            chunks = _chunks_from_region_solid(
                solid, vz_lo=vz_lo, vz_hi=vz_hi, origin_cell=origin, cells=cells, vpc=vpc
            )

            # Drive updates passing chunks until a matching-origin result lands.
            deadline = time.monotonic() + 5.0
            while field._venturi_origin != origin and time.monotonic() < deadline:
                field.update(0.016, 5.0, sky, player, chunks=chunks)
                time.sleep(0.002)
            assert field._venturi_origin == origin, "venturi result never landed"
            assert field._venturi_speedup.max() > 1.3

            # World XY of the wall cell's gap centre vs an open cell at same y.
            ox, oy = field._region.origin_m
            wall_world_x = (origin[0] + wall_x + 0.5) * cell_m
            gap_world_y = (origin[1] + gap_y + 0.5) * cell_m
            z = ground + 1.0

            gap_pt = np.array([[wall_world_x, gap_world_y, z]], dtype=np.float32)
            # Open ground at the same y but well upwind of the wall.
            open_world_x = (origin[0] + 1 + 0.5) * cell_m
            open_pt = np.array([[open_world_x, gap_world_y, z]], dtype=np.float32)

            gap_v = field.sample(gap_pt)[0]
            open_v = field.sample(open_pt)[0]
            gap_speed = float(np.hypot(gap_v[0], gap_v[1]))
            open_speed = float(np.hypot(open_v[0], open_v[1]))
            assert gap_speed > open_speed, (gap_speed, open_speed)

            # Updraft is positive somewhere over the constriction (speedup>1).
            # Sample a grid of points over the wall row at the gap edges.
            ys = (origin[1] + np.arange(cells) + 0.5) * cell_m
            pts = np.stack(
                [
                    np.full(cells, wall_world_x, dtype=np.float32),
                    ys.astype(np.float32),
                    np.full(cells, z, dtype=np.float32),
                ],
                axis=1,
            )
            vz = field.sample(pts)[:, 2]
            assert vz.max() > 0.0, "no positive updraft over the constriction"
        finally:
            worker.stop()
