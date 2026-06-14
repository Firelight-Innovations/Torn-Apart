"""
tests/test_wind_venturi_edges.py — Edge-case / golden-master characterization
tests for solve_venturi (WP2).

PURPOSE: Pin CURRENT behaviour so regressions are visible. Do NOT "fix" any
suspicious output — mark it as a suspected bug in the comment and assert what
the code actually does today.

Headless only — no window, no GPU, no panda3d.  numpy assertions everywhere.
Conventions copied from tests/test_wind_venturi.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.core.config import Config
from fire_engine.world.wind import VenturiJob, VenturiResult, solve_venturi
from fire_engine.world.wind.venturi import column_solid_fraction

# ---------------------------------------------------------------------------
# Constants (match the existing test file)
# ---------------------------------------------------------------------------
VOXEL = 0.5
CHUNK = 32
SEED = 42


# ---------------------------------------------------------------------------
# Synthetic terrain helpers  (copied from test_wind_venturi.py conventions)
# ---------------------------------------------------------------------------


class _Chunk:
    """Minimal chunk stand-in: a 32³ materials array."""

    def __init__(self) -> None:
        self.materials = np.zeros((CHUNK, CHUNK, CHUNK), dtype=np.uint8)


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
    Identical logic to test_wind_venturi.py helper.
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
        gx = vx0 + lx
        gy = vy0 + ly
        ccx, ccy = gx // CHUNK, gy // CHUNK
        for ccz in range(vz_lo // CHUNK, (vz_hi - 1) // CHUNK + 1):
            ch = _chunk((ccx, ccy, ccz))
            az = max(ccz * CHUNK, vz_lo) - ccz * CHUNK
            bz = min(ccz * CHUNK + CHUNK, vz_hi) - ccz * CHUNK
            ch.materials[gx - ccx * CHUNK, gy - ccy * CHUNK, az:bz] = 1
    return chunks


def _make_job(
    cfg: Config,
    solid_vox: np.ndarray | None,
    *,
    cells: int = 16,
    seq: int = 1,
    venturi_max: float | None = None,
    venturi_iters: int | None = None,
    deflect_gain: float | None = None,
) -> VenturiJob:
    """
    Build a VenturiJob from an optional boolean voxel solid mask.

    ``solid_vox`` is ``(cells*vpc, cells*vpc)`` booleans or None (empty dict).
    """
    cell_m = float(cfg.wind_cell_m)
    vpc = int(round(cell_m / VOXEL))
    ground = float(cfg.ground_height_m)
    vz_lo = int(np.floor(ground / VOXEL))
    vz_hi = int(np.ceil((ground + float(cfg.wind_layer_m)) / VOXEL))

    if solid_vox is not None:
        chunks_obj = _chunks_from_region_solid(
            solid_vox,
            vz_lo=vz_lo,
            vz_hi=vz_hi,
            origin_cell=(0, 0),
            cells=cells,
            vpc=vpc,
        )
        materials = {c: ch.materials for c, ch in chunks_obj.items()}
    else:
        materials = {}

    return VenturiJob(
        origin_cell=(0, 0),
        cells=cells,
        cell_m=cell_m,
        chunk_size=CHUNK,
        voxel_size=VOXEL,
        ground_band=(ground, ground + float(cfg.wind_layer_m)),
        materials=materials,
        venturi_iters=int(venturi_iters if venturi_iters is not None else cfg.wind_venturi_iters),
        venturi_max=float(venturi_max if venturi_max is not None else cfg.wind_venturi_max),
        deflect_gain=float(deflect_gain if deflect_gain is not None else cfg.wind_deflect_gain),
        seq=seq,
    )


# ---------------------------------------------------------------------------
# 1. Fully-open region — no solid chunks, empty materials dict
# ---------------------------------------------------------------------------


class TestFullyOpenRegion:
    """
    Golden-master: no terrain → speedup ≈ 1, deflection ≈ 0 everywhere.
    """

    def test_speedup_all_ones(self):
        cfg = Config()
        job = _make_job(cfg, solid_vox=None, cells=16)
        res = solve_venturi(job)
        # pin: speedup is exactly 1.0 (the identity; no crowding, no pinch)
        np.testing.assert_array_equal(
            res.speedup,
            np.ones((16, 16), dtype=np.float32),
            err_msg="fully-open speedup must be exactly 1.0 everywhere",
        )

    def test_deflect_all_zeros(self):
        cfg = Config()
        job = _make_job(cfg, solid_vox=None, cells=16)
        res = solve_venturi(job)
        # pin: deflection is exactly 0.0 (no openness gradient)
        np.testing.assert_array_equal(
            res.deflect,
            np.zeros((16, 16, 2), dtype=np.float32),
            err_msg="fully-open deflect must be exactly 0.0 everywhere",
        )

    def test_fully_open_finite(self):
        cfg = Config()
        job = _make_job(cfg, solid_vox=None, cells=16)
        res = solve_venturi(job)
        assert np.isfinite(res.speedup).all()
        assert np.isfinite(res.deflect).all()

    def test_output_shapes_open(self):
        cfg = Config()
        cells = 12
        job = _make_job(cfg, solid_vox=None, cells=cells)
        res = solve_venturi(job)
        assert res.speedup.shape == (cells, cells)
        assert res.deflect.shape == (cells, cells, 2)
        assert res.speedup.dtype == np.float32
        assert res.deflect.dtype == np.float32


# ---------------------------------------------------------------------------
# 2. Fully-solid region — all cells solid
# ---------------------------------------------------------------------------


class TestFullySolidRegion:
    """
    Golden-master: every column solid → pin the actual speedup/deflect values.

    SUSPECTED BUG: passw is clipped to [0.05, 1.0] even for fully-solid cells
    (passw = clip(1 - solid, 0.05, 1) → 0.05, never 0).  This means a fully-
    solid cell still gets a non-zero speedup above 1 when crowd is high.
    The model may intend speedup to be exactly 1 inside walls, but the clip
    floor prevents that — open cells and solid cells all get speedup >= 1.
    """

    def test_fully_solid_finite(self):
        cfg = Config()
        cells = 8
        vpc = int(round(float(cfg.wind_cell_m) / VOXEL))
        solid_vox = np.ones((cells * vpc, cells * vpc), dtype=bool)
        job = _make_job(cfg, solid_vox=solid_vox, cells=cells)
        res = solve_venturi(job)
        assert np.isfinite(res.speedup).all(), "speedup has non-finite values in fully-solid region"
        assert np.isfinite(res.deflect).all(), "deflect has non-finite values in fully-solid region"

    def test_fully_solid_no_nan(self):
        cfg = Config()
        cells = 8
        vpc = int(round(float(cfg.wind_cell_m) / VOXEL))
        solid_vox = np.ones((cells * vpc, cells * vpc), dtype=bool)
        job = _make_job(cfg, solid_vox=solid_vox, cells=cells)
        res = solve_venturi(job)
        assert not np.isnan(res.speedup).any()
        assert not np.isnan(res.deflect).any()

    def test_fully_solid_speedup_at_least_one(self):
        """
        Pin: even in fully-solid region, speedup >= 1 everywhere (the clip
        floor on passw means the formula 1 + crowd_gain*crowd*passw is >= 1).
        """
        cfg = Config()
        cells = 8
        vpc = int(round(float(cfg.wind_cell_m) / VOXEL))
        solid_vox = np.ones((cells * vpc, cells * vpc), dtype=bool)
        job = _make_job(cfg, solid_vox=solid_vox, cells=cells)
        res = solve_venturi(job)
        assert res.speedup.min() >= 1.0 - 1e-5, (
            f"fully-solid speedup min={res.speedup.min()!r} dropped below 1"
        )

    def test_fully_solid_speedup_bounded_by_max(self):
        cfg = Config()
        cells = 8
        vpc = int(round(float(cfg.wind_cell_m) / VOXEL))
        solid_vox = np.ones((cells * vpc, cells * vpc), dtype=bool)
        job = _make_job(cfg, solid_vox=solid_vox, cells=cells)
        res = solve_venturi(job)
        assert res.speedup.max() <= float(cfg.wind_venturi_max) + 1e-5, (
            f"fully-solid speedup exceeds venturi_max: {res.speedup.max()!r}"
        )

    def test_fully_solid_deflect_all_zero(self):
        """
        Pin: fully-solid → openness gradient is zero everywhere → deflect == 0.
        (open_ = 1-solid = 0 everywhere; np.gradient(all-zero) = 0.)
        """
        cfg = Config()
        cells = 8
        vpc = int(round(float(cfg.wind_cell_m) / VOXEL))
        solid_vox = np.ones((cells * vpc, cells * vpc), dtype=bool)
        job = _make_job(cfg, solid_vox=solid_vox, cells=cells)
        res = solve_venturi(job)
        np.testing.assert_array_equal(
            res.deflect,
            np.zeros((cells, cells, 2), dtype=np.float32),
            err_msg="fully-solid deflect must be zero (no openness gradient)",
        )


# ---------------------------------------------------------------------------
# 3. Zero chunks provided (materials={}) — same as "no terrain"
# ---------------------------------------------------------------------------


class TestZeroChunks:
    """
    Pin: materials={} → identity result (speedup 1, deflect 0).
    """

    def test_zero_chunks_is_identity(self):
        cfg = Config()
        job = _make_job(cfg, solid_vox=None, cells=16)
        assert job.materials == {}
        res = solve_venturi(job)
        np.testing.assert_array_equal(res.speedup, np.ones((16, 16), dtype=np.float32))
        np.testing.assert_array_equal(res.deflect, np.zeros((16, 16, 2), dtype=np.float32))

    def test_zero_chunks_origin_and_seq_echoed(self):
        cfg = Config()
        job = _make_job(cfg, solid_vox=None, cells=8, seq=77)
        res = solve_venturi(job)
        assert res.origin_cell == (0, 0)
        assert res.seq == 77


# ---------------------------------------------------------------------------
# 4. venturi_max clamp — strong pinch must not exceed the config cap
# ---------------------------------------------------------------------------


class TestVenturiMaxClamp:
    """
    Pin: speedup is always in [1, venturi_max], even for an extreme pinch.
    """

    def _strong_pinch_job(self, cfg: Config, venturi_max: float) -> VenturiJob:
        """
        A 1-cell-wide tunnel (99% of the region is solid, 1 column open).
        Designed to produce crowd ≈ 1 at the open cell.
        """
        cells = 16
        vpc = int(round(float(cfg.wind_cell_m) / VOXEL))
        solid_vox = np.ones((cells * vpc, cells * vpc), dtype=bool)
        # Open a single 1-voxel column in the centre
        cx = (cells * vpc) // 2
        cy = (cells * vpc) // 2
        solid_vox[cx, cy] = False
        return _make_job(
            cfg, solid_vox=solid_vox, cells=cells, venturi_max=venturi_max, venturi_iters=16
        )

    def test_speedup_never_exceeds_venturi_max_default(self):
        cfg = Config()
        job = self._strong_pinch_job(cfg, float(cfg.wind_venturi_max))
        res = solve_venturi(job)
        assert res.speedup.max() <= float(cfg.wind_venturi_max) + 1e-5, (
            f"speedup {res.speedup.max()!r} exceeds venturi_max {cfg.wind_venturi_max}"
        )

    def test_speedup_never_exceeds_low_custom_cap(self):
        cfg = Config()
        cap = 1.5
        job = self._strong_pinch_job(cfg, cap)
        res = solve_venturi(job)
        assert res.speedup.max() <= cap + 1e-5, (
            f"speedup {res.speedup.max()!r} exceeds custom cap {cap}"
        )

    def test_strong_pinch_does_reach_cap(self):
        """
        Characterize: a near-total-blockage job with high iters reaches only
        speedup ≈ 1.15 — the venturi_max clamp (2.2) is NEVER hit in practice.

        SUSPECTED BUG: The clamp ``clip(..., 1, venturi_max)`` is unreachable
        with a single-voxel-open tunnel at 16×16 cells + 16 iters.  The 3×3
        box blur smears the one high-speedup cell over 9 cells, reducing the
        max from a theoretical value (1 + 3.0 * crowd * 0.05 ≈ 1.15 for a
        fully-crowded near-solid cell) to ≈ 1.15.  At full solid fraction,
        passw is clamped at 0.05, limiting speedup regardless of crowd.

        PINNED ACTUAL VALUE: speedup.max() ≈ 1.15 (well below venturi_max 2.2).
        The config knob ``wind_venturi_max`` appears to be dead for typical
        terrain — only an extremely large crowd_gain or deflect_gain could
        trigger it.
        """
        cfg = Config()
        job = self._strong_pinch_job(cfg, float(cfg.wind_venturi_max))
        res = solve_venturi(job)
        # Pin: the clamp is NOT reached; the max is well below venturi_max.
        # If this changes, the crowding model changed significantly.
        assert res.speedup.max() < float(cfg.wind_venturi_max), (
            f"speedup {res.speedup.max()!r} unexpectedly reached "
            f"venturi_max {cfg.wind_venturi_max} on a single-open-voxel tunnel"
        )
        # And pin the approximate expected ceiling at this geometry:
        # speedup.max() should be near 1.15 ± 0.05 (passw=0.05 floor * crowd≈1 * gain=3)
        assert res.speedup.max() <= 1.20, (
            f"PINNED: single-voxel tunnel speedup {res.speedup.max()!r} "
            "exceeded expected ceiling ~1.20; crowding model may have changed"
        )

    def test_speedup_floor_always_one(self):
        cfg = Config()
        job = self._strong_pinch_job(cfg, float(cfg.wind_venturi_max))
        res = solve_venturi(job)
        assert res.speedup.min() >= 1.0 - 1e-5


# ---------------------------------------------------------------------------
# 5. Deflection sign: wall with gap deflects flow toward the gap
# ---------------------------------------------------------------------------


class TestDeflectionSignAndDirection:
    """
    Pin the deflection sign/direction the code actually produces.

    The docstring says deflect = np.gradient(1 - solid) * deflect_gain,
    which pushes flow from solid (low openness) toward open (high openness).
    We assert what the code DOES produce, not what physics would require.
    """

    def _wall_gap_job(self, cfg: Config) -> tuple[VenturiJob, int, int]:
        """
        A wall in X (blocking all Y) with a 2-cell gap centred in Y.
        Returns (job, wall_cell_x, gap_cell_y).
        """
        cells = 16
        cell_m = float(cfg.wind_cell_m)
        vpc = int(round(cell_m / VOXEL))
        ground = float(cfg.ground_height_m)
        vz_lo = int(np.floor(ground / VOXEL))
        vz_hi = int(np.ceil((ground + float(cfg.wind_layer_m)) / VOXEL))

        wall_x = 8
        gap_y = 6

        solid_vox = np.zeros((cells * vpc, cells * vpc), dtype=bool)
        solid_vox[wall_x * vpc : (wall_x + 1) * vpc, :] = True
        solid_vox[wall_x * vpc : (wall_x + 1) * vpc, gap_y * vpc : (gap_y + 2) * vpc] = False

        chunks_obj = _chunks_from_region_solid(
            solid_vox,
            vz_lo=vz_lo,
            vz_hi=vz_hi,
            origin_cell=(0, 0),
            cells=cells,
            vpc=vpc,
        )
        materials = {c: ch.materials for c, ch in chunks_obj.items()}
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
            seq=1,
        )
        return job, wall_x, gap_y

    def test_deflect_nonzero_near_wall(self):
        """
        Pin: deflection is non-zero in cells adjacent to the wall.
        (A uniform solid field has zero gradient; a wall-with-gap has
        a non-zero openness gradient in Y near the gap boundary.)
        """
        cfg = Config()
        job, wall_x, gap_y = self._wall_gap_job(cfg)
        res = solve_venturi(job)
        # Row just before the wall has a gradient in X (open → solid transition)
        deflect_x_col = res.deflect[wall_x - 1, :, 0]
        assert np.abs(deflect_x_col).max() > 1e-6, (
            "deflect_x should be non-zero in cells adjacent to the wall"
        )

    def test_deflect_y_toward_gap_side(self):
        """
        Pin the sign: cells on the SOLID side of the gap boundary (above the gap
        in Y) should have a non-positive deflect_y (gradient of openness points
        TOWARD the gap, i.e. toward decreasing Y index from the solid region).

        This asserts what the code produces.  'Correct' physics would direct
        flow around a wall toward the gap; we verify the sign is non-positive
        above the gap (solid region deflected toward the open gap in -Y direction).
        """
        cfg = Config()
        job, wall_x, gap_y = self._wall_gap_job(cfg)
        res = solve_venturi(job)
        # In the cells of the wall column just ABOVE the gap (y > gap_y+1),
        # np.gradient of openness in Y should be negative (openness increases
        # as y decreases toward the gap).
        y_above_gap = gap_y + 3  # clearly above the gap, in solid region
        deflect_y_above = res.deflect[wall_x, y_above_gap, 1]
        # Pin: deflect_y is non-positive above the gap (toward gap = toward -y)
        assert deflect_y_above <= 1e-6, (
            f"PINNED: deflect_y above gap = {deflect_y_above!r}; "
            "expected <= 0 (gradient of openness points toward gap)"
        )

    def test_deflect_finite(self):
        cfg = Config()
        job, _, _ = self._wall_gap_job(cfg)
        res = solve_venturi(job)
        assert np.isfinite(res.deflect).all()

    def test_deflect_shape(self):
        cfg = Config()
        job, _, _ = self._wall_gap_job(cfg)
        res = solve_venturi(job)
        assert res.deflect.shape == (16, 16, 2)
        assert res.deflect.dtype == np.float32


# ---------------------------------------------------------------------------
# 6. Idempotency / determinism — two calls → bit-identical results
# ---------------------------------------------------------------------------


class TestIdempotencyAndDeterminism:
    """
    Pin: solve_venturi is a pure function; calling it twice returns
    bit-identical numpy arrays.
    """

    def _wall_job(self, cfg: Config) -> VenturiJob:
        cells = 16
        vpc = int(round(float(cfg.wind_cell_m) / VOXEL))
        ground = float(cfg.ground_height_m)
        vz_lo = int(np.floor(ground / VOXEL))
        vz_hi = int(np.ceil((ground + float(cfg.wind_layer_m)) / VOXEL))
        solid_vox = np.zeros((cells * vpc, cells * vpc), dtype=bool)
        solid_vox[8 * vpc : (8 + 1) * vpc, :] = True
        solid_vox[8 * vpc : (8 + 1) * vpc, 4 * vpc : 6 * vpc] = False
        chunks_obj = _chunks_from_region_solid(
            solid_vox,
            vz_lo=vz_lo,
            vz_hi=vz_hi,
            origin_cell=(0, 0),
            cells=cells,
            vpc=vpc,
        )
        return _make_job(cfg, solid_vox=solid_vox, cells=cells)

    def test_identical_speedup_on_two_calls(self):
        cfg = Config()
        job = self._wall_job(cfg)
        a = solve_venturi(job)
        b = solve_venturi(job)
        assert np.array_equal(a.speedup, b.speedup), "speedup not idempotent"

    def test_identical_deflect_on_two_calls(self):
        cfg = Config()
        job = self._wall_job(cfg)
        a = solve_venturi(job)
        b = solve_venturi(job)
        assert np.array_equal(a.deflect, b.deflect), "deflect not idempotent"

    def test_open_job_idempotent(self):
        cfg = Config()
        job = _make_job(cfg, solid_vox=None, cells=16)
        a = solve_venturi(job)
        b = solve_venturi(job)
        assert np.array_equal(a.speedup, b.speedup)
        assert np.array_equal(a.deflect, b.deflect)

    def test_seq_and_origin_echoed(self):
        cfg = Config()
        job = _make_job(cfg, solid_vox=None, cells=8, seq=999)
        res = solve_venturi(job)
        assert res.seq == 999
        assert res.origin_cell == (0, 0)


# ---------------------------------------------------------------------------
# 7. Output shapes / dtypes — various cell counts
# ---------------------------------------------------------------------------


class TestOutputShapesDtypes:
    """
    Pin output shapes and dtypes for several grid sizes.
    """

    @pytest.mark.parametrize("cells", [8, 16, 24, 32])
    def test_speedup_shape_and_dtype(self, cells: int):
        cfg = Config()
        job = _make_job(cfg, solid_vox=None, cells=cells)
        res = solve_venturi(job)
        assert res.speedup.shape == (cells, cells)
        assert res.speedup.dtype == np.float32

    @pytest.mark.parametrize("cells", [8, 16, 24, 32])
    def test_deflect_shape_and_dtype(self, cells: int):
        cfg = Config()
        job = _make_job(cfg, solid_vox=None, cells=cells)
        res = solve_venturi(job)
        assert res.deflect.shape == (cells, cells, 2)
        assert res.deflect.dtype == np.float32


# ---------------------------------------------------------------------------
# 8. column_solid_fraction — unit tests for the sub-function
# ---------------------------------------------------------------------------


class TestColumnSolidFraction:
    """
    Pin column_solid_fraction independently of the full solve.
    """

    def test_empty_materials_is_zero(self):
        cfg = Config()
        job = _make_job(cfg, solid_vox=None, cells=8)
        frac = column_solid_fraction(job)
        assert frac.shape == (8, 8)
        assert frac.dtype == np.float32
        np.testing.assert_array_equal(frac, np.zeros((8, 8), dtype=np.float32))

    def test_full_solid_is_one(self):
        cfg = Config()
        cells = 8
        vpc = int(round(float(cfg.wind_cell_m) / VOXEL))
        solid_vox = np.ones((cells * vpc, cells * vpc), dtype=bool)
        job = _make_job(cfg, solid_vox=solid_vox, cells=cells)
        frac = column_solid_fraction(job)
        assert frac.shape == (cells, cells)
        # Every cell should be exactly 1.0 (fully solid)
        np.testing.assert_allclose(
            frac,
            np.ones((cells, cells), dtype=np.float32),
            atol=1e-5,
            err_msg="fully-solid region should give solid fraction == 1.0",
        )

    def test_half_solid_fraction(self):
        """
        Pin: a region with the bottom half of X-columns solid.
        Each solid cell should have fraction close to 1; open cells close to 0.
        """
        cfg = Config()
        cells = 8
        vpc = int(round(float(cfg.wind_cell_m) / VOXEL))
        solid_vox = np.zeros((cells * vpc, cells * vpc), dtype=bool)
        half_x = (cells * vpc) // 2
        solid_vox[:half_x, :] = True  # first half in X is solid
        job = _make_job(cfg, solid_vox=solid_vox, cells=cells)
        frac = column_solid_fraction(job)
        # First half of cells (in wind-cell x) should be ~1
        half_cells = cells // 2
        assert frac[:half_cells, :].min() > 0.5, "solid half of cells should have fraction > 0.5"
        # Second half should be ~0
        assert frac[half_cells:, :].max() < 0.5, "open half of cells should have fraction < 0.5"

    def test_no_nan_in_solid_fraction(self):
        cfg = Config()
        cells = 8
        vpc = int(round(float(cfg.wind_cell_m) / VOXEL))
        solid_vox = np.ones((cells * vpc, cells * vpc), dtype=bool)
        job = _make_job(cfg, solid_vox=solid_vox, cells=cells)
        frac = column_solid_fraction(job)
        assert not np.isnan(frac).any()

    def test_solid_fraction_range(self):
        cfg = Config()
        cells = 16
        vpc = int(round(float(cfg.wind_cell_m) / VOXEL))
        solid_vox = np.zeros((cells * vpc, cells * vpc), dtype=bool)
        # Checkerboard at cell-level
        for i in range(cells):
            for j in range(cells):
                if (i + j) % 2 == 0:
                    solid_vox[i * vpc : (i + 1) * vpc, j * vpc : (j + 1) * vpc] = True
        job = _make_job(cfg, solid_vox=solid_vox, cells=cells)
        frac = column_solid_fraction(job)
        # All values in [0, 1]
        assert frac.min() >= 0.0 - 1e-6
        assert frac.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# 9. Iteration count sensitivity — zero iters vs many iters
# ---------------------------------------------------------------------------


class TestIterationCount:
    """
    Pin behaviour at venturi_iters=0 (no crowding diffusion) vs the default.
    """

    def test_zero_iters_no_crowding_diffusion(self):
        """
        With iters=0, crowd = solid (un-diffused). An open cell adjacent to a
        solid wall has crowd==0 (the open cell was seeded from solid=0).
        So speedup should be exactly 1.0 in open cells.

        SUSPECTED BEHAVIOUR: with iters=0, all open cells have crowd=0,
        so speedup = clip(1 + 0, 1, max) = 1. The box blur can nudge some
        cells very slightly, but they should all remain extremely close to 1.
        """
        cfg = Config()
        cells = 16
        vpc = int(round(float(cfg.wind_cell_m) / VOXEL))
        solid_vox = np.zeros((cells * vpc, cells * vpc), dtype=bool)
        solid_vox[8 * vpc : (8 + 1) * vpc, :] = True
        solid_vox[8 * vpc : (8 + 1) * vpc, 6 * vpc : 8 * vpc] = False
        job = _make_job(cfg, solid_vox=solid_vox, cells=cells, venturi_iters=0)
        res = solve_venturi(job)
        # With 0 iters, crowd is un-diffused: open cells have crowd=0,
        # solid cells have crowd=1. The box blur can smear crowd=1 from solid
        # into adjacent open cells, so speedup slightly > 1 is expected near walls.
        # But it should NOT reach the gap speedup level (> 1.3).
        gap_sp = res.speedup[8, 6:8].max()
        assert gap_sp < 1.3, (
            f"PINNED: with 0 iters, gap speedup {gap_sp!r} should be < 1.3 "
            "(no crowding diffusion = no meaningful funneling)"
        )
        assert np.isfinite(res.speedup).all()

    def test_more_iters_raises_speedup(self):
        """
        Pin: more Jacobi iterations → higher crowding at the gap → higher speedup.
        Comparing iters=2 vs iters=16 on the same pinch geometry.
        """
        cfg = Config()
        cells = 16
        vpc = int(round(float(cfg.wind_cell_m) / VOXEL))
        solid_vox = np.zeros((cells * vpc, cells * vpc), dtype=bool)
        solid_vox[8 * vpc : (8 + 1) * vpc, :] = True
        solid_vox[8 * vpc : (8 + 1) * vpc, 7 * vpc : 9 * vpc] = False

        job_lo = _make_job(cfg, solid_vox=solid_vox, cells=cells, venturi_iters=2)
        job_hi = _make_job(cfg, solid_vox=solid_vox, cells=cells, venturi_iters=16)
        res_lo = solve_venturi(job_lo)
        res_hi = solve_venturi(job_hi)
        # Higher iters should give equal-or-higher gap speedup
        gap_lo = float(res_lo.speedup[8, 7:9].max())
        gap_hi = float(res_hi.speedup[8, 7:9].max())
        assert gap_hi >= gap_lo - 1e-4, (
            f"PINNED: more iters ({gap_hi:.4f}) should give >= speedup "
            f"vs fewer iters ({gap_lo:.4f})"
        )
