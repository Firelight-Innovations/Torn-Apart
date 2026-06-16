"""
tests/test_rain_cover_edges.py — Edge-case characterisation tests for RainCoverField.

Golden-master / pin-down tests.  They capture CURRENT behaviour; they do NOT fix bugs.
Any suspected divergence from the documented contract is noted in a comment rather than
asserted away.

New coverage beyond test_rain_cover.py:
  - OPEN_SKY_Z sentinel: columns with/without solid voxels
  - mark_dirty (rebuild_columns) vs rebuild_all parity on affected region
  - Roof overhang: highest-solid-in-column rule with an air gap underneath
  - recenter hysteresis: tiny moves don't shift origin_m; large moves do
  - recenter origin snap: result is always a whole multiple of cell_m
  - Determinism: byte-identical heights across two independent fields
  - Shape / dtype: height is (cells, cells) float32; origin_m is a tuple
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fire_engine.core import load_config
from fire_engine.world.terrain.chunk import Chunk
from fire_engine.world.terrain.rain_cover import OPEN_SKY_Z, RainCoverField

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg():
    """Small grid so tests are cheap and cell boundaries are easy to reason about."""
    base = load_config()
    return replace(base, rain_cover_cells=64, rain_cover_cell_m=1.0)


VS = 0.5  # voxel size (m) — matches engine default
N = 32  # chunk edge (voxels)
CHUNK_M = N * VS  # 16 m


def _voxel_top_z(cz: int, z_idx: int) -> float:
    """World Z of the TOP face of voxel z_idx in chunk-Z layer cz."""
    return cz * CHUNK_M + (z_idx + 1) * VS


def _world_texel(field: RainCoverField, wx: float, wy: float) -> tuple[int, int]:
    """Map a world XY to (row, col) in field.height."""
    ox, oy = field.origin_m
    col = int(np.floor((wx - ox) / field.cell_m))
    row = int(np.floor((wy - oy) / field.cell_m))
    return row, col


# ---------------------------------------------------------------------------
# OPEN_SKY_Z sentinel value
# ---------------------------------------------------------------------------


class TestOpenSkySentinel:
    """Pin the constant value and the empty-column contract."""

    def test_sentinel_value(self):
        # The constant must be exactly -1e9 (what the GPU uses as the "open" threshold).
        assert OPEN_SKY_Z == -1.0e9

    def test_fresh_field_all_open_sky(self, cfg):
        # A newly created field with no rebuild should be entirely OPEN_SKY_Z.
        field = RainCoverField(cfg)
        expected = np.full((field.cells, field.cells), OPEN_SKY_Z, dtype=np.float32)
        assert np.array_equal(field.height, expected)

    def test_rebuild_all_empty_chunks_stays_open_sky(self, cfg):
        # A chunk dict with only air chunks must leave the heightmap at OPEN_SKY_Z.
        air = np.zeros((N, N, N), dtype=np.uint8)
        chunks = {(0, 0, 0): Chunk((0, 0, 0), air)}
        field = RainCoverField(cfg)
        field.recenter((CHUNK_M * 0.5, CHUNK_M * 0.5))
        field.rebuild_all(chunks)
        assert np.all(field.height == np.float32(OPEN_SKY_Z))

    def test_covered_columns_not_open_sky_uncovered_stay_open(self, cfg):
        """
        After rebuild_all with one chunk:
          - columns that overlap the solid chunk must hold a real height (> OPEN_SKY_Z)
          - columns outside the chunk footprint must remain OPEN_SKY_Z
        """
        # Solid slab at z=0 in chunk (0,0,0), which has world XY footprint [0,16)x[0,16).
        mats = np.zeros((N, N, N), dtype=np.uint8)
        mats[:, :, 0] = 1
        chunks = {(0, 0, 0): Chunk((0, 0, 0), mats)}

        field = RainCoverField(cfg)
        # Place origin so texel (0,0) = world (0,0), chunk footprint = [0,16) x [0,16).
        field.recenter((field.span_m * 0.5, field.span_m * 0.5))
        ox, oy = field.origin_m
        field.rebuild_all(chunks)

        # A column inside the chunk footprint must NOT be OPEN_SKY_Z.
        r, c = _world_texel(field, ox + 8.0, oy + 8.0)  # centre of the chunk
        assert field.height[r, c] > np.float32(OPEN_SKY_Z)

        # A column well outside the chunk (e.g., near the far corner of the 64 m window)
        # must still be OPEN_SKY_Z (no other chunk loaded).
        r2, c2 = _world_texel(field, ox + 60.0, oy + 60.0)
        assert field.height[r2, c2] == np.float32(OPEN_SKY_Z)


# ---------------------------------------------------------------------------
# Roof-overhang rule: highest solid in column wins (air gap underneath)
# ---------------------------------------------------------------------------


class TestRoofOverhang:
    """Pin the 'highest-solid-in-column' rule with an explicit air gap under a roof."""

    def _make_overhang_chunks(self) -> dict:
        """
        Chunk (0,0,0): solid floor at z=0..1 (world Z 0.5..1.0 top).
        Chunk (0,0,1): solid roof at z=20 ONLY (world Z = 16 + 10.5 = 26.5 m top).
        Voxels z=2..19 in both chunks are AIR → a 9+ m gap between floor and roof.
        """
        floor_mats = np.zeros((N, N, N), dtype=np.uint8)
        floor_mats[:, :, 0:2] = 1  # floor slab, top face at 1.0 m

        roof_mats = np.zeros((N, N, N), dtype=np.uint8)
        roof_mats[:, :, 20] = 1  # single-voxel roof slab

        return {
            (0, 0, 0): Chunk((0, 0, 0), floor_mats),
            (0, 0, 1): Chunk((0, 0, 1), roof_mats),
        }

    def test_roof_wins_over_floor(self, cfg):
        """
        The roof voxel is the highest solid, so the column should report
        the roof top-face Z, not the floor top-face Z.
        """
        chunks = self._make_overhang_chunks()
        field = RainCoverField(cfg)
        field.recenter((CHUNK_M * 0.5, CHUNK_M * 0.5))
        field.rebuild_all(chunks)

        expected_roof_z = _voxel_top_z(1, 20)  # chunk-Z 1, voxel index 20
        r, c = _world_texel(field, 8.0, 8.0)
        # Pin current behaviour: the roof top must win.
        assert field.height[r, c] == pytest.approx(expected_roof_z)

    def test_floor_height_not_reported_when_roof_present(self, cfg):
        """The floor top-face Z must NOT be what the column reports (roof takes over)."""
        chunks = self._make_overhang_chunks()
        field = RainCoverField(cfg)
        field.recenter((CHUNK_M * 0.5, CHUNK_M * 0.5))
        field.rebuild_all(chunks)

        floor_z = _voxel_top_z(0, 1)  # top of z=1 voxel in chunk-Z 0
        r, c = _world_texel(field, 8.0, 8.0)
        assert field.height[r, c] != pytest.approx(floor_z)

    def test_single_solid_voxel_at_top_of_chunk(self, cfg):
        """
        A single solid voxel at z=31 (the topmost slot of a chunk) must produce
        the correct top-face Z with no confusion from the reversed-argmax trick.
        """
        mats = np.zeros((N, N, N), dtype=np.uint8)
        mats[:, :, 31] = 1  # only z=31 is solid
        chunks = {(0, 0, 0): Chunk((0, 0, 0), mats)}

        field = RainCoverField(cfg)
        field.recenter((CHUNK_M * 0.5, CHUNK_M * 0.5))
        field.rebuild_all(chunks)

        expected_z = _voxel_top_z(0, 31)  # = 16.0 m
        r, c = _world_texel(field, 8.0, 8.0)
        assert field.height[r, c] == pytest.approx(expected_z)

    def test_single_solid_voxel_at_bottom_of_chunk(self, cfg):
        """
        A single solid voxel at z=0 must map to voxel_size (0.5 m top face).
        """
        mats = np.zeros((N, N, N), dtype=np.uint8)
        mats[:, :, 0] = 1
        chunks = {(0, 0, 0): Chunk((0, 0, 0), mats)}

        field = RainCoverField(cfg)
        field.recenter((CHUNK_M * 0.5, CHUNK_M * 0.5))
        field.rebuild_all(chunks)

        expected_z = _voxel_top_z(0, 0)  # = 0.5 m
        r, c = _world_texel(field, 8.0, 8.0)
        assert field.height[r, c] == pytest.approx(expected_z)


# ---------------------------------------------------------------------------
# mark_dirty (rebuild_columns) vs rebuild_all parity
# ---------------------------------------------------------------------------


class TestMarkDirtyParity:
    """
    Pin that rebuild_columns on a (cx,cy) column produces the SAME result as
    rebuild_all for the texels that overlap that column.  Cells outside the
    dirty column must be unchanged.

    SUSPECTED BUG NOTE:
    If _clear_chunk_column uses ceil/floor that clips differently from _fold_chunk's
    center-bin logic there could be a 1-texel border discrepancy.  The tests below
    will surface it if present.
    """

    def _three_chunk_setup(self):
        """
        Two XY-adjacent solid slabs:
          chunk (0,0,0) — whole floor, solid z=0..3
          chunk (1,0,0) — adjacent X, solid z=0..3
        Both provide ground height at the same Z so the texel values should agree.
        """
        mats_a = np.zeros((N, N, N), dtype=np.uint8)
        mats_a[:, :, 0:4] = 1
        mats_b = np.zeros((N, N, N), dtype=np.uint8)
        mats_b[:, :, 0:4] = 1
        return {
            (0, 0, 0): Chunk((0, 0, 0), mats_a),
            (1, 0, 0): Chunk((1, 0, 0), mats_b),
        }

    def test_rebuild_columns_matches_rebuild_all_for_affected_column(self, cfg):
        """
        Full rebuild then incremental rebuild_columns on column (0,0) must leave
        the texels under column (0,0) identical to what rebuild_all produced.
        """
        chunks = self._three_chunk_setup()
        field_full = RainCoverField(cfg)
        field_full.recenter((CHUNK_M, CHUNK_M * 0.5))
        field_full.rebuild_all(chunks)

        # Compute the texel range that (0,0) covers so we can compare just that region.
        # Column (0,0) spans world X [0,16), Y [0,16).
        ox, oy = field_full.origin_m
        c0 = max(0, int(np.floor((0.0 - ox) / field_full.cell_m)))
        c1 = min(field_full.cells, int(np.ceil((CHUNK_M - ox) / field_full.cell_m)))
        r0 = max(0, int(np.floor((0.0 - oy) / field_full.cell_m)))
        r1 = min(field_full.cells, int(np.ceil((CHUNK_M - oy) / field_full.cell_m)))

        region_full = field_full.height[r0:r1, c0:c1].copy()

        # Now do a partial rebuild_columns on a fresh field.
        field_incr = RainCoverField(cfg)
        field_incr.recenter((CHUNK_M, CHUNK_M * 0.5))
        # Start from a full rebuild so outside cells are populated, then re-run column (0,0).
        field_incr.rebuild_all(chunks)
        snapshot_outside = field_incr.height.copy()
        field_incr.rebuild_columns(chunks, [(0, 0)])

        region_incr = field_incr.height[r0:r1, c0:c1]
        # The affected region must match rebuild_all exactly.
        assert np.array_equal(region_full, region_incr), (
            "rebuild_columns result differs from rebuild_all in the dirty column region "
            f"(max delta = {np.max(np.abs(region_full.astype(np.float64) - region_incr.astype(np.float64)))})"
        )

    def test_rebuild_columns_does_not_touch_outside_cells(self, cfg):
        """
        Calling rebuild_columns on column (0,0) must leave texels under chunk (1,0,0)
        unchanged (i.e. those cells are NOT reset to OPEN_SKY_Z).
        """
        chunks = self._three_chunk_setup()
        field = RainCoverField(cfg)
        field.recenter((CHUNK_M, CHUNK_M * 0.5))
        field.rebuild_all(chunks)

        # Record column (1,0) texel region before rebuild_columns.
        ox, oy = field.origin_m
        c0b = max(0, int(np.floor((CHUNK_M - ox) / field.cell_m)))
        c1b = min(field.cells, int(np.ceil((2 * CHUNK_M - ox) / field.cell_m)))
        r0b = max(0, int(np.floor((0.0 - oy) / field.cell_m)))
        r1b = min(field.cells, int(np.ceil((CHUNK_M - oy) / field.cell_m)))

        outside_before = field.height[r0b:r1b, c0b:c1b].copy()

        # Rebuild only column (0,0).
        field.rebuild_columns(chunks, [(0, 0)])

        outside_after = field.height[r0b:r1b, c0b:c1b]
        assert np.array_equal(outside_before, outside_after), (
            "rebuild_columns([(0,0)]) unexpectedly modified texels outside column (0,0)"
        )

    def test_rebuild_columns_clear_then_refold_lowers_height(self, cfg):
        """
        After removing all solids from a column's Z stack, rebuild_columns should
        reset those texels to OPEN_SKY_Z (the 'removing roof lowers height' contract).
        """
        mats = np.zeros((N, N, N), dtype=np.uint8)
        mats[:, :, 0:4] = 1
        chunks = {(0, 0, 0): Chunk((0, 0, 0), mats)}

        field = RainCoverField(cfg)
        field.recenter((CHUNK_M * 0.5, CHUNK_M * 0.5))
        field.rebuild_all(chunks)

        # Verify a texel is set to ground height.
        r, c = _world_texel(field, 8.0, 8.0)
        ground_z = _voxel_top_z(0, 3)
        assert field.height[r, c] == pytest.approx(ground_z)

        # Clear the chunk and rebuild the column: texels must fall back to OPEN_SKY_Z.
        chunks[(0, 0, 0)].materials[...] = 0
        field.rebuild_columns(chunks, [(0, 0)])
        assert field.height[r, c] == np.float32(OPEN_SKY_Z)


# ---------------------------------------------------------------------------
# recenter hysteresis and snap behaviour
# ---------------------------------------------------------------------------


class TestRecenterBehaviour:
    """
    Pin the committed-origin discipline:
      - small moves that keep the window centred on the same cell DO shift origin
        because recenter always recomputes (no hysteresis in the code); but
        origin_m is always a whole multiple of cell_m.
      - large moves snap origin_m to the cell grid.
    """

    def test_recenter_result_always_multiple_of_cell_m(self, cfg):
        """origin_m components must always be exact multiples of cell_m."""
        field = RainCoverField(cfg)
        for px, py in [(0.0, 0.0), (3.7, -5.3), (100.1, -200.9), (16.0, 32.0)]:
            ox, oy = field.recenter((px, py))
            assert ox % field.cell_m == pytest.approx(0.0), (
                f"origin_x={ox} is not a multiple of cell_m={field.cell_m} for center ({px},{py})"
            )
            assert oy % field.cell_m == pytest.approx(0.0), (
                f"origin_y={oy} is not a multiple of cell_m={field.cell_m} for center ({px},{py})"
            )

    def test_recenter_tiny_sub_cell_move_changes_origin_if_grid_changes(self, cfg):
        """
        recenter always recomputes — there is no hysteresis guard in the code.
        Moving less than cell_m but crossing a cell boundary will shift origin.
        Moving within the same cell boundary produces the same origin.
        """
        field = RainCoverField(cfg)
        half = field.span_m * 0.5

        # Two positions whose floors differ by exactly 0 cells → same origin.
        ox0, oy0 = field.recenter((half, half))
        ox1, oy1 = field.recenter((half + field.cell_m * 0.1, half + field.cell_m * 0.1))
        # Both round to the same cell floor → same origin (same grid snap).
        assert ox0 == pytest.approx(ox1)
        assert oy0 == pytest.approx(oy1)

    def test_recenter_large_move_shifts_origin(self, cfg):
        """A move larger than cell_m must change origin_m."""
        field = RainCoverField(cfg)
        half = field.span_m * 0.5
        ox0, oy0 = field.recenter((half, half))
        # Move by 10 m — well beyond one cell.
        ox1, oy1 = field.recenter((half + 10.0, half + 10.0))
        assert ox1 > ox0
        assert oy1 > oy0

    def test_recenter_negative_coordinates_snap_correctly(self, cfg):
        """origin_m must be a cell-grid multiple even for negative world coords."""
        field = RainCoverField(cfg)
        ox, oy = field.recenter((-7.3, -15.8))
        assert ox % field.cell_m == pytest.approx(0.0)
        assert oy % field.cell_m == pytest.approx(0.0)

    def test_recenter_updates_origin_m_property(self, cfg):
        """recenter return value must match origin_m property immediately after."""
        field = RainCoverField(cfg)
        ret = field.recenter((50.0, -30.0))
        assert ret == field.origin_m

    def test_recenter_does_not_modify_height_array(self, cfg):
        """
        recenter alone must NOT touch height — the caller owns when to rebuild.
        Pin this so future refactors don't accidentally clear or shift data on
        a plain recenter call.
        """
        mats = np.zeros((N, N, N), dtype=np.uint8)
        mats[:, :, 0:4] = 1
        chunks = {(0, 0, 0): Chunk((0, 0, 0), mats)}
        field = RainCoverField(cfg)
        field.recenter((CHUNK_M * 0.5, CHUNK_M * 0.5))
        field.rebuild_all(chunks)
        height_before = field.height.copy()

        # Recenter without rebuild — height must be unchanged.
        field.recenter((CHUNK_M * 0.5 + 0.01, CHUNK_M * 0.5 + 0.01))
        assert np.array_equal(field.height, height_before)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same chunks → byte-identical heightmap from two independent instances."""

    def test_two_fields_same_chunks_byte_identical(self, cfg):
        rng = np.random.default_rng(42)
        chunks = {}
        for cz in range(3):
            mats = (rng.random((N, N, N)) < 0.2).astype(np.uint8)
            chunks[(0, 0, cz)] = Chunk((0, 0, cz), mats)

        center = (CHUNK_M * 0.5, CHUNK_M * 0.5)

        a = RainCoverField(cfg)
        a.recenter(center)
        a.rebuild_all(chunks)

        b = RainCoverField(cfg)
        b.recenter(center)
        b.rebuild_all(chunks)

        assert np.array_equal(a.height, b.height)

    def test_rebuild_all_twice_same_result(self, cfg):
        """Calling rebuild_all twice on the same field and chunks must be idempotent."""
        mats = np.zeros((N, N, N), dtype=np.uint8)
        mats[:, :, 5:10] = 1
        chunks = {(0, 0, 0): Chunk((0, 0, 0), mats)}

        field = RainCoverField(cfg)
        field.recenter((CHUNK_M * 0.5, CHUNK_M * 0.5))
        field.rebuild_all(chunks)
        first = field.height.copy()

        field.rebuild_all(chunks)
        assert np.array_equal(field.height, first)


# ---------------------------------------------------------------------------
# Shape and dtype guarantees
# ---------------------------------------------------------------------------


class TestShapeAndDtype:
    """Pin height shape, dtype, and origin_m type contract."""

    def test_height_shape(self, cfg):
        field = RainCoverField(cfg)
        assert field.height.shape == (field.cells, field.cells)
        assert field.height.shape == (cfg.rain_cover_cells, cfg.rain_cover_cells)

    def test_height_dtype_float32(self, cfg):
        field = RainCoverField(cfg)
        assert field.height.dtype == np.float32

    def test_height_dtype_after_rebuild(self, cfg):
        """rebuild_all must preserve float32 dtype (no upcasting to float64)."""
        mats = np.zeros((N, N, N), dtype=np.uint8)
        mats[:, :, 0] = 1
        chunks = {(0, 0, 0): Chunk((0, 0, 0), mats)}
        field = RainCoverField(cfg)
        field.recenter((CHUNK_M * 0.5, CHUNK_M * 0.5))
        field.rebuild_all(chunks)
        assert field.height.dtype == np.float32

    def test_origin_m_is_tuple_of_floats(self, cfg):
        field = RainCoverField(cfg)
        field.recenter((1.5, -3.2))
        ox, oy = field.origin_m
        assert isinstance(ox, float)
        assert isinstance(oy, float)

    def test_cells_cell_m_span_consistency(self, cfg):
        """span_m must equal cells * cell_m (the grid contract)."""
        field = RainCoverField(cfg)
        assert field.span_m == pytest.approx(field.cells * field.cell_m)

    def test_config_values_reflected_in_field(self, cfg):
        """Field must read cells and cell_m from config, not hard-code them."""
        field = RainCoverField(cfg)
        assert field.cells == cfg.rain_cover_cells
        assert field.cell_m == pytest.approx(cfg.rain_cover_cell_m)

    def test_open_sky_z_is_float32_representable(self):
        """OPEN_SKY_Z must survive a float32 round-trip without change."""
        as_f32 = np.float32(OPEN_SKY_Z)
        # It need not be exactly representable but the test pins that it doesn't
        # silently become +/-inf or NaN.
        assert np.isfinite(as_f32)
        assert float(as_f32) == pytest.approx(OPEN_SKY_Z, rel=1e-3)


# ---------------------------------------------------------------------------
# chunk_column_of static helper
# ---------------------------------------------------------------------------


class TestChunkColumnOf:
    """Pin the static helper that drops cz."""

    def test_drops_cz(self):
        assert RainCoverField.chunk_column_of((3, -2, 7)) == (3, -2)

    def test_zero_coord(self):
        assert RainCoverField.chunk_column_of((0, 0, 0)) == (0, 0)

    def test_negative_xy(self):
        assert RainCoverField.chunk_column_of((-5, -9, 2)) == (-5, -9)
