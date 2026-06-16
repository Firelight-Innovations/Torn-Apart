"""
tests/lighting/test_occluders.py — Headless tests for fire_engine.lighting.occluders.

Covers:
- TreeOccluderSet construction (single, merge, empty, count).
- splat_tree_occluders: trunk presence, canopy presence, empty set, out-of-window.
- Determinism: same inputs → byte-identical output.
- Constants: TRUNK_TOP_FRAC, TRUNK_SIDE_M, CANOPY_CENTER_FRAC, CANOPY_HALF_HEIGHT_FRAC.

No panda3d imports.  No per-voxel Python loops in the tests — correctness
is asserted on aggregated array properties.
"""

from __future__ import annotations

import numpy as np

from fire_engine.lighting.occluders import (
    CANOPY_CENTER_FRAC,
    CANOPY_HALF_HEIGHT_FRAC,
    TRUNK_SIDE_M,
    TRUNK_TOP_FRAC,
    TreeOccluderSet,
    splat_tree_occluders,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CELL_M = 0.5  # cascade-0 fine cell size


def _empty_vol(n: int = 32) -> np.ndarray:
    """Return a zero RGBA uint8 volume of shape (n, n, n, 4)."""
    return np.zeros((n, n, n, 4), dtype=np.uint8)


def _splat_single(
    x: float = 8.0,
    y: float = 8.0,
    z: float = 0.0,
    height_m: float = 6.0,
    canopy_r_m: float = 2.0,
    canopy_sigma: float = 0.25,
    cell_m: float = CELL_M,
    n: int = 32,
    origin_cell: tuple[int, int, int] = (0, 0, 0),
    trunk_occ: float = 0.85,
    canopy_gain: float = 1.0,
) -> np.ndarray:
    occ = TreeOccluderSet.single(x, y, z, height_m, canopy_r_m, canopy_sigma)
    vol = _empty_vol(n)
    splat_tree_occluders(vol, origin_cell, cell_m, occ, trunk_occ, canopy_gain)
    return vol


# ---------------------------------------------------------------------------
# TreeOccluderSet construction
# ---------------------------------------------------------------------------


class TestTreeOccluderSet:
    def test_single_count_is_one(self):
        occ = TreeOccluderSet.single(0.0, 0.0, 0.0, height_m=5.0, canopy_r_m=2.0)
        assert occ.count == 1

    def test_empty_count_is_zero(self):
        occ = TreeOccluderSet.empty()
        assert occ.count == 0

    def test_empty_arrays_have_correct_shape(self):
        occ = TreeOccluderSet.empty()
        assert occ.x.shape == (0,)
        assert occ.bark_rgb.shape == (0, 3)
        assert occ.leaf_rgb.shape == (0, 3)

    def test_single_position_stored_correctly(self):
        occ = TreeOccluderSet.single(3.0, 4.0, 5.0, height_m=8.0, canopy_r_m=3.0)
        assert float(occ.x[0]) == 3.0
        assert float(occ.y[0]) == 4.0
        assert float(occ.z[0]) == 5.0

    def test_merge_two_sets(self):
        a = TreeOccluderSet.single(0.0, 0.0, 0.0, height_m=5.0, canopy_r_m=2.0)
        b = TreeOccluderSet.single(10.0, 10.0, 0.0, height_m=6.0, canopy_r_m=2.5)
        merged = TreeOccluderSet.merge([a, b])
        assert merged.count == 2
        assert float(merged.x[0]) == 0.0
        assert float(merged.x[1]) == 10.0

    def test_merge_empty_list_returns_empty(self):
        merged = TreeOccluderSet.merge([])
        assert merged.count == 0

    def test_single_defaults_dtype_float32(self):
        occ = TreeOccluderSet.single(0.0, 0.0, 0.0, height_m=5.0, canopy_r_m=2.0)
        assert occ.x.dtype == np.float32
        assert occ.bark_rgb.dtype == np.float32


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------


class TestConstants:
    def test_trunk_top_frac_in_range(self):
        assert 0.0 < TRUNK_TOP_FRAC < 1.0

    def test_canopy_center_frac_above_trunk_top(self):
        assert CANOPY_CENTER_FRAC > TRUNK_TOP_FRAC

    def test_trunk_side_m_positive(self):
        assert TRUNK_SIDE_M > 0.0

    def test_canopy_half_height_frac_positive(self):
        assert CANOPY_HALF_HEIGHT_FRAC > 0.0


# ---------------------------------------------------------------------------
# splat_tree_occluders: trunk correctness
# ---------------------------------------------------------------------------


class TestSplatTrunk:
    def test_trunk_alpha_nonzero_in_column(self):
        """A tree centred in the volume should have non-zero alpha in trunk cells."""
        vol = _splat_single(x=8.0, y=8.0, z=0.0, height_m=6.0)
        # Trunk base is at z=0 m → cell z=0 (at CELL_M=0.5). Alpha channel is index 3.
        trunk_col_alpha = vol[16, 16, :, 3]
        assert trunk_col_alpha.any(), "Expected non-zero alpha in the trunk column"

    def test_trunk_disabled_by_zero_trunk_occ(self):
        """trunk_occ=0 and canopy_sigma=0 → no cells written."""
        vol = _splat_single(x=8.0, y=8.0, z=0.0, height_m=6.0, trunk_occ=0.0, canopy_sigma=0.0)
        assert not vol.any(), "With trunk_occ=0 and sigma=0, volume should be all zeros"

    def test_trunk_alpha_at_correct_z_range(self):
        """Trunk should NOT appear above TRUNK_TOP_FRAC * height_m."""
        height_m = 8.0
        occ = TreeOccluderSet.single(
            8.0, 8.0, 0.0, height_m=height_m, canopy_r_m=0.1, canopy_sigma=0.0
        )
        vol = _empty_vol(32)
        splat_tree_occluders(vol, (0, 0, 0), CELL_M, occ, trunk_occ=0.85, canopy_gain=0.0)
        trunk_top_cell = int(TRUNK_TOP_FRAC * height_m / CELL_M)
        # All cells above trunk top should be zero (canopy disabled).
        above = vol[:, :, trunk_top_cell + 1 :, 3]
        assert not above.any(), "Trunk alpha should not appear above trunk top"


# ---------------------------------------------------------------------------
# splat_tree_occluders: canopy correctness
# ---------------------------------------------------------------------------


class TestSplatCanopy:
    def test_canopy_alpha_nonzero_near_canopy_centre(self):
        """Canopy alpha should be nonzero near the canopy centre cell."""
        height_m = 8.0
        canopy_center_m = CANOPY_CENTER_FRAC * height_m  # ~5.2 m
        canopy_center_cell = int(canopy_center_m / CELL_M)
        vol = _splat_single(
            x=8.0, y=8.0, z=0.0, height_m=height_m, canopy_r_m=2.5, canopy_sigma=0.5, trunk_occ=0.0
        )
        region = vol[:, :, max(0, canopy_center_cell - 2) : canopy_center_cell + 2, 3]
        assert region.any(), "Expected non-zero canopy alpha near canopy centre"

    def test_canopy_gain_zero_disables_canopy(self):
        """canopy_gain=0 → no canopy alpha; trunk_occ=0 → no trunk alpha → all zero."""
        vol = _splat_single(x=8.0, y=8.0, z=0.0, trunk_occ=0.0, canopy_gain=0.0)
        assert not vol.any(), "canopy_gain=0 + trunk_occ=0 must leave volume empty"


# ---------------------------------------------------------------------------
# splat_tree_occluders: out-of-window rejection
# ---------------------------------------------------------------------------


class TestSplatOutOfWindow:
    def test_tree_entirely_outside_window_leaves_volume_empty(self):
        """A tree positioned 100 m outside the window must not write any cell."""
        occ = TreeOccluderSet.single(x=200.0, y=0.0, z=0.0, height_m=6.0, canopy_r_m=2.0)
        vol = _empty_vol(32)
        # Window covers cells (0..31) * 0.5 m = [0, 16) m in x.
        splat_tree_occluders(vol, (0, 0, 0), CELL_M, occ, trunk_occ=0.85, canopy_gain=1.0)
        assert not vol.any(), "Out-of-window tree must not modify the volume"

    def test_empty_occluder_set_leaves_volume_empty(self):
        empty = TreeOccluderSet.empty()
        vol = _empty_vol(32)
        splat_tree_occluders(vol, (0, 0, 0), CELL_M, empty, trunk_occ=0.85, canopy_gain=1.0)
        assert not vol.any()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_inputs_byte_identical(self):
        """Two identical calls must produce byte-identical volumes."""
        occ = TreeOccluderSet.single(8.0, 8.0, 0.0, height_m=6.0, canopy_r_m=2.0, canopy_sigma=0.25)
        vol1 = _empty_vol(32)
        vol2 = _empty_vol(32)
        splat_tree_occluders(vol1, (0, 0, 0), CELL_M, occ, trunk_occ=0.85, canopy_gain=1.0)
        splat_tree_occluders(vol2, (0, 0, 0), CELL_M, occ, trunk_occ=0.85, canopy_gain=1.0)
        assert vol1.tobytes() == vol2.tobytes()

    def test_different_positions_differ(self):
        """Trees at different X positions must produce different volumes."""
        occ_a = TreeOccluderSet.single(4.0, 8.0, 0.0, height_m=6.0, canopy_r_m=2.0)
        occ_b = TreeOccluderSet.single(12.0, 8.0, 0.0, height_m=6.0, canopy_r_m=2.0)
        vol_a = _empty_vol(32)
        vol_b = _empty_vol(32)
        splat_tree_occluders(vol_a, (0, 0, 0), CELL_M, occ_a, 0.85, 1.0)
        splat_tree_occluders(vol_b, (0, 0, 0), CELL_M, occ_b, 0.85, 1.0)
        assert vol_a.tobytes() != vol_b.tobytes()
