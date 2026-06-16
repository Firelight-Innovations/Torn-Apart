"""
tests/world/wind/test_region.py — Golden-master / characterisation tests for
fire_engine.world.wind.region.WindRegion.

Headless; numpy only.  No per-element loops.
These tests PIN current behaviour — they do NOT fix bugs.
Suspected deviations are noted inline and in the module docstring.

Suspected issues (DO NOT FIX HERE — report only):
  1. Snap rounding uses floor() for *negative* player positions too.
     ``floor(-1/8) * 8 == -8`` not ``0``, so a player at a small negative
     coord snaps to a very different origin than one at the equivalent small
     positive coord.  May or may not be intentional; pinned here as-is.
  2. The margin threshold is *strictly greater than* (``>``) not ``>=``.
     A player standing exactly ``margin_cells * cell_m`` from the centre does
     NOT trigger a recenter.  Pinned here as-is.
  3. First call always returns True (via ``needs_recenter`` checking
     ``origin_cell is None``) — even if player_xy is (0, 0) and the grid would
     snap to exactly the same origin on a second call.
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.world.wind.region import WindRegion

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _region(cells=64, cell_m=4.0, snap_cells=8, margin_cells=8) -> WindRegion:
    return WindRegion(cells=cells, cell_m=cell_m, snap_cells=snap_cells, margin_cells=margin_cells)


def _placed(player_xy=(0.0, 0.0), **kw) -> WindRegion:
    r = _region(**kw)
    r.maybe_recenter(player_xy)
    return r


# ===========================================================================
# 1. Construction — shape, spacing, indexing convention
# ===========================================================================


class TestConstruction:
    def test_meshgrid_shape_after_first_recenter(self):
        r = _placed()
        assert r.X.shape == (64, 64)
        assert r.Y.shape == (64, 64)

    def test_x_spacing_equals_cell_m(self):
        # X varies along axis-0 (i index) — ij indexing.
        r = _placed()
        diff = np.diff(r.X[:, 0])  # vary i, fix j=0
        assert np.allclose(diff, 4.0)

    def test_y_spacing_equals_cell_m(self):
        # Y varies along axis-1 (j index) — ij indexing.
        r = _placed()
        diff = np.diff(r.Y[0, :])  # fix i=0, vary j
        assert np.allclose(diff, 4.0)

    def test_x_constant_along_j_axis(self):
        # ij order: X[i, :] should be constant for each fixed i.
        r = _placed()
        assert np.allclose(r.X - r.X[:, :1], 0.0)

    def test_y_constant_along_i_axis(self):
        # ij order: Y[:, j] should be constant for each fixed j.
        r = _placed()
        assert np.allclose(r.Y - r.Y[:1, :], 0.0)

    def test_cell_centres_match_origin_formula(self):
        # Cell (i, j) centre should be (origin_cell + (i,j) + 0.5) * cell_m.
        r = _placed()
        ox, oy = r.origin_cell
        cell_m = r.cell_m
        cells = r.cells
        i_idx = np.arange(cells)
        j_idx = np.arange(cells)
        expected_x = (ox + i_idx + 0.5) * cell_m  # shape (cells,)
        expected_y = (oy + j_idx + 0.5) * cell_m
        # Broadcast to (cells, cells)
        assert np.allclose(r.X, expected_x[:, np.newaxis])
        assert np.allclose(r.Y, expected_y[np.newaxis, :])

    def test_origin_m_property_equals_origin_cell_times_cell_m(self):
        r = _placed(player_xy=(50.0, -30.0))
        ox, oy = r.origin_cell
        cell_m = r.cell_m
        assert r.origin_m == (ox * cell_m, oy * cell_m)

    def test_cells_not_multiple_of_snap_raises(self):
        with pytest.raises(ValueError):
            WindRegion(cells=10, cell_m=4.0, snap_cells=8)

    def test_origin_m_before_first_recenter_raises(self):
        r = _region()
        with pytest.raises(ValueError):
            _ = r.origin_m


# ===========================================================================
# 2. maybe_recenter — return value + first-call semantics
# ===========================================================================


class TestMaybeRecenterReturnValue:
    def test_first_call_always_returns_true(self):
        r = _region()
        assert r.maybe_recenter((0.0, 0.0)) is True

    def test_second_call_same_position_returns_false(self):
        r = _region()
        r.maybe_recenter((0.0, 0.0))
        assert r.maybe_recenter((0.0, 0.0)) is False

    def test_tiny_move_returns_false(self):
        r = _placed()
        # 1 m move, margin is 32 m — should not trigger.
        assert r.maybe_recenter((1.0, 1.0)) is False

    def test_exactly_at_margin_boundary_does_not_trigger(self):
        # Threshold is STRICTLY GREATER THAN margin_cells * cell_m.
        # Pin: player at exactly margin_cells * cell_m from centre → False.
        r = _placed(player_xy=(0.0, 0.0))
        cx, _cy = r.origin_cell
        cell_m = r.cell_m
        half = r.cells * 0.5
        centre_x = (cx + half) * cell_m
        margin_m = r.margin_cells * cell_m  # 32.0 m
        # Player sits exactly at margin — one tolerance inside.
        player_x = centre_x + margin_m
        assert r.maybe_recenter((player_x, 0.0)) is False

    def test_just_past_margin_triggers(self):
        r = _placed(player_xy=(0.0, 0.0))
        cx, _cy = r.origin_cell
        cell_m = r.cell_m
        half = r.cells * 0.5
        centre_x = (cx + half) * cell_m
        margin_m = r.margin_cells * cell_m  # 32.0 m
        player_x = centre_x + margin_m + 1e-3  # just beyond
        assert r.maybe_recenter((player_x, 0.0)) is True

    def test_large_jump_returns_true(self):
        r = _placed()
        assert r.maybe_recenter((10_000.0, 10_000.0)) is True


# ===========================================================================
# 3. Snap granularity — pin exact rounding rule
# ===========================================================================


class TestSnapGranularity:
    def test_origin_cell_multiple_of_snap_cells(self):
        for px in (0.0, 37.5, 123.4, 999.9, -64.0):
            r = _placed(player_xy=(px, 0.0))
            assert r.origin_cell[0] % r.snap_cells == 0, (
                f"origin_cell[0]={r.origin_cell[0]} not multiple of "
                f"snap_cells={r.snap_cells} for px={px}"
            )

    def test_origin_cell_multiple_of_snap_cells_y(self):
        for py in (0.0, 50.0, -50.0, 255.0):
            r = _placed(player_xy=(0.0, py))
            assert r.origin_cell[1] % r.snap_cells == 0

    def test_snap_formula_positive_player(self):
        # Pin the exact arithmetic:
        #   cell index = floor(px/cell_m) - cells//2, then rounded down to snap_cells multiple.
        cells, cell_m, snap_cells = 64, 4.0, 8
        px = 100.0
        cell = int(np.floor(px / cell_m)) - cells // 2  # 25 - 32 = -7
        expected = int(np.floor(cell / snap_cells)) * snap_cells  # floor(-7/8)*8 = -8
        r = _placed(player_xy=(px, 0.0), cells=cells, cell_m=cell_m, snap_cells=snap_cells)
        assert r.origin_cell[0] == expected

    def test_snap_formula_large_positive_player(self):
        cells, cell_m, snap_cells = 64, 4.0, 8
        px = 512.0
        cell = int(np.floor(px / cell_m)) - cells // 2  # 128 - 32 = 96
        expected = int(np.floor(cell / snap_cells)) * snap_cells  # 96
        r = _placed(player_xy=(px, 0.0), cells=cells, cell_m=cell_m, snap_cells=snap_cells)
        assert r.origin_cell[0] == expected

    def test_snap_formula_negative_player(self):
        # Pin negative-coordinate floor behaviour.
        cells, cell_m, snap_cells = 64, 4.0, 8
        px = -100.0
        cell = int(np.floor(px / cell_m)) - cells // 2  # -25 - 32 = -57
        expected = int(np.floor(cell / snap_cells)) * snap_cells  # floor(-57/8)*8 = -64
        r = _placed(player_xy=(px, 0.0), cells=cells, cell_m=cell_m, snap_cells=snap_cells)
        assert r.origin_cell[0] == expected

    def test_player_at_origin_snaps_to_expected(self):
        # px=0: cell = floor(0/4) - 32 = -32; snapped = floor(-32/8)*8 = -32
        r = _placed(player_xy=(0.0, 0.0))
        assert r.origin_cell == (-32, -32)


# ===========================================================================
# 4. Hysteresis — no thrash on small back-and-forth moves
# ===========================================================================


class TestHysteresis:
    def test_back_and_forth_within_margin_no_recenter(self):
        r = _placed(player_xy=(0.0, 0.0))
        origin_after_place = r.origin_cell
        margin_m = r.margin_cells * r.cell_m  # 32 m
        # Oscillate at ±(margin_m - 1) — never leaves the margin.
        for sign in (+1, -1, +1, -1, +1):
            moved = r.maybe_recenter((sign * (margin_m - 1.0), 0.0))
            assert moved is False
        assert r.origin_cell == origin_after_place

    def test_move_out_then_exactly_back_same_origin(self):
        # Move far enough to recenter, then move exactly back to the
        # starting player pos; the origin should be the same as the
        # original placement (since the rounding maps them identically).
        start = (0.0, 0.0)
        r = _region()
        r.maybe_recenter(start)
        origin_start = r.origin_cell

        # Jump far to trigger recenter.
        r.maybe_recenter((10_000.0, 0.0))
        assert r.origin_cell != origin_start  # sanity: it actually moved

        # Jump back to the original player position.
        r.maybe_recenter(start)
        origin_back = r.origin_cell
        assert origin_back == origin_start  # round-trip


# ===========================================================================
# 5. X/Y meshgrids update correctly after recenter
# ===========================================================================


class TestMeshgridAfterRecenter:
    def test_meshgrid_shifts_by_origin_delta(self):
        r = _placed(player_xy=(0.0, 0.0))
        X_before = r.X.copy()
        Y_before = r.Y.copy()
        origin_before = r.origin_cell

        # Force a recenter.
        r.maybe_recenter((10_000.0, 10_000.0))
        origin_after = r.origin_cell

        delta_x = (origin_after[0] - origin_before[0]) * r.cell_m
        delta_y = (origin_after[1] - origin_before[1]) * r.cell_m

        assert np.allclose(r.X - X_before, delta_x)
        assert np.allclose(r.Y - Y_before, delta_y)

    def test_meshgrid_unchanged_when_no_recenter(self):
        r = _placed(player_xy=(0.0, 0.0))
        X_before = r.X.copy()
        Y_before = r.Y.copy()
        r.maybe_recenter((1.0, 1.0))  # within margin
        assert np.array_equal(r.X, X_before)
        assert np.array_equal(r.Y, Y_before)


# ===========================================================================
# 6. Determinism — same path → same origin sequence
# ===========================================================================


class TestDeterminism:
    def test_same_player_path_same_origin_sequence(self):
        path = [(0.0, 0.0), (50.0, 0.0), (200.0, 100.0), (-64.0, -64.0), (1000.0, -500.0)]
        seq_a, seq_b = [], []
        for positions, out in ((path, seq_a), (path, seq_b)):
            r = _region()
            for pos in positions:
                r.maybe_recenter(pos)
                out.append(r.origin_cell)
        assert seq_a == seq_b


# ===========================================================================
# 7. Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_player_at_exactly_origin_m(self):
        # Player standing at the tile-corner (origin_m).
        # Pin CURRENT behaviour: origin_m is the far corner of the tile relative
        # to the tile centre, so moving there is past the hysteresis margin and
        # DOES trigger a recenter.
        # Suspected oddity: if origin_m is the (0,0)-corner of a 64-cell tile
        # placed at origin_cell=(-32,-32), origin_m == (-128, -128).  The tile
        # centre sits at (0,0) so moving to (-128,-128) is 128 m away — well
        # beyond the 32 m margin.  Behaviour is correct given the semantics; pin
        # it explicitly so future refactors don't silently change it.
        r = _placed(player_xy=(0.0, 0.0))
        ox, oy = r.origin_m
        moved = r.maybe_recenter((ox, oy))
        # Pin: moving to the tile corner triggers a recenter (it is outside the
        # margin from the original tile centre).
        assert moved is True

    def test_large_negative_coordinates(self):
        r = _placed(player_xy=(-10_000.0, -10_000.0))
        assert r.origin_cell[0] % r.snap_cells == 0
        assert r.origin_cell[1] % r.snap_cells == 0
        # origin_m consistent.
        ox, oy = r.origin_cell
        assert r.origin_m == (ox * r.cell_m, oy * r.cell_m)

    def test_single_large_jump_snaps_correctly(self):
        r = _region()
        r.maybe_recenter((0.0, 0.0))
        returned = r.maybe_recenter((50_000.0, -30_000.0))
        assert returned is True
        assert r.origin_cell[0] % r.snap_cells == 0
        assert r.origin_cell[1] % r.snap_cells == 0

    def test_size_m_property(self):
        r = _region(cells=64, cell_m=4.0)
        assert r.size_m == 256.0

    def test_needs_recenter_non_mutating(self):
        # needs_recenter() must NOT change origin_cell or meshes.
        r = _placed(player_xy=(0.0, 0.0))
        origin_before = r.origin_cell
        X_id = id(r.X)
        r.needs_recenter((9999.0, 9999.0))  # would trigger if mutating
        assert r.origin_cell == origin_before
        assert id(r.X) == X_id  # same object — not rebuilt
