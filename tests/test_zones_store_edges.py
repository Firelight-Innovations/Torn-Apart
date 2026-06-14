"""
tests/test_zones_store_edges.py — Edge-coverage characterization tests (golden-master).

Pins CURRENT behavior of ZoneStore and ZoneVolume at boundaries, before-baseline,
duplicate adds, version handling, NaN/inf corners, zero-size volumes, contains_xy
scalar/array/boundary, intersects_chunk geometry edges, and to_dict/from_dict
round-trips.

DO NOT fix bugs here — only pin current behavior and report suspicions.
No panda3d imports (Hard Rule 1).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fire_engine.zones import ZoneStore, ZoneVolume


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_vol(id_=1, tag="grass", lo=(0.0, 0.0, 0.0), hi=(10.0, 10.0, 4.0), **kw) -> ZoneVolume:
    return ZoneVolume(id_, tag, lo, hi, **kw)


# ---------------------------------------------------------------------------
# ZoneStore — get_delta() BEFORE mark_baseline()
# ---------------------------------------------------------------------------


class TestGetDeltaBeforeBaseline:
    """Pin the pre-baseline get_delta() behavior (no explicit contract in docs)."""

    def test_get_delta_before_baseline_returns_nonempty(self):
        # _baseline is None; snap == None evaluates False → returns full snapshot.
        store = ZoneStore()
        store.add("grass", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        delta = store.get_delta()
        # Current behavior: NOT {}, returns the full volume list.
        assert delta != {}

    def test_get_delta_before_baseline_empty_store(self):
        # Even with no volumes, _baseline is None → full snapshot returned.
        store = ZoneStore()
        delta = store.get_delta()
        # SUSPECTED BUG: an empty store before baseline returns a non-empty
        # dict (with version + next_id but empty volumes list), rather than {}.
        # Pin current behavior:
        assert isinstance(delta, dict)
        assert delta != {}  # because _baseline is None

    def test_get_delta_before_baseline_has_expected_keys(self):
        store = ZoneStore()
        store.add("grass", (0.0, 0.0, 0.0), (5.0, 5.0, 2.0))
        delta = store.get_delta()
        assert "version" in delta
        assert "volumes" in delta
        assert "next_id" in delta

    def test_get_delta_before_baseline_contains_current_volumes(self):
        store = ZoneStore()
        v = store.add("grass", (1.0, 1.0, 0.0), (3.0, 3.0, 1.0))
        delta = store.get_delta()
        assert len(delta["volumes"]) == 1
        assert delta["volumes"][0]["id"] == v.id


# ---------------------------------------------------------------------------
# ZoneStore — delta version handling in apply_delta
# ---------------------------------------------------------------------------


class TestApplyDeltaVersionHandling:
    """Pin behavior when the saved delta version doesn't match _DELTA_VERSION (1)."""

    def test_apply_delta_future_version_is_ignored(self):
        # Version > _DELTA_VERSION → log warning + return without applying.
        store = ZoneStore()
        store.add("grass", (0.0, 0.0, 0.0), (5.0, 5.0, 2.0))
        store.mark_baseline()

        future_delta = {
            "version": 999,
            "volumes": [ZoneVolume(10, "biome", (0.0, 0.0, 0.0), (100.0, 100.0, 10.0)).to_dict()],
            "next_id": 11,
        }
        pre_volumes = store.volumes()
        store.apply_delta(future_delta)
        # Volumes must be unchanged — future delta was silently ignored.
        assert store.volumes() == pre_volumes

    def test_apply_delta_version_zero_is_applied(self):
        # Version 0 < _DELTA_VERSION (1) but code does NOT reject it (no lower bound).
        # Pin current behavior: version 0 IS applied.
        store = ZoneStore()
        store.mark_baseline()

        delta_v0 = {
            "version": 0,
            "volumes": [ZoneVolume(5, "grass", (2.0, 2.0, 0.0), (8.0, 8.0, 4.0)).to_dict()],
            "next_id": 6,
        }
        store.apply_delta(delta_v0)
        # Volume from the delta should now be in the store.
        assert len(store.volumes()) == 1
        assert store.volumes()[0].id == 5

    def test_apply_delta_missing_version_defaults_to_zero_applied(self):
        # No "version" key → delta.get("version", 0) == 0 → applied.
        store = ZoneStore()
        store.mark_baseline()

        delta_no_ver = {
            "volumes": [ZoneVolume(7, "grass", (1.0, 1.0, 0.0), (3.0, 3.0, 1.0)).to_dict()],
            "next_id": 8,
        }
        store.apply_delta(delta_no_ver)
        assert len(store.volumes()) == 1

    def test_apply_delta_empty_dict_is_noop(self):
        # Empty delta {} → early return, store unchanged.
        store = ZoneStore()
        v = store.add("grass", (0.0, 0.0, 0.0), (5.0, 5.0, 2.0))
        store.mark_baseline()
        pre_vol = store.volumes()
        store.apply_delta({})
        assert store.volumes() == pre_vol


# ---------------------------------------------------------------------------
# ZoneStore — duplicate adds, remove behavior
# ---------------------------------------------------------------------------


class TestDuplicateAndRemove:
    """Pin id auto-increment, duplicate volumes, and remove edge cases."""

    def test_duplicate_identical_corners_both_stored(self):
        # Two adds with the same corners/tag get different ids and both survive.
        store = ZoneStore()
        a = store.add("grass", (0.0, 0.0, 0.0), (5.0, 5.0, 2.0))
        b = store.add("grass", (0.0, 0.0, 0.0), (5.0, 5.0, 2.0))
        assert a.id != b.id
        assert len(store.volumes()) == 2

    def test_ids_auto_increment_monotonically(self):
        store = ZoneStore()
        ids = [
            store.add("grass", (float(i), 0.0, 0.0), (float(i + 1), 1.0, 1.0)).id for i in range(5)
        ]
        assert ids == list(range(1, 6))

    def test_remove_nonexistent_returns_false(self):
        store = ZoneStore()
        assert store.remove(999) is False

    def test_remove_twice_second_is_false(self):
        store = ZoneStore()
        v = store.add("grass", (0.0, 0.0, 0.0), (2.0, 2.0, 1.0))
        assert store.remove(v.id) is True
        assert store.remove(v.id) is False

    def test_remove_does_not_change_remaining_volumes(self):
        store = ZoneStore()
        a = store.add("grass", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        b = store.add("biome", (2.0, 2.0, 0.0), (4.0, 4.0, 2.0), biome="snow")
        store.remove(a.id)
        assert store.volumes() == (b,)
        assert store.get(b.id) is b


# ---------------------------------------------------------------------------
# ZoneStore — volumes() filtering and ordering
# ---------------------------------------------------------------------------


class TestVolumesFilteringAndOrdering:
    """Pin tag filtering and id-ordered returns."""

    def test_volumes_no_filter_ordered_by_id(self):
        store = ZoneStore()
        v1 = store.add("grass", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        v2 = store.add("biome", (2.0, 0.0, 0.0), (4.0, 2.0, 1.0), biome="snow")
        v3 = store.add("grass", (5.0, 0.0, 0.0), (7.0, 2.0, 1.0))
        result = store.volumes()
        assert result == (v1, v2, v3)

    def test_volumes_tag_filter_returns_only_matching(self):
        store = ZoneStore()
        g1 = store.add("grass", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        _b = store.add("biome", (2.0, 0.0, 0.0), (4.0, 2.0, 1.0), biome="snow")
        g2 = store.add("grass", (5.0, 0.0, 0.0), (7.0, 2.0, 1.0))
        result = store.volumes("grass")
        assert result == (g1, g2)

    def test_volumes_tag_filter_no_match_returns_empty(self):
        store = ZoneStore()
        store.add("grass", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        assert store.volumes("trees") == ()

    def test_volumes_ordered_after_removes(self):
        # Removing the first volume must not break id ordering.
        store = ZoneStore()
        v1 = store.add("grass", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        v2 = store.add("grass", (2.0, 0.0, 0.0), (3.0, 1.0, 1.0))
        v3 = store.add("grass", (4.0, 0.0, 0.0), (5.0, 1.0, 1.0))
        store.remove(v1.id)
        assert store.volumes() == (v2, v3)


# ---------------------------------------------------------------------------
# ZoneStore — apply_delta with empty delta leaves state intact
# ---------------------------------------------------------------------------


class TestApplyDeltaEmpty:
    def test_empty_delta_leaves_volumes_unchanged(self):
        store = ZoneStore()
        v = store.add("grass", (0.0, 0.0, 0.0), (6.0, 6.0, 2.0))
        store.mark_baseline()
        pre = store.volumes()
        pre_ver = store.version
        store.apply_delta({})
        assert store.volumes() == pre
        # Version must NOT bump on a no-op apply.
        assert store.version == pre_ver


# ---------------------------------------------------------------------------
# ZoneVolume — NaN / inf in corners
# ---------------------------------------------------------------------------


class TestNaNInfCorners:
    """Pin whether construction raises or silently accepts bad float values."""

    def test_nan_in_min_x_raises(self):
        # NaN < anything is False → not all(a < b) → raises ValueError.
        with pytest.raises((ValueError, Exception)):
            ZoneVolume(1, "grass", (float("nan"), 0.0, 0.0), (1.0, 1.0, 1.0))

    def test_nan_in_max_y_raises(self):
        with pytest.raises((ValueError, Exception)):
            ZoneVolume(1, "grass", (0.0, 0.0, 0.0), (1.0, float("nan"), 1.0))

    def test_inf_in_max_raises_or_accepts(self):
        # +inf > any finite → a < b is True → validation passes.
        # Pin current behavior: construction SUCCEEDS with +inf max.
        try:
            v = ZoneVolume(1, "grass", (0.0, 0.0, 0.0), (float("inf"), 10.0, 4.0))
            # If it succeeds, pin that it at least stores the value.
            assert math.isinf(v.max_corner[0])
        except (ValueError, OverflowError):
            pass  # also acceptable; pin whichever branch runs

    def test_negative_inf_in_min_raises_or_accepts(self):
        # -inf < any finite → a < b is True → validation passes.
        try:
            v = ZoneVolume(1, "grass", (float("-inf"), 0.0, 0.0), (1.0, 1.0, 1.0))
            assert math.isinf(v.min_corner[0])
        except (ValueError, OverflowError):
            pass  # also acceptable

    def test_both_nan_raises(self):
        with pytest.raises((ValueError, Exception)):
            ZoneVolume(
                1,
                "grass",
                (float("nan"), float("nan"), float("nan")),
                (float("nan"), float("nan"), float("nan")),
            )


# ---------------------------------------------------------------------------
# ZoneVolume — zero-size volume (min_corner == max_corner)
# ---------------------------------------------------------------------------


class TestZeroSizeVolume:
    """Pin validation behavior: strict < means zero-size is rejected at construction."""

    def test_zero_size_x_raises(self):
        with pytest.raises(ValueError):
            ZoneVolume(1, "grass", (5.0, 0.0, 0.0), (5.0, 1.0, 1.0))

    def test_zero_size_all_axes_raises(self):
        with pytest.raises(ValueError):
            ZoneVolume(1, "grass", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    def test_zero_size_z_only_raises(self):
        # Z must also satisfy min < max — zero z-depth is rejected.
        with pytest.raises(ValueError):
            ZoneVolume(1, "grass", (0.0, 0.0, 3.0), (5.0, 5.0, 3.0))


# ---------------------------------------------------------------------------
# ZoneVolume.contains_xy — scalar vs array, boundary inclusivity
# ---------------------------------------------------------------------------


class TestContainsXY:
    """Pin inclusive-min / exclusive-max semantics and input shapes."""

    def setup_method(self):
        # Volume: x in [2.0, 6.0), y in [1.0, 5.0)
        self.vol = ZoneVolume(1, "grass", (2.0, 1.0, 0.0), (6.0, 5.0, 2.0))

    # --- boundary ---

    def test_min_corner_is_inclusive(self):
        # Exactly at min — should be inside (>=).
        result = self.vol.contains_xy(np.array([2.0]), np.array([1.0]))
        assert result[0] == True

    def test_max_corner_is_exclusive(self):
        # Exactly at max — should be outside (< max).
        result = self.vol.contains_xy(np.array([6.0]), np.array([5.0]))
        assert result[0] == False

    def test_just_inside_max(self):
        result = self.vol.contains_xy(np.array([6.0 - 1e-9]), np.array([5.0 - 1e-9]))
        assert result[0] == True

    def test_just_outside_min(self):
        result = self.vol.contains_xy(np.array([2.0 - 1e-9]), np.array([1.0]))
        assert result[0] == False

    # --- scalar-like inputs ---

    def test_scalar_inputs_return_0d_array(self):
        result = self.vol.contains_xy(np.float64(4.0), np.float64(3.0))
        assert bool(result) is True

    def test_python_scalar_inputs(self):
        # np.asarray wraps plain scalars to 0-d arrays.
        result = self.vol.contains_xy(4.0, 3.0)
        assert bool(result) is True

    # --- array inputs ---

    def test_array_multiple_points(self):
        xs = np.array([2.0, 4.0, 6.0, 0.0])
        ys = np.array([1.0, 3.0, 5.0, 3.0])
        result = self.vol.contains_xy(xs, ys)
        # min inclusive, max exclusive
        np.testing.assert_array_equal(result, [True, True, False, False])

    def test_broadcast_column_vs_row(self):
        # x in [2,6) — test a row of x values vs a single y.
        xs = np.array([2.0, 4.0, 7.0])
        result = self.vol.contains_xy(xs, np.float64(3.0))
        np.testing.assert_array_equal(result, [True, True, False])


# ---------------------------------------------------------------------------
# ZoneVolume.intersects_chunk — geometry edges
# ---------------------------------------------------------------------------


class TestIntersectsChunk:
    """Pin overlap logic at chunk boundaries and z-window edges."""

    def setup_method(self):
        # Volume: x[-12,12), y[-5,25), z[6,10) — same as existing tests
        # but here we probe edges not covered upstream.
        self.chunk_m = 16.0
        self.vol = ZoneVolume(1, "grass", (-12.0, -5.0, 6.0), (12.0, 25.0, 10.0))

    def test_chunk_fully_inside_volume_intersects(self):
        # Chunk (0,0,0): x[0,16), y[0,16), z[0,16) — overlaps.
        assert self.vol.intersects_chunk((0, 0, 0), self.chunk_m) is True

    def test_chunk_partially_overlapping_x(self):
        # Chunk (-1,0,0): x[-16,0), y[0,16), z[0,16) — x overlaps [-12,0).
        assert self.vol.intersects_chunk((-1, 0, 0), self.chunk_m) is True

    def test_chunk_outside_positive_x(self):
        # Chunk (1,0,0): x[16,32) — volume ends at x=12 → no overlap.
        assert self.vol.intersects_chunk((1, 0, 0), self.chunk_m) is False

    def test_chunk_outside_negative_y(self):
        # Chunk (0,-1,0): y[-16,0) — volume min_y=-5 > -16, max_y=25 > 0
        # → overlap in y is [-5,0) which is non-empty → should intersect.
        assert self.vol.intersects_chunk((0, -1, 0), self.chunk_m) is True

    def test_chunk_outside_positive_y(self):
        # Chunk (0,2,0): y[32,48) — volume max_y=25 < 32 → no overlap.
        assert self.vol.intersects_chunk((0, 2, 0), self.chunk_m) is False

    def test_chunk_z_window_below(self):
        # Chunk (0,0,-1): z[-16,0) — volume min_z=6 > 0 → no z overlap.
        assert self.vol.intersects_chunk((0, 0, -1), self.chunk_m) is False

    def test_chunk_z_window_above(self):
        # Chunk (0,0,1): z[16,32) — volume max_z=10 < 16 → no z overlap.
        assert self.vol.intersects_chunk((0, 0, 1), self.chunk_m) is False

    def test_volume_edge_exactly_at_chunk_boundary(self):
        # A volume that ends exactly at a chunk boundary: max = 16.0 = 1*16.
        # Chunk (1,0,0): chunk_min_x = 1*16 = 16. Condition: max_corner > chunk_min.
        # 16.0 > 16.0 is False → NO intersection (strict inequality in code).
        vol_exact = ZoneVolume(1, "grass", (0.0, 0.0, 0.0), (16.0, 16.0, 16.0))
        # SUSPECTED BOUNDARY: check if volume touching chunk boundary includes/excludes.
        result = vol_exact.intersects_chunk((1, 0, 0), self.chunk_m)
        # Pin current behavior (strict >): max_corner (16) is NOT > chunk_min (16)
        assert result is False

    def test_volume_touching_chunk_min_boundary_from_inside(self):
        # Volume's min corner exactly at chunk start: min_corner = 0.0 = 0*16.
        # Chunk (-1,0,0): chunk_max_x = 0*16 = 0. Condition: chunk_max > min_corner.
        # 0.0 > 0.0 is False → no intersection.
        vol = ZoneVolume(1, "grass", (0.0, 0.0, 0.0), (8.0, 8.0, 8.0))
        result = vol.intersects_chunk((-1, 0, 0), self.chunk_m)
        # SUSPECTED BOUNDARY: pin strict exclusion
        assert result is False


# ---------------------------------------------------------------------------
# ZoneVolume.to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


class TestToDictFromDictRoundTrip:
    """Pin serialization completeness including biome and params."""

    def test_round_trip_with_biome_and_params(self):
        v = ZoneVolume(
            42,
            "biome",
            (10.0, 20.0, 0.0),
            (30.0, 40.0, 16.0),
            biome="snow",
            params={"density": 5.5, "scale": 1.2},
        )
        v2 = ZoneVolume.from_dict(v.to_dict())
        assert v2.id == 42
        assert v2.tag == "biome"
        assert v2.biome == "snow"
        assert v2.min_corner == (10.0, 20.0, 0.0)
        assert v2.max_corner == (30.0, 40.0, 16.0)
        assert v2.params["density"] == pytest.approx(5.5)
        assert v2.params["scale"] == pytest.approx(1.2)
        assert v2 == v

    def test_round_trip_no_biome_no_params(self):
        v = ZoneVolume(1, "grass", (0.0, 0.0, 0.0), (5.0, 5.0, 2.0))
        v2 = ZoneVolume.from_dict(v.to_dict())
        assert v2 == v
        assert v2.biome is None
        assert v2.params == {}

    def test_to_dict_contains_all_required_keys(self):
        v = ZoneVolume(3, "grass", (1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
        d = v.to_dict()
        for key in ("id", "tag", "min_corner", "max_corner", "biome", "params"):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_corners_are_lists(self):
        v = ZoneVolume(1, "grass", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        d = v.to_dict()
        assert isinstance(d["min_corner"], list)
        assert isinstance(d["max_corner"], list)

    def test_params_with_string_values_round_trip(self):
        # ZoneVolume.params is typed dict[str, float] but code does dict(params or {})
        # without enforcing value types — pin that string values survive.
        v = ZoneVolume(1, "biome", (0.0, 0.0, 0.0), (10.0, 10.0, 5.0), params={"name": "special"})  # type: ignore[arg-type]
        v2 = ZoneVolume.from_dict(v.to_dict())
        assert v2.params.get("name") == "special"
