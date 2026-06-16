"""
tests/zones/test_volume.py — unit tests for fire_engine/zones/volume.py.

Categories (CLAUDE.md):
- CORRECTNESS: field values, geometry queries (size_m, area_xy_m2,
  contains_xy, intersects_chunk), serialisation (to_dict / from_dict).
- ROUND-TRIP: to_dict -> from_dict restores all fields exactly.
- DETERMINISM: no RNG used in this module; correctness is sufficient.

No panda3d imports (Hard Rule 1).
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.zones.volume import ZoneVolume

# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_basic_fields_stored(self):
        v = ZoneVolume(7, "grass", (1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
        assert v.id == 7
        assert v.tag == "grass"
        assert v.min_corner == (1.0, 2.0, 3.0)
        assert v.max_corner == (4.0, 5.0, 6.0)
        assert v.biome is None
        assert v.params == {}

    def test_biome_and_params_stored(self):
        v = ZoneVolume(
            2,
            "biome",
            (0.0, 0.0, 0.0),
            (10.0, 10.0, 5.0),
            biome="snow",
            params={"density": 3.0},
        )
        assert v.biome == "snow"
        assert v.params["density"] == pytest.approx(3.0)

    def test_corners_normalised_to_float(self):
        v = ZoneVolume(1, "grass", (0, 0, 0), (1, 2, 3))  # int inputs
        assert isinstance(v.min_corner[0], float)
        assert isinstance(v.max_corner[2], float)

    def test_empty_tag_raises(self):
        with pytest.raises(ValueError):
            ZoneVolume(1, "", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))

    def test_min_ge_max_x_raises(self):
        with pytest.raises(ValueError):
            ZoneVolume(1, "grass", (5.0, 0.0, 0.0), (3.0, 1.0, 1.0))

    def test_min_eq_max_raises(self):
        with pytest.raises(ValueError):
            ZoneVolume(1, "grass", (0.0, 0.0, 0.0), (0.0, 1.0, 1.0))

    def test_frozen(self):
        v = ZoneVolume(1, "grass", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        with pytest.raises((AttributeError, TypeError)):
            v.id = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Geometry properties
# ---------------------------------------------------------------------------


class TestGeometryProperties:
    def test_size_m_correct(self):
        v = ZoneVolume(1, "grass", (1.0, 2.0, 3.0), (4.0, 7.0, 9.0))
        assert v.size_m == pytest.approx((3.0, 5.0, 6.0))

    def test_area_xy_m2_correct(self):
        v = ZoneVolume(1, "grass", (0.0, 0.0, 0.0), (10.0, 20.0, 4.0))
        assert v.area_xy_m2 == pytest.approx(200.0)

    def test_area_xy_m2_non_square(self):
        v = ZoneVolume(1, "grass", (-12.0, -5.0, 6.0), (12.0, 25.0, 10.0))
        assert v.area_xy_m2 == pytest.approx(24.0 * 30.0)


# ---------------------------------------------------------------------------
# contains_xy
# ---------------------------------------------------------------------------


class TestContainsXY:
    def setup_method(self):
        self.vol = ZoneVolume(1, "grass", (2.0, 1.0, 0.0), (6.0, 5.0, 2.0))

    def test_interior_point_is_inside(self):
        result = self.vol.contains_xy(np.array([4.0]), np.array([3.0]))
        assert result[0] is np.bool_(True)

    def test_point_outside_x_is_outside(self):
        result = self.vol.contains_xy(np.array([7.0]), np.array([3.0]))
        assert not result[0]

    def test_min_corner_inclusive(self):
        result = self.vol.contains_xy(np.array([2.0]), np.array([1.0]))
        assert result[0]

    def test_max_corner_exclusive(self):
        result = self.vol.contains_xy(np.array([6.0]), np.array([5.0]))
        assert not result[0]

    def test_multiple_points_vectorized(self):
        xs = np.array([2.0, 4.0, 6.0, 0.0])
        ys = np.array([1.0, 3.0, 5.0, 3.0])
        result = self.vol.contains_xy(xs, ys)
        np.testing.assert_array_equal(result, [True, True, False, False])

    def test_scalar_float_inputs(self):
        result = self.vol.contains_xy(4.0, 3.0)
        assert bool(result) is True


# ---------------------------------------------------------------------------
# intersects_chunk
# ---------------------------------------------------------------------------


class TestIntersectsChunk:
    def setup_method(self):
        # x[-12,12), y[-5,25), z[6,10)
        self.vol = ZoneVolume(1, "grass", (-12.0, -5.0, 6.0), (12.0, 25.0, 10.0))
        self.cm = 16.0

    def test_overlapping_chunk_intersects(self):
        assert self.vol.intersects_chunk((0, 0, 0), self.cm) is True

    def test_chunk_on_x_neg_side_intersects(self):
        assert self.vol.intersects_chunk((-1, 0, 0), self.cm) is True

    def test_chunk_too_far_positive_x_no_intersect(self):
        assert self.vol.intersects_chunk((5, 0, 0), self.cm) is False

    def test_chunk_above_z_window_no_intersect(self):
        assert self.vol.intersects_chunk((0, 0, 2), self.cm) is False

    def test_chunk_below_z_window_no_intersect(self):
        assert self.vol.intersects_chunk((0, 0, -1), self.cm) is False


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------


class TestSerialisationRoundTrip:
    def test_grass_volume_round_trip(self):
        v = ZoneVolume(3, "grass", (-12.0, -5.0, 6.0), (12.0, 25.0, 10.0), params={"density": 9.5})
        v2 = ZoneVolume.from_dict(v.to_dict())
        assert v2 == v

    def test_biome_volume_round_trip(self):
        v = ZoneVolume(1, "biome", (0.0, 0.0, 0.0), (50.0, 50.0, 16.0), biome="snow")
        v2 = ZoneVolume.from_dict(v.to_dict())
        assert v2.biome == "snow"
        assert v2 == v

    def test_to_dict_contains_required_keys(self):
        v = ZoneVolume(1, "grass", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        d = v.to_dict()
        for key in ("id", "tag", "min_corner", "max_corner", "biome", "params"):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_corners_are_lists(self):
        v = ZoneVolume(1, "grass", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        d = v.to_dict()
        assert isinstance(d["min_corner"], list)
        assert isinstance(d["max_corner"], list)

    def test_params_preserved_in_round_trip(self):
        v = ZoneVolume(1, "grass", (0.0, 0.0, 0.0), (5.0, 5.0, 2.0), params={"density": 12.0})
        v2 = ZoneVolume.from_dict(v.to_dict())
        assert v2.params["density"] == pytest.approx(12.0)

    def test_no_biome_no_params_round_trip(self):
        v = ZoneVolume(1, "grass", (0.0, 0.0, 0.0), (5.0, 5.0, 2.0))
        v2 = ZoneVolume.from_dict(v.to_dict())
        assert v2 == v
        assert v2.biome is None
        assert v2.params == {}
