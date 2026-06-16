"""
tests/world/sky/test_types.py — Mirror tests for fire_engine/world/sky/types.py.

Covers the SkyState frozen dataclass:
 - All fields are present and have correct types
 - Frozen: assignment raises FrozenInstanceError
 - Equality and hashing (frozen dataclasses are hashable)
 - Default HDR radiance fields are (0, 0, 0)
 - Field values survive construction unchanged
 - __all__ exports SkyState

No panda3d imports. All tests headless.
"""

from __future__ import annotations

import dataclasses

import pytest

from fire_engine.core.math3d import Vec3
from fire_engine.world.sky.types import SkyState, __all__

# A minimal valid SkyState for reuse across tests
_VEC3_UNIT_Z = Vec3(0.0, 0.0, 1.0)
_VEC3_UNIT_NZ = Vec3(0.0, 0.0, -1.0)

_MINIMAL_KWARGS: dict = dict(
    sun_dir=_VEC3_UNIT_Z,
    moon_dir=_VEC3_UNIT_NZ,
    sun_color=(1.0, 0.9, 0.8),
    sun_intensity=0.85,
    moon_phase=0.5,
    daylight=1.0,
    star_visibility=0.0,
    zenith_color=(0.3, 0.46, 0.72),
    horizon_color=(0.62, 0.72, 0.82),
    cloud_coverage=0.0,
    cloud_density=0.0,
    fog_density=0.001,
    fog_color=(0.6, 0.65, 0.7),
    rain_intensity=0.0,
    wind_dir=(1.0, 0.0),
    wind_speed=3.5,
    terrain_light_scale=(1.0, 1.0, 1.0),
)


def _make() -> SkyState:
    return SkyState(**_MINIMAL_KWARGS)


# ---------------------------------------------------------------------------
# Module-level checks
# ---------------------------------------------------------------------------


class TestModule:
    def test_sky_state_in_all(self):
        assert "SkyState" in __all__

    def test_sky_state_is_dataclass(self):
        assert dataclasses.is_dataclass(SkyState)

    def test_sky_state_is_frozen(self):
        assert SkyState.__dataclass_params__.frozen  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Construction and field values
# ---------------------------------------------------------------------------


class TestSkyStateConstruction:
    def test_can_construct_with_minimal_fields(self):
        st = _make()
        assert isinstance(st, SkyState)

    def test_sun_dir_stored(self):
        st = _make()
        assert st.sun_dir is _VEC3_UNIT_Z

    def test_moon_dir_stored(self):
        st = _make()
        assert st.moon_dir is _VEC3_UNIT_NZ

    def test_sun_color_stored(self):
        st = _make()
        assert st.sun_color == (1.0, 0.9, 0.8)

    def test_sun_intensity_stored(self):
        st = _make()
        assert st.sun_intensity == pytest.approx(0.85)

    def test_moon_phase_stored(self):
        st = _make()
        assert st.moon_phase == pytest.approx(0.5)

    def test_daylight_stored(self):
        st = _make()
        assert st.daylight == pytest.approx(1.0)

    def test_star_visibility_stored(self):
        st = _make()
        assert st.star_visibility == pytest.approx(0.0)

    def test_zenith_color_stored(self):
        st = _make()
        assert st.zenith_color == pytest.approx((0.3, 0.46, 0.72))

    def test_horizon_color_stored(self):
        st = _make()
        assert st.horizon_color == pytest.approx((0.62, 0.72, 0.82))

    def test_cloud_coverage_stored(self):
        st = _make()
        assert st.cloud_coverage == pytest.approx(0.0)

    def test_cloud_density_stored(self):
        st = _make()
        assert st.cloud_density == pytest.approx(0.0)

    def test_fog_density_stored(self):
        st = _make()
        assert st.fog_density == pytest.approx(0.001)

    def test_fog_color_stored(self):
        st = _make()
        assert st.fog_color == pytest.approx((0.6, 0.65, 0.7))

    def test_rain_intensity_stored(self):
        st = _make()
        assert st.rain_intensity == pytest.approx(0.0)

    def test_wind_dir_stored(self):
        st = _make()
        assert st.wind_dir == (1.0, 0.0)

    def test_wind_speed_stored(self):
        st = _make()
        assert st.wind_speed == pytest.approx(3.5)

    def test_terrain_light_scale_stored(self):
        st = _make()
        assert st.terrain_light_scale == pytest.approx((1.0, 1.0, 1.0))


# ---------------------------------------------------------------------------
# Default HDR fields
# ---------------------------------------------------------------------------


class TestSkyStateDefaults:
    def test_sun_radiance_default_zero(self):
        st = _make()
        assert st.sun_radiance == (0.0, 0.0, 0.0)

    def test_moon_radiance_default_zero(self):
        st = _make()
        assert st.moon_radiance == (0.0, 0.0, 0.0)

    def test_sky_ambient_default_zero(self):
        st = _make()
        assert st.sky_ambient == (0.0, 0.0, 0.0)

    def test_hdr_fields_can_be_provided(self):
        st = SkyState(
            **_MINIMAL_KWARGS,
            sun_radiance=(3.2, 3.0, 2.6),
            moon_radiance=(0.06, 0.07, 0.10),
            sky_ambient=(0.21, 0.40, 0.71),
        )
        assert st.sun_radiance == pytest.approx((3.2, 3.0, 2.6))
        assert st.moon_radiance == pytest.approx((0.06, 0.07, 0.10))
        assert st.sky_ambient == pytest.approx((0.21, 0.40, 0.71))


# ---------------------------------------------------------------------------
# Immutable / frozen behaviour
# ---------------------------------------------------------------------------


class TestSkyStateFrozen:
    def test_assignment_raises(self):
        st = _make()
        with pytest.raises(dataclasses.FrozenInstanceError):
            st.daylight = 0.5  # type: ignore[misc]

    def test_cannot_add_attribute(self):
        st = _make()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            st.new_field = 42  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Equality and hashing
# ---------------------------------------------------------------------------


class TestSkyStateEquality:
    def test_equal_to_itself(self):
        st = _make()
        assert st == st

    def test_equal_to_identical_copy(self):
        st1 = _make()
        st2 = _make()
        assert st1 == st2

    def test_not_equal_different_daylight(self):
        st1 = _make()
        st2 = SkyState(**{**_MINIMAL_KWARGS, "daylight": 0.5})
        assert st1 != st2

    def test_hashable(self):
        st = _make()
        h = hash(st)
        assert isinstance(h, int)

    def test_can_be_used_in_set(self):
        st1 = _make()
        st2 = _make()
        s = {st1, st2}
        assert len(s) == 1
