"""
tests/test_sky_atmosphere.py — Headless tests for the physical atmosphere.

Covers ``sky/atmosphere.py`` (single-scattering model), the new HDR radiance
fields on ``SkyState`` (``sun_radiance`` / ``moon_radiance`` / ``sky_ambient``),
and the procedural ``"moon_surface"`` texture def.  No panda3d imports.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fire_engine.core import Clock, EventBus, load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.world.sky import atmosphere
from fire_engine.world.sky.sky_state import MOON_CYCLE_DAYS, SkySystem

NOON_Z = 0.94          # sin(sun elevation) at the v0 noon arc peak


def _sky_at(hour: float, day: int = 0, seed: int = 1337):
    """Fresh SkySystem snapshot at the given game hour/day."""
    set_world_seed(seed)
    cfg = load_config()
    bus = EventBus()
    clock = Clock(fixed_dt=0.02, bus=bus)
    clock.game_day = day
    clock.game_time_of_day = hour * 3600.0
    return SkySystem(cfg, clock, bus).update()


# ---------------------------------------------------------------------------
# atmosphere.py — the physical model
# ---------------------------------------------------------------------------

class TestAtmosphere:
    def test_deterministic(self):
        a = atmosphere.sun_radiance(0.5)
        b = atmosphere.sun_radiance(0.5)
        np.testing.assert_array_equal(a, b)
        dirs = np.array([[0.0, 0.0, 1.0], [0.7, 0.0, 0.4]])
        sun = np.array([0.34, 0.0, 0.94])
        np.testing.assert_array_equal(
            atmosphere.sky_radiance(dirs, sun),
            atmosphere.sky_radiance(dirs, sun))

    def test_noon_sun_radiance_near_contract_target(self):
        noon = atmosphere.sun_radiance(NOON_Z)
        assert abs(noon[0] - 3.2) < 0.5
        assert abs(noon[1] - 3.0) < 0.5
        assert abs(noon[2] - 2.6) < 0.5

    def test_sunset_redder_than_noon(self):
        noon = atmosphere.sun_radiance(NOON_Z)
        dusk = atmosphere.sun_radiance(0.03)
        assert dusk[0] / max(dusk[2], 1e-9) > noon[0] / noon[2] * 3.0
        assert dusk.sum() < noon.sum()          # and much dimmer

    def test_sun_extinguished_below_minus_four_degrees(self):
        below = atmosphere.sun_radiance(math.sin(math.radians(-5.0)))
        assert np.all(below == 0.0)
        # ...with a smooth nonzero tail just below the horizon.
        twilight = atmosphere.sun_radiance(math.sin(math.radians(-2.0)))
        assert twilight.sum() > 0.0

    def test_zenith_is_blue_at_midday(self):
        L = atmosphere.sky_radiance(
            np.array([0.0, 0.0, 1.0]), np.array([0.34, 0.0, 0.94]))
        assert L[0, 2] > L[0, 1] > L[0, 0]

    def test_transmittance_favors_red(self):
        T = atmosphere.transmittance(np.array([0.7, 0.0, 0.1]))
        assert T[0, 0] > T[0, 1] > T[0, 2]

    def test_ambient_noon_range_and_night_zero(self):
        amb = atmosphere.sky_ambient(NOON_Z)
        assert 0.15 < amb[0] < 0.45
        assert 0.25 < amb[1] < 0.60
        assert 0.45 < amb[2] < 0.95
        assert amb[2] > amb[1] > amb[0]         # blue-dominant
        night = atmosphere.sky_ambient(-0.3)
        assert np.all(night < 1e-4)


# ---------------------------------------------------------------------------
# SkyState HDR radiance contract
# ---------------------------------------------------------------------------

class TestSkyStateRadiance:
    def test_noon_fields_in_contract_ranges(self):
        st = _sky_at(12.0)
        # Weather on day 0 may dim the sun; check the clear-sky ceiling shape
        # via ratios instead of absolutes where weather interferes.
        assert st.sun_radiance[0] > st.sun_radiance[2] * 0.9
        assert 0.0 < st.sky_ambient[2] <= 1.2
        assert st.sky_ambient[2] > st.sky_ambient[0]    # bluish skylight

    def test_sunset_hue_shift(self):
        noon = _sky_at(12.0)
        dusk = _sky_at(17.9)
        rb_noon = noon.sun_radiance[0] / max(noon.sun_radiance[2], 1e-9)
        rb_dusk = dusk.sun_radiance[0] / max(dusk.sun_radiance[2], 1e-9)
        assert rb_dusk > rb_noon

    def test_night_sun_zero_ambient_floor(self):
        st = _sky_at(0.0)
        assert st.sun_radiance == (0.0, 0.0, 0.0)
        assert all(c > 0.004 for c in st.sky_ambient)   # night floor present
        assert st.sky_ambient[2] < 0.08                 # ...but it IS night

    def test_moon_radiance_follows_phase(self):
        # Day 0 = new moon → no moonlight; day 15 = full moon.
        new = _sky_at(0.0, day=0)
        assert new.moon_radiance == (0.0, 0.0, 0.0)
        full = _sky_at(0.0, day=MOON_CYCLE_DAYS // 2)
        if full.moon_dir.z > 0.05:                      # moon up at midnight
            assert full.moon_radiance[2] > full.moon_radiance[0]  # pale blue
            assert full.moon_radiance[2] > 0.005

    def test_deterministic(self):
        a = _sky_at(9.5)
        b = _sky_at(9.5)
        assert a.sun_radiance == b.sun_radiance
        assert a.sky_ambient == b.sky_ambient
        assert a.zenith_color == b.zenith_color

    def test_gradient_colors_clamped_ldr(self):
        for h in (0.0, 6.0, 12.0, 18.2):
            st = _sky_at(h)
            assert all(0.0 <= c <= 1.0 for c in st.zenith_color)
            assert all(0.0 <= c <= 1.0 for c in st.horizon_color)


# ---------------------------------------------------------------------------
# "moon_surface" procedural texture
# ---------------------------------------------------------------------------

class TestMoonSurface:
    def setup_method(self):
        # Other test modules reset the procedural registry; make sure the
        # moon def is registered regardless of test order.
        from fire_engine.procedural import registry
        from fire_engine.procedural.textures.moon_surface import MoonSurfaceDef
        if "moon_surface" not in registry._registry:
            registry.register(MoonSurfaceDef())

    def test_shape_dtype_and_disc_alpha(self):
        set_world_seed(1337)
        import fire_engine.procedural  # noqa: F401  (registration)
        from fire_engine.procedural import get
        arr = get("moon_surface")
        assert arr.shape == (256, 256, 4) and arr.dtype == np.uint8
        assert arr[128, 128, 3] == 255      # centre inside the disc
        assert arr[0, 0, 3] == 0            # corner outside the disc

    def test_deterministic_per_seed(self):
        import fire_engine.procedural  # noqa: F401
        from fire_engine.procedural import get, clear_cache
        set_world_seed(1337)
        clear_cache()
        a = get("moon_surface").copy()
        set_world_seed(1337)
        clear_cache()
        b = get("moon_surface")
        np.testing.assert_array_equal(a, b)

    def test_different_seed_different_moon(self):
        import fire_engine.procedural  # noqa: F401
        from fire_engine.procedural import get, clear_cache
        set_world_seed(1337)
        clear_cache()
        a = get("moon_surface").copy()
        set_world_seed(4242)
        clear_cache()
        b = get("moon_surface")
        assert not np.array_equal(a, b)

    def test_has_craters_and_maria_variation(self):
        set_world_seed(1337)
        import fire_engine.procedural  # noqa: F401
        from fire_engine.procedural import get
        arr = get("moon_surface")
        inside = arr[arr[..., 3] == 255][:, :3].astype(np.int32)
        assert inside.std() > 8.0           # visible surface detail
