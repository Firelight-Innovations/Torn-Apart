"""
tests/world/sky/test_sky_state.py — Mirror tests for fire_engine/world/sky/sky_state.py.

Covers:
 - SkySystem.update(): returns SkyState with correct field values
 - SkySystem.state: lazy property caches the last update
 - MOON_CYCLE_DAYS constant
 - Determinism: same seed + clock → identical SkyState
 - Color-ramp and weather interaction at key times
 - SkyState re-export (SkyState is importable from sky_state)

No panda3d imports. All tests headless and deterministic.
"""

from __future__ import annotations

import math

from fire_engine.core import Clock, EventBus, load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.world.sky import WeatherType
from fire_engine.world.sky.sky_state import MOON_CYCLE_DAYS, SkyState, SkySystem

HOUR = 3600.0


def _make_sky(seed: int = 1337, day: int = 0, tod: float = 0.0):
    """Fresh SkySystem with the given seed and clock position."""
    set_world_seed(seed)
    cfg = load_config()
    bus = EventBus()
    clock = Clock(fixed_dt=cfg.fixed_dt, bus=bus)
    clock.game_day = day
    clock.game_time_of_day = tod
    return SkySystem(cfg, clock, bus), clock


# ---------------------------------------------------------------------------
# Re-exports and constants
# ---------------------------------------------------------------------------


class TestReExports:
    def test_sky_state_importable_from_sky_state(self):
        from fire_engine.world.sky.sky_state import SkyState as SS

        assert SS is SkyState

    def test_moon_cycle_days_is_30(self):
        assert MOON_CYCLE_DAYS == 30

    def test_moon_cycle_days_is_int(self):
        assert isinstance(MOON_CYCLE_DAYS, int)


# ---------------------------------------------------------------------------
# SkySystem.update — returns valid SkyState
# ---------------------------------------------------------------------------


class TestSkySystemUpdate:
    def test_returns_sky_state_instance(self):
        sky, _ = _make_sky()
        assert isinstance(sky.update(), SkyState)

    def test_update_result_cached_as_state(self):
        sky, _ = _make_sky()
        st = sky.update()
        assert sky.state is st

    def test_state_lazy_property_calls_update(self):
        sky, _ = _make_sky(tod=12 * HOUR)
        st = sky.state
        assert isinstance(st, SkyState)
        assert st.daylight == 1.0

    def test_sun_dir_z_positive_at_noon(self):
        sky, _ = _make_sky(tod=12 * HOUR)
        st = sky.update()
        assert st.sun_dir.z >= 0.9

    def test_sun_dir_z_negative_at_midnight(self):
        sky, _ = _make_sky(tod=0.0)
        st = sky.update()
        assert st.sun_dir.z <= -0.9

    def test_daylight_one_at_noon(self):
        sky, _ = _make_sky(tod=12 * HOUR)
        assert sky.update().daylight == 1.0

    def test_daylight_zero_at_midnight(self):
        sky, _ = _make_sky(tod=0.0)
        assert sky.update().daylight == 0.0

    def test_sun_intensity_zero_at_midnight(self):
        sky, _ = _make_sky(tod=0.0)
        assert sky.update().sun_intensity == 0.0

    def test_star_visibility_zero_at_noon(self):
        sky, _ = _make_sky(tod=12 * HOUR)
        assert sky.update().star_visibility == 0.0

    def test_moon_phase_new_on_day_zero(self):
        sky, _ = _make_sky(day=0, tod=0.0)
        assert sky.update().moon_phase == 0.0

    def test_moon_phase_full_on_day_15(self):
        sky, _ = _make_sky(day=15, tod=0.0)
        assert sky.update().moon_phase == 0.5

    def test_wind_dir_unit_length(self):
        sky, clock = _make_sky()
        for day in range(3):
            clock.game_day = day
            clock.game_time_of_day = 10 * HOUR
            st = sky.update()
            wx, wy = st.wind_dir
            assert abs(math.hypot(wx, wy) - 1.0) < 1e-6

    def test_terrain_light_scale_in_range(self):
        sky, clock = _make_sky(seed=1337)
        for step in range(0, 24 * 3, 1):
            clock.game_time_of_day = step * (20 * 60.0)
            st = sky.update()
            for c in st.terrain_light_scale:
                assert 0.0 <= c <= 1.05

    def test_clear_noon_terrain_scale_near_white(self):
        sky, clock = _make_sky(seed=1337, day=0, tod=12 * HOUR - 1500.0)
        sky.weather.force_weather(WeatherType.CLEAR)
        sky.update()
        clock.game_time_of_day = 12 * HOUR
        st = sky.update()
        for c in st.terrain_light_scale:
            assert abs(c - 1.0) <= 0.03

    def test_night_terrain_scale_cool_blue(self):
        sky, clock = _make_sky(seed=1337, tod=0.0)
        sky.weather.force_weather(WeatherType.CLEAR)
        sky.update()
        clock.game_time_of_day = 1500.0
        st = sky.update()
        r, g, b = st.terrain_light_scale
        assert b > g > r

    def test_gradient_colors_ldr(self):
        for h in (0.0, 6.0, 12.0, 18.2):
            sky, _ = _make_sky(tod=h * HOUR)
            st = sky.update()
            assert all(0.0 <= c <= 1.0 for c in st.zenith_color)
            assert all(0.0 <= c <= 1.0 for c in st.horizon_color)

    def test_sun_radiance_zero_at_night(self):
        sky, _ = _make_sky(tod=0.0)
        st = sky.update()
        assert st.sun_radiance == (0.0, 0.0, 0.0)

    def test_sky_ambient_has_night_floor(self):
        sky, _ = _make_sky(tod=0.0)
        st = sky.update()
        assert all(c > 0.004 for c in st.sky_ambient)

    def test_sky_ambient_blue_dominant_at_noon(self):
        sky, _ = _make_sky(tod=12 * HOUR)
        st = sky.update()
        assert st.sky_ambient[2] > st.sky_ambient[0]

    def test_star_visibility_high_on_clear_night(self):
        sky, clock = _make_sky(seed=5, tod=23 * HOUR)
        sky.weather.force_weather(WeatherType.CLEAR)
        sky.update()
        clock.game_time_of_day = 23 * HOUR + 1500.0
        st = sky.update()
        assert st.daylight == 0.0
        assert st.star_visibility > 0.85


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestSkySystemDeterminism:
    def test_same_seed_same_state(self):
        for day, tod in [(0, 0.0), (1, 9.5 * HOUR), (4, 13.0 * HOUR), (7, 21.25 * HOUR)]:
            sky1, _ = _make_sky(seed=42, day=day, tod=tod)
            st1 = sky1.update()
            sky2, _ = _make_sky(seed=42, day=day, tod=tod)
            st2 = sky2.update()
            assert st1 == st2, f"states differ at day={day} tod={tod}"

    def test_different_seed_different_weather(self):
        def sequence(seed: int) -> list[str]:
            sky, clock = _make_sky(seed=seed)
            out = []
            for day in range(8):
                for seg in range(12):
                    clock.game_day = day
                    clock.game_time_of_day = seg * 2 * HOUR + HOUR
                    sky.update()
                    out.append(sky.weather.current.value)
            return out

        assert sequence(1) != sequence(2)
