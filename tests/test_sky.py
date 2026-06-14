"""
tests/test_sky.py — Headless tests for the sky package (celestial + SkySystem).

No panda3d imports anywhere in this file.  All tests operate on the pure
celestial functions and the SkySystem/SkyState composer.

Test coverage
-------------
- sun_direction: unit length, elevation at 00/06/12/18 h, rise east/set west.
- moon_direction: unit length, roughly opposite the sun, twilight overlap.
- Continuity: a 1-game-minute step moves the sun by only a small angle.
- daylight endpoints (0 at midnight, 1 at noon, ~0.5 at sunrise).
- SkyState determinism: same seed + same clock state → identical fields.
- Different seed → different weather sequence.
- terrain_light_scale: within [0, 1.05] everywhere; ≈ (1, 1, 1) at clear noon.
"""

from __future__ import annotations

import math

import pytest

from fire_engine.core import Clock, EventBus, load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.world.sky import SkySystem, WeatherType, moon_direction, sun_direction
from fire_engine.world.sky.celestial import daylight_factor

HOUR = 3600.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sky(seed: int = 1337, day: int = 0, tod: float = 0.0):
    """Fresh SkySystem with the world seed set and the clock at (day, tod)."""
    set_world_seed(seed)
    cfg = load_config()
    bus = EventBus()
    clock = Clock(fixed_dt=cfg.fixed_dt, bus=bus)
    clock.game_day = day
    clock.game_time_of_day = tod
    return SkySystem(cfg, clock, bus), clock


def _length(v) -> float:
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


# ---------------------------------------------------------------------------
# sun_direction / moon_direction
# ---------------------------------------------------------------------------


class TestSunDirection:
    def test_unit_length_all_day(self):
        for h in range(0, 48):  # every half hour
            d = sun_direction(h * 0.5 * HOUR)
            assert abs(_length(d) - 1.0) < 1e-5, f"non-unit at {h * 0.5} h"

    def test_midnight_below_horizon(self):
        assert sun_direction(0.0).z <= -0.9

    def test_sunrise_on_horizon_rising_east(self):
        d = sun_direction(6 * HOUR)
        assert abs(d.z) < 0.15, "sunrise sun should sit on the horizon"
        assert d.x > 0.9, "sun must rise in +X (east)"

    def test_noon_high(self):
        assert sun_direction(12 * HOUR).z >= 0.9

    def test_sunset_on_horizon_setting_west(self):
        d = sun_direction(18 * HOUR)
        assert abs(d.z) < 0.15, "sunset sun should sit on the horizon"
        assert d.x < -0.9, "sun must set in -X (west)"

    def test_continuity_one_minute_step(self):
        """One game minute moves the sun by a small angle (no snaps)."""
        max_step = math.radians(1.5)  # generous; true step ≈ 0.25°
        for i in range(0, 24 * 60, 7):  # sample across the whole day
            a = sun_direction(i * 60.0)
            b = sun_direction((i + 1) * 60.0)
            dot = a.x * b.x + a.y * b.y + a.z * b.z
            angle = math.acos(min(1.0, max(-1.0, dot)))
            assert angle < max_step, f"snap at minute {i}: {angle} rad"

    def test_periodic_wraps(self):
        a = sun_direction(0.0)
        b = sun_direction(24 * HOUR)
        assert a.approx_eq(b, eps=1e-5)


class TestMoonDirection:
    def test_unit_length(self):
        for h in range(24):
            d = moon_direction(h * HOUR)
            assert abs(_length(d) - 1.0) < 1e-5

    def test_midnight_moon_high(self):
        assert moon_direction(0.0).z > 0.9

    def test_roughly_opposite_sun_at_noon(self):
        s = sun_direction(12 * HOUR)
        m = moon_direction(12 * HOUR)
        dot = s.x * m.x + s.y * m.y + s.z * m.z
        assert dot < -0.8, "moon should be roughly opposite the sun"

    def test_both_visible_at_dusk(self):
        """The moon's phase offset puts both bodies above the horizon at dusk."""
        t = 17.5 * HOUR
        assert sun_direction(t).z > 0.0
        assert moon_direction(t).z > 0.0


# ---------------------------------------------------------------------------
# daylight
# ---------------------------------------------------------------------------


class TestDaylight:
    def test_noon_full(self):
        assert daylight_factor(12 * HOUR) == 1.0

    def test_midnight_zero(self):
        assert daylight_factor(0.0) == 0.0

    def test_sunrise_half(self):
        assert 0.4 <= daylight_factor(6 * HOUR) <= 0.6

    def test_saturates_about_one_hour_past_sunrise(self):
        assert daylight_factor(7.2 * HOUR) > 0.99
        assert daylight_factor(19.2 * HOUR) < 0.01


# ---------------------------------------------------------------------------
# SkyState determinism
# ---------------------------------------------------------------------------


class TestSkyStateDeterminism:
    def test_same_seed_identical_state(self):
        """Two fresh SkySystems, same seed + clock state → identical fields."""
        for day, tod in [(0, 0.0), (1, 9.5 * HOUR), (4, 13.0 * HOUR), (7, 21.25 * HOUR)]:
            sky1, _ = _make_sky(seed=42, day=day, tod=tod)
            st1 = sky1.update()
            sky2, _ = _make_sky(seed=42, day=day, tod=tod)
            st2 = sky2.update()
            assert st1 == st2, f"states differ at day={day} tod={tod}"

    def test_state_property_lazy(self):
        sky, _ = _make_sky(seed=7, day=0, tod=12 * HOUR)
        st = sky.state  # update() never called explicitly — lazy compute
        assert st is sky.state
        assert st.daylight == 1.0


class TestDifferentSeed:
    def test_different_seed_different_weather_sequence(self):
        # Spatial weather model: at a *fixed* world point the timeline is
        # dominated by the per-day regime (storm cells rarely cross any single
        # point), so two seeds can share a short regime run by chance.  Sample
        # a longer window (8 days) so the seed-dependent regime draw has room
        # to diverge — the honest "different world ⇒ different weather"
        # property for a point sample.
        def sequence(seed: int) -> list[str]:
            sky, clock = _make_sky(seed=seed)
            out = []
            for day in range(8):
                for seg in range(12):
                    clock.game_day = day
                    clock.game_time_of_day = seg * 2 * HOUR + HOUR  # mid-segment
                    sky.update()
                    out.append(sky.weather.current.value)
            return out

        assert sequence(1) != sequence(2), (
            "Different world seeds must produce different weather timelines"
        )


# ---------------------------------------------------------------------------
# terrain_light_scale
# ---------------------------------------------------------------------------


class TestTerrainLightScale:
    def test_within_range_over_two_days(self):
        sky, clock = _make_sky(seed=1337)
        for day in range(2):
            for step in range(0, 24 * 3, 1):  # every 20 game minutes
                clock.game_day = day
                clock.game_time_of_day = step * (20 * 60.0)
                st = sky.update()
                for c in st.terrain_light_scale:
                    assert 0.0 <= c <= 1.05, (
                        f"scale {st.terrain_light_scale} out of range at "
                        f"day={day} tod={clock.game_time_of_day}"
                    )

    def test_clear_noon_is_white(self):
        """Forced-clear noon → terrain_light_scale ≈ (1, 1, 1)."""
        sky, clock = _make_sky(seed=1337, day=0, tod=12 * HOUR - 1500.0)
        sky.weather.force_weather(WeatherType.CLEAR)
        sky.update()  # anchors the override blend
        clock.game_time_of_day = 12 * HOUR  # 1500 s later — blend complete
        st = sky.update()
        for c in st.terrain_light_scale:
            assert abs(c - 1.0) <= 0.03, (
                f"clear noon scale should be ~1.0, got {st.terrain_light_scale}"
            )

    def test_night_floor_is_dim_cool_blue(self):
        sky, clock = _make_sky(seed=1337, day=0, tod=0.0)
        sky.weather.force_weather(WeatherType.CLEAR)
        sky.update()
        clock.game_time_of_day = 1500.0  # past the override blend, still night
        st = sky.update()
        r, g, b = st.terrain_light_scale
        assert b > g > r, "night floor should be cool blue (b > g > r)"
        assert r == pytest.approx(0.16, abs=0.02)
        assert b == pytest.approx(0.30, abs=0.02)


# ---------------------------------------------------------------------------
# Misc SkyState invariants
# ---------------------------------------------------------------------------


class TestSkyStateInvariants:
    def test_sun_intensity_zero_below_horizon(self):
        sky, clock = _make_sky(seed=5, day=0, tod=0.0)
        st = sky.update()
        assert st.sun_intensity == 0.0

    def test_star_visibility_high_on_clear_night(self):
        sky, clock = _make_sky(seed=5, day=0, tod=23 * HOUR)
        sky.weather.force_weather(WeatherType.CLEAR)
        sky.update()
        clock.game_time_of_day = 23 * HOUR + 1500.0
        st = sky.update()
        assert st.daylight == 0.0
        assert st.star_visibility > 0.85

    def test_star_visibility_zero_at_noon(self):
        sky, _ = _make_sky(seed=5, day=0, tod=12 * HOUR)
        assert sky.update().star_visibility == 0.0

    def test_moon_phase_cycles_from_game_day(self):
        sky0, _ = _make_sky(seed=5, day=0, tod=0.0)
        assert sky0.update().moon_phase == 0.0  # new moon
        sky15, _ = _make_sky(seed=5, day=15, tod=0.0)
        assert sky15.update().moon_phase == 0.5  # full moon

    def test_wind_dir_unit_length(self):
        sky, clock = _make_sky(seed=5)
        for day in range(3):
            clock.game_day = day
            clock.game_time_of_day = 10 * HOUR
            st = sky.update()
            wx, wy = st.wind_dir
            assert abs(math.hypot(wx, wy) - 1.0) < 1e-6
