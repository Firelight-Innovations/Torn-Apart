"""
tests/world/sky/test_celestial.py — Mirror tests for fire_engine/world/sky/celestial.py.

Covers all public symbols:
 - sun_direction: unit length, schedule (midnight/sunrise/noon/sunset), wrapping
 - moon_direction: unit length, opposite-sun offset, midnight near zenith
 - daylight_factor: endpoints, saturates outside band
 - smoothstep: correctness, clamping, cubic formula
 - lerp_color: endpoints, t clamping, channel independence
 - color_ramp: interpolation, endpoint clamping
 - Constants: GAME_SECONDS_PER_DAY, arc tilts, DAYLIGHT_Z bounds

No panda3d imports. All tests headless and deterministic.
"""

from __future__ import annotations

import math

import numpy as np

from fire_engine.world.sky.celestial import (
    DAYLIGHT_Z_HI,
    DAYLIGHT_Z_LO,
    GAME_SECONDS_PER_DAY,
    MOON_ARC_TILT_RAD,
    MOON_PHASE_OFFSET_RAD,
    SUN_ARC_TILT_RAD,
    color_ramp,
    daylight_factor,
    lerp_color,
    moon_direction,
    smoothstep,
    sun_direction,
)

HOUR = 3600.0


def _len(v) -> float:
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def _dot(a, b) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z


def _to_arr(v):
    return np.array([v.x, v.y, v.z])


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_game_seconds_per_day(self):
        assert GAME_SECONDS_PER_DAY == 24.0 * 3600.0

    def test_sun_arc_tilt_is_20_degrees(self):
        assert abs(SUN_ARC_TILT_RAD - math.radians(20.0)) < 1e-9

    def test_moon_arc_tilt_is_12_degrees(self):
        assert abs(MOON_ARC_TILT_RAD - math.radians(12.0)) < 1e-9

    def test_moon_phase_offset_positive(self):
        assert MOON_PHASE_OFFSET_RAD > 0.0

    def test_daylight_z_lo_negative(self):
        assert DAYLIGHT_Z_LO < 0.0

    def test_daylight_z_hi_positive(self):
        assert DAYLIGHT_Z_HI > 0.0

    def test_daylight_z_lo_less_than_hi(self):
        assert DAYLIGHT_Z_LO < DAYLIGHT_Z_HI


# ---------------------------------------------------------------------------
# sun_direction
# ---------------------------------------------------------------------------


class TestSunDirection:
    def test_unit_length_all_half_hours(self):
        for h in range(48):
            d = sun_direction(h * 0.5 * HOUR)
            assert abs(_len(d) - 1.0) < 1e-5, f"non-unit at {h * 0.5}h"

    def test_midnight_below_horizon(self):
        assert sun_direction(0.0).z <= -0.9

    def test_sunrise_on_horizon_due_east(self):
        d = sun_direction(6 * HOUR)
        assert abs(d.z) < 0.15
        assert d.x > 0.9

    def test_noon_high(self):
        assert sun_direction(12 * HOUR).z >= 0.9

    def test_sunset_on_horizon_due_west(self):
        d = sun_direction(18 * HOUR)
        assert abs(d.z) < 0.15
        assert d.x < -0.9

    def test_periodic_wrap(self):
        a = sun_direction(0.0)
        b = sun_direction(GAME_SECONDS_PER_DAY)
        assert np.allclose(_to_arr(a), _to_arr(b), atol=1e-5)

    def test_noon_is_maximum_elevation(self):
        noon_z = sun_direction(12 * HOUR).z
        for h in range(25):
            assert sun_direction(h * HOUR).z <= noon_z + 1e-6

    def test_continuity_one_minute_step(self):
        max_step = math.radians(1.5)
        for i in range(0, 24 * 60, 7):
            a = sun_direction(i * 60.0)
            b = sun_direction((i + 1) * 60.0)
            dot = _dot(a, b)
            angle = math.acos(min(1.0, max(-1.0, dot)))
            assert angle < max_step, f"snap at minute {i}: {angle} rad"

    def test_deterministic(self):
        for t in [0.0, 6 * HOUR, 12 * HOUR, 18 * HOUR]:
            a = sun_direction(t)
            b = sun_direction(t)
            assert np.allclose(_to_arr(a), _to_arr(b))

    def test_negative_time_wraps_like_python_modulo(self):
        a = sun_direction(-1.0)
        b = sun_direction(GAME_SECONDS_PER_DAY - 1.0)
        assert np.allclose(_to_arr(a), _to_arr(b), atol=1e-6)


# ---------------------------------------------------------------------------
# moon_direction
# ---------------------------------------------------------------------------


class TestMoonDirection:
    def test_unit_length_all_hours(self):
        for h in range(24):
            d = moon_direction(h * HOUR)
            assert abs(_len(d) - 1.0) < 1e-5

    def test_midnight_moon_high(self):
        assert moon_direction(0.0).z > 0.8

    def test_roughly_opposite_sun_at_noon(self):
        s = sun_direction(12 * HOUR)
        m = moon_direction(12 * HOUR)
        assert _dot(s, m) < -0.8

    def test_dusk_overlap_both_above_horizon(self):
        """Phase offset: both sun and moon above horizon around dusk (17.5h)."""
        t = 17.5 * HOUR
        assert sun_direction(t).z > 0.0
        assert moon_direction(t).z > 0.0

    def test_periodic_wrap(self):
        a = moon_direction(0.0)
        b = moon_direction(GAME_SECONDS_PER_DAY)
        assert np.allclose(_to_arr(a), _to_arr(b), atol=1e-6)

    def test_deterministic(self):
        for t in [0.0, 6 * HOUR, 12 * HOUR, 18 * HOUR]:
            a = moon_direction(t)
            b = moon_direction(t)
            assert np.allclose(_to_arr(a), _to_arr(b))


# ---------------------------------------------------------------------------
# daylight_factor
# ---------------------------------------------------------------------------


class TestDaylightFactor:
    def test_noon_returns_one(self):
        assert daylight_factor(12 * HOUR) == 1.0

    def test_midnight_returns_zero(self):
        assert daylight_factor(0.0) == 0.0

    def test_sunrise_near_half(self):
        assert 0.4 <= daylight_factor(6 * HOUR) <= 0.6

    def test_saturates_one_hour_past_sunrise(self):
        assert daylight_factor(7.2 * HOUR) > 0.99

    def test_saturates_zero_one_hour_past_sunset(self):
        assert daylight_factor(19.2 * HOUR) < 0.01

    def test_clamped_0_1_everywhere(self):
        for i in range(0, int(GAME_SECONDS_PER_DAY) + 1, 300):
            f = daylight_factor(float(i))
            assert 0.0 <= f <= 1.0

    def test_equals_smoothstep_on_sun_z(self):
        """daylight_factor must equal smoothstep on sun_dir.z for several times."""
        for t in [5 * HOUR, 8 * HOUR, 12 * HOUR, 16 * HOUR, 20 * HOUR]:
            z = sun_direction(t).z
            expected = smoothstep(float(z), DAYLIGHT_Z_LO, DAYLIGHT_Z_HI)
            got = daylight_factor(t)
            assert abs(got - expected) < 1e-9

    def test_deterministic(self):
        for t in [0.0, 6 * HOUR, 12 * HOUR, 18 * HOUR]:
            assert daylight_factor(t) == daylight_factor(t)


# ---------------------------------------------------------------------------
# smoothstep
# ---------------------------------------------------------------------------


class TestSmoothstep:
    def test_at_lo_is_zero(self):
        assert smoothstep(0.0, 0.0, 1.0) == 0.0

    def test_at_hi_is_one(self):
        assert smoothstep(1.0, 0.0, 1.0) == 1.0

    def test_below_lo_is_zero(self):
        assert smoothstep(-5.0, 0.0, 1.0) == 0.0

    def test_above_hi_is_one(self):
        assert smoothstep(2.0, 0.0, 1.0) == 1.0

    def test_midpoint_is_half(self):
        assert abs(smoothstep(0.5, 0.0, 1.0) - 0.5) < 1e-9

    def test_quarter_cubic_formula(self):
        """t=0.25 → 3(0.0625) - 2(0.015625) = 0.15625."""
        assert abs(smoothstep(0.25, 0.0, 1.0) - 0.15625) < 1e-9

    def test_three_quarter_cubic_formula(self):
        """t=0.75 → 3(0.5625) - 2(0.421875) = 0.84375."""
        assert abs(smoothstep(0.75, 0.0, 1.0) - 0.84375) < 1e-9

    def test_arbitrary_range(self):
        """At midpoint of [2, 5] → should be 0.5."""
        assert abs(smoothstep(3.5, 2.0, 5.0) - 0.5) < 1e-9

    def test_monotonic(self):
        xs = np.linspace(-0.5, 1.5, 200)
        vals = [smoothstep(float(x), 0.0, 1.0) for x in xs]
        for i in range(len(vals) - 1):
            assert vals[i + 1] >= vals[i] - 1e-12

    def test_daylight_z_bounds(self):
        assert smoothstep(DAYLIGHT_Z_LO, DAYLIGHT_Z_LO, DAYLIGHT_Z_HI) == 0.0
        assert smoothstep(DAYLIGHT_Z_HI, DAYLIGHT_Z_LO, DAYLIGHT_Z_HI) == 1.0


# ---------------------------------------------------------------------------
# lerp_color
# ---------------------------------------------------------------------------


class TestLerpColor:
    BLACK = (0.0, 0.0, 0.0)
    WHITE = (1.0, 1.0, 1.0)
    RED = (1.0, 0.0, 0.0)
    BLUE = (0.0, 0.0, 1.0)

    def test_t0_returns_a(self):
        assert lerp_color(self.RED, self.BLUE, 0.0) == self.RED

    def test_t1_returns_b(self):
        assert lerp_color(self.RED, self.BLUE, 1.0) == self.BLUE

    def test_midpoint(self):
        r = lerp_color(self.BLACK, self.WHITE, 0.5)
        assert np.allclose(r, (0.5, 0.5, 0.5), atol=1e-9)

    def test_channels_independent(self):
        a = (0.0, 0.5, 1.0)
        b = (1.0, 0.5, 0.0)
        r = lerp_color(a, b, 0.5)
        assert np.allclose(r, (0.5, 0.5, 0.5), atol=1e-9)

    def test_t_negative_clamped(self):
        r = lerp_color(self.RED, self.BLUE, -1.0)
        assert np.allclose(r, self.RED)

    def test_t_above_one_clamped(self):
        r = lerp_color(self.RED, self.BLUE, 2.0)
        assert np.allclose(r, self.BLUE)

    def test_hdr_components_not_clamped(self):
        """Color components above 1.0 should survive lerp unchanged."""
        a = (3.2, 3.0, 2.6)
        b = (0.0, 0.0, 0.0)
        r = lerp_color(a, b, 0.0)
        assert np.allclose(r, a)

    def test_deterministic(self):
        a = (0.1, 0.2, 0.3)
        b = (0.4, 0.5, 0.6)
        assert lerp_color(a, b, 0.33) == lerp_color(a, b, 0.33)

    def test_example_from_docstring(self):
        r = lerp_color((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0.5)
        assert np.allclose(r, (0.5, 0.5, 0.5))


# ---------------------------------------------------------------------------
# color_ramp
# ---------------------------------------------------------------------------


class TestColorRamp:
    SIMPLE = (
        (0.0, (0.0, 0.0, 0.0)),
        (0.5, (0.5, 0.5, 0.5)),
        (1.0, (1.0, 1.0, 1.0)),
    )

    def test_at_first_key(self):
        r = color_ramp(0.0, self.SIMPLE)
        assert np.allclose(r, (0.0, 0.0, 0.0))

    def test_at_last_key(self):
        r = color_ramp(1.0, self.SIMPLE)
        assert np.allclose(r, (1.0, 1.0, 1.0))

    def test_below_first_key_clamps(self):
        r = color_ramp(-99.0, self.SIMPLE)
        assert np.allclose(r, (0.0, 0.0, 0.0))

    def test_above_last_key_clamps(self):
        r = color_ramp(99.0, self.SIMPLE)
        assert np.allclose(r, (1.0, 1.0, 1.0))

    def test_between_keys_interpolates(self):
        r = color_ramp(0.25, self.SIMPLE)
        assert np.allclose(r, (0.25, 0.25, 0.25), atol=1e-6)

    def test_at_middle_keyframe(self):
        r = color_ramp(0.5, self.SIMPLE)
        assert np.allclose(r, (0.5, 0.5, 0.5))

    def test_nonuniform_channels(self):
        ramp = (
            (0.0, (0.0, 1.0, 0.5)),
            (1.0, (1.0, 0.0, 0.5)),
        )
        r = color_ramp(0.5, ramp)
        assert np.allclose(r, (0.5, 0.5, 0.5), atol=1e-6)

    def test_example_from_docstring(self):
        ramp = ((0.0, (0.0, 0.0, 0.0)), (1.0, (1.0, 0.5, 0.0)))
        r = color_ramp(0.5, ramp)
        assert np.allclose(r, (0.5, 0.25, 0.0))

    def test_deterministic(self):
        r1 = color_ramp(0.3, self.SIMPLE)
        r2 = color_ramp(0.3, self.SIMPLE)
        assert r1 == r2
