"""
tests/test_sky_celestial_edges.py — Golden-master / characterisation tests for
sky/celestial.py edge cases.

DO NOT edit to fix bugs; pin current behaviour and note suspicions in comments.

No panda3d imports anywhere.  All tests are headless and deterministic.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fire_engine.core.math3d import Vec3
from fire_engine.world.sky.celestial import (
    DAYLIGHT_Z_HI,
    DAYLIGHT_Z_LO,
    GAME_SECONDS_PER_DAY,
    MOON_PHASE_OFFSET_RAD,
    color_ramp,
    daylight_factor,
    lerp_color,
    moon_direction,
    smoothstep,
    sun_direction,
)

HOUR = 3600.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(v: Vec3) -> float:
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def _dot(a: Vec3, b: Vec3) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z


def _vec3_to_array(v: Vec3):
    return np.array([v.x, v.y, v.z])


# ===========================================================================
# sun_direction — edge cases
# ===========================================================================

class TestSunDirectionEdges:

    # --- Day-boundary wrap --------------------------------------------------

    def test_day_boundary_t0_equals_t24h(self):
        """time=0 and time=GAME_SECONDS_PER_DAY must return identical dirs (modulo wrap)."""
        a = sun_direction(0.0)
        b = sun_direction(GAME_SECONDS_PER_DAY)
        # Pin current behaviour: wraps via modulo, so they must be equal.
        assert np.allclose(_vec3_to_array(a), _vec3_to_array(b), atol=1e-6), (
            f"sun_direction(0) != sun_direction(24h): {a} vs {b}"
        )

    def test_day_boundary_near_midnight_continuity(self):
        """A tiny step backward from midnight into the previous day should not pop."""
        tiny = 1.0  # 1 game second
        a = sun_direction(GAME_SECONDS_PER_DAY - tiny)
        b = sun_direction(0.0)
        dot = _dot(a, b)
        angle = math.acos(min(1.0, max(-1.0, dot)))
        # 1 s corresponds to 2π/86400 ≈ 0.0000727 rad of arc — allow 10× margin.
        assert angle < math.radians(0.1), (
            f"discontinuity at day boundary: {angle} rad ({math.degrees(angle):.4f}°)"
        )

    def test_day_boundary_5s_before_midnight(self):
        """5 s before midnight and 5 s after should be close (< 0.01 rad)."""
        a = sun_direction(GAME_SECONDS_PER_DAY - 5.0)
        b = sun_direction(5.0)
        dot = _dot(a, b)
        angle = math.acos(min(1.0, max(-1.0, dot)))
        expected_arc = 2.0 * math.pi * 10.0 / GAME_SECONDS_PER_DAY
        # Must be within 3× the expected tiny arc.
        assert angle < 3.0 * expected_arc, (
            f"wrap gap too large: {angle} rad; expected ~{expected_arc} rad"
        )

    def test_negative_time_treated_like_modulo(self):
        """
        Negative time is outside the documented [0, 86400) domain.
        Pin whatever the implementation does so regressions are visible.
        Current implementation: t = time_of_day_s % GAME_SECONDS_PER_DAY uses
        Python %-semantics which returns a non-negative value, so -epsilon wraps
        to GAME_SECONDS_PER_DAY - epsilon ≈ midnight.
        """
        a = sun_direction(-1.0)
        b = sun_direction(GAME_SECONDS_PER_DAY - 1.0)
        # SUSPECTED: these should be equal given Python % semantics — pin it.
        assert np.allclose(_vec3_to_array(a), _vec3_to_array(b), atol=1e-6), (
            "sun_direction(-1) does not equal sun_direction(DAY-1); "
            "Python % wrapping may not be in effect"
        )

    # --- Noon continuity ----------------------------------------------------

    def test_noon_no_snap_1s_window(self):
        """Adjacent 1-second samples around noon must not pop."""
        noon = 12.0 * HOUR
        for offset in (-2, -1, 0, 1, 2):
            a = sun_direction(noon + offset)
            b = sun_direction(noon + offset + 1.0)
            dot = _dot(a, b)
            angle = math.acos(min(1.0, max(-1.0, dot)))
            assert angle < math.radians(0.05), (
                f"snap near noon at offset {offset}s: {angle} rad"
            )

    def test_noon_is_maximum_elevation(self):
        """sun_direction(12h).z should be the maximum over the full day arc."""
        noon_z = sun_direction(12.0 * HOUR).z
        for h in range(25):
            z = sun_direction(h * HOUR).z
            assert z <= noon_z + 1e-6, f"elevation at {h}h ({z}) exceeds noon ({noon_z})"

    # --- Unit length at many times ------------------------------------------

    def test_unit_length_fine_grid(self):
        """sun_direction must be unit-length at every 10-minute mark over 2 days."""
        for i in range(0, int(2 * GAME_SECONDS_PER_DAY), 600):
            v = sun_direction(float(i))
            n = _norm(v)
            assert abs(n - 1.0) < 1e-5, f"non-unit at t={i}s: |v|={n}"

    def test_sun_z_negative_at_midnight(self):
        """Midnight sun should be well below horizon (z <= -0.9)."""
        assert sun_direction(0.0).z <= -0.9

    def test_sun_z_at_sunrise_near_zero(self):
        """Sunrise (06:00) should have sun near the horizon."""
        z = sun_direction(6.0 * HOUR).z
        assert abs(z) < 0.15, f"sunrise z={z} too far from horizon"

    def test_sun_x_positive_at_sunrise(self):
        """Sun rises in +X (east)."""
        assert sun_direction(6.0 * HOUR).x > 0.9

    def test_sun_x_negative_at_sunset(self):
        """Sun sets in -X (west)."""
        assert sun_direction(18.0 * HOUR).x < -0.9

    # --- Determinism --------------------------------------------------------

    def test_sun_direction_deterministic(self):
        """Pure function must return identical results on repeated calls."""
        for t in [0.0, 6.0 * HOUR, 12.0 * HOUR, 18.0 * HOUR, 23.9 * HOUR]:
            a = sun_direction(t)
            b = sun_direction(t)
            assert np.allclose(_vec3_to_array(a), _vec3_to_array(b)), (
                f"sun_direction not deterministic at t={t}"
            )


# ===========================================================================
# moon_direction — edge cases
# ===========================================================================

class TestMoonDirectionEdges:

    def test_unit_length_fine_grid(self):
        """moon_direction must be unit-length at every 10-minute mark."""
        for i in range(0, int(GAME_SECONDS_PER_DAY), 600):
            v = moon_direction(float(i))
            n = _norm(v)
            assert abs(n - 1.0) < 1e-5, f"non-unit at t={i}s: |v|={n}"

    def test_day_wrap_continuity(self):
        """moon_direction(0) == moon_direction(GAME_SECONDS_PER_DAY)."""
        a = moon_direction(0.0)
        b = moon_direction(GAME_SECONDS_PER_DAY)
        assert np.allclose(_vec3_to_array(a), _vec3_to_array(b), atol=1e-6)

    def test_moon_opposite_sun_plus_offset(self):
        """
        Moon is roughly opposite the sun with a phase lead.
        At midnight (sun at nadir) the moon should be near its zenith.
        MOON_PHASE_OFFSET_RAD (~0.26 rad) shifts the peak slightly; z still > 0.8.
        """
        m = moon_direction(0.0)
        assert m.z > 0.8, f"midnight moon not high enough: z={m.z}"

    def test_moon_and_sun_both_visible_at_dusk(self):
        """
        At dusk (~17.5h) the moon's phase offset means both bodies are above the
        horizon.  This pins the documented twilight-overlap property.
        """
        t = 17.5 * HOUR
        assert sun_direction(t).z > 0.0, "sun not above horizon at 17.5h"
        assert moon_direction(t).z > 0.0, "moon not above horizon at 17.5h (phase offset lost?)"

    def test_moon_phase_offset_creates_lead(self):
        """
        Pin the actual moon z at 06:00 (sunrise).

        MOON_PHASE_OFFSET_RAD ≈ 0.26 rad ≈ ~1 game hour of phase lead.
        The moon's arc peak (equivalent to "moon noon") is at roughly
        midnight (00:00), so by 06:00 the moon has descended well past its
        setting point.  Actual measurement: moon_direction(6h).z ≈ -0.25
        (already below the horizon).

        The "both visible at twilight" property mentioned in the docstring refers
        to DUSK (~17:30), not dawn — see test_moon_and_sun_both_visible_at_dusk.

        SUSPICION: the MOON_PHASE_OFFSET_RAD only provides dusk overlap (sun
        setting west, moon rising east at 17:30). At 06:00 the moon has already
        set. The offset direction is FORWARD in phase (moon peaks before the
        sun's nadir), so the moon rises and sets EARLIER than the sun's
        exact-opposite, meaning it has set before sunrise.  Pin it.
        """
        m_at_sunrise = moon_direction(6.0 * HOUR)
        # Moon is below the horizon at 06:00 — pin the approximate z value.
        assert m_at_sunrise.z < 0.0, (
            f"moon z at sunrise = {m_at_sunrise.z}; expected below horizon"
        )
        # Pin the golden-master z more tightly (actual ≈ -0.2515).
        assert abs(m_at_sunrise.z - (-0.2515)) < 0.01, (
            f"moon z at 06:00 drifted from golden-master -0.2515: got {m_at_sunrise.z}"
        )

    def test_moon_deterministic(self):
        """moon_direction is a pure function — same twice."""
        for t in [0.0, 6.0 * HOUR, 12.0 * HOUR, 18.0 * HOUR]:
            a = moon_direction(t)
            b = moon_direction(t)
            assert np.allclose(_vec3_to_array(a), _vec3_to_array(b))


# ===========================================================================
# daylight_factor — edge cases
# ===========================================================================

class TestDaylightFactorEdges:

    def test_midnight_exactly_zero(self):
        """daylight_factor(midnight) must be exactly 0.0 (sun well below horizon)."""
        assert daylight_factor(0.0) == 0.0

    def test_noon_exactly_one(self):
        """daylight_factor(noon) must be exactly 1.0 (sun well above DAYLIGHT_Z_HI)."""
        assert daylight_factor(12.0 * HOUR) == 1.0

    def test_clamped_to_0_1_everywhere(self):
        """daylight_factor must never leave [0, 1] across the full day."""
        for i in range(0, int(GAME_SECONDS_PER_DAY) + 1, 300):
            f = daylight_factor(float(i))
            assert 0.0 <= f <= 1.0, f"out of [0,1] at t={i}s: f={f}"

    def test_monotone_increasing_from_sunrise_to_noon(self):
        """daylight_factor is non-decreasing from 06:00 to 12:00."""
        prev = daylight_factor(6.0 * HOUR)
        for i in range(1, 7 * 2):  # 30-min steps
            t = (6.0 + i * 0.5) * HOUR
            curr = daylight_factor(t)
            assert curr >= prev - 1e-9, (
                f"daylight_factor not monotone at t={t/HOUR:.1f}h: {curr} < {prev}"
            )
            prev = curr

    def test_monotone_decreasing_from_noon_to_midnight(self):
        """daylight_factor is non-increasing from 12:00 to midnight-ish."""
        prev = daylight_factor(12.0 * HOUR)
        for i in range(1, 13 * 2):  # 30-min steps
            t = (12.0 + i * 0.5) * HOUR
            if t >= GAME_SECONDS_PER_DAY:
                break
            curr = daylight_factor(t)
            assert curr <= prev + 1e-9, (
                f"daylight_factor not monotone decreasing at t={t/HOUR:.1f}h: {curr} > {prev}"
            )
            prev = curr

    def test_about_half_at_sunrise_sunset(self):
        """At sunrise (06:00) and sunset (18:00) the factor should be near 0.5."""
        f_rise = daylight_factor(6.0 * HOUR)
        f_set = daylight_factor(18.0 * HOUR)
        assert abs(f_rise - 0.5) < 0.1, f"sunrise daylight_factor={f_rise}, expected ~0.5"
        assert abs(f_set - 0.5) < 0.1, f"sunset daylight_factor={f_set}, expected ~0.5"

    def test_fully_one_about_one_hour_after_sunrise(self):
        """Should reach 1.0 around 7:00–7:30."""
        assert daylight_factor(7.5 * HOUR) == 1.0

    def test_fully_zero_about_one_hour_after_sunset(self):
        """Should reach 0.0 around 19:00–19:30."""
        assert daylight_factor(19.5 * HOUR) == 0.0

    def test_sun_elevation_drives_daylight(self):
        """
        daylight_factor is smoothstep on sun_dir.z between DAYLIGHT_Z_LO and HI.
        Verify the relationship holds directly for a few samples.
        """
        for t in [5.0 * HOUR, 8.0 * HOUR, 12.0 * HOUR, 16.0 * HOUR, 20.0 * HOUR]:
            z = sun_direction(t).z
            expected = smoothstep(z, DAYLIGHT_Z_LO, DAYLIGHT_Z_HI)
            got = daylight_factor(t)
            assert abs(got - expected) < 1e-9, (
                f"daylight_factor({t/HOUR}h) = {got}, expected {expected} from z={z}"
            )

    def test_deterministic(self):
        for t in [0.0, 6.0 * HOUR, 12.0 * HOUR, 18.0 * HOUR]:
            assert daylight_factor(t) == daylight_factor(t)


# ===========================================================================
# smoothstep — edges
# ===========================================================================

class TestSmootstepEdges:

    def test_at_lo_returns_zero(self):
        assert smoothstep(0.0, 0.0, 1.0) == 0.0

    def test_at_hi_returns_one(self):
        assert smoothstep(1.0, 0.0, 1.0) == 1.0

    def test_below_lo_returns_zero(self):
        assert smoothstep(-5.0, 0.0, 1.0) == 0.0
        assert smoothstep(-1e-9, 0.0, 1.0) == 0.0

    def test_above_hi_returns_one(self):
        assert smoothstep(2.0, 0.0, 1.0) == 1.0
        assert smoothstep(1.0 + 1e-9, 0.0, 1.0) == 1.0

    def test_midpoint_is_0_5(self):
        """At the exact midpoint t=0.5, smoothstep = 3(0.5)^2 - 2(0.5)^3 = 0.5."""
        result = smoothstep(0.5, 0.0, 1.0)
        assert abs(result - 0.5) < 1e-9

    def test_quarter_point(self):
        """t=0.25 → 3(0.0625) - 2(0.015625) = 0.1875 - 0.03125 = 0.15625."""
        result = smoothstep(0.25, 0.0, 1.0)
        assert abs(result - 0.15625) < 1e-9

    def test_three_quarter_point(self):
        """t=0.75 → 3(0.5625) - 2(0.421875) = 1.6875 - 0.84375 = 0.84375."""
        result = smoothstep(0.75, 0.0, 1.0)
        assert abs(result - 0.84375) < 1e-9

    def test_arbitrary_range_lo_hi(self):
        """smoothstep(lo=2, hi=5): at x=3.5 (midpoint) should be 0.5."""
        result = smoothstep(3.5, 2.0, 5.0)
        assert abs(result - 0.5) < 1e-9

    def test_monotonic_across_range(self):
        """smoothstep must be non-decreasing across [lo, hi]."""
        N = 200
        xs = np.linspace(-0.5, 1.5, N)
        vals = [smoothstep(float(x), 0.0, 1.0) for x in xs]
        for i in range(len(vals) - 1):
            assert vals[i + 1] >= vals[i] - 1e-12, (
                f"not monotonic at i={i}: {vals[i]} -> {vals[i+1]}"
            )

    def test_negative_range_lo_hi(self):
        """smoothstep with negative lo/hi."""
        # lo=-1, hi=1: at x=0 (midpoint) should be 0.5
        assert abs(smoothstep(0.0, -1.0, 1.0) - 0.5) < 1e-9

    def test_daylight_z_bounds(self):
        """Verify DAYLIGHT_Z constants are the smoothstep edges for daylight_factor."""
        assert smoothstep(DAYLIGHT_Z_LO, DAYLIGHT_Z_LO, DAYLIGHT_Z_HI) == 0.0
        assert smoothstep(DAYLIGHT_Z_HI, DAYLIGHT_Z_LO, DAYLIGHT_Z_HI) == 1.0


# ===========================================================================
# color_ramp — edges
# ===========================================================================

class TestColorRampEdges:

    SIMPLE = (
        (0.0, (0.0, 0.0, 0.0)),
        (0.5, (0.5, 0.5, 0.5)),
        (1.0, (1.0, 1.0, 1.0)),
    )

    def test_at_first_key_returns_first_color(self):
        r = color_ramp(0.0, self.SIMPLE)
        assert np.allclose(r, (0.0, 0.0, 0.0))

    def test_at_last_key_returns_last_color(self):
        r = color_ramp(1.0, self.SIMPLE)
        assert np.allclose(r, (1.0, 1.0, 1.0))

    def test_below_first_key_clamps_to_first_color(self):
        r = color_ramp(-99.0, self.SIMPLE)
        assert np.allclose(r, (0.0, 0.0, 0.0))

    def test_above_last_key_clamps_to_last_color(self):
        r = color_ramp(99.0, self.SIMPLE)
        assert np.allclose(r, (1.0, 1.0, 1.0))

    def test_midpoint_interpolates(self):
        """x=0.5 is a keyframe — should return that key's exact color."""
        r = color_ramp(0.5, self.SIMPLE)
        assert np.allclose(r, (0.5, 0.5, 0.5))

    def test_between_keys_interpolates(self):
        """x=0.25 → midpoint between key[0] and key[1] → (0.25, 0.25, 0.25)."""
        r = color_ramp(0.25, self.SIMPLE)
        assert np.allclose(r, (0.25, 0.25, 0.25), atol=1e-6)

    def test_nonuniform_channels(self):
        """Color channels interpolate independently."""
        ramp = (
            (0.0, (0.0, 1.0, 0.5)),
            (1.0, (1.0, 0.0, 0.5)),
        )
        r = color_ramp(0.5, ramp)
        assert np.allclose(r, (0.5, 0.5, 0.5), atol=1e-6)

    def test_two_key_ramp_endpoint_equality(self):
        ramp = ((0.0, (0.1, 0.2, 0.3)), (1.0, (0.9, 0.8, 0.7)))
        assert np.allclose(color_ramp(0.0, ramp), (0.1, 0.2, 0.3))
        assert np.allclose(color_ramp(1.0, ramp), (0.9, 0.8, 0.7))

    def test_deterministic(self):
        """Pure function — identical calls return identical results."""
        r1 = color_ramp(0.3, self.SIMPLE)
        r2 = color_ramp(0.3, self.SIMPLE)
        assert r1 == r2


# ===========================================================================
# lerp_color — edges
# ===========================================================================

class TestLerpColorEdges:

    BLACK = (0.0, 0.0, 0.0)
    WHITE = (1.0, 1.0, 1.0)
    RED   = (1.0, 0.0, 0.0)
    BLUE  = (0.0, 0.0, 1.0)

    def test_t0_returns_a(self):
        assert lerp_color(self.RED, self.BLUE, 0.0) == self.RED

    def test_t1_returns_b(self):
        assert lerp_color(self.RED, self.BLUE, 1.0) == self.BLUE

    def test_t_half_midpoint(self):
        r = lerp_color(self.BLACK, self.WHITE, 0.5)
        assert np.allclose(r, (0.5, 0.5, 0.5), atol=1e-9)

    def test_channels_interpolate_independently(self):
        a = (0.0, 0.5, 1.0)
        b = (1.0, 0.5, 0.0)
        r = lerp_color(a, b, 0.5)
        assert np.allclose(r, (0.5, 0.5, 0.5), atol=1e-9)

    def test_t_negative_clamped_to_a(self):
        """
        t < 0 is clamped to 0 (docs say clamped to [0,1]).
        Pin current behaviour: returns a unchanged.
        """
        r = lerp_color(self.RED, self.BLUE, -1.0)
        assert np.allclose(r, self.RED), f"t<0 not clamped to a: {r}"

    def test_t_above_one_clamped_to_b(self):
        """
        t > 1 is clamped to 1 (docs say clamped to [0,1]).
        Pin current behaviour: returns b unchanged.
        """
        r = lerp_color(self.RED, self.BLUE, 2.0)
        assert np.allclose(r, self.BLUE), f"t>1 not clamped to b: {r}"

    def test_hdr_values_not_clamped(self):
        """
        Docstring says "no clamping" on color components — HDR > 1 should
        survive lerp unchanged.
        """
        a = (3.2, 3.0, 2.6)  # typical sun_radiance at noon
        b = (0.0, 0.0, 0.0)
        r = lerp_color(a, b, 0.0)
        assert np.allclose(r, a), f"HDR component clamped: {r}"

    def test_lerp_color_deterministic(self):
        a = (0.1, 0.2, 0.3)
        b = (0.4, 0.5, 0.6)
        r1 = lerp_color(a, b, 0.33)
        r2 = lerp_color(a, b, 0.33)
        assert r1 == r2
