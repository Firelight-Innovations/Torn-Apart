"""
tests/test_weather_humidity_edges.py — Characterisation / golden-master tests
for the pure humidity functions in fire_engine.weather.humidity.

These tests PIN CURRENT BEHAVIOUR.  They do NOT fix bugs — suspected anomalies
are noted in comments and reported in the module docstring.  No panda3d imports;
all assertions use numpy so that vectorisation is fully exercised.

Suspected bugs (do NOT fix here — reported to swarm coordinator):
- saturation_humidity clamps *lower* bound to 0.5, so passing extremely cold
  temperatures (e.g. T << sat_ref_c) will silently saturate at 0.5 rather than
  going negative.  The clamp is intentional per docstring, but means the model
  is flat below a certain temperature — worth flagging.
- relative_humidity accepts RH > 1.0 as input without raising; the clamp at the
  output means the excess is silently swallowed.  Pinned here.
- condense_fraction with humidity > h_sat + condense_band returns 1.0 (per
  smoothstep contract); pinned.  If caller passes humidity > 1.0 (un-clamped),
  condense_fraction still returns a value in [0,1]; pinned as current behaviour.
- wind_gate with wind_speed < 0 currently returns 1.0 (extrapolation below the
  full-fog threshold); negative wind speed is physically impossible but the
  function does not raise.  Pinned.
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.core import load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.world.weather import humidity as H

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# ---------------------------------------------------------------------------
# 1. saturation_humidity — monotonicity, finiteness, non-negativity
# ---------------------------------------------------------------------------


class TestSaturationHumidity:
    def test_monotonically_non_decreasing_over_range(self, cfg):
        """h_sat rises (weakly) with temperature — core invariant."""
        T = np.linspace(-30.0, 50.0, 200)
        h_sat = H.saturation_humidity(T, cfg)
        diffs = np.diff(h_sat)
        # Non-decreasing: every step is >= 0 (or within float rounding)
        assert np.all(diffs >= -1e-15), (
            f"saturation_humidity not monotone: min diff = {diffs.min()}"
        )

    def test_finite_and_non_negative_everywhere(self, cfg):
        T = np.array([-100.0, -30.0, -10.0, 0.0, 5.0, 20.0, 50.0, 100.0])
        h_sat = H.saturation_humidity(T, cfg)
        assert np.all(np.isfinite(h_sat))
        assert np.all(h_sat >= 0.0)

    def test_clamped_to_half_one(self, cfg):
        """Docstring guarantees output is in [0.5, 1.0]."""
        T = np.linspace(-200.0, 200.0, 500)
        h_sat = H.saturation_humidity(T, cfg)
        assert np.all(h_sat >= 0.5 - 1e-12)
        assert np.all(h_sat <= 1.0 + 1e-12)

    def test_cool_below_warm(self, cfg):
        """Cool pre-dawn saturates lower than warm afternoon (fog-friendly)."""
        h_cool = H.saturation_humidity(np.array([4.0]), cfg)[0]
        h_warm = H.saturation_humidity(np.array([19.0]), cfg)[0]
        assert h_cool < h_warm

    def test_scalar_via_array_matches_direct(self, cfg):
        """Scalar passed as float vs wrapped in len-1 array gives same result."""
        T_arr = np.array([12.0])
        h_arr = H.saturation_humidity(T_arr, cfg)
        h_scalar = H.saturation_humidity(12.0, cfg)
        assert np.allclose(h_arr, np.atleast_1d(h_scalar))

    def test_extreme_cold_clamp(self, cfg):
        """Very cold temperature should saturate at exactly 0.5."""
        h_sat = H.saturation_humidity(np.array([-200.0]), cfg)[0]
        assert h_sat == pytest.approx(0.5)

    def test_extreme_warm_clamp(self, cfg):
        """Very warm temperature should saturate at exactly 1.0."""
        h_sat = H.saturation_humidity(np.array([200.0]), cfg)[0]
        assert h_sat == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. wind_gate — monotonic non-increasing, bounded [0,1]
# ---------------------------------------------------------------------------


class TestWindGate:
    def test_monotonically_non_increasing(self, cfg):
        """Higher wind → less gate (or equal — it's flat in the plateau zones)."""
        full = cfg.weather_fog_wind_full_ms
        none = cfg.weather_fog_wind_none_ms
        speeds = np.linspace(0.0, none + 5.0, 200)
        gate = H.wind_gate(speeds, cfg)
        diffs = np.diff(gate)
        assert np.all(diffs <= 1e-12), (
            f"wind_gate not monotone non-increasing: max diff = {diffs.max()}"
        )

    def test_bounded_zero_to_one(self, cfg):
        speeds = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 100.0])
        gate = H.wind_gate(speeds, cfg)
        assert np.all(gate >= 0.0 - 1e-12)
        assert np.all(gate <= 1.0 + 1e-12)

    def test_full_at_below_full_threshold(self, cfg):
        """At exactly the full-fog threshold, gate should be 1.0."""
        full = cfg.weather_fog_wind_full_ms
        gate = H.wind_gate(np.array([full - 0.01]), cfg)[0]
        assert gate == pytest.approx(1.0)

    def test_zero_at_above_none_threshold(self, cfg):
        """At and above the no-fog threshold, gate should be 0.0."""
        none = cfg.weather_fog_wind_none_ms
        gate = H.wind_gate(np.array([none + 0.01]), cfg)[0]
        assert gate == pytest.approx(0.0)

    def test_wind_zero(self, cfg):
        """Zero wind → full gate (no mixing)."""
        gate = H.wind_gate(np.array([0.0]), cfg)[0]
        assert gate == pytest.approx(1.0)

    def test_very_large_wind(self, cfg):
        """Very large wind speed → zero gate (all fog mixed away)."""
        gate = H.wind_gate(np.array([1000.0]), cfg)[0]
        assert gate == pytest.approx(0.0)

    def test_negative_wind_pinned(self, cfg):
        """Negative wind speed is unphysical; pin current behaviour (returns 1.0)."""
        gate = H.wind_gate(np.array([-5.0]), cfg)[0]
        # Current behaviour: smoothstep extrapolates below lo → result is 1.0
        assert gate == pytest.approx(1.0)

    def test_vectorisation_shape(self, cfg):
        """Array input gives same-shape output."""
        speeds_n = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        gate_n = H.wind_gate(speeds_n, cfg)
        assert gate_n.shape == speeds_n.shape

        speeds_1 = np.array([2.0])
        gate_1 = H.wind_gate(speeds_1, cfg)
        assert gate_1.shape == (1,)

        # Scalar → 0-d array or broadcastable; just check value matches length-1
        gate_s = H.wind_gate(np.array([2.0]), cfg)
        assert np.allclose(gate_1, gate_s)


# ---------------------------------------------------------------------------
# 3. relative_humidity — vectorisation, monotonicity in rain/wetness, clamp
# ---------------------------------------------------------------------------


class TestRelativeHumidity:
    def test_scalar_input_gives_scalar_like(self, cfg):
        """Length-1 arrays behave consistently."""
        rr = np.array([0.5])
        wet = np.array([0.0])
        h = H.relative_humidity(rr, wet, 0.4, cfg)
        assert h.shape == (1,)

    def test_array_input_preserves_shape(self, cfg):
        N = 7
        rr = np.linspace(0.0, 1.0, N)
        wet = np.zeros(N)
        h = H.relative_humidity(rr, wet, 0.4, cfg)
        assert h.shape == (N,)

    def test_more_rain_higher_humidity(self, cfg):
        """More recent rain → strictly higher relative humidity."""
        rr = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        wet = np.zeros(5)
        h = H.relative_humidity(rr, wet, 0.3, cfg)
        # Strictly increasing (rain_gain > 0 guarantees this, unless already at 1.0)
        assert np.all(np.diff(h) >= 0.0)

    def test_more_wetness_higher_humidity(self, cfg):
        """More ground wetness → same or higher humidity."""
        rr = np.zeros(5)
        wet = np.linspace(0.0, 1.0, 5)
        h = H.relative_humidity(rr, wet, 0.3, cfg)
        assert np.all(np.diff(h) >= 0.0)

    def test_clamped_to_zero_one(self, cfg):
        """Output is always in [0, 1] regardless of inputs."""
        rr = np.array([0.0, 1.0, 2.0])  # 2.0 > 1 is over-range
        wet = np.array([0.0, 0.0, 1.0])
        h = H.relative_humidity(rr, wet, 0.9, cfg)
        assert np.all(h >= 0.0)
        assert np.all(h <= 1.0)

    def test_zero_rain_zero_wetness(self, cfg):
        """Zero rain + zero wetness → exactly h_base (if h_base is in [0,1])."""
        h_base = 0.45
        h = H.relative_humidity(np.array([0.0]), np.array([0.0]), h_base, cfg)
        assert h[0] == pytest.approx(h_base)

    def test_determinism(self, cfg):
        """Pure function — same inputs always give identical outputs."""
        rr = np.array([0.3, 0.7])
        wet = np.array([0.1, 0.5])
        a = H.relative_humidity(rr, wet, 0.4, cfg)
        b = H.relative_humidity(rr, wet, 0.4, cfg)
        assert np.array_equal(a, b)

    def test_rh_over_one_input_pinned(self, cfg):
        """RH input > 1.0 is clamped at output; no raise."""
        h = H.relative_humidity(np.array([10.0]), np.array([10.0]), 1.0, cfg)
        assert h[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. condense_fraction — monotonicity, boundaries, shape
# ---------------------------------------------------------------------------


class TestCondenseFraction:
    def test_zero_below_saturation(self, cfg):
        """When humidity < h_sat, condensation fraction is zero."""
        h_sat = np.array([0.8])
        hum = np.array([0.7])
        cf = H.condense_fraction(hum, h_sat, cfg)
        assert cf[0] == pytest.approx(0.0)

    def test_one_well_above_saturation(self, cfg):
        """When humidity >> h_sat + band, condensation fraction is 1.0."""
        h_sat = np.array([0.6])
        hum = np.array([0.9])  # over by 0.3, band = 0.10
        cf = H.condense_fraction(hum, h_sat, cfg)
        assert cf[0] == pytest.approx(1.0)

    def test_monotonically_non_decreasing_in_humidity(self, cfg):
        """More humidity past saturation → more condensation."""
        h_sat = np.full(40, 0.7)
        hum = np.linspace(0.5, 1.0, 40)
        cf = H.condense_fraction(hum, h_sat, cfg)
        assert np.all(np.diff(cf) >= -1e-12)

    def test_bounded_zero_to_one(self, cfg):
        h_sat = np.array([0.5, 0.7, 0.9])
        hum = np.array([0.0, 0.5, 1.5])  # extreme range including > 1
        cf = H.condense_fraction(hum, h_sat, cfg)
        assert np.all(cf >= 0.0)
        assert np.all(cf <= 1.0)

    def test_vectorisation_shape(self, cfg):
        N = 10
        h_sat = np.full(N, 0.75)
        hum = np.linspace(0.6, 1.0, N)
        cf = H.condense_fraction(hum, h_sat, cfg)
        assert cf.shape == (N,)

    def test_determinism(self, cfg):
        h_sat = np.array([0.7, 0.8])
        hum = np.array([0.75, 0.85])
        a = H.condense_fraction(hum, h_sat, cfg)
        b = H.condense_fraction(hum, h_sat, cfg)
        assert np.array_equal(a, b)

    def test_at_exact_saturation_is_zero(self, cfg):
        """Exactly at saturation (no overshoot) → 0.0."""
        h_sat = np.array([0.75])
        hum = np.array([0.75])
        cf = H.condense_fraction(hum, h_sat, cfg)
        assert cf[0] == pytest.approx(0.0)

    def test_humidity_over_one_pinned(self, cfg):
        """Humidity > 1 (un-clamped input) yields condensation in [0,1]; pin it."""
        h_sat = np.array([0.9])
        hum = np.array([1.5])  # unphysical but we pin the behaviour
        cf = H.condense_fraction(hum, h_sat, cfg)
        assert 0.0 <= cf[0] <= 1.0


# ---------------------------------------------------------------------------
# 5. emergent_fog — directional behaviour and shape
# ---------------------------------------------------------------------------


class TestEmergentFog:
    def test_zero_rh_gives_no_fog(self, cfg):
        """Dry air (RH=0) produces zero emergent fog regardless of temp/wind."""
        h = np.array([0.0])
        T = np.array([5.0])
        w = np.array([0.0])
        f = H.emergent_fog(h, T, w, cfg)
        assert f[0] == pytest.approx(0.0)

    def test_high_rh_low_wind_gives_positive_fog(self, cfg):
        """Humid, calm, cold air → positive emergent fog."""
        h = np.array([0.95])
        T = np.array([0.0])
        w = np.array([0.0])
        f = H.emergent_fog(h, T, w, cfg)
        assert f[0] > 0.0

    def test_high_wind_kills_fog(self, cfg):
        """High wind above none threshold gates fog to zero."""
        h = np.array([1.0])
        T = np.array([0.0])
        none = cfg.weather_fog_wind_none_ms
        w = np.array([none + 1.0])
        f = H.emergent_fog(h, T, w, cfg)
        assert f[0] == pytest.approx(0.0)

    def test_wind_reduces_fog_monotonically(self, cfg):
        """Increasing wind monotonically reduces emergent fog."""
        h = np.full(50, 0.95)
        T = np.full(50, 0.0)
        w = np.linspace(0.0, cfg.weather_fog_wind_none_ms + 2.0, 50)
        f = H.emergent_fog(h, T, w, cfg)
        assert np.all(np.diff(f) <= 1e-12)

    def test_fog_non_negative(self, cfg):
        """Emergent fog can never be negative."""
        h = np.linspace(0.0, 1.0, 20)
        T = np.linspace(-30.0, 50.0, 20)
        w = np.linspace(0.0, 10.0, 20)
        f = H.emergent_fog(h, T, w, cfg)
        assert np.all(f >= 0.0)

    def test_saturated_cold_calm_reaches_max(self, cfg):
        """At full condensation + zero wind, fog should equal emergent_max."""
        # humidity > h_sat + condense_band at freezing, zero wind
        h = np.array([1.0])
        T = np.array([-5.0])
        w = np.array([0.0])
        f = H.emergent_fog(h, T, w, cfg)
        assert f[0] == pytest.approx(cfg.weather_fog_emergent_max)

    def test_warm_air_higher_saturation_requires_more_rh(self, cfg):
        """Warm air saturates higher, so the same RH yields less fog than cold air."""
        h = np.array([0.75, 0.75])
        T = np.array([2.0, 25.0])  # cold vs warm
        w = np.array([0.0, 0.0])
        f = H.emergent_fog(h, T, w, cfg)
        # Cold produces fog (if 0.75 > h_sat at 2°C); warm may produce none
        assert f[0] >= f[1]

    def test_vectorisation_shape(self, cfg):
        """Array inputs of shape (N,) → output shape (N,)."""
        N = 13
        h = np.linspace(0.5, 1.0, N)
        T = np.full(N, 5.0)
        w = np.full(N, 0.5)
        f = H.emergent_fog(h, T, w, cfg)
        assert f.shape == (N,)

    def test_length_one_shape(self, cfg):
        """Length-1 array input → length-1 output."""
        f = H.emergent_fog(np.array([0.9]), np.array([5.0]), np.array([0.5]), cfg)
        assert f.shape == (1,)

    def test_determinism(self, cfg):
        """Pure function — identical result on repeated calls."""
        h = np.array([0.8, 0.9, 1.0])
        T = np.array([0.0, 5.0, 10.0])
        w = np.array([0.0, 0.5, 1.0])
        a = H.emergent_fog(h, T, w, cfg)
        b = H.emergent_fog(h, T, w, cfg)
        assert np.array_equal(a, b)

    def test_boundary_T_extremes(self, cfg):
        """T extremes (-30, +50) remain finite and non-negative."""
        T = np.array([-30.0, 50.0])
        h = np.full(2, 0.95)
        w = np.full(2, 0.0)
        f = H.emergent_fog(h, T, w, cfg)
        assert np.all(np.isfinite(f))
        assert np.all(f >= 0.0)


# ---------------------------------------------------------------------------
# 6. humidity_base — config band, determinism
# ---------------------------------------------------------------------------


class TestHumidityBase:
    def test_within_config_band(self, cfg):
        """Output is in [base_min, base_max] for several days and seeds."""
        for seed in (1, 7, 42, 1337):
            set_world_seed(seed)
            for day in range(0, 10):
                h = H.humidity_base(day, cfg)
                assert cfg.weather_humidity_base_min <= h <= cfg.weather_humidity_base_max

    def test_deterministic_same_seed(self, cfg):
        """Same seed + day → identical base humidity."""
        set_world_seed(7)
        a = H.humidity_base(3, cfg)
        set_world_seed(7)
        b = H.humidity_base(3, cfg)
        assert a == b

    def test_returns_float(self, cfg):
        """humidity_base must return a Python float (not ndarray)."""
        set_world_seed(1)
        h = H.humidity_base(0, cfg)
        assert isinstance(h, float)
