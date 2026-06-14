"""
tests/test_weather_fog.py — Emergent (condensation-driven) ground fog.

No panda3d imports anywhere in this file.

Fog here is NOT a selectable weather state — it condenses from conditions
(humidity past the temperature-dependent saturation, in calm air).  These tests
pin the *behaviour* the model promises:

- A calm, humid night after an evening shower grows ground fog through the cool
  pre-dawn hours, then burns off as the sun warms the air (saturation humidity
  climbs back above the actual humidity).
- Wind mixes fog away: the same humid setup with a stiff wind produces none.
- A dry high-pressure day with no rain history stays at the clear-air baseline.
- Emergent fog is a pure function of (seed, t, pos) and never exceeds the cap.
- The formula functions are monotonic / shaped as documented.

The natural synoptic flow rarely drops below ~3 m/s (so truly calm nights are
rare) and storm cells race across the map (so a fixed ground point rarely gets
sustained rain).  To exercise the model deterministically we build a *calm*
config (a near-still synoptic band — which also keeps an injected cell roughly
stationary) and drop a stationary shower over the sample point; the emergent-fog
code path under test is exactly the production one.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from fire_engine.core import EventBus, load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.world.weather import WeatherSystem
from fire_engine.world.weather.cells import CellKind, StormCell, day_regime
from fire_engine.world.weather.cells import Regime
from fire_engine.world.weather import humidity as H

HOUR = 3600.0
DAY = 24 * HOUR
_FOG_BASELINE = 0.0008  # weather/system.py clear-air baseline (1/m)


def _calm_config(**overrides):
    """Load config, then near-still the synoptic flow (calm wind, still cells)."""
    cfg = load_config()
    base = dict(
        weather_synoptic_speed_min_ms=0.05,
        weather_synoptic_speed_max_ms=0.30,
    )
    base.update(overrides)
    return dataclasses.replace(cfg, **base)


def _windy_config():
    """A config whose synoptic flow is pinned to a stiff, fog-killing wind."""
    cfg = load_config()
    return dataclasses.replace(
        cfg,
        weather_synoptic_speed_min_ms=8.0,
        weather_synoptic_speed_max_ms=10.0,
    )


def _ws(cfg, seed: int) -> WeatherSystem:
    set_world_seed(seed)
    return WeatherSystem(cfg, EventBus())


def _evening_shower(ws: WeatherSystem, day: int, *, peak: float = 1.0) -> None:
    """Inject a stationary heavy shower over the origin, 20:00–23:00 of *day*."""
    ws._summoned.append(
        StormCell(
            id="s:test_shower",
            kind=CellKind.SHOWER,
            spawn_time=day * DAY + 20 * HOUR,
            spawn_pos=(0.0, 0.0),
            duration_s=3 * HOUR,
            radius_m=1800.0,
            peak_intensity=peak,
            drift_bias=(0.0, 0.0),
        )
    )


# ---------------------------------------------------------------------------
# Behaviour: pre-dawn fog after evening rain, burning off after sunrise
# ---------------------------------------------------------------------------


class TestPreDawnFog:
    # Seed 7 has two consecutive high humidity-base days (3 and 4 ≈ 0.59), so a
    # late-evening shower on day 3 leaves the air muggy into the cool small hours.
    SEED = 7
    RAIN_DAY = 3

    def _run(self):
        cfg = _calm_config()
        ws = _ws(cfg, self.SEED)
        _evening_shower(ws, self.RAIN_DAY)
        return ws

    def _fog_at(self, ws, tod_h: float, day: int) -> float:
        t = day * DAY + tod_h * HOUR
        return ws.sample_local((0.0, 0.0), t).fog_density

    def test_fog_grows_in_the_cool_small_hours(self):
        ws = self._run()
        # Late evening / early night, right after the rain, in the cooling air.
        f_2200 = self._fog_at(ws, 22.0, self.RAIN_DAY)
        f_0000 = self._fog_at(ws, 0.0, self.RAIN_DAY + 1)
        f_0200 = self._fog_at(ws, 2.0, self.RAIN_DAY + 1)
        # Fog has condensed well above the clear-air baseline.
        assert f_2200 > _FOG_BASELINE * 3.0
        assert f_0000 > _FOG_BASELINE * 3.0
        assert f_0200 > _FOG_BASELINE * 1.5

    def test_fog_burns_off_by_mid_morning(self):
        ws = self._run()
        f_night = self._fog_at(ws, 22.0, self.RAIN_DAY)
        f_0900 = self._fog_at(ws, 9.0, self.RAIN_DAY + 1)
        f_1400 = self._fog_at(ws, 14.0, self.RAIN_DAY + 1)
        # The warming day burns the fog back down to the clear-air baseline.
        assert f_0900 < f_night
        assert f_0900 == pytest.approx(_FOG_BASELINE, abs=1e-5)
        assert f_1400 == pytest.approx(_FOG_BASELINE, abs=1e-5)

    def test_humidity_is_filled(self):
        ws = self._run()
        # During/after rain the relative humidity is the real emergent value,
        # not the old 0.5 placeholder.
        lw = ws.sample_local((0.0, 0.0), self.RAIN_DAY * DAY + 22 * HOUR)
        assert lw.humidity != 0.5
        assert 0.0 <= lw.humidity <= 1.0
        assert lw.humidity > 0.6  # muggy after the shower


# ---------------------------------------------------------------------------
# Wind kills the fog (gate closed)
# ---------------------------------------------------------------------------


class TestWindKillsFog:
    def test_same_humid_setup_no_fog_when_windy(self):
        # Calm version makes fog…
        calm = _ws(_calm_config(), 7)
        _evening_shower(calm, 3)
        t = 3 * DAY + 22 * HOUR
        assert calm.sample_local((0.0, 0.0), t).fog_density > _FOG_BASELINE * 3.0

        # …a stiff synoptic wind over the *same* humid setup gates it off.
        windy = _ws(_windy_config(), 7)
        _evening_shower(windy, 3)
        lw = windy.sample_local((0.0, 0.0), t)
        assert lw.wind_speed >= 3.0  # well above the fog-none threshold
        # The wind gate is shut at this speed, so however humid the air is, no
        # emergent fog condenses — only the clear-air baseline remains.
        assert H.wind_gate(np.array([lw.wind_speed]), windy._config)[0] == pytest.approx(0.0)
        assert lw.fog_density == pytest.approx(_FOG_BASELINE, abs=1e-5)

    def test_wind_gate_monotonic_and_bounds(self):
        cfg = load_config()
        full = cfg.weather_fog_wind_full_ms
        none = cfg.weather_fog_wind_none_ms
        speeds = np.linspace(0.0, none + 2.0, 50)
        gate = H.wind_gate(speeds, cfg)
        assert np.all(np.diff(gate) <= 1e-12)  # non-increasing
        assert H.wind_gate(np.array([full - 0.1]), cfg)[0] == pytest.approx(1.0)
        assert H.wind_gate(np.array([none + 0.1]), cfg)[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Dry day → no emergent fog
# ---------------------------------------------------------------------------


class TestDryDayNoFog:
    def test_high_pressure_dry_day_stays_at_baseline(self):
        cfg = _calm_config()
        # Find a HIGH_PRESSURE day with a low humidity base and no rain history,
        # then sample it through the cool small hours: no rain ⇒ no condensation.
        seed, day = self._dry_high_pressure_day(cfg)
        ws = _ws(cfg, seed)
        for tod in (0.0, 2.0, 4.0, 22.0):
            t = day * DAY + tod * HOUR
            lw = ws.sample_local((5_000.0, 5_000.0), t)  # away from any cell
            assert lw.rain_intensity == 0.0
            assert lw.fog_density == pytest.approx(_FOG_BASELINE, abs=1e-5)

    @staticmethod
    def _dry_high_pressure_day(cfg):
        for seed in (1337, 42, 11, 5, 99, 7):
            set_world_seed(seed)
            for day in range(2, 8):
                if day_regime(day) is not Regime.HIGH_PRESSURE:
                    continue
                if H.humidity_base(day, cfg) < 0.55:
                    return seed, day
        raise AssertionError("no dry high-pressure day found")


# ---------------------------------------------------------------------------
# Determinism + cap
# ---------------------------------------------------------------------------


class TestDeterminismAndCap:
    def test_emergent_fog_pure_function(self):
        a = _ws(_calm_config(), 7)
        b = _ws(_calm_config(), 7)
        _evening_shower(a, 3)
        _evening_shower(b, 3)
        for tod in (21.0, 23.0, 1.0, 3.0):
            day = 3 if tod >= 12.0 else 4
            t = day * DAY + tod * HOUR
            fa = a.sample_local((0.0, 0.0), t).fog_density
            fb = b.sample_local((0.0, 0.0), t).fog_density
            assert fa == fb

    def test_fog_never_exceeds_cap(self):
        cfg = _calm_config()
        cap = cfg.weather_fog_max_density
        ws = _ws(cfg, 7)
        # Pile on the moisture: a fat, intense, stationary storm, then a FOG_BANK
        # so the baseline + emergent terms both push hard against the cap.
        _evening_shower(ws, 3, peak=1.0)
        ws._summoned.append(
            StormCell(
                "s:fog",
                CellKind.FOG_BANK,
                3 * DAY + 20 * HOUR,
                (0.0, 0.0),
                6 * HOUR,
                2000.0,
                1.0,
                (0.0, 0.0),
            )
        )
        for tod in np.arange(20.0, 30.0, 0.25):
            t = 3 * DAY + tod * HOUR
            f = ws.sample_local((0.0, 0.0), t).fog_density
            assert 0.0 <= f <= cap + 1e-9

    def test_emergent_fog_function_capped_inputs(self):
        cfg = load_config()
        # Saturated, freezing, dead-calm: maximal condensation × gate.
        f = H.emergent_fog(np.array([1.0]), np.array([-5.0]), np.array([0.0]), cfg)[0]
        assert f == pytest.approx(cfg.weather_fog_emergent_max)


# ---------------------------------------------------------------------------
# Formula shape (clarity tests on the pure functions)
# ---------------------------------------------------------------------------


class TestFormulas:
    def test_saturation_rises_with_temperature(self):
        cfg = load_config()
        T = np.array([0.0, 5.0, 10.0, 20.0])
        h_sat = H.saturation_humidity(T, cfg)
        assert np.all(np.diff(h_sat) >= 0.0)  # non-decreasing in T
        assert np.all(h_sat >= 0.5) and np.all(h_sat <= 1.0)
        # Cool pre-dawn saturates lower than the warm afternoon (fog-friendly).
        assert (
            H.saturation_humidity(np.array([4.0]), cfg)[0]
            < H.saturation_humidity(np.array([19.0]), cfg)[0]
        )

    def test_condense_fraction_monotonic(self):
        cfg = load_config()
        h_sat = np.full(40, 0.8)
        hum = np.linspace(0.6, 1.0, 40)
        cf = H.condense_fraction(hum, h_sat, cfg)
        assert np.all(np.diff(cf) >= -1e-12)  # non-decreasing
        assert cf[0] == pytest.approx(0.0)  # below saturation
        assert cf[-1] == pytest.approx(1.0)  # well past saturation

    def test_relative_humidity_rises_with_moisture(self):
        cfg = load_config()
        rr = np.array([0.0, 0.5, 1.0])
        wet = np.zeros(3)
        h = H.relative_humidity(rr, wet, 0.4, cfg)
        assert np.all(np.diff(h) > 0.0)  # more rain ⇒ more humid
        assert np.all(h >= 0.0) and np.all(h <= 1.0)
        # The wetness term also lifts humidity.
        h_wet = H.relative_humidity(np.zeros(1), np.array([1.0]), 0.4, cfg)[0]
        assert h_wet > H.relative_humidity(np.zeros(1), np.zeros(1), 0.4, cfg)[0]

    def test_humidity_base_deterministic_and_banded(self):
        cfg = load_config()
        set_world_seed(7)
        a = H.humidity_base(3, cfg)
        set_world_seed(7)
        b = H.humidity_base(3, cfg)
        assert a == b
        assert cfg.weather_humidity_base_min <= a <= cfg.weather_humidity_base_max
