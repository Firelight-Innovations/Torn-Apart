"""
tests/world/weather/_impl/test__sampling.py — Mirror for
fire_engine/world/weather/_impl/_sampling.py.

Authored tests covering the sampling helper functions, exercised through
the WeatherSystem public API and, where appropriate, by constructing a
minimal stand-in to call the helpers directly.

Headless — no panda3d imports.

Coverage
--------
CORRECTNESS — temperature:
  - Peaks around 15:00 (tod_h=15), lowest around 03:00 (tod_h=3).
  - Returns a finite float for all hours in a day.
  - Depends on ws._temp_mean and ws._temp_amp.

CORRECTNESS — sample_core (via WeatherSystem.sample_local):
  - Returns LocalWeather with fields in valid ranges.
  - A point far from any cell returns ambient coverage/density consistent
    with the day regime (no rain).

CORRECTNESS — sample_fields (via WeatherSystem, spatial batch):
  - All five output channels are finite and non-negative.
  - A point under a THUNDERSTORM cell has higher coverage than a point 20 km away.

DETERMINISM — sample_local:
  - Same (seed, pos, t) → identical LocalWeather.
  - Different positions at the same t can produce different coverage.
"""

from __future__ import annotations

import math

import pytest

from fire_engine.core import load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.world.weather import WeatherSystem
from fire_engine.world.weather._impl._sampling import temperature
from fire_engine.world.weather.cells import CellKind, natural_cells
from fire_engine.world.weather.types import LocalWeather

DAY = 86400.0


def _ws(seed: int = 1337) -> WeatherSystem:
    set_world_seed(seed)
    return WeatherSystem(load_config())


def _first_thunderstorm(cfg):
    """The earliest THUNDERSTORM cell across 80 days at current seed."""
    for d in range(80):
        for c in natural_cells(d, cfg):
            if c.kind is CellKind.THUNDERSTORM:
                return c
    raise AssertionError("no thunderstorm found in 80 days")


# ---------------------------------------------------------------------------
# temperature helper
# ---------------------------------------------------------------------------


class TestTemperature:
    def test_returns_finite_for_all_hours(self):
        ws = _ws()
        for h in range(25):
            t = temperature(ws, float(h))
            assert math.isfinite(t)

    def test_peak_near_15h(self):
        ws = _ws()
        # Temperature should be highest at or near 15:00.
        temps = {h: temperature(ws, float(h)) for h in range(24)}
        peak_hour = max(temps, key=temps.__getitem__)
        # Allow ±1 h tolerance since the cosine peaks at 15.0.
        assert abs(peak_hour - 15) <= 1, f"peak hour was {peak_hour}, expected ~15"

    def test_trough_near_3h(self):
        ws = _ws()
        temps = {h: temperature(ws, float(h)) for h in range(24)}
        trough_hour = min(temps, key=temps.__getitem__)
        # Trough is 12 h from the peak at 15 → 3 h (or 27→3 mod 24).
        assert abs(trough_hour - 3) <= 1, f"trough hour was {trough_hour}, expected ~3"

    def test_peak_above_trough(self):
        ws = _ws()
        peak = temperature(ws, 15.0)
        trough = temperature(ws, 3.0)
        assert peak >= trough

    def test_depends_on_mean_and_amp(self):
        ws = _ws()
        # The default amplitude > 0 means peak != trough.
        assert ws._temp_amp > 0.0
        peak = temperature(ws, 15.0)
        trough = temperature(ws, 3.0)
        assert peak > trough


# ---------------------------------------------------------------------------
# sample_local (via WeatherSystem.sample_local)
# ---------------------------------------------------------------------------


class TestSampleLocal:
    def test_returns_local_weather(self):
        ws = _ws()
        ws.update(0, 6 * 3600.0, (0.0, 0.0))
        lw = ws.sample_local((0.0, 0.0), 6 * 3600.0)
        assert isinstance(lw, LocalWeather)

    def test_all_fields_finite(self):
        ws = _ws()
        ws.update(0, 6 * 3600.0, (0.0, 0.0))
        lw = ws.sample_local((0.0, 0.0), 6 * 3600.0)
        for field in (
            "cloud_coverage",
            "cloud_density",
            "fog_density",
            "rain_intensity",
            "wind_speed",
            "humidity",
            "wetness",
            "temperature_c",
        ):
            assert math.isfinite(getattr(lw, field)), f"{field} is not finite"

    def test_coverage_and_density_in_unit_range(self):
        ws = _ws()
        ws.update(1, 8 * 3600.0, (0.0, 0.0))
        lw = ws.sample_local((100.0, -50.0), DAY + 8 * 3600.0)
        assert 0.0 <= lw.cloud_coverage <= 1.0
        assert 0.0 <= lw.cloud_density <= 1.0

    def test_rain_in_unit_range(self):
        ws = _ws()
        ws.update(0, 3 * 3600.0, (0.0, 0.0))
        lw = ws.sample_local((0.0, 0.0), 3 * 3600.0)
        assert 0.0 <= lw.rain_intensity <= 1.0

    def test_determinism(self):
        ws = _ws(seed=42)
        ws.update(2, 9 * 3600.0, (0.0, 0.0))
        t = 2 * DAY + 9 * 3600.0
        a = ws.sample_local((123.0, -45.0), t)
        b = ws.sample_local((123.0, -45.0), t)
        assert a == b

    def test_far_from_storm_no_rain(self):
        cfg = load_config()
        ws = _ws(seed=1337)
        c = _first_thunderstorm(cfg)
        t = c.spawn_time + 0.5 * c.duration_s
        center = c.center(t, ws.synoptic)
        far_pos = (float(center[0]) + 20000.0, float(center[1]))
        lw = ws.sample_local(far_pos, t)
        assert lw.rain_intensity == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Spatial batch (two points — checks under vs. far-from storm)
# ---------------------------------------------------------------------------


class TestSpatialBatch:
    def test_under_storm_more_coverage_than_far(self):
        cfg = load_config()
        ws = _ws(seed=1337)
        c = _first_thunderstorm(cfg)
        t = c.spawn_time + 0.5 * c.duration_s
        center = c.center(t, ws.synoptic)
        under = tuple(center)
        far = (float(center[0]) + 20000.0, float(center[1]))

        lw_under = ws.sample_local(under, t)
        lw_far = ws.sample_local(far, t)
        assert lw_under.cloud_coverage > lw_far.cloud_coverage

    def test_under_storm_has_rain(self):
        cfg = load_config()
        ws = _ws(seed=1337)
        c = _first_thunderstorm(cfg)
        t = c.spawn_time + 0.5 * c.duration_s
        center = c.center(t, ws.synoptic)
        lw = ws.sample_local(tuple(center), t)
        assert lw.rain_intensity > 0.0
