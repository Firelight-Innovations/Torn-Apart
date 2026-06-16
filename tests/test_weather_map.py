"""
tests/test_weather_map.py — Weather-map raster + ground-wetness quadrature.

No panda3d imports anywhere in this file.

Coverage
--------
- Shape/dtype pinned: (cells, cells, 4) float32.
- raster[row, col] == sample_local at that texel's world center (the map is a
  faithful cache of the field — same routine, exact match).
- Rasterising is time-invariant of call timing: same (center, t_abs) → identical
  array regardless of intervening calls or system state.
- Wetness: 0 with no rain history; rises under sustained rain; closed-form
  (pure function of seed/time/position), in [0, 1].
"""

from __future__ import annotations

import numpy as np

from fire_engine.core import EventBus, load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.world.weather import WeatherMap, WeatherSystem
from fire_engine.world.weather.cells import CellKind, natural_cells

DAY = 24 * 3600.0


def _ws(seed: int = 1337) -> WeatherSystem:
    set_world_seed(seed)
    return WeatherSystem(load_config(), EventBus())


def _first_thunderstorm(cfg):
    for d in range(80):
        for c in natural_cells(d, cfg):
            if c.kind is CellKind.THUNDERSTORM:
                return c
    raise AssertionError("no thunderstorm found")


class TestRasterShape:
    def test_shape_and_dtype(self):
        cfg = load_config()
        ws = _ws()
        wm = WeatherMap(cfg)
        r = wm.rasterize(ws, (0.0, 0.0), 12 * 3600.0)
        assert r.shape == (cfg.weather_map_cells, cfg.weather_map_cells, 4)
        assert r.dtype == np.float32

    def test_channels_in_range(self):
        ws = _ws()
        wm = WeatherMap(load_config())
        # Center on a live storm so coverage/precip channels are exercised.
        cfg = load_config()
        c = _first_thunderstorm(cfg)
        t = c.spawn_time + 0.4 * c.duration_s
        center = tuple(c.center(t, ws.synoptic))
        r = wm.rasterize(ws, center, t)
        assert np.all(r[..., :3] >= 0.0) and np.all(r[..., :3] <= 1.0)
        assert np.all(r[..., 3] >= 0.0)
        assert np.all(r[..., 3] <= cfg.weather_fog_max_density + 1e-6)


class TestRasterMatchesSample:
    def test_texel_equals_sample_local(self):
        cfg = load_config()
        ws = _ws()
        wm = WeatherMap(cfg)
        c = _first_thunderstorm(cfg)
        t = c.spawn_time + 0.4 * c.duration_s
        center = tuple(c.center(t, ws.synoptic))
        r = wm.rasterize(ws, center, t)

        n = wm.cells
        half = 0.5 * wm.span_m
        # Probe a spread of texels (including ones near the storm core).
        for row, col in [
            (0, 0),
            (n - 1, n - 1),
            (n // 2, n // 2),
            (n // 2 + 3, n // 2 - 5),
            (10, n - 10),
        ]:
            wx = center[0] - half + (col + 0.5) * wm.cell_m
            wy = center[1] - half + (row + 0.5) * wm.cell_m
            lw = ws.sample_local((wx, wy), t)
            expected = np.array(
                [lw.cloud_coverage, lw.cloud_density, lw.rain_intensity, lw.fog_density],
                dtype=np.float32,
            )
            assert np.allclose(r[row, col], expected, atol=1e-6), (
                f"texel ({row},{col}) mismatch: {r[row, col]} != {expected}"
            )


class TestRasterTimeInvariant:
    def test_call_timing_irrelevant(self):
        cfg = load_config()
        ws = _ws()
        wm = WeatherMap(cfg)
        c = _first_thunderstorm(cfg)
        t = c.spawn_time + 0.5 * c.duration_s
        center = tuple(c.center(t, ws.synoptic))

        first = wm.rasterize(ws, center, t)
        # Churn the system: many updates at other times/positions.
        for i in range(20):
            ws.update(0, i * 1234.0 % DAY, (i * 100.0, -i * 50.0))
        ws.sample_local((9999.0, 9999.0), 3 * DAY)
        second = wm.rasterize(ws, center, t)
        assert np.array_equal(first, second)


class TestWetness:
    def test_dry_before_any_rain(self):
        ws = _ws()
        # Very early time → no rain history available (quadrature reaches t<0).
        w = ws.wetness_at(np.array([[0.0, 0.0]]), 0.0)
        assert w.shape == (1,)
        assert w[0] == 0.0

    def test_wet_under_sustained_storm(self):
        cfg = load_config()
        ws = _ws()
        c = _first_thunderstorm(cfg)
        # Late in the storm's life, tracking its core: rain has fallen here for
        # a while, so wetness should have built up.
        t = c.spawn_time + 0.8 * c.duration_s
        # Sample wetness following the moving core back through the recent past.
        pts = np.array([c.center(t, ws.synoptic)])
        w_core = ws.wetness_at(pts, t)[0]
        far = pts + np.array([30000.0, 0.0])
        w_far = ws.wetness_at(far, t)[0]
        assert 0.0 < w_core <= 1.0
        assert w_far == 0.0
        assert w_core > w_far

    def test_pure_function(self):
        ws = _ws()
        pts = np.array([[120.0, -30.0]])
        a = ws.wetness_at(pts, 5 * DAY)
        b = ws.wetness_at(pts, 5 * DAY)
        assert np.array_equal(a, b)
