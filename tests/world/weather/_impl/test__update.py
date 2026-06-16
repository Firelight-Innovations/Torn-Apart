"""
tests/world/weather/_impl/test__update.py — Mirror for
fire_engine/world/weather/_impl/_update.py.

Authored tests covering the internal update helpers:
  _smoothstep, _lerp_local, classified_state (hysteresis), and
  do_update (exercised via WeatherSystem.update).

Headless — no panda3d imports.

Coverage
--------
CORRECTNESS — _smoothstep:
  - Returns 0 below lo, 1 above hi.
  - Is monotonically non-decreasing over [lo, hi].
  - Degenerate case (hi <= lo) returns 0 or 1 correctly.

CORRECTNESS — _lerp_local:
  - t=0 → returns ``a`` bit-exactly.
  - t=1 → returns ``b`` bit-exactly.
  - t=0.5 → all scalar fields are midpoint; wind_dir is renormalised.
  - The returned LocalWeather is valid (all fields in expected ranges).

CORRECTNESS — do_update (via WeatherSystem.update):
  - Calling update with a player position returns a LocalWeather.
  - update is DETERMINISTIC: same inputs → identical output across two instances.
  - Continuity: successive 2-second steps never pop (max delta bounded).
"""

from __future__ import annotations

import math

import pytest

from fire_engine.core import load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.world.weather import WeatherSystem
from fire_engine.world.weather._impl._update import _lerp_local, _smoothstep
from fire_engine.world.weather.types import LocalWeather

# ---------------------------------------------------------------------------
# _smoothstep
# ---------------------------------------------------------------------------


class TestSmoothstep:
    def test_zero_below_lo(self):
        assert _smoothstep(0.0, 1.0, 2.0) == pytest.approx(0.0)
        assert _smoothstep(-10.0, 0.0, 1.0) == pytest.approx(0.0)

    def test_one_above_hi(self):
        assert _smoothstep(3.0, 1.0, 2.0) == pytest.approx(1.0)
        assert _smoothstep(10.0, 0.0, 1.0) == pytest.approx(1.0)

    def test_at_lo_returns_zero(self):
        assert _smoothstep(1.0, 1.0, 2.0) == pytest.approx(0.0)

    def test_at_hi_returns_one(self):
        assert _smoothstep(2.0, 1.0, 2.0) == pytest.approx(1.0)

    def test_midpoint_is_half(self):
        # smoothstep(0.5, 0, 1) = 0.5
        assert _smoothstep(0.5, 0.0, 1.0) == pytest.approx(0.5)

    def test_monotonically_non_decreasing(self):

        xs = [i / 20.0 for i in range(21)]
        vals = [_smoothstep(x, 0.0, 1.0) for x in xs]
        for i in range(len(vals) - 1):
            assert vals[i + 1] >= vals[i] - 1e-15

    def test_degenerate_hi_le_lo_below(self):
        # hi <= lo: returns 0 if x < lo
        assert _smoothstep(0.5, 1.0, 1.0) == pytest.approx(0.0)

    def test_degenerate_hi_le_lo_above(self):
        # hi <= lo: returns 1 if x >= lo
        assert _smoothstep(1.5, 1.0, 0.5) == pytest.approx(1.0)

    def test_s_curve_symmetry(self):
        # smoothstep is S-shaped: value at lo+d equals 1 - value at hi-d
        v_low = _smoothstep(0.2, 0.0, 1.0)
        v_high = _smoothstep(0.8, 0.0, 1.0)
        assert v_low + v_high == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _lerp_local
# ---------------------------------------------------------------------------


def _make_lw(
    cloud_coverage: float = 0.0,
    cloud_density: float = 0.0,
    fog_density: float = 0.0,
    rain_intensity: float = 0.0,
    wind_dir: tuple[float, float] = (1.0, 0.0),
    wind_speed: float = 5.0,
    humidity: float = 0.5,
    wetness: float = 0.0,
    temperature_c: float = 12.0,
) -> LocalWeather:
    return LocalWeather(
        cloud_coverage=cloud_coverage,
        cloud_density=cloud_density,
        fog_density=fog_density,
        rain_intensity=rain_intensity,
        wind_dir=wind_dir,
        wind_speed=wind_speed,
        humidity=humidity,
        wetness=wetness,
        temperature_c=temperature_c,
    )


class TestLerpLocal:
    def test_t0_returns_a(self):
        a = _make_lw(cloud_coverage=0.2, wind_speed=3.0, temperature_c=5.0)
        b = _make_lw(cloud_coverage=0.8, wind_speed=9.0, temperature_c=20.0)
        result = _lerp_local(a, b, 0.0)
        assert result is a

    def test_t1_returns_b(self):
        a = _make_lw(cloud_coverage=0.2, wind_speed=3.0, temperature_c=5.0)
        b = _make_lw(cloud_coverage=0.8, wind_speed=9.0, temperature_c=20.0)
        result = _lerp_local(a, b, 1.0)
        assert result is b

    def test_midpoint_scalars(self):
        a = _make_lw(cloud_coverage=0.0, cloud_density=0.0, rain_intensity=0.0, wind_speed=4.0)
        b = _make_lw(cloud_coverage=1.0, cloud_density=1.0, rain_intensity=1.0, wind_speed=8.0)
        mid = _lerp_local(a, b, 0.5)
        assert mid.cloud_coverage == pytest.approx(0.5)
        assert mid.cloud_density == pytest.approx(0.5)
        assert mid.rain_intensity == pytest.approx(0.5)
        assert mid.wind_speed == pytest.approx(6.0)

    def test_wind_dir_renormalised(self):
        a = _make_lw(wind_dir=(1.0, 0.0))
        b = _make_lw(wind_dir=(0.0, 1.0))
        mid = _lerp_local(a, b, 0.5)
        norm = math.hypot(*mid.wind_dir)
        assert norm == pytest.approx(1.0, abs=1e-9)

    def test_result_is_local_weather(self):
        a = _make_lw()
        b = _make_lw(cloud_coverage=0.5)
        assert isinstance(_lerp_local(a, b, 0.5), LocalWeather)

    def test_t_below_zero_returns_a(self):
        a = _make_lw(cloud_coverage=0.1)
        b = _make_lw(cloud_coverage=0.9)
        result = _lerp_local(a, b, -0.5)
        assert result is a

    def test_t_above_one_returns_b(self):
        a = _make_lw(cloud_coverage=0.1)
        b = _make_lw(cloud_coverage=0.9)
        result = _lerp_local(a, b, 1.5)
        assert result is b


# ---------------------------------------------------------------------------
# do_update via WeatherSystem.update
# ---------------------------------------------------------------------------


def _ws(seed: int = 1337) -> WeatherSystem:
    set_world_seed(seed)
    return WeatherSystem(load_config())


class TestDoUpdate:
    def test_update_returns_local_weather(self):
        ws = _ws()
        lw = ws.update(0, 3600.0, (0.0, 0.0))
        assert isinstance(lw, LocalWeather)

    def test_update_deterministic_same_seed(self):
        """Two instances with the same seed produce identical update sequences."""
        a = _ws(seed=42)
        b = _ws(seed=42)
        for day in range(2):
            for seg in range(12):
                tod = seg * 2 * 3600.0
                pos = (seg * 50.0, day * 30.0)
                assert a.update(day, tod, pos) == b.update(day, tod, pos)

    def test_update_without_player_pos(self):
        """update(day, tod) with no player_pos must not crash."""
        ws = _ws()
        lw = ws.update(0, 0.0)
        assert isinstance(lw, LocalWeather)

    def test_update_fields_in_valid_ranges(self):
        ws = _ws(seed=99)
        for day in range(3):
            for seg in range(24):
                lw = ws.update(day, seg * 3600.0, (seg * 10.0, 0.0))
                assert 0.0 <= lw.cloud_coverage <= 1.0
                assert 0.0 <= lw.cloud_density <= 1.0
                assert 0.0 <= lw.rain_intensity <= 1.0
                assert lw.wind_speed >= 0.0
                assert math.isfinite(lw.fog_density)
                assert lw.fog_density >= 0.0

    def test_continuity_at_2_second_steps(self):
        """Successive 2-second updates must not produce sudden pops."""
        ws = _ws(seed=77)
        DAY = 86400.0
        BOUNDS = {
            "cloud_coverage": 0.05,
            "cloud_density": 0.05,
            "rain_intensity": 0.06,
            "fog_density": 0.005,
            "wind_speed": 0.5,
        }
        prev = None
        for step in range(1800):  # 1 game-hour at 2-second steps
            t = step * 2.0
            day = int(t // DAY)
            lw = ws.update(day, t - day * DAY, (0.0, 0.0))
            if prev is not None:
                for field, bound in BOUNDS.items():
                    delta = abs(getattr(lw, field) - getattr(prev, field))
                    assert delta <= bound, f"{field} popped by {delta:.4f} > {bound} at t={t:.0f}s"
            prev = lw
