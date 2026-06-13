"""
tests/test_weather_system.py — Spatial WeatherSystem: sampling, classify,
override shim, Saveable.

No panda3d imports anywhere in this file.

Coverage
--------
- Determinism: same seed + same (day, tod, pos) stream → identical samples.
- sample_local is spatial: a point under a shower is wetter than one far away.
- Continuity: per-(2 game-s) param steps never pop, even as a fast cell sweeps.
- Classification hysteresis: a changed label waits HYSTERESIS_SECONDS before it
  becomes `current`.
- WeatherChangedEvent is deferred and fires exactly once per committed change.
- Saveable: get_delta() == {} naturally; force/release round-trips through a
  fresh instance; a legacy (old-Markov) delta still loads.
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.core import EventBus, WeatherChangedEvent, load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.weather import WeatherSystem, WeatherType
from fire_engine.weather.cells import CellKind, natural_cells
from fire_engine.weather.system import BLEND_SECONDS, HYSTERESIS_SECONDS

HOUR = 3600.0
DAY = 24 * HOUR


def _ws(seed: int = 1337, bus: EventBus | None = None) -> WeatherSystem:
    set_world_seed(seed)
    return WeatherSystem(load_config(), bus)


def _first_thunderstorm(cfg):
    """The earliest THUNDERSTORM cell (smallest, strongest — steepest field)."""
    for d in range(80):
        for c in natural_cells(d, cfg):
            if c.kind is CellKind.THUNDERSTORM:
                return c
    raise AssertionError("no thunderstorm found in 80 days")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_identical_samples(self):
        a = _ws(seed=42)
        b = _ws(seed=42)
        for day in range(2):
            for step in range(0, 24 * 4):           # every 15 game minutes
                tod = step * (15 * 60.0)
                pos = (step * 7.0, -step * 3.0)
                assert a.update(day, tod, pos) == b.update(day, tod, pos)

    def test_sample_local_pure(self):
        ws = _ws(seed=7)
        ws.update(3, 9 * HOUR, (0.0, 0.0))
        s1 = ws.sample_local((123.0, -45.0), 3 * DAY + 9 * HOUR)
        s2 = ws.sample_local((123.0, -45.0), 3 * DAY + 9 * HOUR)
        assert s1 == s2


# ---------------------------------------------------------------------------
# Spatial sampling
# ---------------------------------------------------------------------------

class TestSpatial:
    def test_under_shower_wetter_than_far(self):
        cfg = load_config()
        ws = _ws(seed=1337)
        c = _first_thunderstorm(cfg)
        t = c.spawn_time + 0.5 * c.duration_s
        under = tuple(c.center(t, ws.synoptic))
        far = (under[0] + 20000.0, under[1])         # 20 km away
        s_under = ws.sample_local(under, t)
        s_far = ws.sample_local(far, t)
        assert s_under.rain_intensity > 0.3
        assert s_far.rain_intensity == 0.0
        assert s_under.cloud_coverage > s_far.cloud_coverage

    def test_params_in_valid_ranges(self):
        ws = _ws(seed=11)
        fog_max = load_config().weather_fog_max_density
        for day in range(2):
            for seg in range(48):
                p = ws.update(day, seg * 1800.0, (seg * 50.0, day * 30.0))
                assert 0.0 <= p.cloud_coverage <= 1.0
                assert 0.0 <= p.cloud_density <= 1.0
                assert 0.0 <= p.rain_intensity <= 1.0
                assert 0.0 <= p.fog_density <= fog_max + 1e-9
                assert p.wind_speed >= 0.0
                assert abs(np.hypot(*p.wind_dir) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Continuity (no popping) — even as a fast cell sweeps the sample point
# ---------------------------------------------------------------------------

class TestContinuity:
    STEP = 2.0
    BOUNDS = {
        "cloud_coverage": 0.05,
        "cloud_density": 0.05,
        "rain_intensity": 0.06,
        "fog_density": 0.005,
        "wind_speed": 0.5,
    }

    def _sweep(self, ws, pos, t0, t1):
        prev = None
        t = t0
        while t < t1:
            day = int(t // DAY)
            p = ws.update(day, t - day * DAY, pos)
            if prev is not None:
                for field, bound in self.BOUNDS.items():
                    delta = abs(getattr(p, field) - getattr(prev, field))
                    assert delta <= bound, (
                        f"{field} popped {delta:.4f} (> {bound}) at t={t:.0f}"
                    )
            prev = p
            t += self.STEP

    def test_no_pop_as_thunderstorm_sweeps_point(self):
        cfg = load_config()
        ws = _ws(seed=1337)
        c = _first_thunderstorm(cfg)
        # Sample at the storm's mid-life center: the moving cell sweeps right
        # through this fixed point — the steepest the local field ever gets.
        mid = c.spawn_time + 0.5 * c.duration_s
        pos = tuple(c.center(mid, ws.synoptic))
        self._sweep(ws, pos, c.spawn_time, c.spawn_time + c.duration_s)

    def test_no_pop_over_two_days_at_origin(self):
        ws = _ws(seed=1337)
        self._sweep(ws, (0.0, 0.0), 0.0, 2 * DAY)


# ---------------------------------------------------------------------------
# Classification hysteresis
# ---------------------------------------------------------------------------

class TestHysteresis:
    def test_label_change_waits_for_hysteresis(self):
        cfg = load_config()
        ws = _ws(seed=1337)
        c = _first_thunderstorm(cfg)
        syn = ws.synoptic
        mid = c.spawn_time + 0.5 * c.duration_s      # deep in the plateau

        def _upd(t_abs, pos):
            day = int(t_abs // DAY)
            return ws.update(day, t_abs - day * DAY, pos)

        # The cell moves, so always query its *current* center to stay under
        # the core; a fixed faraway point stays dry across the short window.
        far = tuple(c.center(mid, syn) + np.array([20000.0, 0.0]))

        # Commit the ambient (dry) label away from the cell.
        _upd(mid - 100.0, far)
        baseline = ws.current
        assert baseline not in (WeatherType.RAIN, WeatherType.STORM)

        # Step under the storm core: the raw label flips to wet, but `current`
        # must hold the baseline for HYSTERESIS_SECONDS before it commits.
        _upd(mid, tuple(c.center(mid, syn)))
        assert ws.current is baseline, "label changed with no hysteresis"

        t2 = mid + HYSTERESIS_SECONDS - 5.0
        _upd(t2, tuple(c.center(t2, syn)))
        assert ws.current is baseline, "committed before hysteresis elapsed"

        t3 = mid + HYSTERESIS_SECONDS + 5.0
        _upd(t3, tuple(c.center(t3, syn)))
        assert ws.current in (WeatherType.RAIN, WeatherType.STORM)


# ---------------------------------------------------------------------------
# force_weather (dev override shim)
# ---------------------------------------------------------------------------

class TestForceWeather:
    def test_force_blends_to_target(self):
        ws = _ws(seed=5)
        ws.update(0, 10 * HOUR, (0.0, 0.0))
        ws.force_weather(WeatherType.STORM)
        assert ws.current is WeatherType.STORM

        p_start = ws.update(0, 10 * HOUR, (0.0, 0.0))         # anchors the blend
        p_mid = ws.update(0, 10 * HOUR + BLEND_SECONDS / 2, (0.0, 0.0))
        p_end = ws.update(0, 10 * HOUR + BLEND_SECONDS + 1.0, (0.0, 0.0))

        assert p_start.rain_intensity <= p_mid.rain_intensity <= p_end.rain_intensity
        assert p_end.rain_intensity == pytest.approx(1.0)
        assert p_end.cloud_coverage == pytest.approx(0.98)
        assert p_end.cloud_density == pytest.approx(0.95)

    def test_clear_override_blends_back_to_natural(self):
        ws = _ws(seed=5)
        ws.update(0, 10 * HOUR, (0.0, 0.0))
        ws.force_weather(WeatherType.STORM)
        ws.update(0, 10 * HOUR, (0.0, 0.0))
        ws.update(0, 10 * HOUR + BLEND_SECONDS + 1.0, (0.0, 0.0))

        ws.force_weather(None)
        t_release = 10 * HOUR + BLEND_SECONDS + 2.0
        ws.update(0, t_release, (0.0, 0.0))                  # anchors release
        assert ws.get_delta() != {}, "mid-release blend must be saveable"

        t_done = t_release + BLEND_SECONDS + 120.0
        p = ws.update(0, t_done, (0.0, 0.0))

        natural = _ws(seed=5)
        p_nat = natural.update(0, t_done, (0.0, 0.0))
        assert p == p_nat
        assert ws.get_delta() == {}, "delta must return to {} after release"


# ---------------------------------------------------------------------------
# WeatherChangedEvent
# ---------------------------------------------------------------------------

class TestWeatherChangedEvent:
    def test_event_deferred_and_once_per_change(self):
        bus = EventBus()
        received: list[WeatherChangedEvent] = []
        bus.subscribe(WeatherChangedEvent, received.append)

        ws = _ws(seed=1337, bus=bus)
        ws.update(0, 6 * HOUR, (0.0, 0.0))          # establishes last_state
        ws.force_weather(WeatherType.STORM)
        ws.update(0, 6 * HOUR + 1.0, (0.0, 0.0))    # guaranteed change…
        if received:
            pytest.fail("event delivered before drain()")
        bus.drain()
        assert len(received) == 1
        assert received[0].current == "storm"

        # No further change → no further events.
        ws.update(0, 6 * HOUR + 2.0, (0.0, 0.0))
        bus.drain()
        assert len(received) == 1


# ---------------------------------------------------------------------------
# Saveable protocol
# ---------------------------------------------------------------------------

class TestSaveable:
    def test_save_key(self):
        assert WeatherSystem.save_key == "weather"

    def test_natural_delta_is_empty(self):
        ws = _ws(seed=11)
        for day in range(2):
            for seg in range(12):
                ws.update(day, seg * 2 * HOUR + HOUR, (seg * 40.0, 0.0))
        assert ws.get_delta() == {}

    def test_override_round_trip(self):
        t0 = 9 * HOUR
        ws1 = _ws(seed=11)
        ws1.update(0, t0, (0.0, 0.0))
        ws1.force_weather(WeatherType.RAIN)
        ws1.update(0, t0 + 60.0, (0.0, 0.0))         # anchor mid-blend

        delta = ws1.get_delta()
        assert delta != {} and delta["override"] == "rain"

        ws2 = _ws(seed=11)
        ws2.apply_delta(delta)
        for dt in (120.0, 600.0, BLEND_SECONDS, BLEND_SECONDS + 3600.0):
            p1 = ws1.update(0, t0 + dt, (0.0, 0.0))
            p2 = ws2.update(0, t0 + dt, (0.0, 0.0))
            assert p1 == p2, f"divergence at +{dt} s"
            assert ws1.current is ws2.current

    def test_delta_is_plain_primitives(self):
        ws = _ws(seed=11)
        ws.update(0, 0.0, (0.0, 0.0))
        ws.force_weather(WeatherType.FOG)
        ws.update(0, 60.0, (0.0, 0.0))

        def _check(value):
            if isinstance(value, dict):
                for v in value.values():
                    _check(v)
            elif isinstance(value, list):
                for v in value:
                    _check(v)
            else:
                assert isinstance(value, (int, float, str, bool, type(None)))

        _check(ws.get_delta())

    def test_legacy_markov_delta_loads(self):
        """A delta saved by the old global-Markov system still restores."""
        legacy = {
            "override": "storm",
            "override_start_abs_t": 36000.0,
            "override_from": {                       # old WeatherParams keys —
                "cloud_coverage": 0.5,               # no humidity/wetness/temp
                "cloud_density": 0.6,
                "fog_density": 0.002,
                "rain_intensity": 0.3,
                "wind_dir": [1.0, 0.0],
                "wind_speed": 7.0,
            },
            "last_state": "rain",
        }
        ws = _ws(seed=11)
        ws.apply_delta(legacy)                       # must not raise
        assert ws.current is WeatherType.STORM
        assert ws.get_delta()["override"] == "storm"
