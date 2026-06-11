"""
tests/test_weather.py — Headless tests for the WeatherSystem (sky package).

No panda3d imports anywhere in this file.

Test coverage
-------------
- Schedule determinism across fresh instances (same seed).
- A day's discrete sequence is a pure function of (seed, day) — querying a
  later day directly matches querying it after simulating earlier days.
- Blend continuity: params never jump more than the 20-game-minute blend
  allows per 60-game-second step.
- force_weather blends toward the target and reaches it; clearing blends back.
- Saveable: get_delta() == {} in the natural case; an active override
  round-trips through a fresh instance with identical subsequent behaviour.
- WeatherChangedEvent published (deferred) on discrete transitions.
"""

from __future__ import annotations

import math

import pytest

from fire_engine.core import EventBus, WeatherChangedEvent, load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.sky.weather import (
    BLEND_SECONDS,
    SEGMENT_SECONDS,
    WeatherParams,
    WeatherSystem,
    WeatherType,
)

HOUR = 3600.0
DAY = 24 * HOUR


def _make_ws(seed: int = 1337, bus: EventBus | None = None) -> WeatherSystem:
    set_world_seed(seed)
    return WeatherSystem(load_config(), bus)


def _day_sequence(ws: WeatherSystem, day: int) -> list[str]:
    """Discrete state at each segment midpoint of *day* (12 entries)."""
    out = []
    for seg in range(12):
        ws.update(day, seg * SEGMENT_SECONDS + SEGMENT_SECONDS / 2)
        out.append(ws.current.value)
    return out


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestScheduleDeterminism:
    def test_fresh_instances_same_schedule(self):
        a = _make_ws(seed=42)
        b = _make_ws(seed=42)
        for day in range(3):
            assert _day_sequence(a, day) == _day_sequence(b, day)

    def test_fresh_instances_same_params(self):
        a = _make_ws(seed=42)
        b = _make_ws(seed=42)
        for day in range(2):
            for step in range(0, 24 * 4):  # every 15 game minutes
                tod = step * (15 * 60.0)
                assert a.update(day, tod) == b.update(day, tod)

    def test_day_sequence_pure_function_of_seed_and_day(self):
        """Day 2's schedule is identical whether or not days 0–1 were simulated."""
        a = _make_ws(seed=99)
        for day in range(2):
            _day_sequence(a, day)        # simulate earlier days
        seq_after_history = _day_sequence(a, 2)

        b = _make_ws(seed=99)            # fresh — jump straight to day 2
        seq_direct = _day_sequence(b, 2)
        assert seq_after_history == seq_direct

    def test_different_seed_different_schedule(self):
        # NOTE: set_world_seed is global module state — fully consume each
        # instance's schedule before switching seeds.
        a = _make_ws(seed=1)
        seq_a = [s for d in range(4) for s in _day_sequence(a, d)]
        b = _make_ws(seed=2)
        seq_b = [s for d in range(4) for s in _day_sequence(b, d)]
        assert seq_a != seq_b


# ---------------------------------------------------------------------------
# Blend continuity
# ---------------------------------------------------------------------------

class TestBlendContinuity:
    # Max change per 60-game-second step: the smoothstep's peak slope is
    # 1.5/BLEND_SECONDS, so a full-range (0→1) param can move at most
    # 1.5 * 60/1200 = 0.075 per step.  Bounds below add a little headroom.
    STEP = 60.0
    BOUNDS = {
        "cloud_coverage": 0.08,
        "cloud_density": 0.08,
        "rain_intensity": 0.08,
        "fog_density": 0.0030,    # full range ≈ 0.024 → max step ≈ 0.0018
        "wind_speed": 1.2,        # full range ≈ 13 m/s → max step ≈ 0.98
    }

    def test_no_param_pops_over_three_days(self):
        ws = _make_ws(seed=1337)
        prev: WeatherParams | None = None
        for day in range(3):
            steps = int(DAY / self.STEP)
            for i in range(steps):
                p = ws.update(day, i * self.STEP)
                if prev is not None:
                    for field, bound in self.BOUNDS.items():
                        delta = abs(getattr(p, field) - getattr(prev, field))
                        assert delta <= bound, (
                            f"{field} jumped {delta:.4f} (> {bound}) at "
                            f"day={day} tod={i * self.STEP}"
                        )
                prev = p

    def test_params_in_valid_ranges(self):
        ws = _make_ws(seed=7)
        for day in range(2):
            for seg in range(48):
                p = ws.update(day, seg * 1800.0)
                assert 0.0 <= p.cloud_coverage <= 1.0
                assert 0.0 <= p.cloud_density <= 1.0
                assert 0.0 <= p.rain_intensity <= 1.0
                assert 0.0 <= p.fog_density <= 0.03
                assert p.wind_speed >= 0.0
                assert abs(math.hypot(*p.wind_dir) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# force_weather
# ---------------------------------------------------------------------------

class TestForceWeather:
    def test_force_blends_to_target(self):
        ws = _make_ws(seed=5)
        ws.update(0, 10 * HOUR)
        ws.force_weather(WeatherType.STORM)
        assert ws.current is WeatherType.STORM

        p_start = ws.update(0, 10 * HOUR)            # anchors the blend
        p_mid = ws.update(0, 10 * HOUR + BLEND_SECONDS / 2)
        p_end = ws.update(0, 10 * HOUR + BLEND_SECONDS + 1.0)

        # Monotonic approach to the storm targets.
        assert p_start.rain_intensity <= p_mid.rain_intensity <= p_end.rain_intensity
        assert p_end.rain_intensity == pytest.approx(1.0)
        assert p_end.cloud_coverage == pytest.approx(0.98)
        assert p_end.cloud_density == pytest.approx(0.95)

    def test_clear_override_blends_back_to_natural(self):
        ws = _make_ws(seed=5)
        ws.update(0, 10 * HOUR)
        ws.force_weather(WeatherType.STORM)
        ws.update(0, 10 * HOUR)
        ws.update(0, 10 * HOUR + BLEND_SECONDS + 1.0)   # storm fully in

        ws.force_weather(None)
        t_release = 10 * HOUR + BLEND_SECONDS + 2.0
        ws.update(0, t_release)              # anchors the release blend
        assert ws.get_delta() != {}, "mid-release blend must be saveable"
        # After the release blend, params equal the pure natural schedule.
        t_done = t_release + BLEND_SECONDS + 120.0
        p = ws.update(0, t_done)

        natural = _make_ws(seed=5)
        p_nat = natural.update(0, t_done)
        assert p == p_nat
        assert ws.get_delta() == {}, "delta must return to {} after release"


# ---------------------------------------------------------------------------
# Saveable protocol
# ---------------------------------------------------------------------------

class TestSaveable:
    def test_save_key(self):
        assert WeatherSystem.save_key == "weather"

    def test_natural_delta_is_empty(self):
        ws = _make_ws(seed=11)
        for day in range(2):
            _day_sequence(ws, day)
        assert ws.get_delta() == {}

    def test_override_delta_round_trip(self):
        """apply_delta on a fresh instance → identical subsequent behaviour."""
        t0 = 9 * HOUR
        ws1 = _make_ws(seed=11)
        ws1.update(0, t0)
        ws1.force_weather(WeatherType.RAIN)
        ws1.update(0, t0 + 60.0)             # anchor mid-blend

        delta = ws1.get_delta()
        assert delta != {}
        assert delta["override"] == "rain"

        ws2 = _make_ws(seed=11)
        ws2.apply_delta(delta)

        # Identical params and discrete state from here on (incl. mid-blend).
        for dt in (120.0, 600.0, BLEND_SECONDS, BLEND_SECONDS + 3600.0):
            p1 = ws1.update(0, t0 + dt)
            p2 = ws2.update(0, t0 + dt)
            assert p1 == p2, f"divergence at +{dt} s"
            assert ws1.current is ws2.current

    def test_delta_is_plain_primitives(self):
        ws = _make_ws(seed=11)
        ws.update(0, 0.0)
        ws.force_weather(WeatherType.FOG)
        ws.update(0, 60.0)
        delta = ws.get_delta()

        def _check(value):
            if isinstance(value, dict):
                for v in value.values():
                    _check(v)
            elif isinstance(value, list):
                for v in value:
                    _check(v)
            else:
                assert isinstance(value, (int, float, str, bool, type(None))), (
                    f"non-primitive in delta: {type(value)}"
                )

        _check(delta)


# ---------------------------------------------------------------------------
# WeatherChangedEvent
# ---------------------------------------------------------------------------

class TestWeatherChangedEvent:
    def test_event_published_on_transitions(self):
        bus = EventBus()
        received: list[WeatherChangedEvent] = []
        bus.subscribe(WeatherChangedEvent, received.append)

        ws = _make_ws(seed=1337, bus=bus)
        observed: list[str] = []
        for day in range(5):
            for seg in range(12):
                ws.update(day, seg * SEGMENT_SECONDS + SEGMENT_SECONDS / 2)
                observed.append(ws.current.value)
                bus.drain()

        # The schedule over 5 days must contain at least one transition.
        changes = [
            (observed[i - 1], observed[i], (i * SEGMENT_SECONDS) // DAY)
            for i in range(1, len(observed))
            if observed[i] != observed[i - 1]
        ]
        assert len(changes) >= 1, "5-day schedule had no transitions (suspicious)"
        assert len(received) == len(changes)
        for evt, (prev, cur, _) in zip(received, changes):
            assert evt.previous == prev
            assert evt.current == cur
            assert WeatherType(evt.previous) and WeatherType(evt.current)

    def test_no_event_without_transition(self):
        bus = EventBus()
        received: list[WeatherChangedEvent] = []
        bus.subscribe(WeatherChangedEvent, received.append)

        ws = _make_ws(seed=1337, bus=bus)
        # Many updates inside a single segment → at most the segment's own
        # state; no transition, no events.
        for i in range(20):
            ws.update(0, 100.0 + i * 10.0)
        bus.drain()
        assert received == []

    def test_event_deferred_until_drain(self):
        bus = EventBus()
        received: list[WeatherChangedEvent] = []
        bus.subscribe(WeatherChangedEvent, received.append)

        ws = _make_ws(seed=1337, bus=bus)
        ws.update(0, SEGMENT_SECONDS / 2)
        ws.force_weather(WeatherType.STORM)   # guaranteed discrete change…
        ws.update(0, SEGMENT_SECONDS / 2 + 1.0)
        # …but only on drain (publish_deferred — never mid-frame).
        if ws.current is not WeatherType.STORM or received:
            pytest.fail("event delivered before drain()")
        bus.drain()
        assert len(received) == 1
        assert received[0].current == "storm"
