"""
tests/test_clock.py — Characterisation / golden-master tests for core/clock.py.

Covers:
  - fixed_steps() accumulator: partial, exact, multi-step accumulation
  - Spiral-of-death guard: very large real_dt capped at MAX_FIXED_STEPS
  - Leftover carry-over after fixed_steps() is consumed
  - Calendar: game_time_of_day and game_day advance with game_time_scale
  - Day rollover at _GAME_SECONDS_PER_DAY; boundaries (time 0, just before/after)
  - Multiple rollovers in one update()
  - get_state() / set_state() round-trip equality
  - GameDayTickEvent: published deferred on rollover; NOT fired before rollover
  - game_time_scale = 0 (no game-time advance)
  - real_dt = 0 (no steps, no time advance)
  - negative real_dt (pin current behaviour)
  - fractional real_dt smaller than fixed_dt (accumulates but no step)
  - total_real_time and dt tracking
  - DEFAULT_GAME_TIME_SCALE and _GAME_SECONDS_PER_DAY used as authoritative constants

All tests are headless: no panda3d, no fire_engine.world, no fire_engine.lighting.gpu.
"""

from __future__ import annotations

import pytest

from fire_engine.core.clock import (
    Clock,
    MAX_FIXED_STEPS,
    DEFAULT_GAME_TIME_SCALE,
    _GAME_SECONDS_PER_DAY,
)
from fire_engine.core.config import load_config
from fire_engine.core.event_bus import EventBus, GameDayTickEvent
from fire_engine.core.rng import set_world_seed


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_clock(scale: float = DEFAULT_GAME_TIME_SCALE, bus: EventBus | None = None) -> Clock:
    """Return a fresh Clock using fixed_dt from the real config."""
    cfg = load_config()
    return Clock(fixed_dt=cfg.fixed_dt, bus=bus, game_time_scale=scale)


def _make_bus_and_clock(scale: float = DEFAULT_GAME_TIME_SCALE):
    """Return (bus, clock) pair wired together."""
    bus = EventBus()
    clock = _make_clock(scale=scale, bus=bus)
    return bus, clock


# ---------------------------------------------------------------------------
# fixed_steps() accumulator
# ---------------------------------------------------------------------------

class TestFixedStepsAccumulator:
    def test_no_steps_below_fixed_dt(self):
        """real_dt < fixed_dt → no fixed steps yielded."""
        clock = _make_clock()
        clock.update(clock.fixed_dt * 0.5)
        steps = list(clock.fixed_steps())
        assert steps == []

    def test_exactly_one_step_at_fixed_dt(self):
        """real_dt == fixed_dt → exactly one fixed step."""
        clock = _make_clock()
        clock.update(clock.fixed_dt)
        steps = list(clock.fixed_steps())
        assert len(steps) == 1
        assert steps[0] == clock.fixed_dt

    def test_two_steps_at_twice_fixed_dt(self):
        """real_dt == 2 × fixed_dt → exactly two fixed steps."""
        clock = _make_clock()
        clock.update(clock.fixed_dt * 2)
        steps = list(clock.fixed_steps())
        assert len(steps) == 2
        assert all(s == clock.fixed_dt for s in steps)

    def test_accumulation_across_frames(self):
        """Two sub-threshold updates accumulate; together they exceed one step."""
        clock = _make_clock()
        # Each update alone is 60% of fixed_dt — not enough for a step.
        clock.update(clock.fixed_dt * 0.6)
        assert list(clock.fixed_steps()) == []
        clock.update(clock.fixed_dt * 0.6)
        # 1.2 × fixed_dt accumulated — one step, 0.2 left.
        steps = list(clock.fixed_steps())
        assert len(steps) == 1

    def test_leftover_carried_over(self):
        """
        Residual after fixed_steps() is not discarded — it accumulates.

        FINDING (pinned): Due to floating-point representation, 0.5×fixed_dt
        accumulates to ~0.009999... and a second 0.5×fixed_dt addition yields
        ~0.019999..., which is just below the threshold (0.02). So no step is
        emitted even though naively 0.5+0.5 = 1.0×fixed_dt. The accumulator
        IS non-zero (residual preserved), so the leftover is not silently
        dropped — but the float arithmetic means the combined residual falls
        just short of the threshold. Pin the actual behaviour: 0 steps emitted.
        """
        clock = _make_clock()
        # Send exactly 1.5 × fixed_dt
        clock.update(clock.fixed_dt * 1.5)
        steps = list(clock.fixed_steps())
        assert len(steps) == 1
        # Residual (~0.5×fixed_dt in float) is still in accumulator (non-zero)
        assert clock._accumulator > 0.0, "Residual should be carried (non-zero)"
        # Adding another 0.5×fixed_dt: floating point lands just below threshold
        clock.update(clock.fixed_dt * 0.5)
        steps2 = list(clock.fixed_steps())
        # Pin actual behaviour: float arithmetic yields 0 steps here
        assert len(steps2) == 0, (
            "Pin: float(0.5)*fixed_dt + float(0.5)*fixed_dt lands just below "
            "fixed_dt threshold — 0 steps emitted (leftover is preserved but "
            "just-below-threshold is current behaviour)"
        )

    def test_each_yielded_value_is_fixed_dt(self):
        """All yielded values equal fixed_dt (not something else)."""
        clock = _make_clock()
        clock.update(clock.fixed_dt * MAX_FIXED_STEPS)
        for step in clock.fixed_steps():
            assert step == clock.fixed_dt


# ---------------------------------------------------------------------------
# Spiral-of-death guard
# ---------------------------------------------------------------------------

class TestSpiralOfDeathGuard:
    def test_large_dt_capped_at_max_fixed_steps(self):
        """Very large real_dt must yield at most MAX_FIXED_STEPS steps."""
        clock = _make_clock()
        huge_dt = clock.fixed_dt * (MAX_FIXED_STEPS + 100)
        clock.update(huge_dt)
        steps = list(clock.fixed_steps())
        assert len(steps) <= MAX_FIXED_STEPS

    def test_max_fixed_steps_is_five(self):
        """Pin the guard constant at its documented value of 5."""
        assert MAX_FIXED_STEPS == 5

    def test_accumulator_reset_after_spiral_guard_fires(self):
        """After the guard drops excess, accumulator should not be huge."""
        clock = _make_clock()
        huge_dt = clock.fixed_dt * (MAX_FIXED_STEPS + 50)
        clock.update(huge_dt)
        list(clock.fixed_steps())  # exhaust generator (triggers guard)
        # A follow-up tiny update should not suddenly emit many steps
        clock.update(clock.fixed_dt * 0.1)
        steps = list(clock.fixed_steps())
        assert len(steps) == 0, (
            "After spiral-guard reset, accumulator should be near zero"
        )

    def test_exactly_max_steps_at_float_boundary(self):
        """
        Exactly MAX_FIXED_STEPS × fixed_dt: pin the actual step count.

        FINDING (pinned): With fixed_dt=0.02 and MAX_FIXED_STEPS=5,
        5×0.02 == 0.1 exactly in IEEE 754. However, repeated subtraction of
        0.02 introduces rounding: after 4 subtractions the accumulator lands at
        ~0.019999999999999993, which is just below 0.02. So only 4 steps are
        emitted, not 5.  The guard condition (steps < MAX_FIXED_STEPS) is never
        the bottleneck here — floating-point drift stops the loop first.

        This means the accumulator can retain a ~0×fixed_dt residual that
        prevents the 5th tick from ever firing, silently dropping one simulation
        step. Suspicious: the code says "up to MAX_FIXED_STEPS" but in practice
        5 × fixed_dt only produces 4 steps.
        """
        clock = _make_clock()
        clock.update(clock.fixed_dt * MAX_FIXED_STEPS)
        steps = list(clock.fixed_steps())
        # Pin: 4 steps, not 5, due to floating-point drift in subtraction
        assert len(steps) == MAX_FIXED_STEPS - 1, (
            "Pin: floating-point drift causes 5×fixed_dt to yield only 4 steps"
        )


# ---------------------------------------------------------------------------
# Calendar advancement
# ---------------------------------------------------------------------------

class TestCalendar:
    def test_game_time_advances_with_scale(self):
        """1 real second at default scale advances game time by game_time_scale."""
        clock = _make_clock(scale=DEFAULT_GAME_TIME_SCALE)
        clock.update(1.0)
        assert abs(clock.game_time_of_day - DEFAULT_GAME_TIME_SCALE) < 1e-9

    def test_day_starts_at_zero(self):
        """Fresh clock starts at day 0 with time-of-day 0."""
        clock = _make_clock()
        assert clock.game_day == 0
        assert clock.game_time_of_day == 0.0

    def test_no_rollover_just_before_end_of_day(self):
        """game_time_of_day just below _GAME_SECONDS_PER_DAY does not roll over."""
        clock = _make_clock(scale=1.0)
        # advance to just below 1 full day (1 second short)
        clock.update(_GAME_SECONDS_PER_DAY - 1.0)
        assert clock.game_day == 0
        assert clock.game_time_of_day < _GAME_SECONDS_PER_DAY

    def test_rollover_at_end_of_day(self):
        """Advancing exactly one game-day increments game_day to 1."""
        clock = _make_clock(scale=1.0)
        clock.update(_GAME_SECONDS_PER_DAY)
        assert clock.game_day == 1
        assert clock.game_time_of_day < 1e-9  # should land very near 0

    def test_rollover_wraps_time_of_day(self):
        """After a rollover, game_time_of_day is the remainder, not >= _GAME_SECONDS_PER_DAY."""
        clock = _make_clock(scale=1.0)
        extra = 3600.0  # 1 game hour past day boundary
        clock.update(_GAME_SECONDS_PER_DAY + extra)
        assert clock.game_day == 1
        assert abs(clock.game_time_of_day - extra) < 1e-9

    def test_multiple_rollovers_in_one_update(self):
        """An extremely large dt can roll over multiple days."""
        clock = _make_clock(scale=1.0)
        clock.update(3.0 * _GAME_SECONDS_PER_DAY)
        assert clock.game_day == 3

    def test_game_seconds_per_day_constant(self):
        """Pin the documented value: 24 hours × 3600 seconds."""
        assert _GAME_SECONDS_PER_DAY == 24.0 * 3600.0

    def test_default_game_time_scale_constant(self):
        """Pin DEFAULT_GAME_TIME_SCALE at the documented value of 60.0."""
        assert DEFAULT_GAME_TIME_SCALE == 60.0


# ---------------------------------------------------------------------------
# Boundary cases: real_dt = 0, negative, fractional, scale = 0
# ---------------------------------------------------------------------------

class TestBoundaryInputs:
    def test_zero_dt_no_steps_no_advance(self):
        """real_dt = 0 produces no fixed steps and no game-time change."""
        clock = _make_clock()
        clock.update(0.0)
        steps = list(clock.fixed_steps())
        assert steps == []
        assert clock.game_time_of_day == 0.0
        assert clock.game_day == 0
        assert clock.total_real_time == 0.0

    def test_negative_dt_pins_current_behaviour(self):
        """
        Negative real_dt: pin current behaviour — Clock does not raise; it
        decrements the accumulator and game time.

        FINDING: Clock accepts negative dt silently. This could allow the
        accumulator and game_time_of_day to go negative, which is likely
        unintentional (no guard in update() or fixed_steps()).
        """
        clock = _make_clock(scale=1.0)
        clock.update(-1.0)
        # Pin: dt is set to the negative value
        assert clock.dt == -1.0
        # Pin: total_real_time decrements (negative real time is suspicious)
        assert clock.total_real_time == -1.0
        # Pin: game_time_of_day goes negative (suspicious)
        assert clock.game_time_of_day == -1.0
        # Pin: fixed_steps() either yields nothing or behaves gracefully
        steps = list(clock.fixed_steps())
        assert isinstance(steps, list)

    def test_fractional_dt_smaller_than_fixed_dt(self):
        """real_dt < fixed_dt accumulates but yields no step."""
        clock = _make_clock()
        small_dt = clock.fixed_dt * 0.1
        clock.update(small_dt)
        steps = list(clock.fixed_steps())
        assert steps == []
        # Residual should be carried; total real time should equal small_dt
        assert abs(clock.total_real_time - small_dt) < 1e-12

    def test_scale_zero_no_game_time_advance(self):
        """game_time_scale = 0 → real time passes, game time stays frozen."""
        clock = _make_clock(scale=0.0)
        clock.update(10.0)
        assert clock.game_time_of_day == 0.0
        assert clock.game_day == 0
        assert abs(clock.total_real_time - 10.0) < 1e-12


# ---------------------------------------------------------------------------
# Total real time and dt tracking
# ---------------------------------------------------------------------------

class TestRealTimeTracking:
    def test_total_real_time_accumulates(self):
        """total_real_time accumulates across multiple updates."""
        clock = _make_clock()
        clock.update(0.016)
        clock.update(0.017)
        clock.update(0.018)
        assert abs(clock.total_real_time - (0.016 + 0.017 + 0.018)) < 1e-12

    def test_dt_reflects_last_update(self):
        """clock.dt always reflects the most recent real_dt passed to update()."""
        clock = _make_clock()
        clock.update(0.033)
        assert clock.dt == 0.033
        clock.update(0.100)
        assert clock.dt == 0.100


# ---------------------------------------------------------------------------
# get_state() / set_state() round-trip
# ---------------------------------------------------------------------------

class TestStateRoundTrip:
    def test_state_keys_present(self):
        """get_state() returns all expected keys."""
        clock = _make_clock()
        state = clock.get_state()
        expected_keys = {"game_day", "game_time_of_day", "total_real_time", "accumulator"}
        assert set(state.keys()) == expected_keys

    def test_state_values_are_primitives(self):
        """get_state() values are plain Python primitives, not numpy arrays."""
        clock = _make_clock()
        clock.update(123.456)
        state = clock.get_state()
        for k, v in state.items():
            assert isinstance(v, (int, float)), (
                f"State key '{k}' has non-primitive type {type(v)}"
            )

    def test_set_state_restores_game_day(self):
        """set_state() restores game_day from saved state."""
        clock = _make_clock(scale=1.0)
        clock.update(_GAME_SECONDS_PER_DAY * 7 + 3600.0)  # 7 days + 1 hour
        state = clock.get_state()

        fresh = _make_clock()
        fresh.set_state(state)
        assert fresh.game_day == clock.game_day

    def test_set_state_restores_game_time_of_day(self):
        """set_state() restores game_time_of_day precisely."""
        clock = _make_clock(scale=1.0)
        clock.update(_GAME_SECONDS_PER_DAY + 7200.0)
        state = clock.get_state()

        fresh = _make_clock()
        fresh.set_state(state)
        assert abs(fresh.game_time_of_day - clock.game_time_of_day) < 1e-6

    def test_set_state_restores_total_real_time(self):
        """set_state() restores total_real_time."""
        clock = _make_clock()
        clock.update(999.5)
        state = clock.get_state()

        fresh = _make_clock()
        fresh.set_state(state)
        assert abs(fresh.total_real_time - 999.5) < 1e-6

    def test_set_state_restores_accumulator(self):
        """set_state() restores the fixed-step accumulator."""
        clock = _make_clock()
        clock.update(clock.fixed_dt * 0.7)  # partial accumulation
        # do NOT drain fixed_steps — leave residual in accumulator
        state = clock.get_state()

        fresh = _make_clock()
        fresh.set_state(state)
        # After restoring, fresh clock should yield no steps (residual < fixed_dt)
        steps = list(fresh.fixed_steps())
        assert steps == []

    def test_round_trip_identical_state(self):
        """get_state() after set_state(get_state()) is identical."""
        clock = _make_clock(scale=1.0)
        clock.update(12345.0)
        original_state = clock.get_state()

        fresh = _make_clock()
        fresh.set_state(original_state)
        restored_state = fresh.get_state()

        for k in original_state:
            assert original_state[k] == restored_state[k], (
                f"Mismatch for key '{k}': {original_state[k]} != {restored_state[k]}"
            )


# ---------------------------------------------------------------------------
# game_time_scale read/write property
# ---------------------------------------------------------------------------

class TestGameTimeScaleProperty:
    def test_default_scale_is_constant(self):
        """Clock initialised without explicit scale uses DEFAULT_GAME_TIME_SCALE."""
        clock = _make_clock()
        assert clock.game_time_scale == DEFAULT_GAME_TIME_SCALE

    def test_scale_writable_at_runtime(self):
        """game_time_scale is writable and takes effect on the next update."""
        clock = _make_clock(scale=1.0)
        clock.update(3600.0)
        before = clock.game_time_of_day

        clock.game_time_scale = 2.0
        clock.update(3600.0)
        after = clock.game_time_of_day
        # Second update should advance by 2× more game seconds
        assert abs((after - before) - 2.0 * 3600.0) < 1e-6

    def test_scale_change_does_not_rewind_calendar(self):
        """Changing scale must not rewind or jump the existing calendar."""
        clock = _make_clock(scale=1.0)
        clock.update(7200.0)
        before_time = clock.game_time_of_day
        clock.game_time_scale = 100.0
        # game_time_of_day should be unchanged immediately after the assignment
        assert clock.game_time_of_day == before_time


# ---------------------------------------------------------------------------
# GameDayTickEvent via EventBus
# ---------------------------------------------------------------------------

class TestGameDayTickEvent:
    def test_day_tick_event_not_fired_before_rollover(self):
        """
        No GameDayTickEvent before a full game day elapses.
        """
        bus, clock = _make_bus_and_clock(scale=1.0)
        received = []
        bus.subscribe(GameDayTickEvent, received.append)

        clock.update(_GAME_SECONDS_PER_DAY - 1.0)  # just before rollover
        bus.drain()
        assert received == []

    def test_day_tick_event_published_deferred_on_rollover(self):
        """
        After a day rollover, GameDayTickEvent is in the deferred queue —
        it is NOT delivered until bus.drain() is called.

        Pin: Clock uses publish_deferred (not publish), so the event is only
        delivered after drain().
        """
        bus, clock = _make_bus_and_clock(scale=1.0)
        received: list[GameDayTickEvent] = []
        bus.subscribe(GameDayTickEvent, received.append)

        clock.update(_GAME_SECONDS_PER_DAY)  # exactly one day
        # Event queued but not yet delivered
        assert received == [], (
            "GameDayTickEvent should be deferred, not immediate — pin deferred behaviour"
        )
        bus.drain()
        assert len(received) == 1
        assert received[0].day == 1

    def test_day_tick_event_carries_new_day_number(self):
        """GameDayTickEvent.day reflects the new day number after rollover."""
        bus, clock = _make_bus_and_clock(scale=1.0)
        received: list[GameDayTickEvent] = []
        bus.subscribe(GameDayTickEvent, received.append)

        clock.update(_GAME_SECONDS_PER_DAY * 3)
        bus.drain()
        assert len(received) == 3
        days_in_order = [e.day for e in received]
        assert days_in_order == [1, 2, 3]

    def test_no_bus_no_error_on_rollover(self):
        """Clock without a bus rolls over without raising."""
        clock = _make_clock(scale=1.0, bus=None)
        clock.update(_GAME_SECONDS_PER_DAY * 2)  # two rollovers
        assert clock.game_day == 2  # calendar still advances

    def test_day_tick_event_day_matches_game_day(self):
        """The day in GameDayTickEvent matches clock.game_day after the drain."""
        bus, clock = _make_bus_and_clock(scale=1.0)
        received: list[GameDayTickEvent] = []
        bus.subscribe(GameDayTickEvent, received.append)

        clock.update(_GAME_SECONDS_PER_DAY + 60.0)
        bus.drain()
        assert len(received) == 1
        assert received[0].day == clock.game_day
