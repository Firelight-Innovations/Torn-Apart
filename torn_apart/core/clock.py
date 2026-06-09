"""
core/clock.py — Frame clock and fixed-step accumulator with game calendar.

The Clock drives two parallel time tracks:

1. **Real time** — the raw ``dt`` passed by the OS each frame (seconds).
2. **Game time** — an accelerated in-game calendar.  One real second equals
   ``game_time_scale`` in-game seconds (default: 1 real minute = 1 game hour,
   i.e. ``game_time_scale = 60``).  A full in-game day is 24 game hours =
   24 * 60 = 1440 real seconds ≈ 24 minutes.

The Clock also maintains a **fixed-step accumulator** for physics / AI ticks:
each call to ``update(real_dt)`` accumulates ``real_dt``; every call to
``fixed_steps()`` yields up to ``MAX_FIXED_STEPS`` fixed-timestep intervals
(spiral-of-death guard — see DEVELOPMENT_PLAN.md Known Traps).

Saveable state
--------------
``get_state()`` / ``set_state()`` return / accept plain dicts of primitives
(no live object references) for integration with the ``Saveable`` protocol.

Example
-------
    from torn_apart.core.clock import Clock
    from torn_apart.core.config import load_config
    from torn_apart.core.event_bus import EventBus

    cfg = load_config()
    bus = EventBus()
    clock = Clock(fixed_dt=cfg.fixed_dt, bus=bus)

    # Per-frame update
    clock.update(0.016)          # 16 ms frame

    # Drive fixed_update components
    for _ in clock.fixed_steps():
        physics_tick()           # called 0 or 1 times per frame at 50 Hz
"""

from __future__ import annotations

from typing import Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from torn_apart.core.event_bus import EventBus, GameDayTickEvent


# Spiral-of-death guard: never tick fixed_update more than this per frame.
MAX_FIXED_STEPS: int = 5

# Time scale: 1 real second = this many in-game seconds.
# At 60, one real minute = one in-game hour, one real day = 24 in-game days.
DEFAULT_GAME_TIME_SCALE: float = 60.0

# Seconds in one in-game day.
_GAME_SECONDS_PER_DAY: float = 24.0 * 3600.0


class Clock:
    """
    Frame clock, fixed-step accumulator, and game calendar.

    Parameters
    ----------
    fixed_dt        : float — fixed timestep in seconds (e.g. 0.02 for 50 Hz).
    bus             : EventBus | None — if provided, publishes GameDayTickEvent
                      whenever the in-game day counter increments.
    game_time_scale : float — real seconds per in-game second (default 60.0).

    Attributes (read-only)
    ----------------------
    dt              : float — last real frame duration in seconds.
    fixed_dt        : float — fixed timestep in seconds (from Config).
    game_day        : int   — current in-game day number (starts at 0).
    game_time_of_day: float — elapsed seconds within the current in-game day.
    total_real_time : float — total real seconds elapsed since boot / load.
    """

    def __init__(
        self,
        fixed_dt: float = 0.02,
        bus: "EventBus | None" = None,
        game_time_scale: float = DEFAULT_GAME_TIME_SCALE,
    ) -> None:
        self._fixed_dt:         float = float(fixed_dt)
        self._bus               = bus
        self._game_time_scale:  float = float(game_time_scale)

        self.dt:                float = 0.0
        self._accumulator:      float = 0.0
        self.total_real_time:   float = 0.0

        # Game calendar
        self.game_day:          int   = 0
        self.game_time_of_day:  float = 0.0  # seconds within current day

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def fixed_dt(self) -> float:
        """Fixed timestep in seconds (50 Hz = 0.02 s by default)."""
        return self._fixed_dt

    def update(self, real_dt: float) -> None:
        """
        Advance the clock by one real frame.

        Call this once per frame from the main loop, passing the OS-supplied
        frame duration.

        Parameters
        ----------
        real_dt : float — real elapsed time since the last frame, in **seconds**.
                          Typically 1/60 ≈ 0.0167 s at 60 fps.

        Side effects
        ------------
        - Updates ``dt``, ``total_real_time``, and the fixed-step accumulator.
        - Advances the in-game calendar; publishes ``GameDayTickEvent`` via the
          EventBus (if configured) whenever a new in-game day begins.
        """
        self.dt = float(real_dt)
        self.total_real_time += self.dt
        self._accumulator += self.dt

        # Advance game time
        game_dt = self.dt * self._game_time_scale
        self.game_time_of_day += game_dt

        # Roll over game days
        while self.game_time_of_day >= _GAME_SECONDS_PER_DAY:
            self.game_time_of_day -= _GAME_SECONDS_PER_DAY
            self.game_day += 1
            if self._bus is not None:
                # Import here to avoid a circular import at module level.
                from torn_apart.core.event_bus import GameDayTickEvent
                self._bus.publish_deferred(GameDayTickEvent(day=self.game_day))

    def fixed_steps(self) -> Iterator[float]:
        """
        Yield up to ``MAX_FIXED_STEPS`` fixed-timestep intervals accumulated
        since the last call.

        Each yielded value is ``fixed_dt`` (seconds).  Unconsumed remainder
        carries over to the next frame.

        Spiral-of-death guard: if the real frame is so slow that more than
        ``MAX_FIXED_STEPS`` intervals have accumulated, the excess is silently
        dropped rather than causing a cascade of simulation ticks.

        Yields
        ------
        float — fixed_dt (seconds) for each tick to process.

        Example
        -------
        >>> clock.update(0.05)          # 50 ms frame, fixed_dt = 0.02
        >>> list(clock.fixed_steps())   # → [0.02, 0.02]  (2 ticks, 0.01 s residual)
        """
        steps = 0
        while self._accumulator >= self._fixed_dt and steps < MAX_FIXED_STEPS:
            self._accumulator -= self._fixed_dt
            steps += 1
            yield self._fixed_dt

        # If spiral-of-death guard clamped, drop leftover (don't let it grow)
        if self._accumulator > self._fixed_dt * MAX_FIXED_STEPS:
            self._accumulator = 0.0

    # ------------------------------------------------------------------
    # Saveable protocol support
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """
        Return the clock state as a plain dict of primitives for serialisation.

        Compatible with the ``Saveable`` protocol (no live object references).

        Returns
        -------
        dict with keys:
            game_day        : int
            game_time_of_day: float  (seconds within current day)
            total_real_time : float  (seconds)
            accumulator     : float  (fixed-step residual, seconds)
        """
        return {
            "game_day":         self.game_day,
            "game_time_of_day": self.game_time_of_day,
            "total_real_time":  self.total_real_time,
            "accumulator":      self._accumulator,
        }

    def set_state(self, state: dict) -> None:
        """
        Restore clock state from a plain dict (inverse of ``get_state``).

        Called by SaveManager during a world load after the header is validated.

        Parameters
        ----------
        state : dict — as produced by ``get_state()``.
        """
        self.game_day         = int(state["game_day"])
        self.game_time_of_day = float(state["game_time_of_day"])
        self.total_real_time  = float(state["total_real_time"])
        self._accumulator     = float(state.get("accumulator", 0.0))
        self.dt               = 0.0
