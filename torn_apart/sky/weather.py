"""
sky/weather.py — Deterministic Markov-chain weather schedule + smooth blending.

Design
------
The in-game day is divided into **12 segments of 2 game hours** (7200 s).
Each segment has one discrete :class:`WeatherType`.  Segment states are
sampled from a per-state Markov transition table using
``core.rng.for_domain("weather", game_day, segment_index)``, so the entire
schedule is a **pure function of (world_seed, game_day, segment)** — fully
recomputable on every load, costing ~0 bytes in saves.

Day anchoring (recorded design decision)
----------------------------------------
To keep recomputation bounded (≤ 12 chain steps per queried day, instead of
an unbounded walk from day 0), each day's **segment 0** is drawn from a fixed
initial distribution (≈ the chain's stationary distribution) rather than from
the previous day's final state.  Segments 1–11 follow the Markov transition
table normally.  The midnight hand-off is still smoothed by the standard
20-game-minute parameter blend, so nothing pops visually; the only cost is
that the *discrete* state at 00:00 is not Markov-conditioned on 23:59.
This is the smallest decision that keeps "pure function of (seed, day,
segment)" literally true with O(1) cost per day.

Parameter blending
------------------
The continuous :class:`WeatherParams` at any instant are a pure function of
absolute game time: during the first ``BLEND_SECONDS`` (1200 s = 20 game
minutes) of each segment, params are smoothstep-lerped from the previous
segment's targets to the current segment's targets.  Because blending is
time-derived (not call-history-derived), two fresh instances with the same
seed and the same clock readings produce identical params — and
``get_delta()`` is ``{}`` in the natural case.

Dev override (``force_weather``) is the only runtime state that must be
saved; it blends in and out over the same 20-game-minute window.

Example
-------
    from torn_apart.core import EventBus, load_config, set_world_seed
    from torn_apart.sky.weather import WeatherSystem, WeatherType

    set_world_seed(1337)
    ws = WeatherSystem(load_config(), EventBus())
    params = ws.update(game_day=3, game_time_of_day=8.5 * 3600.0)
    print(ws.current, params.cloud_coverage, params.rain_intensity)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from torn_apart.core.config import Config
from torn_apart.core.event_bus import EventBus, WeatherChangedEvent
from torn_apart.core.rng import for_domain
from torn_apart.sky.celestial import GAME_SECONDS_PER_DAY, smoothstep

__all__ = [
    "WeatherType",
    "WeatherParams",
    "WeatherSystem",
    "SEGMENT_SECONDS",
    "SEGMENTS_PER_DAY",
    "BLEND_SECONDS",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: One weather segment = 2 game hours (seconds).
SEGMENT_SECONDS: float = 2.0 * 3600.0

#: 12 segments per 24-hour game day.
SEGMENTS_PER_DAY: int = int(GAME_SECONDS_PER_DAY // SEGMENT_SECONDS)

#: Parameter blend window at every transition: 20 game minutes (seconds).
BLEND_SECONDS: float = 20.0 * 60.0


class WeatherType(str, Enum):
    """
    Discrete weather states.  ``str`` mixin so ``.value`` round-trips through
    saves and :class:`WeatherChangedEvent` payloads as plain strings.
    """

    CLEAR = "clear"
    CLOUDY = "cloudy"
    OVERCAST = "overcast"
    FOG = "fog"
    RAIN = "rain"
    STORM = "storm"


#: Stable index order for the transition matrix rows/columns.
_STATE_ORDER: tuple[WeatherType, ...] = (
    WeatherType.CLEAR,
    WeatherType.CLOUDY,
    WeatherType.OVERCAST,
    WeatherType.FOG,
    WeatherType.RAIN,
    WeatherType.STORM,
)

#: Per-state parameter targets (documented tuning values):
#: (cloud_coverage 0–1, cloud_density 0–1, fog_density 1/m, rain 0–1,
#:  base wind_speed m/s).
#: fog_density is the exponential fog coefficient: ~0.0008 = crisp clear air,
#: ~0.025 = thick FOG weather (visibility ≈ 3/0.025 = 120 m).
_STATE_TARGETS: dict[WeatherType, tuple[float, float, float, float, float]] = {
    WeatherType.CLEAR:    (0.12, 0.35, 0.0008, 0.0,  2.5),
    WeatherType.CLOUDY:   (0.45, 0.55, 0.0012, 0.0,  3.5),
    WeatherType.OVERCAST: (0.85, 0.80, 0.0030, 0.0,  4.5),
    WeatherType.FOG:      (0.55, 0.50, 0.0250, 0.0,  0.8),   # low wind
    WeatherType.RAIN:     (0.90, 0.85, 0.0060, 0.7,  6.5),
    WeatherType.STORM:    (0.98, 0.95, 0.0080, 1.0, 12.0),   # high wind
}

#: Markov transition matrix — row = from-state, column = to-state, in
#: ``_STATE_ORDER`` order.  Rows sum to 1.  Tuning intent:
#: CLEAR/CLOUDY dominate; OVERCAST is the gateway to RAIN; STORM is rare and
#: (almost) only entered from RAIN; FOG mostly returns to CLEAR/CLOUDY.
_TRANSITIONS: np.ndarray = np.array([
    #  CLEAR CLOUDY OVERC  FOG   RAIN  STORM
    [0.60, 0.28, 0.04, 0.06, 0.02, 0.00],   # from CLEAR
    [0.30, 0.40, 0.18, 0.05, 0.07, 0.00],   # from CLOUDY
    [0.08, 0.30, 0.38, 0.04, 0.18, 0.02],   # from OVERCAST
    [0.30, 0.25, 0.15, 0.28, 0.02, 0.00],   # from FOG
    [0.05, 0.20, 0.25, 0.02, 0.40, 0.08],   # from RAIN
    [0.02, 0.05, 0.18, 0.00, 0.45, 0.30],   # from STORM
], dtype=np.float64)

#: Initial distribution for each day's segment 0 (≈ stationary distribution,
#: CLEAR-heavy).  Sums to 1.
_INITIAL_DIST: np.ndarray = np.array(
    [0.45, 0.30, 0.10, 0.08, 0.06, 0.01], dtype=np.float64
)

#: Segments whose state is sampled with a FOG bias: segment 2 = 04:00–06:00,
#: 3 = 06:00–08:00, 4 = 08:00–10:00 (early-morning ground fog).
_FOG_BIAS_SEGMENTS: frozenset[int] = frozenset({2, 3, 4})

#: FOG-column probability multiplier in biased segments (row re-normalised).
_FOG_BIAS_FACTOR: float = 3.0

_FOG_INDEX: int = _STATE_ORDER.index(WeatherType.FOG)


# ---------------------------------------------------------------------------
# WeatherParams
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WeatherParams:
    """
    Continuous, already-blended weather parameters for one instant.

    Same units and meaning as the matching fields of ``SkyState``:

    Attributes
    ----------
    cloud_coverage : float — 0–1 fraction of cloud cells filled.
    cloud_density : float — 0–1 cloud opacity/darkness.
    fog_density : float — exponential fog coefficient in 1/m (0 = none;
        typical FOG weather ≈ 0.025).
    rain_intensity : float — 0–1 (0 = dry, 1 = torrential).
    wind_dir : tuple[float, float] — unit XY direction the wind blows TOWARD.
    wind_speed : float — m/s.

    Example
    -------
    >>> p = WeatherParams(0.12, 0.35, 0.0008, 0.0, (1.0, 0.0), 2.5)
    >>> p.cloud_coverage
    0.12
    """

    cloud_coverage: float
    cloud_density: float
    fog_density: float
    rain_intensity: float
    wind_dir: tuple[float, float]
    wind_speed: float


def _lerp_params(a: WeatherParams, b: WeatherParams, t: float) -> WeatherParams:
    """
    Component-wise lerp between two :class:`WeatherParams` (``t`` in [0, 1]).

    ``wind_dir`` is lerped then re-normalised to unit length; if the two
    directions are nearly opposite (degenerate lerp), *b*'s direction wins.
    Endpoints short-circuit (``t <= 0 → a``, ``t >= 1 → b``) so a completed
    blend is **bit-exact** equal to its target — required for the Saveable
    "delta returns to {}" guarantee to also mean "params return to the pure
    natural schedule".
    """
    if t <= 0.0:
        return a
    if t >= 1.0:
        return b
    wx = a.wind_dir[0] + (b.wind_dir[0] - a.wind_dir[0]) * t
    wy = a.wind_dir[1] + (b.wind_dir[1] - a.wind_dir[1]) * t
    norm = math.hypot(wx, wy)
    if norm < 1e-6:
        wind_dir = b.wind_dir
    else:
        wind_dir = (wx / norm, wy / norm)
    return WeatherParams(
        cloud_coverage=a.cloud_coverage + (b.cloud_coverage - a.cloud_coverage) * t,
        cloud_density=a.cloud_density + (b.cloud_density - a.cloud_density) * t,
        fog_density=a.fog_density + (b.fog_density - a.fog_density) * t,
        rain_intensity=a.rain_intensity + (b.rain_intensity - a.rain_intensity) * t,
        wind_dir=wind_dir,
        wind_speed=a.wind_speed + (b.wind_speed - a.wind_speed) * t,
    )


def _params_to_dict(p: WeatherParams) -> dict:
    """Serialise a WeatherParams to a plain dict of primitives (Saveable)."""
    return {
        "cloud_coverage": float(p.cloud_coverage),
        "cloud_density": float(p.cloud_density),
        "fog_density": float(p.fog_density),
        "rain_intensity": float(p.rain_intensity),
        "wind_dir": [float(p.wind_dir[0]), float(p.wind_dir[1])],
        "wind_speed": float(p.wind_speed),
    }


def _params_from_dict(d: dict) -> WeatherParams:
    """Inverse of :func:`_params_to_dict`."""
    return WeatherParams(
        cloud_coverage=float(d["cloud_coverage"]),
        cloud_density=float(d["cloud_density"]),
        fog_density=float(d["fog_density"]),
        rain_intensity=float(d["rain_intensity"]),
        wind_dir=(float(d["wind_dir"][0]), float(d["wind_dir"][1])),
        wind_speed=float(d["wind_speed"]),
    )


# ---------------------------------------------------------------------------
# WeatherSystem
# ---------------------------------------------------------------------------

class WeatherSystem:
    """
    Deterministic weather schedule + smooth parameter blending.  Implements
    the ``Saveable`` protocol (``save_key = "weather"``).

    The natural schedule and blended params are **pure functions of the world
    seed and game time** — the only saveable runtime state is an active dev
    override (``force_weather``) or its release blend.  ``get_delta()``
    therefore returns ``{}`` in the un-forced case (~0-byte baseline, the
    engine's save philosophy).

    Parameters
    ----------
    config : Config — engine configuration (reserved for future weather
        tuning fields; cloud geometry fields are consumed by the renderer).
    bus : EventBus | None — if provided, a :class:`WeatherChangedEvent` is
        published via ``bus.publish_deferred`` whenever the discrete state
        changes (state-change notification — never per-frame data).

    Example
    -------
    >>> from torn_apart.core import EventBus, load_config, set_world_seed
    >>> set_world_seed(1337)
    >>> ws = WeatherSystem(load_config(), EventBus())
    >>> p = ws.update(game_day=0, game_time_of_day=3600.0)
    >>> 0.0 <= p.cloud_coverage <= 1.0
    True
    >>> ws.get_delta()
    {}
    """

    save_key: str = "weather"

    def __init__(self, config: Config, bus: EventBus | None = None) -> None:
        self._config = config
        self._bus = bus

        # Memoised discrete schedule: (day, segment) → WeatherType.
        self._sched_cache: dict[tuple[int, int], WeatherType] = {}

        # Runtime tracking (NOT part of the pure schedule).
        self._last_state: WeatherType | None = None
        self._last_params: WeatherParams | None = None
        self._last_abs_t: float | None = None

        # Dev override state (the only saveable deviation).
        self._override: WeatherType | None = None
        self._override_start_abs_t: float | None = None     # anchored lazily
        self._override_from: WeatherParams | None = None
        # Release blend back to the natural schedule after clearing override.
        self._release_start_abs_t: float | None = None
        self._release_from: WeatherParams | None = None

    # ------------------------------------------------------------------
    # Discrete schedule — pure function of (world_seed, day, segment)
    # ------------------------------------------------------------------

    def _scheduled_state(self, day: int, segment: int) -> WeatherType:
        """
        Discrete weather state for (day, segment) — memoised pure function.

        Segment 0 is drawn from ``_INITIAL_DIST``; segments 1–11 from the
        Markov row of the previous segment's state.  Both draws use
        ``for_domain("weather", day, segment)``, so any (seed, day, segment)
        triple always yields the same state across processes and restarts.
        """
        segment = int(segment) % SEGMENTS_PER_DAY
        day = int(day)
        key = (day, segment)
        cached = self._sched_cache.get(key)
        if cached is not None:
            return cached

        # Walk forward from segment 0 of this day (bounded: ≤ 12 steps).
        for seg in range(segment + 1):
            k = (day, seg)
            if k in self._sched_cache:
                continue
            rng = for_domain("weather", day, seg)
            if seg == 0:
                probs = _INITIAL_DIST
            else:
                prev = self._sched_cache[(day, seg - 1)]
                probs = _TRANSITIONS[_STATE_ORDER.index(prev)]
            if seg in _FOG_BIAS_SEGMENTS:
                probs = probs.copy()
                probs[_FOG_INDEX] *= _FOG_BIAS_FACTOR
                probs = probs / probs.sum()
            idx = int(rng.choice(len(_STATE_ORDER), p=probs))
            self._sched_cache[k] = _STATE_ORDER[idx]

        return self._sched_cache[key]

    def _wind_for_day(self, day: int) -> tuple[tuple[float, float], float]:
        """
        Per-day wind: (unit XY direction, speed jitter multiplier).

        Direction is uniform on the circle from ``for_domain("weather",
        "wind", day)``; the jitter multiplier (0.85–1.15) scales each state's
        base wind speed so days feel different even in the same weather.
        """
        rng = for_domain("weather", "wind", int(day))
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        jitter = float(rng.uniform(0.85, 1.15))
        return (math.cos(angle), math.sin(angle)), jitter

    def _target_params(self, state: WeatherType, day: int) -> WeatherParams:
        """Raw (un-blended) parameter targets for *state* on *day*."""
        cov, den, fog, rain, base_wind = _STATE_TARGETS[state]
        wind_dir, jitter = self._wind_for_day(day)
        return WeatherParams(
            cloud_coverage=cov,
            cloud_density=den,
            fog_density=fog,
            rain_intensity=rain,
            wind_dir=wind_dir,
            wind_speed=base_wind * jitter,
        )

    def _natural_params(self, day: int, tod: float) -> WeatherParams:
        """
        Blended natural-schedule params at (day, time_of_day) — pure function.

        During the first ``BLEND_SECONDS`` of each segment the params are
        smoothstep-lerped from the previous segment's targets (previous day's
        segment 11 across midnight, carrying that day's wind) to the current
        segment's targets; afterwards they sit at the current targets.
        """
        seg = min(int(tod // SEGMENT_SECONDS), SEGMENTS_PER_DAY - 1)
        cur_state = self._scheduled_state(day, seg)
        cur = self._target_params(cur_state, day)

        if seg == 0:
            prev_day, prev_seg = day - 1, SEGMENTS_PER_DAY - 1
        else:
            prev_day, prev_seg = day, seg - 1
        if prev_day < 0:
            return cur     # world start: no previous segment to blend from
        prev_state = self._scheduled_state(prev_day, prev_seg)
        prev = self._target_params(prev_state, prev_day)

        t_in_seg = tod - seg * SEGMENT_SECONDS
        blend = smoothstep(t_in_seg, 0.0, BLEND_SECONDS)
        return _lerp_params(prev, cur, blend)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current(self) -> WeatherType:
        """
        The discrete weather state as of the last ``update()`` call (the dev
        override, if one is active).  Before the first update, returns the
        scheduled state for day 0 segment 0.
        """
        if self._override is not None:
            return self._override
        if self._last_state is not None:
            return self._last_state
        return self._scheduled_state(0, 0)

    def update(self, game_day: int, game_time_of_day: float) -> WeatherParams:
        """
        Compute the blended weather parameters for the given game time.

        Call once per frame with ``clock.game_day`` / ``clock.game_time_of_day``.
        Publishes a :class:`WeatherChangedEvent` (deferred) when the discrete
        state differs from the previous call's state.

        Parameters
        ----------
        game_day : int — in-game day number (``clock.game_day``).
        game_time_of_day : float — seconds within the day, [0, 86400)
            (``clock.game_time_of_day``).

        Returns
        -------
        WeatherParams — smoothly blended parameters (no popping: at most a
        20-game-minute crossfade at any transition, including overrides).
        """
        day = int(game_day)
        tod = float(game_time_of_day) % GAME_SECONDS_PER_DAY
        abs_t = day * GAME_SECONDS_PER_DAY + tod
        natural = self._natural_params(day, tod)

        if self._override is not None:
            # Anchor the override blend at the first update after forcing.
            if self._override_start_abs_t is None:
                self._override_start_abs_t = abs_t
                self._override_from = (
                    self._last_params if self._last_params is not None else natural
                )
            target = self._target_params(self._override, day)
            bt = smoothstep(
                abs_t - self._override_start_abs_t, 0.0, BLEND_SECONDS
            )
            params = _lerp_params(self._override_from, target, bt)
            new_state = self._override
        elif self._release_from is not None:
            # Blending back from a cleared override to the natural schedule.
            if self._release_start_abs_t is None:
                self._release_start_abs_t = abs_t
            bt = smoothstep(
                abs_t - self._release_start_abs_t, 0.0, BLEND_SECONDS
            )
            params = _lerp_params(self._release_from, natural, bt)
            if bt >= 1.0:
                # Release complete — back to the ~0-byte natural baseline.
                self._release_from = None
                self._release_start_abs_t = None
            seg = min(int(tod // SEGMENT_SECONDS), SEGMENTS_PER_DAY - 1)
            new_state = self._scheduled_state(day, seg)
        else:
            params = natural
            seg = min(int(tod // SEGMENT_SECONDS), SEGMENTS_PER_DAY - 1)
            new_state = self._scheduled_state(day, seg)

        # State-change notification (never per-frame: at most one per
        # segment boundary / override toggle).
        if (
            self._last_state is not None
            and new_state is not self._last_state
            and self._bus is not None
        ):
            self._bus.publish_deferred(
                WeatherChangedEvent(
                    previous=self._last_state.value,
                    current=new_state.value,
                    day=day,
                )
            )
        self._last_state = new_state
        self._last_params = params
        self._last_abs_t = abs_t
        return params

    def force_weather(self, weather: WeatherType | None) -> None:
        """
        Dev override: pin the weather to *weather*, or pass ``None`` to clear
        the override and return to the natural schedule.

        Both forcing and clearing blend over 20 game minutes starting from
        the next ``update()`` call — no popping.  While an override (or its
        release blend) is active, ``get_delta()`` returns a small snapshot so
        the override survives save/load.

        Parameters
        ----------
        weather : WeatherType | None
            Target state, e.g. ``WeatherType.STORM`` — or ``None`` to clear.

        Example
        -------
        >>> ws.force_weather(WeatherType.STORM)   # blend toward storm
        >>> ws.force_weather(None)                # blend back to schedule
        """
        if weather is not None:
            self._override = WeatherType(weather)
            self._override_start_abs_t = None       # anchored at next update
            self._override_from = None
            self._release_from = None
            self._release_start_abs_t = None
        elif self._override is not None:
            # Begin the release blend from whatever is currently on screen.
            self._release_from = self._last_params
            self._release_start_abs_t = None         # anchored at next update
            self._override = None
            self._override_start_abs_t = None
            self._override_from = None

    # ------------------------------------------------------------------
    # Saveable protocol
    # ------------------------------------------------------------------

    def get_delta(self) -> dict:
        """
        Deviations from the procedural baseline (Saveable protocol).

        The natural schedule is recomputed from the world seed, so the
        baseline costs 0 bytes: returns ``{}`` unless a dev override is
        active or a mid-release blend is in flight.  All values are plain
        primitives (strings, floats, lists) — no live objects, no pickle.
        """
        if self._override is not None:
            delta: dict = {"override": self._override.value}
            if self._override_start_abs_t is not None:
                delta["override_start_abs_t"] = float(self._override_start_abs_t)
            if self._override_from is not None:
                delta["override_from"] = _params_to_dict(self._override_from)
            if self._last_state is not None:
                delta["last_state"] = self._last_state.value
            return delta
        if self._release_from is not None:
            delta = {"release_from": _params_to_dict(self._release_from)}
            if self._release_start_abs_t is not None:
                delta["release_start_abs_t"] = float(self._release_start_abs_t)
            if self._last_state is not None:
                delta["last_state"] = self._last_state.value
            return delta
        return {}

    def apply_delta(self, delta: dict) -> None:
        """
        Restore override/release state from :meth:`get_delta` output.

        Called by SaveManager after the natural baseline (which needs no
        restoration — it is recomputed from the seed) is in place.  After
        this call, subsequent ``update()`` behaviour is identical to the
        instance that produced the delta.
        """
        if not delta:
            return
        if "override" in delta:
            self._override = WeatherType(delta["override"])
            self._override_start_abs_t = (
                float(delta["override_start_abs_t"])
                if "override_start_abs_t" in delta else None
            )
            self._override_from = (
                _params_from_dict(delta["override_from"])
                if "override_from" in delta else None
            )
        elif "release_from" in delta:
            self._release_from = _params_from_dict(delta["release_from"])
            self._release_start_abs_t = (
                float(delta["release_start_abs_t"])
                if "release_start_abs_t" in delta else None
            )
        if "last_state" in delta:
            self._last_state = WeatherType(delta["last_state"])
