"""
weather/system.py — Spatial weather system: local sampling + dev override.

This replaces the old global Markov schedule (``sky/weather.py``) with a
**spatial** model: weather at any instant is sampled *at a world position*
from the day's ambient regime plus every active :class:`StormCell` whose
drifting footprint overlaps that point (see :mod:`fire_engine.weather.cells`).
Stand under a passing shower and it rains; step out from under it and it
doesn't — the same storm is dry a kilometer away.

Everything natural is a closed-form pure function of (world_seed, game time,
position): ``get_delta()`` is ``{}`` for natural weather (zero save bytes).
The only saveable runtime state is the dev override (``force_weather``) — kept
as a compatibility shim so devtools, the F6 cycle, and existing saves keep
working while the spatial summon API (M8) is built on top.

The system hands the rest of the game a single :class:`LocalWeather` snapshot
per frame (the player's local sample) and a :class:`WeatherType` label
(:func:`classify`, with a short hysteresis so the label never flickers).

Units: meters, m/s, game seconds, °C.

Example
-------
    from fire_engine.core import EventBus, load_config, set_world_seed
    from fire_engine.weather import WeatherSystem, WeatherType

    set_world_seed(1337)
    ws = WeatherSystem(load_config(), EventBus())
    lw = ws.update(game_day=3, game_time_of_day=8.5 * 3600.0, player_pos=(0.0, 0.0))
    print(ws.current, lw.cloud_coverage, lw.rain_intensity)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fire_engine.core.config import Config
from fire_engine.core.event_bus import EventBus, WeatherChangedEvent
from fire_engine.weather.cells import (
    CellKind,
    StormCell,
    day_regime,
    natural_cells,
    regime_ambient,
)
from fire_engine.weather.classify import WeatherType, classify
from fire_engine.weather.synoptic import Synoptic

__all__ = ["LocalWeather", "WeatherSystem", "BLEND_SECONDS", "HYSTERESIS_SECONDS"]

#: Game seconds per game day (see ``cells._DAY_S`` — kept local, no sky import).
_DAY_S: float = 24.0 * 3600.0

#: Override blend window: 20 game minutes (seconds).  Forcing/clearing the dev
#: override crossfades over this window so nothing pops.
BLEND_SECONDS: float = 20.0 * 60.0

#: Classification hysteresis (game seconds): a new :func:`classify` label must
#: persist this long before it becomes ``current`` and fires an event — keeps
#: the discrete label from flickering at a threshold boundary.
HYSTERESIS_SECONDS: float = 60.0

#: Clear-air baseline fog coefficient (1/m) — always present, even on a clear
#: day, so distant terrain has a faint aerial perspective.
_FOG_BASELINE: float = 0.0008

#: Per-kind weights: how much a cell at full contribution adds to each channel.
_KIND_COV: dict[CellKind, float] = {
    CellKind.CLOUD_BANK: 0.90,
    CellKind.SHOWER: 0.85,
    CellKind.THUNDERSTORM: 1.00,
    CellKind.FOG_BANK: 0.00,
}
_KIND_DEN: dict[CellKind, float] = {
    CellKind.CLOUD_BANK: 0.60,
    CellKind.SHOWER: 0.80,
    CellKind.THUNDERSTORM: 0.95,
    CellKind.FOG_BANK: 0.00,
}
_KIND_RAIN: dict[CellKind, float] = {
    CellKind.CLOUD_BANK: 0.00,
    CellKind.SHOWER: 0.65,
    CellKind.THUNDERSTORM: 1.00,
    CellKind.FOG_BANK: 0.00,
}
#: Fog coefficient a FOG_BANK adds at full contribution (1/m).
_FOG_BANK_GAIN: float = 0.027

#: Per-state targets for the dev override shim (unchanged from the legacy
#: Markov system so forced-weather screenshots/tests are bit-identical):
#: (cloud_coverage, cloud_density, fog_density 1/m, rain 0–1, synoptic-speed
#: multiplier).  Wind direction/base speed come from the synoptic flow.
_STATE_TARGETS: dict[WeatherType, tuple[float, float, float, float, float]] = {
    WeatherType.CLEAR:    (0.12, 0.35, 0.0008, 0.0, 0.70),
    WeatherType.CLOUDY:   (0.45, 0.55, 0.0012, 0.0, 0.90),
    WeatherType.OVERCAST: (0.85, 0.80, 0.0030, 0.0, 1.00),
    WeatherType.FOG:      (0.55, 0.50, 0.0250, 0.0, 0.30),
    WeatherType.RAIN:     (0.90, 0.85, 0.0060, 0.7, 1.25),
    WeatherType.STORM:    (0.98, 0.95, 0.0080, 1.0, 1.90),
}


def _smoothstep(x: float, lo: float, hi: float) -> float:
    """Hermite smoothstep clamped to [0, 1]."""
    if hi <= lo:
        return 0.0 if x < lo else 1.0
    t = (x - lo) / (hi - lo)
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return t * t * (3.0 - 2.0 * t)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


# ---------------------------------------------------------------------------
# LocalWeather
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LocalWeather:
    """
    Continuous weather sampled at one world position and instant.

    The first six fields match the corresponding ``SkyState`` fields exactly
    (same names, units, meaning) so the sky composer fills ``SkyState``
    one-to-one.  ``humidity``/``wetness`` are placeholders until the emergent
    humidity model (M5/M6) fills them; ``temperature_c`` is already live.

    Attributes
    ----------
    cloud_coverage : float — 0–1 fraction of sky filled.
    cloud_density : float — 0–1 cloud opacity/darkness.
    fog_density : float — exponential fog coefficient, 1/m (0.0008 ≈ clear).
    rain_intensity : float — 0–1 (0 = dry, 1 = torrential).
    wind_dir : tuple[float, float] — unit XY direction the wind blows TOWARD.
    wind_speed : float — m/s.
    humidity : float — 0–1 relative humidity (placeholder 0.5 until M5).
    wetness : float — 0–1 ground wetness (placeholder 0.0 until M6).
    temperature_c : float — local air temperature, °C.

    Example
    -------
    >>> lw = LocalWeather(0.2, 0.4, 0.0008, 0.0, (1.0, 0.0), 3.0)
    >>> lw.temperature_c
    12.0
    """

    cloud_coverage: float
    cloud_density: float
    fog_density: float
    rain_intensity: float
    wind_dir: tuple[float, float]
    wind_speed: float
    humidity: float = 0.5
    wetness: float = 0.0
    temperature_c: float = 12.0


def _lerp_local(a: LocalWeather, b: LocalWeather, t: float) -> LocalWeather:
    """
    Component-wise lerp between two :class:`LocalWeather` (``t`` in [0, 1]).

    ``wind_dir`` is lerped then renormalised; endpoints short-circuit
    (``t<=0 → a``, ``t>=1 → b``) so a completed blend is **bit-exact** equal to
    its target — required for the "delta returns to {}" save guarantee to also
    mean "params return to the pure natural sample".
    """
    if t <= 0.0:
        return a
    if t >= 1.0:
        return b
    wx = a.wind_dir[0] + (b.wind_dir[0] - a.wind_dir[0]) * t
    wy = a.wind_dir[1] + (b.wind_dir[1] - a.wind_dir[1]) * t
    norm = math.hypot(wx, wy)
    wind_dir = (wx / norm, wy / norm) if norm > 1e-6 else b.wind_dir

    def mix(x: float, y: float) -> float:
        return x + (y - x) * t

    return LocalWeather(
        cloud_coverage=mix(a.cloud_coverage, b.cloud_coverage),
        cloud_density=mix(a.cloud_density, b.cloud_density),
        fog_density=mix(a.fog_density, b.fog_density),
        rain_intensity=mix(a.rain_intensity, b.rain_intensity),
        wind_dir=wind_dir,
        wind_speed=mix(a.wind_speed, b.wind_speed),
        humidity=mix(a.humidity, b.humidity),
        wetness=mix(a.wetness, b.wetness),
        temperature_c=mix(a.temperature_c, b.temperature_c),
    )


def _local_to_dict(lw: LocalWeather) -> dict:
    """Serialise a LocalWeather to plain primitives (Saveable)."""
    return {
        "cloud_coverage": float(lw.cloud_coverage),
        "cloud_density": float(lw.cloud_density),
        "fog_density": float(lw.fog_density),
        "rain_intensity": float(lw.rain_intensity),
        "wind_dir": [float(lw.wind_dir[0]), float(lw.wind_dir[1])],
        "wind_speed": float(lw.wind_speed),
        "humidity": float(lw.humidity),
        "wetness": float(lw.wetness),
        "temperature_c": float(lw.temperature_c),
    }


def _local_from_dict(d: dict) -> LocalWeather:
    """Inverse of :func:`_local_to_dict`."""
    return LocalWeather(
        cloud_coverage=float(d["cloud_coverage"]),
        cloud_density=float(d["cloud_density"]),
        fog_density=float(d["fog_density"]),
        rain_intensity=float(d["rain_intensity"]),
        wind_dir=(float(d["wind_dir"][0]), float(d["wind_dir"][1])),
        wind_speed=float(d["wind_speed"]),
        humidity=float(d.get("humidity", 0.5)),
        wetness=float(d.get("wetness", 0.0)),
        temperature_c=float(d.get("temperature_c", 12.0)),
    )


# ---------------------------------------------------------------------------
# WeatherSystem
# ---------------------------------------------------------------------------

class WeatherSystem:
    """
    Spatial weather: local sampling, discrete classification, dev override.
    Implements the ``Saveable`` protocol (``save_key = "weather"``).

    The natural model is a pure function of (world_seed, game time, position),
    so the saved baseline costs ~0 bytes — ``get_delta()`` returns ``{}``
    unless a dev override (or its release blend) is active.

    Parameters
    ----------
    config : Config — engine configuration (``weather_*`` tuning fields).
    bus : EventBus | None — if given, a :class:`WeatherChangedEvent` is
        published (deferred) whenever the discrete label changes.

    Example
    -------
    >>> from fire_engine.core import EventBus, load_config, set_world_seed
    >>> set_world_seed(1337)
    >>> ws = WeatherSystem(load_config(), EventBus())
    >>> lw = ws.update(0, 3600.0, player_pos=(0.0, 0.0))
    >>> 0.0 <= lw.cloud_coverage <= 1.0
    True
    >>> ws.get_delta()
    {}
    """

    save_key: str = "weather"

    def __init__(self, config: Config, bus: EventBus | None = None) -> None:
        self._config = config
        self._bus = bus

        #: Closed-form synoptic flow shared by the cell tracks and the local
        #: wind.  Pure function of (world_seed, game time).
        self.synoptic: Synoptic = Synoptic(config)

        self._temp_mean = float(config.weather_temp_mean_c)
        self._temp_amp = float(config.weather_temp_amp_c)
        self._fog_max = float(config.weather_fog_max_density)
        self._storm_wind_max = float(config.weather_storm_wind_max_ms)
        # Closed-form ground-wetness quadrature over the analytic rain history.
        self._wet_tau = float(config.weather_wetness_tau_s)
        self._wet_step = float(config.weather_wetness_step_s)
        self._wet_samples = int(config.weather_wetness_samples)

        # Per-day natural cell cache (pure fn of seed+day; never saved).
        self._cell_cache: dict[int, list[StormCell]] = {}
        # Summoned cells (M8) — saveable; empty for now.
        self._summoned: list[StormCell] = []

        # Active cells at the last update, sorted nearest-first to the player.
        self._cells: list[StormCell] = []

        # Runtime tracking (not part of the pure model).
        self._last_local: LocalWeather | None = None
        self._last_state: WeatherType | None = None
        self._last_abs_t: float | None = None
        self._last_player: tuple[float, float] = (0.0, 0.0)

        # Classification hysteresis.
        self._committed_state: WeatherType | None = None
        self._pending_state: WeatherType | None = None
        self._pending_since: float = 0.0

        # Dev override (the only saveable deviation).
        self._override: WeatherType | None = None
        self._override_start_abs_t: float | None = None
        self._override_from: LocalWeather | None = None
        self._release_from: LocalWeather | None = None
        self._release_start_abs_t: float | None = None

    # ------------------------------------------------------------------
    # Cells
    # ------------------------------------------------------------------

    def _cells_for_day(self, day: int) -> list[StormCell]:
        """Memoised natural cells of *day* (pure fn of seed+day)."""
        day = int(day)
        if day < 0:
            return []
        cached = self._cell_cache.get(day)
        if cached is None:
            cached = natural_cells(day, self._config)
            self._cell_cache[day] = cached
        return cached

    def _active_cells(self, t: float) -> list[StormCell]:
        """All cells (natural ∪ summoned) alive at absolute time *t*."""
        day = int(t // _DAY_S)
        pool = self._cells_for_day(day - 1) + self._cells_for_day(day)
        out = [c for c in pool if c.active(t)]
        out.extend(c for c in self._summoned if c.active(t))
        return out

    @property
    def cells(self) -> list[StormCell]:
        """Active cells as of the last :meth:`update`, nearest player first."""
        return self._cells

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _temperature(self, tod_h: float) -> float:
        """Local air temperature (°C): daily cosine peaking at 15:00."""
        return self._temp_mean + self._temp_amp * math.cos(
            2.0 * math.pi * (tod_h - 15.0) / 24.0
        )

    def sample_fields(
        self, points_xy: np.ndarray, t_abs: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorised core sampling: the raster channels at every query point.

        This is the single source of truth for the spatial weather field —
        :meth:`sample_local` calls it with one point and the weather-map raster
        (M3) calls it over a whole grid, so a texel's rasterised value equals
        ``sample_local`` at that texel center by construction.

        Composition: the day regime sets the ambient cloud cover/density
        (cosine-blended from the previous day's regime over the first game hour
        after midnight); every active cell adds its footprint contribution to
        coverage/density/rain, FOG_BANKs to fog, THUNDERSTORMs to a core gust.

        Parameters
        ----------
        points_xy : np.ndarray — shape ``(N, 2)`` world-XY query points (m).
        t_abs : float — absolute game seconds.

        Returns
        -------
        tuple of five ``(N,)`` arrays — ``(coverage, density, rain, fog,
        storm_gust)``; the first four are the raster channels (coverage,
        density, rain, fog clamped/capped), the last feeds local wind speed.
        """
        pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        n = pts.shape[0]
        t = float(t_abs)
        day = int(t // _DAY_S)
        tod = t - day * _DAY_S

        # Ambient regime, cosine-blended across the midnight hand-off so the
        # base sky never snaps at 00:00.
        cov_cur, den_cur = regime_ambient(day_regime(day))
        if day > 0:
            cov_prev, den_prev = regime_ambient(day_regime(day - 1))
        else:
            cov_prev, den_prev = cov_cur, den_cur
        blend = 0.5 - 0.5 * math.cos(math.pi * min(tod / 3600.0, 1.0))
        coverage = np.full(n, cov_prev + (cov_cur - cov_prev) * blend)
        density = np.full(n, den_prev + (den_cur - den_prev) * blend)
        rain = np.zeros(n)
        fog_extra = np.zeros(n)
        storm_gust = np.zeros(n)

        for cell in self._active_cells(t):
            c = cell.contribution(pts, t, self.synoptic)          # (N,)
            coverage += c * _KIND_COV[cell.kind]
            density += c * _KIND_DEN[cell.kind]
            rain += c * _KIND_RAIN[cell.kind]
            if cell.kind is CellKind.FOG_BANK:
                fog_extra += c
            elif cell.kind is CellKind.THUNDERSTORM:
                storm_gust += c

        np.clip(coverage, 0.0, 1.0, out=coverage)
        np.clip(density, 0.0, 1.0, out=density)
        np.clip(rain, 0.0, 1.0, out=rain)
        fog = np.minimum(_FOG_BASELINE + fog_extra * _FOG_BANK_GAIN, self._fog_max)
        return coverage, density, rain, fog, storm_gust

    def wetness_at(self, points_xy: np.ndarray, t_abs: float) -> np.ndarray:
        """
        Closed-form ground wetness 0–1 at each query point.

        Fixed-offset quadrature over the **analytic rain history** at each fixed
        world point: the rain that fell there over the recent past, exponentially
        decayed (recent rain weighs most).  No integrated state — wetness is a
        pure function of (seed, time, position), so it recomputes for free on
        load, like everything else here.

        Parameters
        ----------
        points_xy : np.ndarray — shape ``(N, 2)`` world-XY query points (m).
        t_abs : float — absolute game seconds.

        Returns
        -------
        np.ndarray — shape ``(N,)``, ground wetness clamped to [0, 1].
        """
        pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        t = float(t_abs)
        acc = np.zeros(pts.shape[0])
        for k in range(1, self._wet_samples + 1):
            tk = t - k * self._wet_step
            if tk < 0.0:                       # before world start: no history
                break
            weight = (self._wet_step / self._wet_tau) * math.exp(
                -k * self._wet_step / self._wet_tau
            )
            _, _, rain_k, _, _ = self.sample_fields(pts, tk)
            acc += weight * rain_k
        np.clip(acc, 0.0, 1.0, out=acc)
        return acc

    def sample_local(
        self, pos_xy: tuple[float, float], t_abs: float | None = None
    ) -> LocalWeather:
        """
        Sample the full natural weather at world position *pos_xy* and *t_abs*.

        A single-point wrapper over :meth:`sample_fields` (so it matches the
        weather-map raster exactly) that also resolves wind, wetness, and
        temperature into a complete :class:`LocalWeather`.

        Parameters
        ----------
        pos_xy : tuple[float, float] — world XY query point (meters).
        t_abs : float | None — absolute game seconds; defaults to the time of
            the last :meth:`update` (or 0.0 before the first update).

        Returns
        -------
        LocalWeather
        """
        t = float(t_abs) if t_abs is not None else (
            self._last_abs_t if self._last_abs_t is not None else 0.0
        )
        pt = np.array([[float(pos_xy[0]), float(pos_xy[1])]], dtype=np.float64)
        cov, den, rain, fog, gust = self.sample_fields(pt, t)
        coverage = float(cov[0])

        wind_dir, syn_speed = self.synoptic.wind(t)
        wind_speed = (
            syn_speed * (0.7 + 0.5 * coverage)
            + float(gust[0]) * self._storm_wind_max
        )
        tod_h = (t % _DAY_S) / 3600.0
        return LocalWeather(
            cloud_coverage=coverage,
            cloud_density=float(den[0]),
            fog_density=float(fog[0]),
            rain_intensity=float(rain[0]),
            wind_dir=wind_dir,
            wind_speed=wind_speed,
            humidity=0.5,
            wetness=float(self.wetness_at(pt, t)[0]),
            temperature_c=self._temperature(tod_h),
        )

    # ------------------------------------------------------------------
    # Classification (hysteresis)
    # ------------------------------------------------------------------

    def _classified_state(self, lw: LocalWeather, abs_t: float) -> WeatherType:
        """
        Hysteresis-stabilised :func:`classify`: a changed label must persist
        ``HYSTERESIS_SECONDS`` before it becomes the committed state.
        """
        raw = classify(lw)
        if self._committed_state is None:
            self._committed_state = raw
            self._pending_state = raw
            self._pending_since = abs_t
            return raw
        if raw is self._committed_state:
            self._pending_state = raw
            self._pending_since = abs_t
            return raw
        if raw is not self._pending_state:
            self._pending_state = raw
            self._pending_since = abs_t
        if abs_t - self._pending_since >= HYSTERESIS_SECONDS:
            self._committed_state = raw
        return self._committed_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current(self) -> WeatherType:
        """
        Discrete label as of the last :meth:`update` (the dev override if one
        is active).  Before the first update, classifies the natural sample at
        the origin at t = 0.
        """
        if self._override is not None:
            return self._override
        if self._committed_state is not None:
            return self._committed_state
        return classify(self.sample_local((0.0, 0.0), 0.0))

    def update(
        self,
        game_day: int,
        game_time_of_day: float,
        player_pos: tuple[float, float] | None = None,
    ) -> LocalWeather:
        """
        Sample the weather at the player and advance the override/label state.

        Call once per frame.  Publishes a :class:`WeatherChangedEvent`
        (deferred) when the committed label changes.

        Parameters
        ----------
        game_day : int — in-game day number (``clock.game_day``).
        game_time_of_day : float — seconds within the day, [0, 86400)
            (``clock.game_time_of_day``).
        player_pos : tuple[float, float] | None — world XY to sample at;
            defaults to the origin (sufficient until the renderer threads the
            camera position through in M4).

        Returns
        -------
        LocalWeather — the local sample (override-blended if forced).
        """
        day = int(game_day)
        tod = float(game_time_of_day) % _DAY_S
        abs_t = day * _DAY_S + tod
        pos = (
            (float(player_pos[0]), float(player_pos[1]))
            if player_pos is not None else (0.0, 0.0)
        )
        self._last_player = pos

        natural = self.sample_local(pos, abs_t)

        # Active cells for the `.cells` readout, nearest player first.
        cells = self._active_cells(abs_t)
        origin = np.array(pos, dtype=np.float64)
        cells.sort(key=lambda c: float(
            np.hypot(*(c.center(abs_t, self.synoptic) - origin))
        ))
        self._cells = cells

        if self._override is not None:
            if self._override_start_abs_t is None:
                self._override_start_abs_t = abs_t
                self._override_from = (
                    self._last_local if self._last_local is not None else natural
                )
            target = self._target_local(self._override, abs_t)
            bt = _smoothstep(abs_t - self._override_start_abs_t, 0.0, BLEND_SECONDS)
            local = _lerp_local(self._override_from, target, bt)
            new_state = self._override
        elif self._release_from is not None:
            if self._release_start_abs_t is None:
                self._release_start_abs_t = abs_t
            bt = _smoothstep(abs_t - self._release_start_abs_t, 0.0, BLEND_SECONDS)
            local = _lerp_local(self._release_from, natural, bt)
            if bt >= 1.0:
                self._release_from = None
                self._release_start_abs_t = None
            new_state = self._classified_state(local, abs_t)
        else:
            local = natural
            new_state = self._classified_state(local, abs_t)

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
        self._last_local = local
        self._last_abs_t = abs_t
        return local

    def _target_local(self, state: WeatherType, abs_t: float) -> LocalWeather:
        """Override target: a :class:`LocalWeather` from ``_STATE_TARGETS``."""
        cov, den, fog, rain, wind_mult = _STATE_TARGETS[state]
        wind_dir, syn_speed = self.synoptic.wind(abs_t)
        tod_h = (abs_t % _DAY_S) / 3600.0
        return LocalWeather(
            cloud_coverage=cov,
            cloud_density=den,
            fog_density=fog,
            rain_intensity=rain,
            wind_dir=wind_dir,
            wind_speed=syn_speed * wind_mult,
            humidity=0.5,
            wetness=0.0,
            temperature_c=self._temperature(tod_h),
        )

    def force_weather(self, weather: WeatherType | None) -> None:
        """
        Dev override: pin the weather to *weather*, or pass ``None`` to clear
        it and blend back to the natural spatial sample.

        Both forcing and clearing crossfade over ``BLEND_SECONDS`` from the
        next :meth:`update`.  While active (or mid-release), ``get_delta()``
        returns a small snapshot so the override survives save/load.

        Note
        ----
        This is a **compatibility shim** over the legacy global states — the
        spatial summon API (M8) is the real replacement.

        Example
        -------
        >>> ws.force_weather(WeatherType.STORM)   # blend toward storm
        >>> ws.force_weather(None)                # blend back to natural
        """
        if weather is not None:
            self._override = WeatherType(weather)
            self._override_start_abs_t = None
            self._override_from = None
            self._release_from = None
            self._release_start_abs_t = None
        elif self._override is not None:
            self._release_from = self._last_local
            self._release_start_abs_t = None
            self._override = None
            self._override_start_abs_t = None
            self._override_from = None

    # ------------------------------------------------------------------
    # Saveable protocol
    # ------------------------------------------------------------------

    def get_delta(self) -> dict:
        """
        Deviations from the procedural baseline (Saveable protocol).

        Natural weather recomputes from the seed, so it costs 0 bytes —
        returns ``{}`` unless a dev override is active or a release blend is in
        flight.  All values are plain primitives (no live objects, no pickle).
        """
        if self._override is not None:
            delta: dict = {"override": self._override.value}
            if self._override_start_abs_t is not None:
                delta["override_start_abs_t"] = float(self._override_start_abs_t)
            if self._override_from is not None:
                delta["override_from"] = _local_to_dict(self._override_from)
            if self._last_state is not None:
                delta["last_state"] = self._last_state.value
            return delta
        if self._release_from is not None:
            delta = {"release_from": _local_to_dict(self._release_from)}
            if self._release_start_abs_t is not None:
                delta["release_start_abs_t"] = float(self._release_start_abs_t)
            if self._last_state is not None:
                delta["last_state"] = self._last_state.value
            return delta
        return {}

    def apply_delta(self, delta: dict) -> None:
        """
        Restore override/release state from :meth:`get_delta` output.

        The natural baseline needs no restoration (recomputed from the seed).
        Legacy deltas from the old Markov system are accepted: their
        ``override``/``release_from`` keys map straight onto this shim; any
        extra keys are ignored.
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
                _local_from_dict(delta["override_from"])
                if "override_from" in delta else None
            )
        elif "release_from" in delta:
            self._release_from = _local_from_dict(delta["release_from"])
            self._release_start_abs_t = (
                float(delta["release_start_abs_t"])
                if "release_start_abs_t" in delta else None
            )
        if "last_state" in delta:
            self._last_state = WeatherType(delta["last_state"])
            self._committed_state = self._last_state
