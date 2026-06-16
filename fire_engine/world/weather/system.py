"""
weather/system.py — Spatial weather system: local sampling + dev override.

This replaces the old global Markov schedule (``sky/weather.py``) with a
**spatial** model: weather at any instant is sampled *at a world position*
from the day's ambient regime plus every active :class:`StormCell` whose
drifting footprint overlaps that point (see :mod:`fire_engine.world.weather.cells`).

Everything natural is a closed-form pure function of (world_seed, game time,
position): ``get_delta()`` is ``{}`` for natural weather (zero save bytes).

Heavy clusters are extracted into private helpers under
``fire_engine.world.weather._impl`` (_sampling, _summon, _save, _update).

Units: meters, m/s, game seconds, °C.

Example
-------
    from fire_engine.core import EventBus, load_config, set_world_seed
    from fire_engine.world.weather import WeatherSystem, WeatherType

    set_world_seed(1337)
    ws = WeatherSystem(load_config(), EventBus())
    lw = ws.update(game_day=3, game_time_of_day=8.5 * 3600.0, player_pos=(0.0, 0.0))
    print(ws.current, lw.cloud_coverage, lw.rain_intensity)

Docs: docs/systems/world.weather.md
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fire_engine.core.config import Config
from fire_engine.core.event_bus import EventBus
from fire_engine.world.weather._impl import _sampling, _save, _summon, _update
from fire_engine.world.weather.cells import CellKind, StormCell, natural_cells
from fire_engine.world.weather.classify import WeatherType, classify
from fire_engine.world.weather.synoptic import Synoptic
from fire_engine.world.weather.types import LocalWeather  # re-exported below

__all__ = ["BLEND_SECONDS", "HYSTERESIS_SECONDS", "LocalWeather", "WeatherSystem"]

#: Game seconds per game day (see ``cells._DAY_S`` — kept local, no sky import).
_DAY_S: float = 24.0 * 3600.0

#: Override blend window: 20 game minutes (seconds).
BLEND_SECONDS: float = 20.0 * 60.0

#: Classification hysteresis (game seconds).
HYSTERESIS_SECONDS: float = 60.0

#: Per-state override targets (unchanged from legacy Markov system).
_STATE_TARGETS: dict[WeatherType, tuple[float, float, float, float, float]] = {
    WeatherType.CLEAR: (0.12, 0.35, 0.0008, 0.0, 0.70),
    WeatherType.CLOUDY: (0.45, 0.55, 0.0012, 0.0, 0.90),
    WeatherType.OVERCAST: (0.85, 0.80, 0.0030, 0.0, 1.00),
    WeatherType.FOG: (0.55, 0.50, 0.0250, 0.0, 0.30),
    WeatherType.RAIN: (0.90, 0.85, 0.0060, 0.7, 1.25),
    WeatherType.STORM: (0.98, 0.95, 0.0080, 1.0, 1.90),
}


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

    Docs: docs/systems/world.weather.md
    """

    save_key: str = "weather"

    # Class-level attribute annotations required so that external functions in
    # _impl/ can read/write these without mypy "cannot declare in non-self" errors.
    _config: Config
    _bus: EventBus | None
    synoptic: Synoptic
    _temp_mean: float
    _temp_amp: float
    _fog_max: float
    _storm_wind_max: float
    _wet_tau: float
    _wet_step: float
    _wet_samples: int
    _recent_tau: float
    _recent_step: float
    _recent_samples: int
    _cell_cache: dict[int, list[StormCell]]
    _summoned: list[StormCell]
    _summon_seq: int
    _suppressed: set[str]
    _summon_defaults: dict[CellKind, tuple[float, float, float]]
    _summon_upwind_m: float
    _wind_field: Any
    _gustfront_range_m: float
    _gustfront_strength: float
    _gustfront_width_m: float
    _active_fronts: dict[str, object]
    _cells: list[StormCell]
    _last_local: LocalWeather | None
    _last_state: WeatherType | None
    _last_abs_t: float | None
    _last_player: tuple[float, float]
    _last_strike_time: float | None
    _committed_state: WeatherType | None
    _pending_state: WeatherType | None
    _pending_since: float
    _override: WeatherType | None
    _override_start_abs_t: float | None
    _override_from: LocalWeather | None
    _release_from: LocalWeather | None
    _release_start_abs_t: float | None

    def __init__(self, config: Config, bus: EventBus | None = None) -> None:
        self._config = config
        self._bus = bus
        self.synoptic: Synoptic = Synoptic(config)

        self._temp_mean = float(config.weather_temp_mean_c)
        self._temp_amp = float(config.weather_temp_amp_c)
        self._fog_max = float(config.weather_fog_max_density)
        self._storm_wind_max = float(config.weather_storm_wind_max_ms)
        self._wet_tau = float(config.weather_wetness_tau_s)
        self._wet_step = float(config.weather_wetness_step_s)
        self._wet_samples = int(config.weather_wetness_samples)
        self._recent_tau = float(config.weather_humidity_recent_tau_s)
        self._recent_step = float(config.weather_humidity_recent_step_s)
        self._recent_samples = int(config.weather_humidity_recent_samples)
        self._cell_cache: dict[int, list[StormCell]] = {}
        self._summoned: list[StormCell] = []
        self._summon_seq: int = 0
        self._suppressed: set[str] = set()
        self._summon_defaults: dict[CellKind, tuple[float, float, float]] = {
            CellKind.SHOWER: (
                float(config.weather_summon_rain_radius_m),
                float(config.weather_summon_rain_duration_s),
                float(config.weather_summon_rain_intensity),
            ),
            CellKind.THUNDERSTORM: (
                float(config.weather_summon_storm_radius_m),
                float(config.weather_summon_storm_duration_s),
                float(config.weather_summon_storm_intensity),
            ),
            CellKind.FOG_BANK: (
                float(config.weather_summon_fog_radius_m),
                float(config.weather_summon_fog_duration_s),
                float(config.weather_summon_fog_intensity),
            ),
            CellKind.CLOUD_BANK: (
                float(config.weather_summon_rain_radius_m),
                float(config.weather_summon_rain_duration_s),
                float(config.weather_summon_rain_intensity),
            ),
        }
        self._summon_upwind_m = float(config.weather_summon_upwind_m)
        self._wind_field = None
        self._gustfront_range_m = float(config.weather_gustfront_range_m)
        self._gustfront_strength = float(config.weather_gustfront_strength_ms)
        self._gustfront_width_m = float(config.weather_gustfront_width_m)
        self._active_fronts: dict[str, object] = {}
        self._cells: list[StormCell] = []
        self._last_local: LocalWeather | None = None
        self._last_state: WeatherType | None = None
        self._last_abs_t: float | None = None
        self._last_player: tuple[float, float] = (0.0, 0.0)
        self._last_strike_time: float | None = None
        self._committed_state: WeatherType | None = None
        self._pending_state: WeatherType | None = None
        self._pending_since: float = 0.0
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
        out = [c for c in pool if c.active(t) and c.id not in self._suppressed]
        out.extend(c for c in self._summoned if c.active(t))
        return out

    @property
    def cells(self) -> list[StormCell]:
        """Active cells as of the last :meth:`update`, nearest player first."""
        return self._cells

    def cell_eta_s(self, cell: StormCell, t: float, player_pos: tuple[float, float]) -> float:
        """
        Approximate game seconds until *cell*'s leading edge reaches the player.

        Returns ``0.0`` if already under the cell, ``inf`` if receding.
        """
        origin = np.array(player_pos, dtype=np.float64)
        center = cell.center(t, self.synoptic)
        to_player = origin - center
        dist = float(np.hypot(to_player[0], to_player[1]))
        edge = dist - cell.radius(t)
        if edge <= 0.0:
            return 0.0
        vel = self.synoptic.wind_vec(t) + np.array(cell.drift_bias)
        if dist < 1e-6:
            return 0.0
        closing = float(np.dot(vel, to_player / dist))
        if closing <= 1e-6:
            return float("inf")
        return edge / closing

    # ------------------------------------------------------------------
    # M8 — Summon API (delegates to _impl/_summon.py)
    # ------------------------------------------------------------------

    def summon_cell(
        self,
        kind: CellKind,
        *,
        time_abs: float,
        player_pos: tuple[float, float],
        radius_m: float | None = None,
        duration_s: float | None = None,
        peak_intensity: float | None = None,
        upwind_m: float | None = None,
    ) -> str:
        """
        Spawn a summoned :class:`StormCell` UPWIND of the player and return its id.

        Example
        -------
        >>> cid = ws.summon_cell(CellKind.THUNDERSTORM, time_abs=3600.0,
        ...                      player_pos=(0.0, 0.0))
        >>> cid.startswith("s:")
        True
        """
        return _summon.summon_cell(
            self,
            kind,
            time_abs=time_abs,
            player_pos=player_pos,
            radius_m=radius_m,
            duration_s=duration_s,
            peak_intensity=peak_intensity,
            upwind_m=upwind_m,
        )

    def summon_rainstorm(
        self, *, time_abs: float, player_pos: tuple[float, float], **kw: Any
    ) -> str:
        """Convenience: summon a SHOWER (rainstorm) drifting toward the player."""
        return self.summon_cell(CellKind.SHOWER, time_abs=time_abs, player_pos=player_pos, **kw)

    def summon_thunderstorm(
        self, *, time_abs: float, player_pos: tuple[float, float], **kw: Any
    ) -> str:
        """Convenience: summon a THUNDERSTORM (rain + lightning + gust)."""
        return self.summon_cell(
            CellKind.THUNDERSTORM, time_abs=time_abs, player_pos=player_pos, **kw
        )

    def summon_fog_bank(
        self, *, time_abs: float, player_pos: tuple[float, float], **kw: Any
    ) -> str:
        """Convenience: summon a FOG_BANK drifting toward the player."""
        return self.summon_cell(CellKind.FOG_BANK, time_abs=time_abs, player_pos=player_pos, **kw)

    def suppress(self, cell_id: str) -> None:
        """Hide a cell from all future samples (see _impl/_summon.py)."""
        _summon.suppress(self, cell_id)

    def clear_all(self) -> None:
        """Clear every summoned cell and suppress every natural cell active now."""
        _summon.clear_all(self)

    # ------------------------------------------------------------------
    # M8 — GustFront coupling
    # ------------------------------------------------------------------

    def attach_wind_field(self, wind_field: Any) -> None:
        """
        Wire the wind field so approaching storm cells register a gust front.

        ``None`` detaches and clears any live fronts.
        """
        if wind_field is None:
            for cid in list(self._active_fronts):
                self._release_front(cid)
        self._wind_field = wind_field

    def _release_front(self, cell_id: str) -> None:
        """Remove the gust-front modifier registered for *cell_id*, if any."""
        front = self._active_fronts.pop(cell_id, None)
        if front is not None and self._wind_field is not None:
            self._wind_field.remove_modifier(front)

    def _update_gust_fronts(self, t: float, player_pos: tuple[float, float]) -> None:
        """Register / release gust-front wind modifiers for nearby storm cells."""
        _update.update_gust_fronts(self, t, player_pos)

    # ------------------------------------------------------------------
    # Sampling (delegates to _impl/_sampling.py)
    # ------------------------------------------------------------------

    def _sample_core(
        self, pts: np.ndarray, t: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Recursion-safe core fields (no emergent fog) at *pts*, absolute *t*."""
        return _sampling.sample_core(self, pts, t)

    def _local_wind_speed(
        self, coverage: np.ndarray, storm_gust: np.ndarray, t: float
    ) -> np.ndarray:
        """Vectorised local wind speed (m/s) at *t* for each point."""
        return _sampling.local_wind_speed(self, coverage, storm_gust, t)

    def _emergent_fog(
        self,
        pts: np.ndarray,
        coverage: np.ndarray,
        storm_gust: np.ndarray,
        t: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Emergent fog coefficient (1/m) and relative humidity at each point."""
        return _sampling.emergent_fog(self, pts, coverage, storm_gust, t)

    def sample_fields(
        self, points_xy: np.ndarray, t_abs: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorised core sampling: the raster channels at every query point.

        Returns ``(coverage, density, rain, fog, storm_gust)``.
        """
        return _sampling.sample_fields(self, points_xy, t_abs)

    def wetness_at(self, points_xy: np.ndarray, t_abs: float) -> np.ndarray:
        """Closed-form ground wetness 0–1 at each query point."""
        pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        return _sampling.wetness_at(self, pts, float(t_abs))

    def rain_recent_at(self, points_xy: np.ndarray, t_abs: float) -> np.ndarray:
        """Closed-form recent-rain measure 0–1 at each query point."""
        pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        return _sampling.wetness_at(self, pts, float(t_abs), use_recent=True)

    def sample_local(self, pos_xy: tuple[float, float], t_abs: float | None = None) -> LocalWeather:
        """
        Sample the full natural weather at world position *pos_xy* and *t_abs*.

        Single-point wrapper that resolves wind, wetness, and temperature into a
        complete :class:`LocalWeather`.
        """
        return _sampling.sample_local(self, pos_xy, t_abs)

    # ------------------------------------------------------------------
    # Classification + override hysteresis (delegates to _impl/_update.py)
    # ------------------------------------------------------------------

    def _classified_state(self, lw: LocalWeather, abs_t: float) -> WeatherType:
        """Hysteresis-stabilised classify: changed label must persist HYSTERESIS_SECONDS."""
        return _update.classified_state(self, lw, abs_t)

    def _emit_lightning(self, abs_t: float, cells: list[StormCell]) -> None:
        """Publish LightningStrikeEvent per scheduled strike since last update."""
        _update.emit_lightning(self, abs_t, cells)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current(self) -> WeatherType:
        """
        Discrete label as of the last :meth:`update` (the dev override if one
        is active).  Before the first update, classifies at origin, t = 0.
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
        game_time_of_day : float — seconds within the day, [0, 86400).
        player_pos : tuple[float, float] | None — world XY; defaults to origin.

        Returns
        -------
        LocalWeather — the local sample (override-blended if forced).
        """
        return _update.do_update(
            self, game_day, game_time_of_day, player_pos, BLEND_SECONDS, _STATE_TARGETS
        )

    def force_weather(self, weather: WeatherType | None) -> None:
        """
        Dev override: pin the weather to *weather*, or pass ``None`` to clear
        it and blend back to the natural spatial sample.

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
    # Saveable protocol (delegates to _impl/_save.py)
    # ------------------------------------------------------------------

    def get_delta(self) -> dict[str, Any]:
        """
        Deviations from the procedural baseline (Saveable protocol).

        Returns ``{}`` when no summons, no suppressions, and no dev override.
        """
        return _save.get_delta(self)

    def _apply_delta_summons(self, delta: dict[str, Any]) -> None:
        """Rebuild summoned cells + suppression set from delta (M8)."""
        _save.apply_delta_summons(self, delta)

    def _apply_delta_override(self, delta: dict[str, Any]) -> None:
        """Restore legacy dev-override shim state from delta."""
        _save.apply_delta_override(self, delta)

    def apply_delta(self, delta: dict[str, Any]) -> None:
        """
        Restore summons / suppressions / override from :meth:`get_delta` output.

        A malformed summoned-cell entry is skipped rather than crashing the load.
        """
        if not isinstance(delta, dict) or not delta:
            return
        self._apply_delta_summons(delta)
        self._apply_delta_override(delta)
