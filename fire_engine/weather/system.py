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
from fire_engine.weather.humidity import (
    emergent_fog,
    humidity_base,
    relative_humidity,
)
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
    one-to-one.  ``humidity`` is the live emergent relative humidity (M5);
    ``wetness`` and ``temperature_c`` are already live.

    Attributes
    ----------
    cloud_coverage : float — 0–1 fraction of sky filled.
    cloud_density : float — 0–1 cloud opacity/darkness.
    fog_density : float — exponential fog coefficient, 1/m (0.0008 ≈ clear).
    rain_intensity : float — 0–1 (0 = dry, 1 = torrential).
    wind_dir : tuple[float, float] — unit XY direction the wind blows TOWARD.
    wind_speed : float — m/s.
    humidity : float — 0–1 emergent relative humidity (base + recent rain +
        ground wetness).
    wetness : float — 0–1 ground wetness.
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


# --- M8 summon/save ---------------------------------------------------------
# Serialise a summoned :class:`StormCell` to/from a plain primitive dict (~80
# bytes): every field is a float/str/list of floats — no live object refs, no
# pickle (Hard Rule 3).  These params fully determine the cell's closed-form
# track + footprint, so a round-trip reproduces the IDENTICAL future (positions
# *and* the would-be strike schedule M7 derives from them).

def _cell_to_dict(c: StormCell) -> dict:
    """Serialise a summoned :class:`StormCell` to plain primitives (Saveable)."""
    return {
        "id": str(c.id),
        "kind": c.kind.value,
        "spawn_time": float(c.spawn_time),
        "spawn_pos": [float(c.spawn_pos[0]), float(c.spawn_pos[1])],
        "duration_s": float(c.duration_s),
        "radius_m": float(c.radius_m),
        "peak_intensity": float(c.peak_intensity),
        "drift_bias": [float(c.drift_bias[0]), float(c.drift_bias[1])],
    }


def _cell_from_dict(d: dict) -> StormCell:
    """Inverse of :func:`_cell_to_dict` (raises ``KeyError``/``ValueError`` on a
    malformed dict — the caller guards the whole delta)."""
    return StormCell(
        id=str(d["id"]),
        kind=CellKind(d["kind"]),
        spawn_time=float(d["spawn_time"]),
        spawn_pos=(float(d["spawn_pos"][0]), float(d["spawn_pos"][1])),
        duration_s=float(d["duration_s"]),
        radius_m=float(d["radius_m"]),
        peak_intensity=float(d["peak_intensity"]),
        drift_bias=(float(d["drift_bias"][0]), float(d["drift_bias"][1])),
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
        # "Recent rain" quadrature for emergent humidity — the same exponential
        # form as the wetness window but with its own (longer) decay constant:
        # the air stays muggy for hours after a shower even as the ground dries,
        # so evening rain can still feed pre-dawn fog.
        self._recent_tau = float(config.weather_humidity_recent_tau_s)
        self._recent_step = float(config.weather_humidity_recent_step_s)
        self._recent_samples = int(config.weather_humidity_recent_samples)

        # Per-day natural cell cache (pure fn of seed+day; never saved).
        self._cell_cache: dict[int, list[StormCell]] = {}

        # --- M8 summon/save ---
        # Summoned cells (saveable deviation). Each is a first-class StormCell
        # consumed by `_active_cells` exactly like a natural one, so it shows up
        # in `_sample_core`, the `.cells` readout and the weather-map raster.
        self._summoned: list[StormCell] = []
        # Monotonic counter for stable summoned-cell ids ("s:{n}").  Saved so
        # ids never collide with pre-existing summoned cells after a load.
        self._summon_seq: int = 0
        # Suppressed natural-cell ids ("n:{day}:{slot}"): filtered out of every
        # sample so a dev can "clear skies" without touching the seed.
        self._suppressed: set[str] = set()
        # Summon defaults per kind: (radius_m, duration_s, peak_intensity).
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

        # Gust-front coupling (M8): the wind field whose modifiers we register
        # so a storm's leading edge kicks the grass.  Set by the world layer via
        # `attach_wind_field`; None keeps the system fully headless / decoupled.
        self._wind_field = None  # fire_engine.wind.WindField | None
        self._gustfront_range_m = float(config.weather_gustfront_range_m)
        self._gustfront_strength = float(config.weather_gustfront_strength_ms)
        self._gustfront_width_m = float(config.weather_gustfront_width_m)
        # cell.id -> live GustFront modifier currently registered for it.
        self._active_fronts: dict[str, object] = {}

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
        """
        All cells (natural ∪ summoned) alive at absolute time *t*.

        Natural cells whose id is in :attr:`_suppressed` are filtered out (a dev
        "clear skies" hides the day's natural weather without touching the seed,
        so the suppression survives save/load as a small id list).  Summoned
        cells are appended verbatim — they are first-class participants in every
        downstream sample (``_sample_core``, ``.cells``, the weather-map raster).
        """
        day = int(t // _DAY_S)
        pool = self._cells_for_day(day - 1) + self._cells_for_day(day)
        out = [
            c for c in pool
            if c.active(t) and c.id not in self._suppressed
        ]
        out.extend(c for c in self._summoned if c.active(t))
        return out

    @property
    def cells(self) -> list[StormCell]:
        """Active cells as of the last :meth:`update`, nearest player first."""
        return self._cells

    def cell_eta_s(
        self, cell: StormCell, t: float, player_pos: tuple[float, float]
    ) -> float:
        """
        Approximate game seconds until *cell*'s leading edge reaches the player.

        A first-order estimate: the cell's current edge distance divided by its
        closing speed (the component of the cell's instantaneous velocity toward
        the player).  Returns ``0.0`` if the edge already covers the player and
        ``inf`` if the cell is receding (closing speed ≤ 0).  Drives the
        devtools "ETA" read-out; not part of the deterministic sim.

        Parameters
        ----------
        cell : StormCell — the cell to time.
        t : float — absolute game seconds.
        player_pos : tuple[float, float] — world XY of the player (m).
        """
        origin = np.array(player_pos, dtype=np.float64)
        center = cell.center(t, self.synoptic)
        to_player = origin - center
        dist = float(np.hypot(to_player[0], to_player[1]))
        edge = dist - cell.radius(t)
        if edge <= 0.0:
            return 0.0
        # Cell velocity = synoptic wind + drift_bias (dD/dt ≡ wind_vec).
        vel = self.synoptic.wind_vec(t) + np.array(cell.drift_bias)
        if dist < 1e-6:
            return 0.0
        closing = float(np.dot(vel, to_player / dist))   # m/s toward player
        if closing <= 1e-6:
            return float("inf")
        return edge / closing

    # ------------------------------------------------------------------
    # M8 — Summon API (spatial dev/scripting control over weather)
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

        The cell is placed ``upwind_m`` meters from *player_pos* in the
        **opposite** direction to the synoptic wind at *time_abs*, so it drifts
        *toward* the player on the steering current (``cell.center`` rides the
        raw synoptic displacement — see :mod:`fire_engine.weather.cells`).  It is
        a first-class participant in every sample from the moment it spawns
        (``time_abs`` is its ``spawn_time``); radius/duration/intensity default
        to the per-kind ``weather_summon_*`` config values.

        The cell is fully described by its stored params, so it survives
        save/load bit-identically (:meth:`get_delta` / :meth:`apply_delta`) —
        the load-resume invariant: future positions *and* the would-be strike
        schedule M7 derives from them round-trip exactly.

        Parameters
        ----------
        kind : CellKind — what the summoned cell does.
        time_abs : float — absolute game seconds = the cell's ``spawn_time``.
        player_pos : tuple[float, float] — world XY the cell is aimed at (m).
        radius_m, duration_s, peak_intensity : float | None — override the
            per-kind ``weather_summon_*`` defaults when given.
        upwind_m : float | None — spawn distance upwind; defaults to
            ``weather_summon_upwind_m``.

        Returns
        -------
        str — the new cell's stable id (``"s:{n}"``).

        Example
        -------
        >>> cid = ws.summon_cell(CellKind.THUNDERSTORM, time_abs=3600.0,
        ...                      player_pos=(0.0, 0.0))
        >>> cid.startswith("s:")
        True
        """
        kind = CellKind(kind)
        r_def, d_def, p_def = self._summon_defaults[kind]
        radius = float(radius_m) if radius_m is not None else r_def
        duration = float(duration_s) if duration_s is not None else d_def
        peak = float(peak_intensity) if peak_intensity is not None else p_def
        dist = float(upwind_m) if upwind_m is not None else self._summon_upwind_m

        # Spawn upwind: the synoptic wind blows TOWARD (ux, uy), so place the
        # cell at player − dist·(ux, uy); it then drifts back over the player.
        (ux, uy), _ = self.synoptic.wind(float(time_abs))
        spawn_pos = (
            float(player_pos[0]) - dist * ux,
            float(player_pos[1]) - dist * uy,
        )

        cell_id = f"s:{self._summon_seq}"
        self._summon_seq += 1
        self._summoned.append(StormCell(
            id=cell_id,
            kind=kind,
            spawn_time=float(time_abs),
            spawn_pos=spawn_pos,
            duration_s=duration,
            radius_m=radius,
            peak_intensity=peak,
            drift_bias=(0.0, 0.0),
        ))
        return cell_id

    def summon_rainstorm(
        self, *, time_abs: float, player_pos: tuple[float, float], **kw
    ) -> str:
        """Convenience: summon a SHOWER (rainstorm) drifting toward the player."""
        return self.summon_cell(
            CellKind.SHOWER, time_abs=time_abs, player_pos=player_pos, **kw
        )

    def summon_thunderstorm(
        self, *, time_abs: float, player_pos: tuple[float, float], **kw
    ) -> str:
        """Convenience: summon a THUNDERSTORM (rain + lightning + gust)."""
        return self.summon_cell(
            CellKind.THUNDERSTORM, time_abs=time_abs, player_pos=player_pos, **kw
        )

    def summon_fog_bank(
        self, *, time_abs: float, player_pos: tuple[float, float], **kw
    ) -> str:
        """Convenience: summon a FOG_BANK drifting toward the player."""
        return self.summon_cell(
            CellKind.FOG_BANK, time_abs=time_abs, player_pos=player_pos, **kw
        )

    def suppress(self, cell_id: str) -> None:
        """
        Hide a cell from all future samples.

        A natural-cell id (``"n:{day}:{slot}"``) is added to the suppression set
        (it is filtered out of :meth:`_active_cells` — survives save/load as a
        small id list); a summoned-cell id (``"s:{n}"``) is removed outright.
        No-op for an unknown id.
        """
        cid = str(cell_id)
        if cid.startswith("s:"):
            self._summoned = [c for c in self._summoned if c.id != cid]
            self._release_front(cid)
        else:
            self._suppressed.add(cid)

    def clear_all(self) -> None:
        """
        Clear every summoned cell and suppress every natural cell active *now*.

        Gives a dev a one-call "clear skies": summoned cells are dropped and the
        natural cells alive at the last :meth:`update` are added to the
        suppression set so the sky reads clear.  The suppression persists across
        subsequent updates (and save/load) until new cells spawn into days that
        were not suppressed — i.e. it clears the *current* weather, not all
        future weather forever.  Registered gust fronts are released cleanly.
        """
        self._summoned.clear()
        # Suppress the natural cells active at the last sampled time (the ones
        # currently making weather); future days resume their natural schedule.
        t = self._last_abs_t if self._last_abs_t is not None else 0.0
        day = int(t // _DAY_S)
        for c in self._cells_for_day(day - 1) + self._cells_for_day(day):
            self._suppressed.add(c.id)
        # Drop all live gust fronts (their cells are gone / suppressed).
        for cid in list(self._active_fronts):
            self._release_front(cid)

    # ------------------------------------------------------------------
    # M8 — GustFront coupling (a storm's leading edge kicks the grass)
    # ------------------------------------------------------------------

    def attach_wind_field(self, wind_field) -> None:
        """
        Wire the wind field so approaching storm cells register a gust front.

        Called once by the world layer (the only place that holds both the
        weather system and the :class:`~fire_engine.wind.WindField`).  ``None``
        detaches and clears any live fronts — the system stays fully headless
        and decoupled when no field is attached, so the headless test suite
        never needs panda3d.
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
        """
        Register / release gust-front wind modifiers for nearby storm cells.

        For each active cell whose **leading edge** (center distance − radius) is
        within ``weather_gustfront_range_m`` of the player, ensure a
        :class:`~fire_engine.wind.GustFront` is registered on the attached wind
        field, travelling along the synoptic wind direction.  Cells that pass,
        decay, or drift out of range have their front removed — balanced
        register/remove so modifiers never accumulate or leak.  No-op when no
        wind field is attached.
        """
        if self._wind_field is None:
            return
        from fire_engine.wind import GustFront

        (ux, uy), _ = self.synoptic.wind(t)
        origin = np.array(player_pos, dtype=np.float64)
        active = self._active_cells(t)
        near_ids: set[str] = set()
        for cell in active:
            center = cell.center(t, self.synoptic)
            edge = float(np.hypot(*(center - origin))) - cell.radius(t)
            if edge <= self._gustfront_range_m:
                near_ids.add(cell.id)
                if cell.id not in self._active_fronts:
                    front = GustFront(
                        seed_key=("weather", cell.id),
                        direction=(ux, uy),
                        speed=max(1.0, self._gustfront_strength),
                        strength=self._gustfront_strength * cell.intensity(t),
                        width_m=self._gustfront_width_m,
                    )
                    self._wind_field.add_modifier(front)
                    self._active_fronts[cell.id] = front
        # Release fronts whose cell is no longer near (passed / decayed / gone).
        for cid in list(self._active_fronts):
            if cid not in near_ids:
                self._release_front(cid)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _temperature(self, tod_h: float) -> float:
        """Local air temperature (°C): daily cosine peaking at 15:00."""
        return self._temp_mean + self._temp_amp * math.cos(
            2.0 * math.pi * (tod_h - 15.0) / 24.0
        )

    def _sample_core(
        self, pts: np.ndarray, t: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Recursion-safe core fields (no emergent fog) at *pts*, absolute *t*.

        Returns ``(coverage, density, rain, fog_bank, storm_gust)`` where
        ``fog_bank`` is the FOG_BANK + baseline fog coefficient *before* the
        emergent humidity-condensation term is added.  The rain-history
        quadratures (:meth:`wetness_at`, :meth:`rain_recent_at`) call this
        instead of :meth:`sample_fields` so emergent fog — which *depends* on
        that history — never recurses back into it.  ``pts`` is assumed already
        a ``float64`` ``(N, 2)`` array.
        """
        n = pts.shape[0]
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
        fog_bank = _FOG_BASELINE + fog_extra * _FOG_BANK_GAIN
        return coverage, density, rain, fog_bank, storm_gust

    def _local_wind_speed(
        self, coverage: np.ndarray, storm_gust: np.ndarray, t: float
    ) -> np.ndarray:
        """
        Vectorised local wind speed (m/s) at *t* for each point.

        The single definition shared by :meth:`sample_fields` (the emergent-fog
        wind gate) and :meth:`sample_local` (the returned ``wind_speed``), so a
        texel's fog equals ``sample_local`` fog at its center:
        ``syn_speed·(0.7 + 0.5·coverage) + storm_gust·storm_wind_max``.
        """
        _, syn_speed = self.synoptic.wind(t)
        return syn_speed * (0.7 + 0.5 * coverage) + storm_gust * self._storm_wind_max

    def _emergent_fog(
        self,
        pts: np.ndarray,
        coverage: np.ndarray,
        storm_gust: np.ndarray,
        t: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Emergent fog coefficient (1/m) and relative humidity at each point.

        Closed-form condensation (see :mod:`fire_engine.weather.humidity`): the
        local relative humidity (per-day baseline + recent rain + ground
        wetness) condenses into ground fog where it exceeds the
        temperature-dependent saturation humidity, gated off by wind.  Returns
        ``(emergent_fog, humidity)`` — both shape ``(N,)``.
        """
        day = int(t // _DAY_S)
        tod = t - day * _DAY_S
        tod_h = tod / 3600.0
        # Per-day calm-air humidity baseline, cosine-blended across the midnight
        # hand-off over the first game hour (same shape as the regime ambient
        # blend) so humidity — and the fog it drives — never snaps at 00:00.
        h_cur = humidity_base(day, self._config)
        h_prev = humidity_base(day - 1, self._config) if day > 0 else h_cur
        blend = 0.5 - 0.5 * math.cos(math.pi * min(tod / 3600.0, 1.0))
        h_base = h_prev + (h_cur - h_prev) * blend

        temp = np.full(pts.shape[0], self._temperature(tod_h))
        wind_speed = self._local_wind_speed(coverage, storm_gust, t)
        rain_recent = self.rain_recent_at(pts, t)
        wetness = self.wetness_at(pts, t)
        humidity = relative_humidity(rain_recent, wetness, h_base, self._config)
        fog = emergent_fog(humidity, temp, wind_speed, self._config)
        return fog, humidity

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
        The fog channel also gains the **emergent** condensation term (humidity
        past the temperature-dependent saturation in calm air — see
        :mod:`fire_engine.weather.humidity`), so calm humid nights grow ground
        fog with no scheduled "fog state"; the total is capped at
        ``weather_fog_max_density``.

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
        t = float(t_abs)
        coverage, density, rain, fog_bank, storm_gust = self._sample_core(pts, t)
        emergent, _ = self._emergent_fog(pts, coverage, storm_gust, t)
        fog = np.minimum(fog_bank + emergent, self._fog_max)
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
            _, _, rain_k, _, _ = self._sample_core(pts, tk)
            acc += weight * rain_k
        np.clip(acc, 0.0, 1.0, out=acc)
        return acc

    def rain_recent_at(self, points_xy: np.ndarray, t_abs: float) -> np.ndarray:
        """
        Closed-form "recent rain" measure 0–1 at each query point.

        The same fixed-offset exponential quadrature as :meth:`wetness_at`, but
        with its own (longer) decay constant ``weather_humidity_recent_tau_s``:
        the air stays muggy for hours after a shower while the ground dries in
        ~1 h, so evening rain still registers in the pre-dawn humidity and can
        feed ground fog.  Feeds the emergent-humidity model
        (:func:`fire_engine.weather.humidity.relative_humidity`).  Pure function
        of (seed, time, position); recomputes for free on load.

        Parameters
        ----------
        points_xy : np.ndarray — shape ``(N, 2)`` world-XY query points (m).
        t_abs : float — absolute game seconds.

        Returns
        -------
        np.ndarray — shape ``(N,)`` recent-rain measure clamped to [0, 1].
        """
        pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        t = float(t_abs)
        acc = np.zeros(pts.shape[0])
        for k in range(1, self._recent_samples + 1):
            tk = t - k * self._recent_step
            if tk < 0.0:                       # before world start: no history
                break
            weight = (self._recent_step / self._recent_tau) * math.exp(
                -k * self._recent_step / self._recent_tau
            )
            _, _, rain_k, _, _ = self._sample_core(pts, tk)
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
        cov, den, rain, fog_bank, gust = self._sample_core(pt, t)
        coverage = float(cov[0])

        emergent, humidity = self._emergent_fog(pt, cov, gust, t)
        fog = float(min(fog_bank[0] + emergent[0], self._fog_max))

        wind_dir, _ = self.synoptic.wind(t)
        wind_speed = float(self._local_wind_speed(cov, gust, t)[0])
        tod_h = (t % _DAY_S) / 3600.0
        return LocalWeather(
            cloud_coverage=coverage,
            cloud_density=float(den[0]),
            fog_density=fog,
            rain_intensity=float(rain[0]),
            wind_dir=wind_dir,
            wind_speed=wind_speed,
            humidity=float(humidity[0]),
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

        # --- M8 summon/save: keep gust-front wind modifiers in sync with the
        # nearby cells (no-op when no wind field is attached). Self-contained so
        # it never collides with M7's separate edit to this method. ---
        self._update_gust_fronts(abs_t, pos)

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
        returns ``{}`` when no summons, no suppressions, and no dev override (or
        release blend) exist.  Otherwise a small dict of plain primitives (no
        live objects, no pickle, Hard Rule 3):

        - ``summoned``: list of ~80-byte cell param dicts (:func:`_cell_to_dict`),
        - ``summon_seq``: the id counter (so post-load summons keep unique ids),
        - ``suppressed``: list of suppressed natural-cell ids,
        - the legacy ``override`` / ``release_from`` / ``last_state`` keys when a
          dev override (``force_weather``) is in flight.

        The summoned-cell params fully determine each cell's closed-form track +
        footprint, so a save→load mid-storm reproduces the IDENTICAL future
        (positions and the would-be M7 strike schedule).
        """
        delta: dict = {}

        # --- M8 summon/save: spatial summons + natural-cell suppressions. ---
        if self._summoned:
            delta["summoned"] = [_cell_to_dict(c) for c in self._summoned]
            delta["summon_seq"] = int(self._summon_seq)
        if self._suppressed:
            delta["suppressed"] = sorted(self._suppressed)

        # Legacy dev-override shim (force_weather) — unchanged contract.
        if self._override is not None:
            delta["override"] = self._override.value
            if self._override_start_abs_t is not None:
                delta["override_start_abs_t"] = float(self._override_start_abs_t)
            if self._override_from is not None:
                delta["override_from"] = _local_to_dict(self._override_from)
            if self._last_state is not None:
                delta["last_state"] = self._last_state.value
        elif self._release_from is not None:
            delta["release_from"] = _local_to_dict(self._release_from)
            if self._release_start_abs_t is not None:
                delta["release_start_abs_t"] = float(self._release_start_abs_t)
            if self._last_state is not None:
                delta["last_state"] = self._last_state.value

        return delta

    def apply_delta(self, delta: dict) -> None:
        """
        Restore summons / suppressions / override from :meth:`get_delta` output.

        The natural baseline needs no restoration (recomputed from the seed).
        Summoned cells are reconstructed from their param dicts and the
        suppression set from its id list, so the loaded system reproduces the
        identical future.  A legacy or unknown-shaped delta is tolerated: a
        malformed summoned-cell entry is skipped rather than crashing the load,
        and legacy Markov deltas' ``override`` / ``release_from`` keys still map
        straight onto the shim.
        """
        if not isinstance(delta, dict) or not delta:
            return

        # --- M8 summon/save: rebuild summoned cells + suppression set. ---
        summoned: list[StormCell] = []
        for d in delta.get("summoned", ()) or ():
            try:
                summoned.append(_cell_from_dict(d))
            except (KeyError, ValueError, TypeError, IndexError):
                continue  # skip a malformed/legacy entry rather than crash
        if summoned:
            self._summoned = summoned
        if "summon_seq" in delta:
            try:
                self._summon_seq = int(delta["summon_seq"])
            except (TypeError, ValueError):
                pass
        # Never let a stale counter alias an existing summoned id.
        for c in self._summoned:
            if c.id.startswith("s:"):
                try:
                    self._summon_seq = max(self._summon_seq, int(c.id[2:]) + 1)
                except ValueError:
                    pass
        supp = delta.get("suppressed")
        if isinstance(supp, (list, tuple)):
            self._suppressed = {str(s) for s in supp}

        # Legacy dev-override shim.
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
