"""
Engine-level frozen event dataclasses for fire_engine.core.

All event types are frozen dataclasses named ``*Event`` that are
published/consumed via :class:`~fire_engine.core.event_bus.EventBus`.
This grouping module is exempt from the one-public-class limit — it
collects related trivial support types that would otherwise clutter their
primary modules.

Docs: docs/systems/core.md
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Engine-level event types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkLoadedEvent:
    """
    Published when a terrain chunk has been generated, meshed, and is ready.

    Attributes
    ----------
    coord : tuple[int, int, int]
        Chunk grid coordinate (cx, cy, cz). Each unit = 16 meters.

    Docs: docs/systems/core.md
    """

    coord: tuple[int, int, int]


@dataclass(frozen=True)
class ChunkUnloadedEvent:
    """
    Published when a terrain chunk has been evicted from memory.

    Attributes
    ----------
    coord : tuple[int, int, int]
        Chunk grid coordinate (cx, cy, cz).

    Docs: docs/systems/core.md
    """

    coord: tuple[int, int, int]


@dataclass(frozen=True)
class TerrainEditedEvent:
    """
    Published after a brush edit has been applied to terrain voxel data.

    Attributes
    ----------
    chunk_coords : tuple
        The set (or single) chunk coordinates affected by this edit.
        May be a single (cx, cy, cz) or a frozenset of coordinates.
    brush : object
        The brush object used (SphereBrush, BoxBrush, etc.).
        Typed loosely as ``object`` to avoid an import cycle with terrain/.

    Docs: docs/systems/core.md
    """

    chunk_coords: tuple[int, ...]
    brush: object


@dataclass(frozen=True)
class WeatherChangedEvent:
    """
    Published by the WeatherSystem (``fire_engine.world.sky``) whenever the discrete
    weather state changes — at most once per 2-game-hour segment boundary,
    or when a dev override is applied/cleared.  Published via
    ``bus.publish_deferred`` (state-change notification; never per-frame).

    Subscribers: ambience/audio, AI behaviour (seek shelter), render layer
    (particle systems), gameplay (crop growth, fire spread).

    Attributes
    ----------
    previous : str
        The ``WeatherType.value`` string of the outgoing state
        (e.g. ``"clear"``, ``"rain"``).
    current : str
        The ``WeatherType.value`` string of the incoming state.
    day : int
        In-game day number on which the change occurred.

    Docs: docs/systems/core.md
    """

    previous: str
    current: str
    day: int


@dataclass(frozen=True)
class LightningStrikeEvent:
    """
    Published by the WeatherSystem (M7) when a thunderstorm cell fires a strike.

    One event per scheduled strike (deterministic Poisson schedule per active
    THUNDERSTORM cell — see ``fire_engine.world.weather.lightning``).  Carried via
    ``bus.publish_deferred`` (state-change notification, never per-frame data
    plumbing).  The render half (``world/lightning_renderer.py``) subscribes,
    generates the bolt geometry from ``seed`` (deterministic), animates the
    flash, adds a transient scene light, and re-publishes a
    :class:`ThunderEvent` for the delayed audio crack.

    All randomness for the bolt geometry is keyed off ``seed`` so the same
    strike renders byte-identically on every machine / after a save-load.

    Attributes
    ----------
    pos : tuple[float, float, float]
        Strike origin (cloud-base) world XYZ, meters — the top of the bolt.
    ground_pos : tuple[float, float, float]
        Where the bolt terminates on the ground / roof, world XYZ, meters.
    seed : int
        Bolt RNG seed — deterministic channel geometry
        (``for_domain("weather", "bolt", seed, ...)``).
    time_abs : float
        Absolute game time of the strike, seconds (day·86400 + time-of-day).
    cell_id : int
        Source :class:`~fire_engine.world.weather.StormCell` id, hashed to an int
        (the cell's stable string id digested deterministically).
    intensity : float
        0–1 strike brightness / scale (peak cell intensity at strike time).

    Docs: docs/systems/core.md
    """

    pos: tuple[float, float, float]
    ground_pos: tuple[float, float, float]
    seed: int
    time_abs: float
    cell_id: int
    intensity: float


@dataclass(frozen=True)
class ThunderEvent:
    """
    Published by the render half (M7) after a :class:`LightningStrikeEvent`,
    carrying the delayed audio crack for the strike.

    The thunder is heard ``delay_s`` after the flash (``distance_m / 343`` — the
    speed of sound), so a distant storm rumbles seconds after its lightning and
    a close one cracks almost immediately.  Audio subscribers schedule a rumble
    that long after the event arrives.  Via ``bus.publish_deferred``.

    Attributes
    ----------
    pos : tuple[float, float, float]
        Strike origin world XYZ, meters (same as the strike's ``pos``).
    distance_m : float
        Distance from the listener (camera) to the strike, meters.
    delay_s : float
        Audio delay before the crack is heard, seconds (``distance_m / 343``).
    time_abs : float
        Absolute game time of the originating flash, seconds.
    intensity : float
        0–1 strike intensity (drives the rumble loudness / low-pass roll-off).

    Docs: docs/systems/core.md
    """

    pos: tuple[float, float, float]
    distance_m: float
    delay_s: float
    time_abs: float
    intensity: float


@dataclass(frozen=True)
class GameDayTickEvent:
    """
    Published by the Clock once per in-game day elapsed.

    Subscribers: world-map AI tier (statistical outcomes), Economy, Politics.

    Attributes
    ----------
    day : int
        The new day number (starts at 0; increments each in-game day).

    Docs: docs/systems/core.md
    """

    day: int


@dataclass(frozen=True)
class BuildingChangedEvent:
    """
    Published by ``fire_engine.buildings.BuildingManager`` whenever a building
    is added, modified, or removed.  A state-change notification (never per
    frame) — the World layer's building renderer rebuilds the affected node,
    and the lighting layer can invalidate the cascades overlapping the bounds.

    Attributes
    ----------
    building_id : int
        Manager-assigned id of the building that changed.
    change : str
        ``"added"``, ``"modified"``, or ``"removed"``.
    bounds_min : tuple[float, float, float]
        Conservative world-space AABB minimum corner in meters
        (``Building.world_aabb()``); for ``"removed"`` it is the last-known
        bounds so listeners can invalidate the vacated region.
    bounds_max : tuple[float, float, float]
        World-space AABB maximum corner in meters.

    Docs: docs/systems/core.md
    """

    building_id: int
    change: str
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]


__all__ = [
    "BuildingChangedEvent",
    "ChunkLoadedEvent",
    "ChunkUnloadedEvent",
    "GameDayTickEvent",
    "LightningStrikeEvent",
    "TerrainEditedEvent",
    "ThunderEvent",
    "WeatherChangedEvent",
]
