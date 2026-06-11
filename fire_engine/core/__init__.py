"""
fire_engine.core — Foundation layer, callable from any layer.

Exports the full public API for the core package.  Import from here rather
than from submodules directly whenever practical.

Contents
--------
Math
    Vec3        — float32 3-vector (Z-up: forward=+Y, right=+X, up=+Z)
    Quat        — float32 unit quaternion, scalar-first [w,x,y,z]

Events
    EventBus                — publish/subscribe event dispatcher
    ChunkLoadedEvent        — terrain chunk ready
    ChunkUnloadedEvent      — terrain chunk evicted
    TerrainEditedEvent      — brush edit applied to terrain
    GameDayTickEvent        — in-game day elapsed
    WeatherChangedEvent     — discrete weather state changed (sky package)

RNG
    set_world_seed          — set the global world seed at boot
    for_domain              — get a deterministic Generator for a domain key

Config
    Config                  — frozen typed config dataclass
    load_config             — load Config from config.toml

Clock
    Clock                   — frame dt, fixed-step accumulator, game calendar

LOD
    LODPolicy               — distance-band policy (shared by World + Terrain)

Logging
    get_logger              — obtain a named Logger with sane formatting
"""

from fire_engine.core.math3d import Vec3, Quat
from fire_engine.core.event_bus import (
    EventBus,
    ChunkLoadedEvent,
    ChunkUnloadedEvent,
    TerrainEditedEvent,
    GameDayTickEvent,
    WeatherChangedEvent,
)
from fire_engine.core.rng import set_world_seed, for_domain
from fire_engine.core.config import Config, load_config
from fire_engine.core.clock import Clock
from fire_engine.core.lod import LODPolicy
from fire_engine.core.log import get_logger

__all__ = [
    # Math
    "Vec3",
    "Quat",
    # Events
    "EventBus",
    "ChunkLoadedEvent",
    "ChunkUnloadedEvent",
    "TerrainEditedEvent",
    "GameDayTickEvent",
    "WeatherChangedEvent",
    # RNG
    "set_world_seed",
    "for_domain",
    # Config
    "Config",
    "load_config",
    # Clock
    "Clock",
    # LOD
    "LODPolicy",
    # Logging
    "get_logger",
]
