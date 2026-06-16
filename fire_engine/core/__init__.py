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

Profiler
    Profiler                — per-frame hierarchical timer + numpy ring buffer
    get_profiler            — the process-wide Profiler singleton (no-op until init)
    init_profiler           — configure the singleton from Config at boot
"""

from fire_engine.core.clock import Clock
from fire_engine.core.config import Config, load_config
from fire_engine.core.event_bus import (
    BuildingChangedEvent,
    ChunkLoadedEvent,
    ChunkUnloadedEvent,
    EventBus,
    GameDayTickEvent,
    LightningStrikeEvent,
    TerrainEditedEvent,
    ThunderEvent,
    WeatherChangedEvent,
)
from fire_engine.core.lod import LODPolicy
from fire_engine.core.log import get_logger
from fire_engine.core.math3d import Quat, Vec3
from fire_engine.core.profiler import Profiler, get_profiler, init_profiler
from fire_engine.core.rng import for_domain, set_world_seed

__all__ = [
    "BuildingChangedEvent",
    "ChunkLoadedEvent",
    "ChunkUnloadedEvent",
    "Clock",
    "Config",
    "EventBus",
    "GameDayTickEvent",
    "LODPolicy",
    "LightningStrikeEvent",
    "Profiler",
    "Quat",
    "TerrainEditedEvent",
    "ThunderEvent",
    "Vec3",
    "WeatherChangedEvent",
    "for_domain",
    "get_logger",
    "get_profiler",
    "init_profiler",
    "load_config",
    "set_world_seed",
]
