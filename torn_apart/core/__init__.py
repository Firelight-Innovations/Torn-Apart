"""
torn_apart.core — Foundation layer, callable from any layer.

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

from torn_apart.core.math3d import Vec3, Quat
from torn_apart.core.event_bus import (
    EventBus,
    ChunkLoadedEvent,
    ChunkUnloadedEvent,
    TerrainEditedEvent,
    GameDayTickEvent,
)
from torn_apart.core.rng import set_world_seed, for_domain
from torn_apart.core.config import Config, load_config
from torn_apart.core.clock import Clock
from torn_apart.core.lod import LODPolicy
from torn_apart.core.log import get_logger

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
