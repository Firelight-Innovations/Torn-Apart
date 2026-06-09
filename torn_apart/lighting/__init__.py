"""
torn_apart.lighting — Voxel light grid (Phase 4 v0: CPU sunlight).

Exports
-------
LightGrid
    Per-chunk ``uint8 (16, 16, 16)`` light array store.
occupancy_from_materials
    Downsample a 32³ terrain material array to a 16³ occupancy grid.
SunlightComputer
    Vectorised column-pass + box-blur sunlight; subscribes to terrain events.
make_light_sampler
    Factory returning a ``light_sampler`` callable for ``build_mesh``.

Quick start
-----------
>>> from torn_apart.core import load_config, EventBus
>>> from torn_apart.core.rng import set_world_seed
>>> from torn_apart.terrain import ChunkManager
>>> from torn_apart.lighting import LightGrid, SunlightComputer, make_light_sampler
>>> set_world_seed(1337)
>>> cfg = load_config()
>>> bus = EventBus()
>>> cm = ChunkManager(cfg, bus)
>>> lg = LightGrid()
>>> sc = SunlightComputer(cfg, cm, lg, bus)
>>> sc.recompute_all_loaded()
>>> sampler = make_light_sampler(lg, cfg)
>>> # pass sampler to cm.stream_frame or cm.mesh_chunk
"""

from torn_apart.lighting.light_grid import (
    LightGrid,
    occupancy_from_materials,
    LIGHT_FULL,
    LIGHT_AMBIENT,
)
from torn_apart.lighting.sunlight import (
    SunlightComputer,
    make_light_sampler,
)

__all__ = [
    "LightGrid",
    "occupancy_from_materials",
    "LIGHT_FULL",
    "LIGHT_AMBIENT",
    "SunlightComputer",
    "make_light_sampler",
]
