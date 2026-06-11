"""
fire_engine.zones — tagged world-space volumes (grass regions, biomes).

Headless foundation package (numpy + core only — NO panda3d, Hard Rule 1).
A :class:`ZoneVolume` is a tagged AABB in world meters; the :class:`ZoneStore`
registry holds them and rides delta saves (``save_key="zones"``).  The
``grass_placement`` module is the testable math behind the GPU-instanced
grass renderer (``world/grass_renderer.py``).

See ``docs/systems/zones.md`` for the full reference.
"""

from fire_engine.zones.volume import ZoneVolume
from fire_engine.zones.store import ZoneStore
from fire_engine.zones.grass_placement import (
    HEIGHT_SENTINEL,
    bake_grass_height_field,
    grass_hash_seed,
    grass_instance_count,
    hash_lowbias32,
    instance_attribs,
)

__all__ = [
    "ZoneVolume",
    "ZoneStore",
    "HEIGHT_SENTINEL",
    "bake_grass_height_field",
    "grass_hash_seed",
    "grass_instance_count",
    "hash_lowbias32",
    "instance_attribs",
]
