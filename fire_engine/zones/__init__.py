"""
fire_engine.zones — tagged world-space volumes (grass regions, biomes).

Headless foundation package (numpy + core only — NO panda3d, Hard Rule 1).
A :class:`ZoneVolume` is a tagged AABB in world meters; the :class:`ZoneStore`
registry holds them and rides delta saves (``save_key="zones"``).  The
``grass_placement`` module is the testable math behind the GPU-instanced
grass renderer (``world/grass_renderer.py``).

See ``docs/systems/zones.md`` for the full reference.
"""

from fire_engine.zones.flora_placement import (
    FLORA_KINDS,
    flora_hash_seed,
    flora_instance_attribs,
    flora_instance_count,
)
from fire_engine.zones.grass_placement import (
    HEIGHT_SENTINEL,
    bake_grass_height_field,
    grass_hash_seed,
    grass_instance_count,
    hash_lowbias32,
    instance_attribs,
    leaf_hash_seed,
    leaf_instance_count,
)
from fire_engine.zones.store import ZoneStore
from fire_engine.zones.tree_placement import (
    SCALE_JITTER,
    TREE_KINDS,
    TreeInstances,
    bake_tree_instances,
    instances_data_block,
    species_mix_from_params,
)
from fire_engine.zones.volume import ZoneVolume

__all__ = [
    "FLORA_KINDS",
    "HEIGHT_SENTINEL",
    "SCALE_JITTER",
    "TREE_KINDS",
    "TreeInstances",
    "ZoneStore",
    "ZoneVolume",
    "bake_grass_height_field",
    "bake_tree_instances",
    "flora_hash_seed",
    "flora_instance_attribs",
    "flora_instance_count",
    "grass_hash_seed",
    "grass_instance_count",
    "hash_lowbias32",
    "instance_attribs",
    "instances_data_block",
    "leaf_hash_seed",
    "leaf_instance_count",
    "species_mix_from_params",
]
