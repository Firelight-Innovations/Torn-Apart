"""
procedural/textures/ground — ground-surface texture definitions.

Registers:
    ``"dirt_ground"``       — 64×64 RGBA dry post-apocalyptic dirt.
    ``"grass_ground"``      — 64×64 RGBA weathered grass.
    ``"wasteland_ground"``  — 256×256 RGBA cracked dead earth.
    ``"plaster_wall"``      — 64×64 RGBA weathered lime plaster wall.
    ``"roof_shingle"``      — 64×64 RGBA weathered slate roof shingles.
    ``"wood_floor"``        — 64×64 RGBA warm timber floorboards.
    ``"stone_foundation"``  — 64×64 RGBA coursed grey rubble masonry.

Private sub-package of ``fire_engine.procedural.textures``; import the
parent package or ``fire_engine.procedural`` instead of this directly.

Docs: docs/systems/procedural.md
"""

from fire_engine.procedural.textures.ground import (
    dirt_ground,
    grass_ground,
    plaster_wall,
    roof_shingle,
    stone_foundation,
    wasteland_ground,
    wood_floor,
)

__all__ = [
    "dirt_ground",
    "grass_ground",
    "plaster_wall",
    "roof_shingle",
    "stone_foundation",
    "wasteland_ground",
    "wood_floor",
]
