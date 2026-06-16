"""
procedural/textures/ground — ground-surface texture definitions.

Registers:
    ``"dirt_ground"``       — 64×64 RGBA dry post-apocalyptic dirt.
    ``"grass_ground"``      — 64×64 RGBA weathered grass.
    ``"wasteland_ground"``  — 256×256 RGBA cracked dead earth.
    ``"plaster_wall"``      — 64×64 RGBA weathered lime plaster wall.

Private sub-package of ``fire_engine.procedural.textures``; import the
parent package or ``fire_engine.procedural`` instead of this directly.

Docs: docs/systems/procedural.md
"""

from fire_engine.procedural.textures.ground import (
    dirt_ground,
    grass_ground,
    plaster_wall,
    wasteland_ground,
)

__all__ = ["dirt_ground", "grass_ground", "plaster_wall", "wasteland_ground"]
