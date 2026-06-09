"""
procedural/textures — ProceduralTextureDef base class and built-in texture defs.

Importing this package registers all built-in textures in the global
procedural registry:

    * ``"wasteland_ground"`` — 256×256 RGBA dirt/dead-grass ground texture.

Additional textures can be added by creating a new module in this package and
importing it here.  See ``docs/systems/procedural.md`` for the authoring guide.
"""

from torn_apart.procedural.textures.base import ProceduralTextureDef, value_noise
from torn_apart.procedural.textures import wasteland_ground  # registers "wasteland_ground"

__all__ = [
    "ProceduralTextureDef",
    "value_noise",
    "wasteland_ground",
]
