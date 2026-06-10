"""
procedural/textures — ProceduralTextureDef base class and built-in texture defs.

Importing this package registers all built-in textures in the global
procedural registry:

    * ``"wasteland_ground"`` — 256×256 RGBA dirt/dead-grass ground texture.
    * ``"night_sky"``        — 1024×512 RGBA equirect star field + galaxy band.
    * ``"rain_streak"``      — 128×512 RGBA tiling rain streaks (U+V tileable).

Additional textures can be added by creating a new module in this package and
importing it here.  See ``docs/systems/procedural.md`` for the authoring guide.
"""

from torn_apart.procedural.textures.base import ProceduralTextureDef, value_noise
from torn_apart.procedural.textures import wasteland_ground  # registers "wasteland_ground"
from torn_apart.procedural.textures import night_sky         # registers "night_sky"
from torn_apart.procedural.textures import rain_streak       # registers "rain_streak"

__all__ = [
    "ProceduralTextureDef",
    "value_noise",
    "wasteland_ground",
    "night_sky",
    "rain_streak",
]
