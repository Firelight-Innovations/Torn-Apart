"""
procedural/textures — ProceduralTextureDef base class and built-in texture defs.

Importing this package registers all built-in textures in the global
procedural registry:

    * ``"wasteland_ground"`` — 256×256 RGBA dirt/dead-grass ground texture.
    * ``"night_sky"``        — 1024×512 RGBA equirect star field + galaxy band.
    * ``"night_sky_cube"``   — (6, 512, 512, 4) RGBA cube-map star field +
      galaxy band (GL face order; no pole distortion — the renderer's pick).
    * ``"rain_streak"``      — 128×512 RGBA tiling rain streaks (U+V tileable).
    * ``"grass_ground"``     — 64×64 RGBA pixel-art weathered grass ground.
    * ``"dirt_ground"``      — 64×64 RGBA pixel-art dry dirt/clod ground.
    * ``"moon_surface"``     — 256×256 RGBA lunar disc (maria + craters).
    * ``"grass_tuft"``       — 32×32 RGBA pixel-art grass-blade alpha cutout.
    * ``"dust_mote"``        — 32×32 RGBA soft radial dust/pollen speck (wind).
    * ``"leaf_sprite"``      — 32×96 RGBA leaf-litter atlas, 3 hue variants (wind).
    * ``"flower_sprite"``    — 32×128 RGBA wildflower atlas, 4 hue variants (flora).

(Tree and bush sprites retired: 3-D trees/bushes live in
``procedural/flora/`` — see ``docs/content/tree_species_authoring.md``.)

Additional textures can be added by creating a new module in this package and
importing it here.  See ``docs/systems/procedural.md`` for the authoring guide.
"""

from fire_engine.procedural.textures import (
    dirt_ground,  # registers "dirt_ground"
    dust_mote,  # registers "dust_mote"
    flower_sprite,  # registers "flower_sprite"
    grass_ground,  # registers "grass_ground"
    grass_tuft,  # registers "grass_tuft"
    leaf_sprite,  # registers "leaf_sprite"
    moon_surface,  # registers "moon_surface"
    night_sky,  # registers "night_sky"
    plaster_wall,  # registers "plaster_wall"
    rain_streak,  # registers "rain_streak"
    wasteland_ground,  # registers "wasteland_ground"
)
from fire_engine.procedural.textures.base import ProceduralTextureDef, pixel_noise, value_noise

__all__ = [
    "ProceduralTextureDef",
    "dirt_ground",
    "dust_mote",
    "flower_sprite",
    "grass_ground",
    "grass_tuft",
    "leaf_sprite",
    "moon_surface",
    "night_sky",
    "pixel_noise",
    "plaster_wall",
    "rain_streak",
    "value_noise",
    "wasteland_ground",
]
