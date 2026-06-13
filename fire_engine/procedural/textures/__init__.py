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

from fire_engine.procedural.textures.base import ProceduralTextureDef, value_noise, pixel_noise
from fire_engine.procedural.textures import wasteland_ground  # registers "wasteland_ground"
from fire_engine.procedural.textures import night_sky         # registers "night_sky"
from fire_engine.procedural.textures import rain_streak       # registers "rain_streak"
from fire_engine.procedural.textures import grass_ground      # registers "grass_ground"
from fire_engine.procedural.textures import dirt_ground       # registers "dirt_ground"
from fire_engine.procedural.textures import moon_surface      # registers "moon_surface"
from fire_engine.procedural.textures import grass_tuft        # registers "grass_tuft"
from fire_engine.procedural.textures import dust_mote         # registers "dust_mote"
from fire_engine.procedural.textures import leaf_sprite       # registers "leaf_sprite"
from fire_engine.procedural.textures import flower_sprite     # registers "flower_sprite"
from fire_engine.procedural.textures import plaster_wall      # registers "plaster_wall"

__all__ = [
    "ProceduralTextureDef",
    "value_noise",
    "pixel_noise",
    "wasteland_ground",
    "night_sky",
    "rain_streak",
    "grass_ground",
    "dirt_ground",
    "moon_surface",
    "grass_tuft",
    "dust_mote",
    "leaf_sprite",
    "flower_sprite",
    "plaster_wall",
]
