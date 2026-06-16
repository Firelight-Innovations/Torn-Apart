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

Texture def modules are organised into category sub-packages for the deep-and-
narrow structure rule: ``ground/`` (dirt/grass/wasteland/plaster), ``sprites/``
(dust/flower/grass-tuft/leaf) and ``sky/`` (moon/rain-streak/night-sky). Import
a def's symbols from its sub-package, e.g.
``from fire_engine.procedural.textures.ground.dirt_ground import DIRT_PALETTE``;
the registry itself is keyed by def NAME, so ``get("dirt_ground")`` is unchanged.

Docs: docs/systems/procedural.md
"""

from __future__ import annotations

# Import sub-packages first — triggers the @register_def decorators for all
# built-ins so a bare ``import fire_engine.procedural`` registers everything.
import fire_engine.procedural.textures.ground
import fire_engine.procedural.textures.sky
import fire_engine.procedural.textures.sprites  # noqa: F401  triggers sub-pkg @register_def
from fire_engine.procedural.textures.base import ProceduralTextureDef, pixel_noise, value_noise
from fire_engine.procedural.textures.ground import (
    dirt_ground,
    grass_ground,
    plaster_wall,
    wasteland_ground,
)
from fire_engine.procedural.textures.sky import (
    moon_surface,
    night_sky,
    rain_streak,
)
from fire_engine.procedural.textures.sprites import (
    dust_mote,
    flower_sprite,
    grass_tuft,
    leaf_sprite,
)

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
