"""
procedural/textures/sprites — sprite / particle texture definitions.

Registers:
    ``"dust_mote"``      — 32×32 RGBA soft radial dust/pollen speck.
    ``"flower_sprite"``  — 32×128 RGBA wildflower atlas, 4 hue variants.
    ``"grass_tuft"``     — 32×32 RGBA pixel-art grass-blade alpha cutout.
    ``"leaf_sprite"``    — 32×96 RGBA leaf-litter atlas, 3 hue variants.

Private sub-package of ``fire_engine.procedural.textures``; import the
parent package or ``fire_engine.procedural`` instead of this directly.

Docs: docs/systems/procedural.md
"""

from fire_engine.procedural.textures.sprites import (
    dust_mote,
    flower_sprite,
    grass_tuft,
    leaf_sprite,
)

__all__ = ["dust_mote", "flower_sprite", "grass_tuft", "leaf_sprite"]
