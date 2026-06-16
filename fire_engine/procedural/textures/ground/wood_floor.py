"""
procedural/textures/ground/wood_floor.py — "wood_floor" texture.

A 64×64 RGBA pixel-art floorboard texture for buildings: warm worn timber planks
running in parallel boards with subtle grain.  The building renderer samples this
as the albedo for every ``SurfaceMaterial.FLOOR`` face (the per-storey floor
slabs) so interiors read as wooden floors rather than plaster.

Generation algorithm
--------------------
A stretched ``pixel_noise`` field (low frequency across the boards, higher along
them) gives lengthwise grain; a **plank banding** term (a sawtooth in u) darkens
the seam between adjacent boards.  Combined and posterised to a 6-colour warm
timber palette.

Registered as ``"wood_floor"`` at import time via ``@register_def``.

Usage
-----
::

    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural import get

    set_world_seed(1337)
    arr = get("wood_floor")        # np.ndarray (64, 64, 4) uint8
    # python tools/preview_texture.py wood_floor

Docs: docs/systems/procedural.textures.ground.md
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fire_engine.procedural.defs import register_def
from fire_engine.procedural.textures.base import ProceduralTextureDef, pixel_noise, posterise

__all__ = ["WOOD_FLOOR_PALETTE", "WOOD_FLOOR_THRESHOLDS", "WoodFloorDef"]

# Warm worn timber — dark gap → honeyed highlight.
_PALETTE = np.array(
    [
        (74, 50, 32),  # 0 — dark plank seam / knot
        (104, 72, 44),  # 1 — shaded board
        (132, 94, 58),  # 2 — mid timber (dominant)
        (158, 116, 74),  # 3 — worn board
        (182, 142, 96),  # 4 — sun-faded board
        (204, 168, 120),  # 5 — bleached highlight
    ],
    dtype=np.uint8,
)
_THRESHOLDS = np.array([0.16, 0.34, 0.54, 0.74, 0.90], dtype=np.float32)

WOOD_FLOOR_PALETTE = _PALETTE
WOOD_FLOOR_THRESHOLDS = _THRESHOLDS

# Boards across a 64-px tile (≈ one board every ~13 px).
_PLANKS = 5


@register_def
class WoodFloorDef(ProceduralTextureDef):
    """
    Warm worn-timber floorboard texture for buildings.

    Registered name
    ---------------
    ``"wood_floor"``

    Output
    ------
    ``numpy.ndarray`` of shape ``(64, 64, 4)``, dtype ``uint8``; alpha 255.

    Example
    -------
    ::

        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural import get

        set_world_seed(7)
        arr = get("wood_floor")        # (64, 64, 4) uint8

    Docs: docs/systems/procedural.textures.ground.md
    """

    name = "wood_floor"

    DEFAULT_WIDTH = 64
    DEFAULT_HEIGHT = 64

    def generate(self, rng: np.random.Generator, **params: Any) -> np.ndarray:
        """
        Generate the wood floor texture.

        Parameters
        ----------
        rng : numpy.random.Generator
            Seeded generator injected by the registry.
        **params : any
            ``width`` / ``height`` (int) overrides — default 64×64.

        Returns
        -------
        numpy.ndarray
            ``(H, W, 4)`` uint8.

        Docs: docs/systems/procedural.textures.ground.md
        """
        w = int(params.get("width", self.DEFAULT_WIDTH))
        h = int(params.get("height", self.DEFAULT_HEIGHT))
        shape = (h, w)

        grain = pixel_noise(
            rng, shape=shape, octaves=4, persistence=0.5, lacunarity=2.0, base_freq=3
        )
        fleck = pixel_noise(
            rng, shape=shape, octaves=2, persistence=0.5, lacunarity=2.0, base_freq=10
        )
        # Plank seams: sawtooth across u, dark at the board edge.
        u = (np.arange(w, dtype=np.float32) / w)[None, :] * np.ones((h, 1), dtype=np.float32)
        seam = (u * _PLANKS + fleck * 0.15) % 1.0
        plank = 0.4 + 0.6 * np.minimum(seam, 1.0 - seam) * 2.0  # dark at both edges
        combined = grain * 0.6 + np.clip(plank, 0.0, 1.0) * 0.4

        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = posterise(combined, _PALETTE, _THRESHOLDS)
        rgba[..., 3] = 255
        return rgba
