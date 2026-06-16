"""
procedural/textures/ground/stone_foundation.py — "stone_foundation" texture.

A 64×64 RGBA pixel-art stone texture for buildings: rough grey coursed rubble
masonry.  The building renderer samples this as the albedo for every
``SurfaceMaterial.FOUNDATION`` face (the foundation slab below local z=0) so the
base course reads as stone footing under the plastered walls.

Generation algorithm
--------------------
Two ``pixel_noise`` fields drive a blocky mortar grid: a low-frequency tonal
field for stone-to-stone colour variation, and a **block grid** term (the
distance to the nearest mortar line on a jittered lattice) that darkens the
mortar joints.  Combined and posterised to a 6-colour cool-grey stone palette.

Registered as ``"stone_foundation"`` at import time via ``@register_def``.

Usage
-----
::

    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural import get

    set_world_seed(1337)
    arr = get("stone_foundation")        # np.ndarray (64, 64, 4) uint8
    # python tools/preview_texture.py stone_foundation

Docs: docs/systems/procedural.textures.ground.md
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fire_engine.procedural.defs import register_def
from fire_engine.procedural.textures.base import ProceduralTextureDef, pixel_noise, posterise

__all__ = ["STONE_FOUNDATION_PALETTE", "STONE_FOUNDATION_THRESHOLDS", "StoneFoundationDef"]

# Cool grey rubble — dark mortar joint → pale weathered stone.
_PALETTE = np.array(
    [
        (52, 52, 56),  # 0 — deep mortar joint
        (78, 78, 82),  # 1 — shaded stone
        (104, 104, 106),  # 2 — mid stone (dominant)
        (128, 127, 124),  # 3 — exposed stone
        (152, 150, 144),  # 4 — sun-touched stone
        (176, 173, 165),  # 5 — pale weathered highlight
    ],
    dtype=np.uint8,
)
_THRESHOLDS = np.array([0.18, 0.36, 0.56, 0.74, 0.90], dtype=np.float32)

STONE_FOUNDATION_PALETTE = _PALETTE
STONE_FOUNDATION_THRESHOLDS = _THRESHOLDS

# Stone blocks across a 64-px tile.
_BLOCKS = 6


@register_def
class StoneFoundationDef(ProceduralTextureDef):
    """
    Coursed grey rubble-masonry foundation texture for buildings.

    Registered name
    ---------------
    ``"stone_foundation"``

    Output
    ------
    ``numpy.ndarray`` of shape ``(64, 64, 4)``, dtype ``uint8``; alpha 255.

    Example
    -------
    ::

        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural import get

        set_world_seed(7)
        arr = get("stone_foundation")        # (64, 64, 4) uint8

    Docs: docs/systems/procedural.textures.ground.md
    """

    name = "stone_foundation"

    DEFAULT_WIDTH = 64
    DEFAULT_HEIGHT = 64

    def generate(self, rng: np.random.Generator, **params: Any) -> np.ndarray:
        """
        Generate the stone foundation texture.

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

        tone = pixel_noise(
            rng, shape=shape, octaves=4, persistence=0.5, lacunarity=2.0, base_freq=5
        )
        jitter = pixel_noise(
            rng, shape=shape, octaves=1, persistence=0.5, lacunarity=2.0, base_freq=8
        )
        # Blocky mortar grid: brighten toward block centres, dark at the joints.
        u = (np.arange(w, dtype=np.float32) / w)[None, :] * np.ones((h, 1), dtype=np.float32)
        v = (np.arange(h, dtype=np.float32) / h)[:, None] * np.ones((1, w), dtype=np.float32)
        # Offset every other course by half a block (running-bond masonry).
        row = np.floor(v * _BLOCKS)
        u_off = u + 0.5 * (row % 2.0) / _BLOCKS
        cu = np.abs(((u_off * _BLOCKS + jitter * 0.2) % 1.0) - 0.5) * 2.0
        cv = np.abs(((v * _BLOCKS) % 1.0) - 0.5) * 2.0
        mortar = np.minimum(cu, cv)  # 0 at block centre → 1 at joint
        block = 0.85 - 0.65 * mortar  # bright centre, dark joint
        combined = tone * 0.55 + np.clip(block, 0.0, 1.0) * 0.45

        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = posterise(combined, _PALETTE, _THRESHOLDS)
        rgba[..., 3] = 255
        return rgba
