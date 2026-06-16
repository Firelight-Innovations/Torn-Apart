"""
procedural/textures/ground/roof_shingle.py — "roof_shingle" texture.

A 64×64 RGBA pixel-art roof texture for buildings: weathered grey-brown
overlapping shingles/slates laid in horizontal courses.  The building renderer
samples this as the albedo for every ``SurfaceMaterial.ROOF`` face (pitched
planes and flat roof slabs) so roofs read distinctly from the plastered walls.

Generation algorithm
--------------------
A low-frequency ``pixel_noise`` field gives broad tonal weathering; a horizontal
**course banding** term (a sawtooth in v, jittered per row by a second noise
field) darkens the bottom edge of each shingle row so the courses read as
overlapping tiles.  Combined and posterised to a 6-colour slate palette.

Registered as ``"roof_shingle"`` at import time via ``@register_def``.

Usage
-----
::

    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural import get

    set_world_seed(1337)
    arr = get("roof_shingle")        # np.ndarray (64, 64, 4) uint8
    # python tools/preview_texture.py roof_shingle

Docs: docs/systems/procedural.textures.ground.md
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fire_engine.procedural.defs import register_def
from fire_engine.procedural.textures.base import ProceduralTextureDef, pixel_noise, posterise

__all__ = ["ROOF_SHINGLE_PALETTE", "ROOF_SHINGLE_THRESHOLDS", "RoofShingleDef"]

# Weathered slate — cool grey-brown, dark course shadow → lichen-touched highlight.
_PALETTE = np.array(
    [
        (54, 52, 58),  # 0 — deep course shadow
        (74, 70, 74),  # 1 — shaded slate
        (92, 86, 86),  # 2 — mid slate (dominant)
        (110, 103, 99),  # 3 — exposed slate
        (128, 122, 112),  # 4 — sun-touched / faded
        (146, 142, 126),  # 5 — lichen / bleached highlight
    ],
    dtype=np.uint8,
)
_THRESHOLDS = np.array([0.16, 0.34, 0.54, 0.74, 0.90], dtype=np.float32)

ROOF_SHINGLE_PALETTE = _PALETTE
ROOF_SHINGLE_THRESHOLDS = _THRESHOLDS

# Shingle courses per 64-px tile (≈ one course every 8 px).
_COURSES = 8


@register_def
class RoofShingleDef(ProceduralTextureDef):
    """
    Weathered slate/shingle roof texture for buildings.

    Registered name
    ---------------
    ``"roof_shingle"``

    Output
    ------
    ``numpy.ndarray`` of shape ``(64, 64, 4)``, dtype ``uint8``; alpha 255.

    Example
    -------
    ::

        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural import get

        set_world_seed(7)
        arr = get("roof_shingle")        # (64, 64, 4) uint8

    Docs: docs/systems/procedural.textures.ground.md
    """

    name = "roof_shingle"

    DEFAULT_WIDTH = 64
    DEFAULT_HEIGHT = 64

    def generate(self, rng: np.random.Generator, **params: Any) -> np.ndarray:
        """
        Generate the roof shingle texture.

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

        weather = pixel_noise(
            rng, shape=shape, octaves=4, persistence=0.5, lacunarity=2.0, base_freq=4
        )
        jitter = pixel_noise(
            rng, shape=shape, octaves=1, persistence=0.5, lacunarity=2.0, base_freq=6
        )
        # Horizontal course banding: sawtooth in v darkened at each row's base.
        v = (np.arange(h, dtype=np.float32) / h)[:, None] * np.ones((1, w), dtype=np.float32)
        band = (v * _COURSES + jitter * 0.35) % 1.0  # 0 at top of course → 1 at base
        course = 0.35 + 0.65 * band  # dark line at the overlap (band→0)
        combined = weather * 0.6 + course * 0.4

        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = posterise(combined, _PALETTE, _THRESHOLDS)
        rgba[..., 3] = 255
        return rgba
