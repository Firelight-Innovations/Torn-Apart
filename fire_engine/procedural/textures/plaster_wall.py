"""
procedural/textures/plaster_wall.py — "plaster_wall" texture definition.

A 64×64 RGBA pixel-art wall texture for buildings: weathered off-white lime
plaster with subtle warm/cool tonal drift, faint hairline cracks and grime, and
occasional darker patches where the render has flaked — a quiet, low-contrast
surface that reads as "rendered wall" without fighting the lighting.  The
building renderer samples this as the wall/slab albedo; if it is unavailable the
shader falls back to a flat albedo, so this def is a nice-to-have, not load
bearing.

Generation algorithm
--------------------
Two layered ``pixel_noise`` fields drive a **posterised** 6-colour ramp (hard
thresholds, no interpolation — the engine-wide pixel-art look):

``base_noise``
    Low-frequency field (4 octaves, base_freq=4) — broad lime-wash tonal drift
    (sun-bleached highlights ↔ shaded hollows).
``grime_noise``
    High-frequency field (2 octaves, base_freq=12) — fine speckle: hairline
    cracks, dirt flecks, flaked-render pocks as few-pixel clusters.

Combined ``base*0.7 + grime*0.3`` then posterised to the plaster palette.
Alpha is always 255.

Registered as ``"plaster_wall"`` at import time via ``@register_def``.

Usage
-----
::

    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural import get

    set_world_seed(1337)
    arr = get("plaster_wall")        # np.ndarray (64, 64, 4) uint8
    # python tools/preview_texture.py plaster_wall
"""

from __future__ import annotations

import numpy as np

from fire_engine.procedural.defs import register_def
from fire_engine.procedural.textures.base import ProceduralTextureDef, pixel_noise

__all__ = ["PlasterWallDef", "PLASTER_PALETTE", "PLASTER_THRESHOLDS"]


# Weathered lime plaster — 6 colours, dark flake → bleached highlight.
# Low chroma, slightly warm; no saturated tones (matches the muted world).
_PLASTER_PALETTE = np.array(
    [
        (150, 142, 128),  # 0 — dark flaked patch / deep grime
        (172, 164, 149),  # 1 — shaded hollow
        (193, 185, 170),  # 2 — mid plaster (shadow side)
        (210, 203, 189),  # 3 — base plaster (dominant)
        (224, 218, 205),  # 4 — sun-touched plaster
        (236, 231, 220),  # 5 — bleached highlight
    ],
    dtype=np.uint8,
)

# 5 thresholds → 6 buckets. Weighted toward the mid tones so the wall reads as
# a calm field with only occasional dark flecks / bright catches.
_THRESHOLDS = np.array([0.12, 0.30, 0.52, 0.74, 0.90], dtype=np.float32)

PLASTER_PALETTE = _PLASTER_PALETTE
PLASTER_THRESHOLDS = _THRESHOLDS


def _posterise(field: np.ndarray, palette: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Map a float32 ``(H, W)`` field in [0,1] to a fixed RGB palette."""
    h, w = field.shape
    idx = np.searchsorted(thresholds, field.ravel(), side="right").astype(np.int32)
    np.clip(idx, 0, len(palette) - 1, out=idx)
    return palette[idx].reshape(h, w, 3)


@register_def
class PlasterWallDef(ProceduralTextureDef):
    """
    Weathered lime-plaster wall texture for buildings.

    Registered name
    ---------------
    ``"plaster_wall"``

    Output
    ------
    ``numpy.ndarray`` of shape ``(64, 64, 4)``, dtype ``uint8``; alpha 255.

    Example
    -------
    ::

        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural import get

        set_world_seed(7)
        arr = get("plaster_wall")        # (64, 64, 4) uint8
    """

    name = "plaster_wall"

    DEFAULT_WIDTH = 64
    DEFAULT_HEIGHT = 64

    def generate(self, rng: np.random.Generator, **params) -> np.ndarray:
        """
        Generate the plaster wall texture.

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
        """
        w = int(params.get("width", self.DEFAULT_WIDTH))
        h = int(params.get("height", self.DEFAULT_HEIGHT))
        shape = (h, w)

        base_noise = pixel_noise(
            rng, shape=shape, octaves=4, persistence=0.5, lacunarity=2.0, base_freq=4
        )
        grime_noise = pixel_noise(
            rng, shape=shape, octaves=2, persistence=0.5, lacunarity=2.0, base_freq=12
        )
        combined = base_noise * 0.7 + grime_noise * 0.3

        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = _posterise(combined, _PLASTER_PALETTE, _THRESHOLDS)
        rgba[..., 3] = 255
        return rgba
