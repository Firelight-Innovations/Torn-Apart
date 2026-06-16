"""
procedural/textures/dust_mote.py — "dust_mote" soft radial speck texture.

Produces a small RGBA texture of a single soft dust/pollen mote: a radial
falloff disc (bright opaque centre fading to transparent edge) with a slight
deterministic noise asymmetry so a field of them does not read as a grid of
identical perfect circles.  The dust-mote renderer (``world/mote_renderer.py``)
billboards it on every instance and **additive**-blends the lot, so the alpha
channel here is the falloff mask the additive blend reads (the RGB is a warm
off-white pollen tint, scaled by alpha at composite time).

Unlike ``grass_tuft`` this is NOT a binary cutout — additive blending wants a
smooth alpha ramp so motes glow softly at the centre and vanish at the rim with
no hard edge.

Generation
----------
Pure numpy, no per-pixel Python loops:

1. Radial distance field from the texture centre (one ``np.hypot`` of two
   ``np.linspace`` meshes).
2. A smooth ``smoothstep``-style radial falloff → the base alpha disc.
3. A low-frequency :func:`value_noise` field warps the radius slightly
   (asymmetry) and dapples the alpha so each speck is a touch irregular.
4. RGB is a constant warm off-white; alpha carries the soft mask.

Registered as ``"dust_mote"`` at import time.

Usage
-----
::

    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural import get

    set_world_seed(1337)
    arr = get("dust_mote")        # (32, 32, 4) uint8, soft radial alpha
    # Preview: python tools/preview_texture.py dust_mote

Docs: docs/systems/procedural.textures.sprites.md
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fire_engine.procedural.defs import register_def
from fire_engine.procedural.textures.base import ProceduralTextureDef, value_noise

__all__ = ["DustMoteDef"]

# Warm off-white pollen/dust tint (RGB).  Additive blend scales this by alpha,
# so a slightly warm, slightly bright colour reads as glowing motes in sunlight.
_MOTE_RGB = np.array([238, 232, 214], dtype=np.float32)


@register_def
class DustMoteDef(ProceduralTextureDef):
    """
    Soft radial dust/pollen speck (RGBA, smooth alpha falloff).

    Registered name
    ---------------
    ``"dust_mote"``

    Output
    ------
    ``numpy.ndarray (32, 32, 4) uint8``.  RGB is a constant warm off-white;
    **alpha** is a smooth radial falloff (opaque-ish centre → 0 at the rim)
    with a slight per-texture noise asymmetry.  Render **additively** with
    depth-write off (the renderer reads the alpha as the additive mask).

    Example
    -------
    ::

        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural import get

        set_world_seed(7)
        arr = get("dust_mote")
        assert arr.shape == (32, 32, 4)
        assert arr[16, 16, 3] > arr[0, 0, 3]   # centre brighter than corner

    Docs: docs/systems/procedural.textures.sprites.md
    """

    name = "dust_mote"

    DEFAULT_WIDTH = 32
    DEFAULT_HEIGHT = 32

    def generate(self, rng: np.random.Generator, **params: Any) -> np.ndarray:
        """
        Generate the soft radial speck.

        Parameters
        ----------
        rng : numpy.random.Generator
            Seeded generator injected by the registry.
        **params : any
            Optional ``width`` / ``height`` (int) pixel-size overrides.

        Returns
        -------
        numpy.ndarray
            ``(H, W, 4) uint8`` RGBA, smooth alpha mask.

        Docs: docs/systems/procedural.textures.sprites.md
        """
        W = int(params.get("width", self.DEFAULT_WIDTH))
        H = int(params.get("height", self.DEFAULT_HEIGHT))

        # Normalised coordinates in [-1, 1] about the centre.
        ys = np.linspace(-1.0, 1.0, H, dtype=np.float32)
        xs = np.linspace(-1.0, 1.0, W, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)  # (H, W) each

        # Low-frequency noise (in [0,1]) → ±asymmetry on the radius + alpha
        # dapple, so specks are subtly irregular rather than perfect circles.
        noise = value_noise(rng, (H, W), octaves=2, base_freq=2)  # (H, W)
        warp = (noise - 0.5) * 0.22  # ~±0.11 radial wobble

        radius = np.hypot(gx, gy) * (1.0 + warp)

        # Smooth radial falloff: 1 at centre, 0 by the rim (smoothstep).
        # Inner core stays near-opaque, outer half fades to nothing.
        t = np.clip((1.0 - radius) / 0.85, 0.0, 1.0)
        alpha = t * t * (3.0 - 2.0 * t)  # smoothstep(0,1,t)
        # Dapple the alpha a little with the same noise so motes shimmer.
        alpha = alpha * (0.82 + 0.18 * noise)
        alpha = np.clip(alpha, 0.0, 1.0)

        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[..., 0] = np.uint8(_MOTE_RGB[0])
        rgba[..., 1] = np.uint8(_MOTE_RGB[1])
        rgba[..., 2] = np.uint8(_MOTE_RGB[2])
        rgba[..., 3] = np.clip(np.round(alpha * 255.0), 0, 255).astype(np.uint8)
        return rgba
