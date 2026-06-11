"""
procedural/textures/grass_tuft.py — "grass_tuft" blade-silhouette texture.

Produces a small RGBA **alpha-cutout** texture of a grass tuft: a handful of
thin pixel-art blades rising from the bottom edge, each leaning slightly and
shading from a dark base to a pale dried tip.  The GPU grass renderer
(``world/grass_renderer.py``) maps it onto three crossed quads per instance;
the fragment shader discards texels with alpha < 0.5, so the silhouette IS
the blade geometry — Daggerfall/Morrowind-style chunky vegetation.

Orientation contract
--------------------
Blade **bases sit on the bottom row of the image**.  ``texture_bridge.
to_panda_texture`` flips images vertically on upload (UV origin bottom-left),
so the bases land at V=0 — matching the tuft quads, whose V=0 edge is at
ground level.

Generation
----------
A loop over ~9 blades (a handful of iterations — not a per-pixel loop; the
per-blade pixel work is numpy).  Each blade gets a deterministic base column,
height, lean and width from the injected RNG; lateral offset grows
quadratically with height (blades curve, not tilt), and colour is posterised
into a fixed 5-green ramp by height tier.

Registered as ``"grass_tuft"`` at import time.

Usage
-----
::

    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural import get

    set_world_seed(1337)
    arr = get("grass_tuft")       # (32, 32, 4) uint8, alpha mask
    # Preview: python tools/preview_texture.py grass_tuft
"""

from __future__ import annotations

import numpy as np

from fire_engine.procedural.defs import register_def
from fire_engine.procedural.textures.base import ProceduralTextureDef

__all__ = ["GrassTuftDef"]


# Blade colour ramp, base (dark) → tip (pale dried highlight).  Slightly
# lighter than the ground's _GRASS_PALETTE mid-greens so tufts read against it.
_BLADE_RAMP = np.array([
    ( 52,  72,  36),   # 0 — base shadow green
    ( 74,  96,  46),   # 1 — lower stem
    ( 98, 118,  56),   # 2 — mid blade
    (130, 142,  70),   # 3 — upper blade
    (158, 152,  86),   # 4 — dried tip highlight
], dtype=np.uint8)

_BLADE_COUNT = 9


@register_def
class GrassTuftDef(ProceduralTextureDef):
    """
    Pixel-art grass tuft silhouette (RGBA alpha cutout).

    Registered name
    ---------------
    ``"grass_tuft"``

    Output
    ------
    ``numpy.ndarray (32, 32, 4) uint8``.  Alpha is 255 on blade pixels and
    0 everywhere else (binary mask — render with discard, never blending).
    Blade bases touch the bottom image row (V=0 after upload flip).

    Example
    -------
    ::

        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural import get

        set_world_seed(7)
        arr = get("grass_tuft")
        assert ((arr[..., 3] == 0) | (arr[..., 3] == 255)).all()
    """

    name = "grass_tuft"

    DEFAULT_WIDTH = 32
    DEFAULT_HEIGHT = 32

    def generate(self, rng: np.random.Generator, **params) -> np.ndarray:
        """
        Generate the tuft silhouette.

        Parameters
        ----------
        rng : numpy.random.Generator
            Seeded generator injected by the registry.
        **params : any
            Optional ``width`` / ``height`` (int) pixel-size overrides and
            ``blades`` (int) blade-count override.

        Returns
        -------
        numpy.ndarray
            ``(H, W, 4) uint8`` RGBA with binary alpha.
        """
        W = int(params.get("width", self.DEFAULT_WIDTH))
        H = int(params.get("height", self.DEFAULT_HEIGHT))
        blades = int(params.get("blades", _BLADE_COUNT))

        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        n_tiers = len(_BLADE_RAMP)

        for _ in range(blades):                      # ~9 iterations, not per-pixel
            base_x = float(rng.uniform(W * 0.12, W * 0.88))
            height_px = int(rng.uniform(H * 0.45, H * 0.95))
            lean_px = float(rng.uniform(-W * 0.18, W * 0.18))
            wide = bool(rng.uniform() < 0.4)         # some blades 2 px wide

            ys = np.arange(height_px)                # 0 = base
            t = ys / max(height_px - 1, 1)           # 0..1 along the blade
            xs = np.clip(np.round(base_x + lean_px * t * t), 0, W - 1)
            xs = xs.astype(np.intp)
            rows = (H - 1 - ys).astype(np.intp)      # base on bottom image row
            tiers = np.minimum((t * n_tiers).astype(np.intp), n_tiers - 1)

            rgba[rows, xs, :3] = _BLADE_RAMP[tiers]
            rgba[rows, xs, 3] = 255
            if wide:
                lower = ys < height_px * 0.6         # widen the lower stem only
                xs2 = np.clip(xs[lower] + 1, 0, W - 1)
                rgba[rows[lower], xs2, :3] = _BLADE_RAMP[tiers[lower]]
                rgba[rows[lower], xs2, 3] = 255

        return rgba
