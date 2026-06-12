"""
procedural/textures/bush_sprite.py — "bush_sprite" shrub atlas texture.

Produces an **atlas** of pixel-art bush/shrub sprites: one row of 3 variants,
each a 48×48 RGBA cell (144×48 total).  Each cell is a clumpy foliage mass —
a union of seeded elliptical lobes anchored to the bottom of the cell, its
edge and interior broken by crisp :func:`pixel_noise` (holes where the sky
shows through), shaded with a posterised 4-tone ramp (dark interior/bottom,
pale dusty highlights on top) — over a few dark twig pixels at the base.
Variants run living-green → dry-olive → dead-brown, so a scatter of bushes
reads as patchy half-dead scrubland (Vintage Story × RimWorld wasteland).

The flora renderer (``world/flora_renderer.py``) maps one atlas cell per
instance (selected from the instance hash) onto crossed quads inside
``tag="bushes"`` zone volumes; the canopy top sways gently in the wind field.

Atlas layout (the renderer's UV contract)
-----------------------------------------
3 cells laid left→right in one row.  Cell ``k`` occupies horizontal texels
``[k*48, (k+1)*48)``; in the shader, variant ``k`` samples
``u = (k + frac_u) / 3``.  The foliage mass sits on the **bottom image row**
(V=0 after the upload flip), matching the quad geometry's ground edge.

Registered as ``"bush_sprite"`` at import time.

Usage
-----
::

    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural import get

    set_world_seed(1337)
    arr = get("bush_sprite")       # (48, 144, 4) uint8 — 3 bushes in a row
    # Preview: python tools/preview_texture.py bush_sprite
"""

from __future__ import annotations

import numpy as np

from fire_engine.procedural.defs import register_def
from fire_engine.procedural.textures.base import ProceduralTextureDef, pixel_noise

__all__ = ["BushSpriteDef"]

_CELL = 48                 # one bush cell is 48×48
_VARIANTS = 3

# 4-tone posterised foliage ramps per variant, dark → light (RGB).
_BUSH_RAMPS = np.array([
    [(36, 50, 28), (54, 72, 36), (76, 94, 46), (102, 116, 60)],     # 0 living green
    [(46, 50, 26), (70, 74, 36), (96, 94, 46), (124, 114, 58)],     # 1 dry olive
    [(48, 40, 28), (72, 60, 40), (96, 80, 52), (120, 102, 64)],     # 2 dead brown
], dtype=np.uint8)

# Per-variant hole threshold: dead bushes are sparser (more sky shows through).
_HOLE_THRESH = (0.14, 0.18, 0.26)

_TWIG_RGB = np.array((56, 44, 30), dtype=np.uint8)
_BERRY_RGB = np.array((150, 74, 60), dtype=np.uint8)   # dusty rosehip red


@register_def
class BushSpriteDef(ProceduralTextureDef):
    """
    Shrub atlas: one row of 3 condition variants (RGBA alpha cutout).

    Registered name
    ---------------
    ``"bush_sprite"``

    Output
    ------
    ``numpy.ndarray (48, 144, 4) uint8`` — three 48×48 bush cells side by
    side (living green / dry olive / dead brown).  Alpha is binary — render
    with discard, never blending.  Foliage sits on the bottom image row.

    Example
    -------
    ::

        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural import get

        set_world_seed(7)
        arr = get("bush_sprite")
        assert arr.shape == (48, 144, 4)
        assert ((arr[..., 3] == 0) | (arr[..., 3] == 255)).all()
    """

    name = "bush_sprite"

    DEFAULT_WIDTH = _CELL * _VARIANTS          # 144
    DEFAULT_HEIGHT = _CELL                     # 48

    def generate(self, rng: np.random.Generator, **params) -> np.ndarray:
        """
        Generate the 3-variant bush atlas.

        Parameters
        ----------
        rng : numpy.random.Generator
            Seeded generator injected by the registry.
        **params : any
            Optional ``width`` / ``height`` (int) overrides; the width is
            split into 3 equal cells.

        Returns
        -------
        numpy.ndarray
            ``(H, W, 4) uint8`` RGBA bush atlas with binary alpha.
        """
        W = int(params.get("width", self.DEFAULT_WIDTH))
        H = int(params.get("height", self.DEFAULT_HEIGHT))
        cell_w = max(1, W // _VARIANTS)

        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        yy, xx = np.mgrid[0:H, 0:cell_w]
        h_up = (H - 1 - yy).astype(np.float32)         # height above ground row

        for k in range(_VARIANTS):                     # fixed 3-iteration loop
            x0 = k * cell_w

            # --- foliage mass: union of seeded elliptical lobes ------------
            mask = np.zeros((H, cell_w), dtype=bool)
            n_lobes = int(rng.integers(4, 7))
            for _ in range(n_lobes):                   # ≤6 iterations
                cx = float(rng.uniform(cell_w * 0.28, cell_w * 0.72))
                ch = float(rng.uniform(H * 0.22, H * 0.55))    # lobe centre height
                rx = float(rng.uniform(cell_w * 0.18, cell_w * 0.30))
                ry = float(rng.uniform(H * 0.16, H * 0.26))
                mask |= ((xx - cx) / rx) ** 2 + ((h_up - ch) / ry) ** 2 <= 1.0

            # Crisp pixel noise breaks the smooth ellipse edges and punches
            # ragged sky holes through the mass — the chunky retro read.
            noise = pixel_noise(rng, (H, cell_w), octaves=3, base_freq=6)
            mask &= noise > _HOLE_THRESH[k]

            # --- twigs: short dark verticals from the ground into the mass --
            for _ in range(3):                          # 3 iterations
                tx = int(rng.integers(int(cell_w * 0.35), int(cell_w * 0.65)))
                th = int(rng.integers(int(H * 0.12), int(H * 0.3)))
                trows = np.arange(H - th, H)
                rgba[trows, x0 + tx, :3] = _TWIG_RGB
                rgba[trows, x0 + tx, 3] = 255

            # --- posterised shading: vertical light + noise blotch ----------
            ramp = _BUSH_RAMPS[k]
            tier = np.clip(((h_up / H) * 0.5 + noise * 0.6) * len(ramp),
                           0, len(ramp) - 1).astype(np.intp)
            cell_rgb = ramp[tier]                       # (H, cell_w, 3)
            rgba[:, x0:x0 + cell_w, :3][mask] = cell_rgb[mask]
            rgba[:, x0:x0 + cell_w, 3][mask] = 255

            # --- a few berry specks on the living variant -------------------
            if k == 0:
                bx = rng.integers(0, cell_w, size=12)
                by = rng.integers(0, H, size=12)
                on = mask[by, bx]
                rgba[by[on], x0 + bx[on], :3] = _BERRY_RGB

        return rgba
