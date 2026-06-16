"""
procedural/textures/flower_sprite.py — "flower_sprite" wildflower atlas texture.

Produces a small **atlas** of pixel-art wildflower sprites: one row of 4
variants, each a 32×32 RGBA cell (128×32 total).  Each cell holds a single
flower — a leaning green stem with a leaf nub or two, topped by a chunky
petal rosette around a dark seed-head — in one of four post-apocalyptic-muted
hues (yarrow off-white, dusty yellow, faded violet, washed poppy red).  The
silhouette IS the geometry (binary alpha, discard-rendered), the same
Daggerfall-style cutout idiom as ``grass_tuft``.

The flora renderer (``world/flora_renderer.py``) maps one atlas cell per
instance (selected from the instance hash) onto crossed quads inside
``tag="flowers"`` zone volumes; flowers sway with the wind field exactly like
grass blades.

Atlas layout (the renderer's UV contract)
-----------------------------------------
4 cells laid left→right in one row.  Cell ``k`` occupies horizontal texels
``[k*32, (k+1)*32)``; in the shader, variant ``k`` samples
``u = (k + frac_u) / 4``.  Stem bases sit on the **bottom image row** (V=0
after ``texture_bridge.to_panda_texture``'s upload flip), matching the quad
geometry whose V=0 edge is at ground level.

Generation
----------
A fixed loop over the 4 cells (and a handful of petals each) — never
per-pixel.  Stem lean/height, petal count/radius and leaf placement come from
the injected RNG; colours are fixed posterised ramps so every seed stays
inside the art direction.

Registered as ``"flower_sprite"`` at import time.

Usage
-----
::

    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural import get

    set_world_seed(1337)
    arr = get("flower_sprite")     # (32, 128, 4) uint8 — 4 flowers in a row
    # Preview: python tools/preview_texture.py flower_sprite
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fire_engine.procedural.defs import register_def
from fire_engine.procedural.textures.base import ProceduralTextureDef

__all__ = ["FlowerSpriteDef"]

_CELL = 32  # one flower cell is 32×32
_VARIANTS = 4

# Petal hue per variant (RGB).  Everything dusty and desaturated — RimWorld ×
# Morrowind wildflowers, nothing saturated enough to read as a power-up.
_PETAL_HUES = np.array(
    [
        (198, 190, 170),  # 0 — yarrow off-white
        (190, 158, 66),  # 1 — dusty yellow
        (138, 112, 158),  # 2 — faded violet
        (168, 86, 72),  # 3 — washed poppy red
    ],
    dtype=np.float32,
)

# Stem colour ramp, base (dark) → top (lighter); matches grass_tuft greens.
_STEM_RAMP = np.array(
    [
        (46, 64, 34),
        (68, 90, 44),
        (94, 112, 54),
    ],
    dtype=np.uint8,
)

_CENTER_RGB = np.array((96, 76, 36), dtype=np.uint8)  # seed-head amber


@register_def
class FlowerSpriteDef(ProceduralTextureDef):
    """
    Wildflower atlas: one row of 4 hue variants (RGBA alpha cutout).

    Registered name
    ---------------
    ``"flower_sprite"``

    Output
    ------
    ``numpy.ndarray (32, 128, 4) uint8`` — four 32×32 flower cells side by
    side.  Alpha is binary (255 on flower pixels, 0 elsewhere) — render with
    discard, never blending.  Stem bases touch the bottom image row.

    Example
    -------
    ::

        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural import get

        set_world_seed(7)
        arr = get("flower_sprite")
        assert arr.shape == (32, 128, 4)
        assert ((arr[..., 3] == 0) | (arr[..., 3] == 255)).all()
    """

    name = "flower_sprite"

    DEFAULT_WIDTH = _CELL * _VARIANTS  # 128
    DEFAULT_HEIGHT = _CELL  # 32

    def generate(self, rng: np.random.Generator, **params: Any) -> np.ndarray:
        """
        Generate the 4-variant wildflower atlas.

        Parameters
        ----------
        rng : numpy.random.Generator
            Seeded generator injected by the registry.
        **params : any
            Optional ``width`` / ``height`` (int) overrides; the width is
            split into 4 equal cells.

        Returns
        -------
        numpy.ndarray
            ``(H, W, 4) uint8`` RGBA flower atlas with binary alpha.
        """
        W = int(params.get("width", self.DEFAULT_WIDTH))
        H = int(params.get("height", self.DEFAULT_HEIGHT))
        cell_w = max(1, W // _VARIANTS)

        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        n_tiers = len(_STEM_RAMP)
        yy, xx = np.mgrid[0:H, 0:cell_w]  # cell-local pixel grids

        for k in range(_VARIANTS):  # fixed 4-iteration loop
            x0 = k * cell_w

            # --- head size first, so the stem leaves room for it -----------
            head_r = float(rng.uniform(3.5, 5.5))
            petal_r = float(rng.uniform(1.8, 2.8))
            margin = head_r + petal_r + 1.0

            # --- stem: leaning quadratic curve, like a grass blade ---------
            base_x = float(rng.uniform(cell_w * 0.38, cell_w * 0.62))
            height_px = int(rng.uniform(H * 0.5, H - margin))
            lean_px = float(rng.uniform(-cell_w * 0.12, cell_w * 0.12))

            ys = np.arange(height_px)
            t = ys / max(height_px - 1, 1)
            sx = np.clip(np.round(base_x + lean_px * t * t), 0, cell_w - 1).astype(np.intp)
            rows = (H - 1 - ys).astype(np.intp)
            tiers = np.minimum((t * n_tiers).astype(np.intp), n_tiers - 1)
            rgba[rows, x0 + sx, :3] = _STEM_RAMP[tiers]
            rgba[rows, x0 + sx, 3] = 255

            # --- one or two leaf nubs: short diagonal runs off the stem ----
            for frac in (0.35, 0.6)[: int(rng.integers(1, 3))]:
                iy = int(frac * (height_px - 1))
                side = 1 if rng.uniform() < 0.5 else -1
                run = np.arange(1, int(rng.integers(3, 5)))
                lx = np.clip(sx[iy] + side * run, 0, cell_w - 1)
                lr = np.clip(rows[iy] - (run // 2), 0, H - 1)
                rgba[lr, x0 + lx, :3] = _STEM_RAMP[1]
                rgba[lr, x0 + lx, 3] = 255

            # --- petal rosette around the stem tip --------------------------
            hx = float(sx[-1])
            hy = float(rows[-1]) - 1.0  # head centre, just above tip
            hue = _PETAL_HUES[k]
            n_petals = int(rng.integers(5, 8))
            angles = np.arange(n_petals) * 2.0 * np.pi / n_petals + float(
                rng.uniform(0.0, 2.0 * np.pi)
            )
            # Two-tone petal shading: upper half of the head catches the sky.
            shade = np.where(yy < hy, 1.06, 0.85)
            for a in angles:  # ≤7 iterations, blobs are numpy
                px = hx + np.cos(a) * head_r * 0.62
                py = hy + np.sin(a) * head_r * 0.45  # squash → rosette reads from the side too
                blob = (xx - px) ** 2 + (yy - py) ** 2 <= petal_r**2
                rgb = np.clip(hue[None, None, :] * shade[..., None], 0, 255)
                rgba[:, x0 : x0 + cell_w, :3][blob] = rgb[blob].astype(np.uint8)
                rgba[:, x0 : x0 + cell_w, 3][blob] = 255

            # --- dark seed-head centre overwrites the petal inner edge -----
            centre = (xx - hx) ** 2 + (yy - hy) ** 2 <= (head_r * 0.4) ** 2
            rgba[:, x0 : x0 + cell_w, :3][centre] = _CENTER_RGB
            rgba[:, x0 : x0 + cell_w, 3][centre] = 255

        return rgba
