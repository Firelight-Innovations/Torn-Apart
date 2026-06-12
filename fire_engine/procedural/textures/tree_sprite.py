"""
procedural/textures/tree_sprite.py — "tree_sprite" tree atlas texture.

Produces an **atlas** of pixel-art whole-tree sprites: one row of 3 variants,
each a 64-wide × 96-tall RGBA cell (192×96 total).  Each cell is a complete
tree — a tapered, leaning bark-striated trunk with a couple of branch stubs,
under a blobby canopy built from seeded elliptical lobes whose edges and
interior are broken by crisp :func:`pixel_noise` (ragged outline + sky
holes), shaded with a posterised 5-tone foliage ramp (dark under-canopy,
pale dusty crown).  Variants run full green → autumn ochre → scraggly
half-dead, so a treeline reads as patchy recovering wasteland — the
Daggerfall-billboard × Vintage Story art direction.

The flora renderer (``world/flora_renderer.py``) maps one atlas cell per
instance (selected from the instance hash) onto large crossed quads inside
``tag="trees"`` zone volumes — the same volumes the wind system's leaf
litter already scatters under, so trees and their fallen leaves arrive
together.  The canopy sways in the wind field; the trunk stays pinned.

Atlas layout (the renderer's UV contract)
-----------------------------------------
3 cells laid left→right in one row.  Cell ``k`` occupies horizontal texels
``[k*64, (k+1)*64)``; in the shader, variant ``k`` samples
``u = (k + frac_u) / 3``.  Trunk bases sit on the **bottom image row** (V=0
after the upload flip).  The cell is 2:3 wide:tall — the renderer's quads
keep that aspect so texels stay square.

Registered as ``"tree_sprite"`` at import time.

Usage
-----
::

    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural import get

    set_world_seed(1337)
    arr = get("tree_sprite")       # (96, 192, 4) uint8 — 3 trees in a row
    # Preview: python tools/preview_texture.py tree_sprite
"""

from __future__ import annotations

import numpy as np

from fire_engine.procedural.defs import register_def
from fire_engine.procedural.textures.base import ProceduralTextureDef, pixel_noise

__all__ = ["TreeSpriteDef"]

_CELL_W = 64               # one tree cell is 64 wide ...
_CELL_H = 96               # ... × 96 tall (2:3 — quads keep this aspect)
_VARIANTS = 3

# Bark ramp, shadow side → lit side (RGB).
_BARK_RAMP = np.array([
    (38, 30, 22),
    (56, 44, 32),
    (76, 60, 42),
], dtype=np.uint8)

# 5-tone posterised canopy ramps per variant, dark under-canopy → pale crown.
_CANOPY_RAMPS = np.array([
    [(30, 44, 26), (44, 62, 34), (60, 80, 42), (80, 98, 52), (104, 116, 64)],   # 0 full green
    [(48, 42, 24), (72, 60, 32), (98, 80, 40), (124, 100, 50), (148, 122, 62)], # 1 autumn ochre
    [(40, 42, 30), (56, 58, 40), (74, 74, 50), (92, 90, 60), (110, 106, 72)],   # 2 scraggly grey-olive
], dtype=np.uint8)

# Per-variant canopy hole threshold: the scraggly tree is mostly gaps.
_HOLE_THRESH = (0.16, 0.20, 0.30)
# Per-variant lobe count range: scraggly trees grow fewer clumps.
_LOBES = ((6, 9), (5, 8), (3, 6))


@register_def
class TreeSpriteDef(ProceduralTextureDef):
    """
    Whole-tree atlas: one row of 3 condition variants (RGBA alpha cutout).

    Registered name
    ---------------
    ``"tree_sprite"``

    Output
    ------
    ``numpy.ndarray (96, 192, 4) uint8`` — three 64×96 tree cells side by
    side (full green / autumn ochre / scraggly half-dead).  Alpha is binary —
    render with discard, never blending.  Trunk bases touch the bottom image
    row; cells are 2:3 wide:tall.

    Example
    -------
    ::

        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural import get

        set_world_seed(7)
        arr = get("tree_sprite")
        assert arr.shape == (96, 192, 4)
        assert ((arr[..., 3] == 0) | (arr[..., 3] == 255)).all()
    """

    name = "tree_sprite"

    DEFAULT_WIDTH = _CELL_W * _VARIANTS        # 192
    DEFAULT_HEIGHT = _CELL_H                   # 96

    def generate(self, rng: np.random.Generator, **params) -> np.ndarray:
        """
        Generate the 3-variant tree atlas.

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
            ``(H, W, 4) uint8`` RGBA tree atlas with binary alpha.
        """
        W = int(params.get("width", self.DEFAULT_WIDTH))
        H = int(params.get("height", self.DEFAULT_HEIGHT))
        cell_w = max(1, W // _VARIANTS)

        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        yy, xx = np.mgrid[0:H, 0:cell_w]
        h_up = (H - 1 - yy).astype(np.float32)         # height above ground row

        for k in range(_VARIANTS):                     # fixed 3-iteration loop
            x0 = k * cell_w

            # --- trunk: tapered, leaning quadratic column -------------------
            base_x = float(rng.uniform(cell_w * 0.42, cell_w * 0.58))
            trunk_h = float(rng.uniform(H * 0.52, H * 0.68))
            lean_px = float(rng.uniform(-cell_w * 0.08, cell_w * 0.08))
            w_base = float(rng.uniform(2.2, 3.2))      # half-width at the base
            w_top = 1.0                                # half-width at the crown

            t = np.clip(h_up / trunk_h, 0.0, 1.0)      # 0 ground → 1 crown
            cx_line = base_x + lean_px * t * t
            half_w = w_base * (1.0 - t) + w_top * t
            trunk = (h_up < trunk_h) & (np.abs(xx - cx_line) <= half_w)

            # Bark: vertical pixel-noise striations + a darker shadow side.
            bark_noise = pixel_noise(rng, (H, cell_w), octaves=2, base_freq=8)
            bark_tier = np.clip((bark_noise * 3), 0, 2).astype(np.intp)
            bark_tier = np.where(xx < cx_line, np.maximum(bark_tier - 1, 0),
                                 bark_tier)
            rgba[:, x0:x0 + cell_w, :3][trunk] = _BARK_RAMP[bark_tier][trunk]
            rgba[:, x0:x0 + cell_w, 3][trunk] = 255

            # --- branch stubs: diagonal runs from the upper trunk -----------
            for _ in range(int(rng.integers(2, 4))):    # ≤3 iterations
                bh = float(rng.uniform(trunk_h * 0.55, trunk_h * 0.95))
                side = 1 if rng.uniform() < 0.5 else -1
                run = np.arange(int(rng.integers(4, 9)))
                bx_px = np.clip(np.round(base_x + lean_px * (bh / trunk_h) ** 2
                                         + side * run), 0, cell_w - 1).astype(np.intp)
                brows = np.clip(H - 1 - (bh + run * 0.5).astype(np.intp),
                                0, H - 1)
                rgba[brows, x0 + bx_px, :3] = _BARK_RAMP[1]
                rgba[brows, x0 + bx_px, 3] = 255

            # --- canopy: union of seeded lobes around the crown -------------
            crown_x = base_x + lean_px
            lo, hi = _LOBES[k]
            canopy = np.zeros((H, cell_w), dtype=bool)
            for j in range(int(rng.integers(lo, hi))):  # ≤8 iterations
                if j == 0:
                    # Anchor lobe sits ON the crown so the canopy always
                    # connects to the trunk, whatever the other lobes do.
                    cx, ch = crown_x, trunk_h + H * 0.06
                else:
                    cx = float(crown_x
                               + rng.uniform(-cell_w * 0.20, cell_w * 0.20))
                    ch = float(trunk_h + rng.uniform(-H * 0.02, H * 0.22))
                rx = float(rng.uniform(cell_w * 0.16, cell_w * 0.30))
                ry = float(rng.uniform(H * 0.08, H * 0.16))
                canopy |= ((xx - cx) / rx) ** 2 + ((h_up - ch) / ry) ** 2 <= 1.0

            # Ragged outline + sky holes from crisp pixel noise.
            leaf_noise = pixel_noise(rng, (H, cell_w), octaves=3, base_freq=5)
            canopy &= leaf_noise > _HOLE_THRESH[k]

            # --- posterised canopy shading: vertical light + noise blotch ---
            ramp = _CANOPY_RAMPS[k]
            c_lo, c_hi = trunk_h * 0.8, H * 0.95
            vert = np.clip((h_up - c_lo) / max(c_hi - c_lo, 1.0), 0.0, 1.0)
            tier = np.clip((vert * 0.45 + leaf_noise * 0.6) * len(ramp),
                           0, len(ramp) - 1).astype(np.intp)
            cell_rgb = ramp[tier]
            rgba[:, x0:x0 + cell_w, :3][canopy] = cell_rgb[canopy]
            rgba[:, x0:x0 + cell_w, 3][canopy] = 255

        return rgba
