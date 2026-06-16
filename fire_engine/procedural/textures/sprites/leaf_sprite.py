"""
procedural/textures/leaf_sprite.py — "leaf_sprite" leaf-litter atlas texture.

Produces a tiny **atlas** of leaf silhouettes: one row of 3 variants, each a
32×32 RGBA cell (96×32 total).  Each leaf is an elliptical blade with a darker
**midrib** down its long axis and a **serrated, noise-broken edge** so the
silhouette reads as a real leaf rather than a smooth ellipse.  The three cells
carry 3 autumn-muted **hue variants** (a desaturated green, an ochre, and a
russet-brown) so a scatter of leaf-litter instances looks like mixed fallen
leaves, not clones.

The leaf-litter renderer (``world/mote_renderer.py``) billboards one atlas cell
per instance (selected from the instance hash) and **alpha-blends** with an
alpha-test-style discard in the fragment shader (same threshold as grass), lit
by the same radiance cascades as the grass — so leaves sit in the scene and
catch the same light as the ground.

Atlas layout (the renderer's UV contract)
-----------------------------------------
3 cells laid left→right in one row.  Cell ``k`` (0,1,2) occupies horizontal
texels ``[k*32, (k+1)*32)``.  In the shader, variant ``k`` samples
``u = (k + frac_u) / 3``.  Leaf long axis runs **vertically** (stem at the
bottom row, tip at the top) — the same V-up orientation as ``grass_tuft`` —
though a billboarded leaf is rotated freely by the tumble shader so the
orientation is cosmetic.

Generation
----------
Pure numpy, no per-pixel Python loops (a fixed 3-iteration loop over the cells,
each cell's pixel work fully vectorised):

1. Per-cell normalised coordinates; an elliptical mask (narrow in X, tall in Y).
2. A serrated edge: modulate the ellipse radius by a few sine lobes along the
   long axis plus a low-freq :func:`value_noise` nibble → leaf teeth + an
   irregular, asymmetric outline.
3. A midrib: a thin vertical dark band down the leaf centre, plus a couple of
   diagonal vein hints, darkening the base hue.
4. The cell's hue variant tints the whole leaf; alpha is the (binary-ish) leaf
   mask with a 1-texel soft rim so the alpha-test discard has a clean edge.

Registered as ``"leaf_sprite"`` at import time.

Usage
-----
::

    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural import get

    set_world_seed(1337)
    arr = get("leaf_sprite")      # (32, 96, 4) uint8 — 3 leaves in a row
    # Preview: python tools/preview_texture.py leaf_sprite

Docs: docs/systems/procedural.textures.sprites.md
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fire_engine.procedural.defs import register_def
from fire_engine.procedural.textures.base import ProceduralTextureDef, value_noise

__all__ = ["LeafSpriteDef"]

_CELL = 32  # one leaf cell is 32×32
_VARIANTS = 3  # green / ochre / russet

# Autumn-muted, slightly desaturated leaf hues (RGB), base colour per variant.
# Minecraft × Morrowind art direction: nothing saturated, everything dusty.
_LEAF_HUES = np.array(
    [
        (104, 122, 66),  # 0 — muted olive green
        (170, 138, 62),  # 1 — ochre / amber
        (148, 86, 52),  # 2 — russet brown
    ],
    dtype=np.float32,
)

# Midrib / vein darkening factor (multiplies the base hue along the rib).
_RIB_DARKEN = 0.62


@register_def
class LeafSpriteDef(ProceduralTextureDef):
    """
    Leaf-litter atlas: one row of 3 hue variants (RGBA alpha cutout).

    Registered name
    ---------------
    ``"leaf_sprite"``

    Output
    ------
    ``numpy.ndarray (32, 96, 4) uint8`` — three 32×32 leaf cells side by side.
    Each leaf is an elliptical silhouette with a serrated edge and a darker
    midrib; alpha is ~binary (leaf vs. background) with a soft 1-texel rim.
    Render alpha-blended with a discard threshold (like grass), lit by the
    scene cascades.

    Example
    -------
    ::

        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural import get

        set_world_seed(7)
        arr = get("leaf_sprite")
        assert arr.shape == (32, 96, 4)
        assert arr[16, 16, 3] == 255           # centre of leaf 0 is opaque

    Docs: docs/systems/procedural.textures.sprites.md
    """

    name = "leaf_sprite"

    DEFAULT_WIDTH = _CELL * _VARIANTS  # 96
    DEFAULT_HEIGHT = _CELL  # 32

    def generate(self, rng: np.random.Generator, **params: Any) -> np.ndarray:
        """
        Generate the 3-variant leaf atlas.

        Parameters
        ----------
        rng : numpy.random.Generator
            Seeded generator injected by the registry.
        **params : any
            Optional ``width`` / ``height`` overrides (the width is split into
            ``_VARIANTS`` equal cells; non-multiples truncate the last cell).

        Returns
        -------
        numpy.ndarray
            ``(H, W, 4) uint8`` RGBA leaf atlas.

        Docs: docs/systems/procedural.textures.sprites.md
        """
        W = int(params.get("width", self.DEFAULT_WIDTH))
        H = int(params.get("height", self.DEFAULT_HEIGHT))
        cell_w = max(1, W // _VARIANTS)

        rgba = np.zeros((H, W, 4), dtype=np.uint8)

        # Per-cell normalised coords reused each variant: x in [-1,1] (narrow
        # axis), y in [-1,1] (long axis).
        ys = np.linspace(-1.0, 1.0, H, dtype=np.float32)
        xs = np.linspace(-1.0, 1.0, cell_w, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)  # (H, cell_w)

        for k in range(_VARIANTS):  # fixed 3-iteration loop
            x0 = k * cell_w
            x1 = min((k + 1) * cell_w, W)
            cw = x1 - x0
            cgx = gx[:, :cw]
            cgy = gy[:, :cw]

            # Per-variant low-freq noise → edge nibble + a little blotch in the
            # body so leaves aren't flat-coloured.
            noise = value_noise(rng, (H, cw), octaves=2, base_freq=3)

            # Elliptical leaf: narrow in X (0.62), full in Y, tapered to a point
            # at both ends.  taper widens the middle, pinches the tip/stem.
            taper = np.sqrt(np.clip(1.0 - cgy * cgy, 0.0, 1.0))  # 0 at ends, 1 mid
            half_w = 0.62 * taper

            # Serrated edge: a few sine teeth along the long axis modulating the
            # half-width, plus a noise nibble — gives leaf teeth + asymmetry.
            teeth = 0.10 * np.sin(cgy * 9.0) * taper
            nibble = (noise[:, :cw] - 0.5) * 0.16
            edge = half_w + teeth + nibble

            inside = np.abs(cgx) <= np.maximum(edge, 0.0)

            # Base hue, blotched slightly by the body noise.
            hue = _LEAF_HUES[k]
            shade = 0.82 + 0.30 * noise[:, :cw]  # (H, cw) body shading
            body = hue[None, None, :] * shade[..., None]  # (H, cw, 3)

            # Midrib: thin dark vertical band down the centre (|x| small),
            # fading out near the tip/stem so it doesn't poke past the outline.
            rib = np.clip(1.0 - np.abs(cgx) / 0.10, 0.0, 1.0) * taper
            # A couple of diagonal vein hints branching off the midrib.
            veins = (
                np.clip(1.0 - np.abs(np.abs(cgx) - 0.30 * (1.0 - np.abs(cgy))) / 0.05, 0.0, 1.0)
                * taper
                * 0.6
            )
            rib = np.clip(rib + veins, 0.0, 1.0)
            rib_mul = (1.0 - rib) + rib * _RIB_DARKEN  # darken along ribs
            body = body * rib_mul[..., None]

            cell_rgb = np.clip(body, 0.0, 255.0).astype(np.uint8)
            # Alpha: binary leaf mask with a soft 1-texel rim near the edge so
            # the fragment discard has a clean (non-aliased-to-death) boundary.
            margin = np.maximum(edge, 0.0) - np.abs(cgx)  # >0 inside, 0 at edge
            rim = np.clip(margin / 0.06, 0.0, 1.0)  # soft last ~2 texels
            alpha = np.where(inside, rim, 0.0)
            cell_a = np.clip(np.round(alpha * 255.0), 0, 255).astype(np.uint8)

            rgba[:, x0:x1, :3] = cell_rgb
            rgba[:, x0:x1, 3] = cell_a

        return rgba
