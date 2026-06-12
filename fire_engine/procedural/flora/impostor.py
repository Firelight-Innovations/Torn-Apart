"""
procedural/flora/impostor.py — software-rasterized far-LOD tree sprites.

Beyond the mesh fade distance the renderer crossfades each tree to an
instanced crossed-quad billboard — the **impostor**.  This module renders
those sprites headlessly: an orthographic XZ projection of the SAME skeleton
and individual leaves the mesh was built from, colored with the SAME
palettes as the species atlas, so the handoff at distance is silhouette-
and hue-consistent without any GPU bake (deterministic, pytest-testable; a
boot-time offscreen render bake is the documented upgrade if mismatch ever
shows at closer ranges).

Conventions match the retired ``tree_sprite`` atlas so the impostor shader
path stays simple: cells laid left→right, trunk base on the **bottom image
row** (V = 0 after the upload flip), binary alpha.

The per-segment branch loop runs over *tens* of elements with vectorized
work inside each bounding box (a bounded recipe loop, not per-pixel
iteration — Hard Rule 4); the leaf pass is a fully-vectorized point
scatter + diamond dilation because leaves number in the hundreds.
"""

from __future__ import annotations

import numpy as np

from fire_engine.procedural.flora.leaves import Leaves
from fire_engine.procedural.flora.skeleton import TreeSkeleton
from fire_engine.procedural.textures.base import pixel_noise

__all__ = ["rasterize_impostor", "impostor_atlas"]


def rasterize_impostor(
    sk: TreeSkeleton,
    leaves: Leaves,
    bark_palette: np.ndarray,
    leaf_palette: np.ndarray,
    rng: np.random.Generator,
    *,
    cell_wh: tuple[int, int] = (64, 96),
    hole_thresh: float = 0.2,
    px_per_m: float | None = None,
) -> np.ndarray:
    """
    Render one variant's far-LOD sprite cell — ``(H, W, 4) uint8``.

    Orthographic projection onto the XZ plane: branch segments become
    tapered capsules (bark mid-tone, shadowed left of the trunk axis); the
    individual leaves scatter as points, dilated to their pixel radius and
    broken by pixel noise, drawn over the wood — the canopy silhouette at
    distance is literally the leaf point-cloud's.  Texels are square in
    meters (aspect-preserving), the trunk axis is horizontally centered
    and the trunk base touches the bottom row.

    Parameters
    ----------
    sk : TreeSkeleton
        The variant's skeleton (same one the mesh was built from).
    leaves : Leaves
        The variant's individual leaves (may be empty — dead trees).
    bark_palette / leaf_palette : numpy.ndarray
        ``uint8 (T, 3)`` ramps — pass the SAME palettes as the atlas so
        mesh and impostor agree at the crossfade.
    rng : numpy.random.Generator
        Deterministic generator (canopy hole noise).
    cell_wh : tuple[int, int]
        Sprite cell (width, height) texels.  Default (64, 96) — 2:3, the
        impostor quads keep this aspect.
    hole_thresh : float
        Canopy hole noise cut.  Default 0.2.
    px_per_m : float | None
        Pixel scale.  ``TreeSpeciesDef.generate`` passes the POOL-COMMON
        scale so every variant cell shares one meters-per-texel and the
        impostor quad (``TreeVariantSet.impostor_height_m`` tall) overlays
        the mesh exactly.  ``None`` = fit this tree alone.

    Returns
    -------
    numpy.ndarray
        ``(cell_wh[1], cell_wh[0], 4) uint8`` with binary alpha.
    """
    W, H = int(cell_wh[0]), int(cell_wh[1])
    bark_palette = np.asarray(bark_palette, dtype=np.uint8)
    leaf_palette = np.asarray(leaf_palette, dtype=np.uint8)
    rgba = np.zeros((H, W, 4), dtype=np.uint8)

    if px_per_m is None:
        # Self-fit (aspect-preserving): bound the tree, take the tighter axis.
        reach = np.abs(np.concatenate([sk.start[:, 0:2],
                                       sk.end[:, 0:2]])).max()
        top = float(sk.end[:, 2].max())
        if leaves.n_leaves:
            reach = max(reach,
                        float((np.abs(leaves.center[:, 0:2]).max(axis=1)
                               + leaves.radius).max()))
            top = max(top,
                      float((leaves.center[:, 2] + leaves.radius).max()))
        reach = max(float(reach), 0.25) * 1.05
        top = max(top, 0.5) * 1.02
        px_per_m = min((W - 1) / (2.0 * reach), (H - 1) / top)
    s = float(px_per_m)

    def px(p3: np.ndarray) -> tuple[float, float]:
        """Tree-local (x, _, z) → image (col, row); row H-1 = ground."""
        return float(p3[0]) * s + (W - 1) * 0.5, (H - 1) - float(p3[2]) * s

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # --- branches: tapered capsules (tens of segments — bounded loop) ----
    mid_tier = len(bark_palette) // 2
    for i in range(sk.n_segments):
        x0, y0 = px(sk.start[i])
        x1, y1 = px(sk.end[i])
        r0 = max(float(sk.radius_start[i]) * s, 0.6)
        r1 = max(float(sk.radius_end[i]) * s, 0.6)
        lo_x = max(int(min(x0, x1) - max(r0, r1) - 1), 0)
        hi_x = min(int(max(x0, x1) + max(r0, r1) + 2), W)
        lo_y = max(int(min(y0, y1) - max(r0, r1) - 1), 0)
        hi_y = min(int(max(y0, y1) + max(r0, r1) + 2), H)
        if lo_x >= hi_x or lo_y >= hi_y:
            continue
        bx = xx[lo_y:hi_y, lo_x:hi_x]
        by = yy[lo_y:hi_y, lo_x:hi_x]
        dx, dy = x1 - x0, y1 - y0
        denom = max(dx * dx + dy * dy, 1e-6)
        t = np.clip(((bx - x0) * dx + (by - y0) * dy) / denom, 0.0, 1.0)
        dist = np.hypot(bx - (x0 + dx * t), by - (y0 + dy * t))
        inside = dist <= (r0 + (r1 - r0) * t)
        if not inside.any():
            continue
        shade = np.where(bx < (x0 + dx * t), max(mid_tier - 1, 0), mid_tier)
        region = rgba[lo_y:hi_y, lo_x:hi_x]
        region[..., :3][inside] = bark_palette[shade.astype(np.intp)][inside]
        region[..., 3][inside] = 255

    # --- leaves: vectorized point scatter + diamond dilation -------------
    if leaves.n_leaves:
        cols = np.clip(np.round(leaves.center[:, 0] * s
                                + (W - 1) * 0.5).astype(np.int64), 0, W - 1)
        rows = np.clip(np.round((H - 1)
                                - leaves.center[:, 2] * s).astype(np.int64),
                       0, H - 1)
        mask = np.zeros((H, W), dtype=bool)
        mask[rows, cols] = True

        # Dilate each dot to the (median) leaf pixel radius — a diamond
        # per round, capped tiny: leaves are 1–3 px at impostor scale.
        r_px = float(np.median(leaves.radius)) * s
        for _ in range(int(np.clip(round(r_px), 1, 3))):
            grown = mask.copy()
            grown[1:, :] |= mask[:-1, :]
            grown[:-1, :] |= mask[1:, :]
            grown[:, 1:] |= mask[:, :-1]
            grown[:, :-1] |= mask[:, 1:]
            mask = grown

        noise = pixel_noise(rng, (H, W), octaves=3, base_freq=5)
        inside = mask & (noise > hole_thresh * 0.8)
        if inside.any():
            light = 1.0 - yy / max(H - 1, 1)         # higher = lighter
            tier = np.clip(((noise * 0.55 + light * 0.5)
                            * len(leaf_palette)),
                           0, len(leaf_palette) - 1).astype(np.intp)
            rgba[..., :3][inside] = leaf_palette[tier][inside]
            rgba[..., 3][inside] = 255

    return rgba


def impostor_atlas(cells: list[np.ndarray]) -> np.ndarray:
    """
    Lay variant sprite cells left→right into one atlas row.

    Cell ``k`` occupies horizontal texels ``[k·W, (k+1)·W)``; the impostor
    shader samples ``u = (variant + frac_u) / n_variants`` — the exact UV
    contract the old ``tree_sprite`` atlas used.

    Parameters
    ----------
    cells : list[numpy.ndarray]
        Equal-shape ``(H, W, 4) uint8`` cells from
        :func:`rasterize_impostor`, variant order.

    Returns
    -------
    numpy.ndarray
        ``(H, W × len(cells), 4) uint8``.
    """
    if not cells:
        raise ValueError("impostor_atlas: no cells")
    return np.ascontiguousarray(np.hstack(cells))
