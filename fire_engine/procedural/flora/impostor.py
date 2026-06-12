"""
procedural/flora/impostor.py — software-rasterized far-LOD tree sprites.

Beyond the mesh fade distance the renderer crossfades each tree to an
instanced crossed-quad billboard — the **impostor**.  This module renders
those sprites headlessly: an orthographic XZ projection of the SAME skeleton
and leaf clusters the mesh was built from, colored with the SAME palettes as
the species atlas, so the handoff at distance is silhouette- and
hue-consistent without any GPU bake (deterministic, pytest-testable; a
boot-time offscreen render bake is the documented upgrade if mismatch ever
shows at closer ranges).

Conventions match the retired ``tree_sprite`` atlas so the impostor shader
path stays simple: cells laid left→right, trunk base on the **bottom image
row** (V = 0 after the upload flip), binary alpha.

Per-segment / per-cluster Python loops here run over *tens* of elements with
vectorized work inside each bounding box — bounded recipe loops, not
per-pixel iteration (Hard Rule 4).
"""

from __future__ import annotations

import numpy as np

from fire_engine.procedural.flora.leaves import LeafClusters
from fire_engine.procedural.flora.skeleton import TreeSkeleton
from fire_engine.procedural.textures.base import pixel_noise

__all__ = ["rasterize_impostor", "impostor_atlas"]


def rasterize_impostor(
    sk: TreeSkeleton,
    clusters: LeafClusters,
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
    tapered capsules (bark mid-tone, shadowed left of the trunk axis),
    leaf clusters become pixel-noise-broken posterised discs drawn over the
    wood.  Texels are square in meters (aspect-preserving), the trunk axis
    is horizontally centered and the trunk base touches the bottom row.

    Parameters
    ----------
    sk : TreeSkeleton
        The variant's skeleton (same one the mesh was built from).
    clusters : LeafClusters
        The variant's foliage (may be empty — dead trees).
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
        if clusters.n_clusters:
            reach = max(reach,
                        float((np.abs(clusters.center[:, 0:2]).max(axis=1)
                               + clusters.radius).max()))
            top = max(top,
                      float((clusters.center[:, 2] + clusters.radius).max()))
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

    # --- leaf clusters: noisy posterised discs over the wood -------------
    if clusters.n_clusters:
        noise = pixel_noise(rng, (H, W), octaves=3, base_freq=5)
        for j in range(clusters.n_clusters):
            cx_px, cy_px = px(clusters.center[j])
            r_px = max(float(clusters.radius[j]) * s, 1.5)
            lo_x = max(int(cx_px - r_px - 1), 0)
            hi_x = min(int(cx_px + r_px + 2), W)
            lo_y = max(int(cy_px - r_px - 1), 0)
            hi_y = min(int(cy_px + r_px + 2), H)
            if lo_x >= hi_x or lo_y >= hi_y:
                continue
            bx = xx[lo_y:hi_y, lo_x:hi_x]
            by = yy[lo_y:hi_y, lo_x:hi_x]
            d = np.hypot(bx - cx_px, by - cy_px) / r_px
            n = noise[lo_y:hi_y, lo_x:hi_x]
            inside = (d < 1.0) & (n > hole_thresh + d * 0.25)
            if not inside.any():
                continue
            light = 1.0 - by / max(H - 1, 1)         # higher = lighter
            tier = np.clip(((n * 0.55 + light * 0.5)
                            * len(leaf_palette)),
                           0, len(leaf_palette) - 1).astype(np.intp)
            region = rgba[lo_y:hi_y, lo_x:hi_x]
            region[..., :3][inside] = leaf_palette[tier][inside]
            region[..., 3][inside] = 255

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
