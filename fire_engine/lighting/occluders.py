"""
lighting/occluders.py — static tree/bush occupancy splats for the cascades.

Trees are real geometry but not voxels, so until now they were invisible to
the lighting volumes: the sun marched straight through a canopy, the ground
under an oak was noon-bright, and crowns never self-shadowed.  This module
splats each baked tree instance (``zones/tree_placement.py``) into a
cascade's geometry volume as **fractional occupancy** (the same A-channel
contract the chunk downsampler emits), so INJECT's visibility march, the
GATHER bounce, the surface shaders' refinement march and the voxel AO all
see trees with zero shader changes.

Shape model (deliberately crude — light cells are 0.5–8 m)
----------------------------------------------------------
- **Trunk**: a vertical column from the base to ``TRUNK_TOP_FRAC × height``
  with a ``TRUNK_SIDE_M`` square cross-section, at ``trunk_occ`` opacity
  (scaled by the cross-section ÷ cell area at coarse cells — a 0.5 m trunk
  blocks little of an 8 m cell).
- **Canopy**: an ellipsoid centred at ``CANOPY_CENTER_FRAC × height`` with
  horizontal semi-axis ``canopy_r`` and vertical semi-axis
  ``CANOPY_HALF_HEIGHT_FRAC × height``, at ``canopy_occ`` opacity.  Leaves
  ATTENUATE light rather than block it — keep ``canopy_occ`` well under 1
  or the under-canopy goes pitch black and crowns shade themselves to mud.
  At cells larger than the canopy the contribution is scaled by the
  ellipsoid volume ÷ cell volume (a bush is a wisp inside an 8 m cell).

Occupancy combines with ``max`` (a tree inside a hill stays hill-solid) and
albedo is written only where the splat RAISES the cell's occupancy, so
terrain bounce colour is never repainted.

Determinism: pure numpy function of its inputs — same instances, same
volume → byte-identical output (Hard Rule 2 is upstream, in the placement
bake).  Python iterates *instances* (≤ a few thousand, like the chunk loop
in ``assemble_geometry``); all per-cell work is vectorised (Hard Rule 4).

Example
-------
>>> import numpy as np
>>> from fire_engine.lighting.occluders import TreeOccluderSet, splat_tree_occluders
>>> occ = TreeOccluderSet.single(x=8.0, y=8.0, z=4.0, height_m=6.0,
...                              canopy_r_m=2.5)
>>> vol = np.zeros((32, 32, 32, 4), dtype=np.uint8)
>>> splat_tree_occluders(vol, (0, 0, 0), 0.5, occ,
...                      trunk_occ=0.85, canopy_occ=0.3)
>>> int(vol[16, 16, 9, 3]) > 0    # trunk cell just above the 4 m base
True
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["TreeOccluderSet", "splat_tree_occluders",
           "TRUNK_TOP_FRAC", "TRUNK_SIDE_M",
           "CANOPY_CENTER_FRAC", "CANOPY_HALF_HEIGHT_FRAC"]

# Trunk column reaches this fraction of the tree height (the rest is canopy).
TRUNK_TOP_FRAC: float = 0.45
# Trunk square cross-section edge (m) — one near-cascade cell.
TRUNK_SIDE_M: float = 0.5
# Canopy ellipsoid centre height as a fraction of tree height.
CANOPY_CENTER_FRAC: float = 0.65
# Canopy ellipsoid vertical semi-axis as a fraction of tree height.
CANOPY_HALF_HEIGHT_FRAC: float = 0.35

# Default splat colours (linear RGB 0–1) when a set has no per-instance
# colours: muted bark brown / dark foliage green.
_DEFAULT_BARK_RGB = (0.16, 0.11, 0.07)
_DEFAULT_LEAF_RGB = (0.08, 0.16, 0.06)


@dataclass(frozen=True)
class TreeOccluderSet:
    """
    Struct-of-arrays description of every tree/bush the cascades should see.

    Built by the tree renderer from its baked placements (one merged set for
    all volumes) and handed to ``GpuLightingPipeline.set_static_occluders``;
    the assembly path splats it into every (re)assembled geometry volume.

    Attributes
    ----------
    x, y, z : numpy.ndarray
        ``float32 (N,)`` trunk-base world positions (m, Z-up).
    height_m : numpy.ndarray
        ``float32 (N,)`` per-instance tree height (m) — species max height ×
        instance scale.
    canopy_r_m : numpy.ndarray
        ``float32 (N,)`` canopy horizontal semi-axis (m).
    bark_rgb, leaf_rgb : numpy.ndarray
        ``float32 (N, 3)`` linear-light splat albedo (0–1) for trunk /
        canopy cells (the GI bounce colour).

    Example
    -------
    >>> occ = TreeOccluderSet.single(0.0, 0.0, 0.0, height_m=6.0,
    ...                              canopy_r_m=2.0)
    >>> occ.count
    1
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    height_m: np.ndarray
    canopy_r_m: np.ndarray
    bark_rgb: np.ndarray
    leaf_rgb: np.ndarray

    @property
    def count(self) -> int:
        """Instance count ``N``."""
        return int(self.x.shape[0])

    @classmethod
    def single(cls, x: float, y: float, z: float, height_m: float,
               canopy_r_m: float,
               bark_rgb: tuple = _DEFAULT_BARK_RGB,
               leaf_rgb: tuple = _DEFAULT_LEAF_RGB) -> "TreeOccluderSet":
        """One-instance set (tests / tools)."""
        f = np.float32
        return cls(
            x=np.asarray([x], f), y=np.asarray([y], f), z=np.asarray([z], f),
            height_m=np.asarray([height_m], f),
            canopy_r_m=np.asarray([canopy_r_m], f),
            bark_rgb=np.asarray([bark_rgb], f),
            leaf_rgb=np.asarray([leaf_rgb], f))

    @classmethod
    def merge(cls, sets: "list[TreeOccluderSet]") -> "TreeOccluderSet":
        """Concatenate several sets (one per zone volume) into one."""
        if not sets:
            return cls.empty()
        return cls(
            x=np.concatenate([s.x for s in sets]),
            y=np.concatenate([s.y for s in sets]),
            z=np.concatenate([s.z for s in sets]),
            height_m=np.concatenate([s.height_m for s in sets]),
            canopy_r_m=np.concatenate([s.canopy_r_m for s in sets]),
            bark_rgb=np.concatenate([s.bark_rgb for s in sets]),
            leaf_rgb=np.concatenate([s.leaf_rgb for s in sets]))

    @classmethod
    def empty(cls) -> "TreeOccluderSet":
        """Zero-instance set."""
        f = np.empty(0, dtype=np.float32)
        c = np.empty((0, 3), dtype=np.float32)
        return cls(x=f, y=f.copy(), z=f.copy(), height_m=f.copy(),
                   canopy_r_m=f.copy(), bark_rgb=c, leaf_rgb=c.copy())


def _cell_range(lo_m: float, hi_m: float, origin_cell: int, cell_m: float,
                n: int) -> tuple[int, int]:
    """Clamped half-open cell-index range covering world meters [lo, hi)."""
    a = int(np.floor(lo_m / cell_m)) - origin_cell
    b = int(np.floor(hi_m / cell_m)) - origin_cell + 1
    return max(a, 0), min(b, n)


def splat_tree_occluders(
    albedo_occ: np.ndarray,
    origin_cell: tuple[int, int, int],
    cell_m: float,
    occluders: TreeOccluderSet,
    trunk_occ: float,
    canopy_occ: float,
) -> None:
    """
    Splat ``occluders`` into a cascade geometry block, in place.

    Parameters
    ----------
    albedo_occ : numpy.ndarray
        ``uint8 (N, N, N, 4)`` ``[x, y, z]`` block from
        :func:`fire_engine.lighting.volume.assemble_geometry` — RGB linear
        albedo, A fractional occupancy ×255.  Modified in place.
    origin_cell : tuple[int, int, int]
        World cell index of texel (0,0,0) (``VolumeWindow.origin_cell``).
    cell_m : float
        Cell edge in meters.
    occluders : TreeOccluderSet
        The merged instance set.  Instances outside the window are skipped.
    trunk_occ, canopy_occ : float
        Splat opacities in [0, 1] (``config.light_tree_trunk_occ`` /
        ``light_tree_canopy_occ``).  0 disables that shape entirely.

    Notes
    -----
    Occupancy is ``max``-combined and albedo written only where the splat
    raises occupancy — terrain solids win.  Deterministic, no RNG.
    """
    if occluders.count == 0 or (trunk_occ <= 0.0 and canopy_occ <= 0.0):
        return
    n = albedo_occ.shape[0]
    ox, oy, oz = origin_cell
    win_lo = (ox * cell_m, oy * cell_m, oz * cell_m)
    win_hi = tuple(win_lo[i] + n * cell_m for i in range(3))
    cell_vol = cell_m ** 3
    # Trunk opacity scaled by its cross-section share of a cell (clamped 1).
    trunk_eff = trunk_occ * min(1.0, (TRUNK_SIDE_M / cell_m) ** 2)
    trunk_byte = np.uint8(round(255.0 * min(1.0, trunk_eff)))

    for i in range(occluders.count):
        tx = float(occluders.x[i])
        ty = float(occluders.y[i])
        tz = float(occluders.z[i])
        h = float(occluders.height_m[i])
        cr = float(occluders.canopy_r_m[i])
        cv = CANOPY_HALF_HEIGHT_FRAC * h          # canopy vertical semi-axis
        reach = max(cr, TRUNK_SIDE_M)
        # Cheap whole-instance rejection against the window box.
        if (tx + reach <= win_lo[0] or tx - reach >= win_hi[0]
                or ty + reach <= win_lo[1] or ty - reach >= win_hi[1]
                or tz + h <= win_lo[2] or tz >= win_hi[2]):
            continue

        # --- trunk column -------------------------------------------------
        if trunk_byte > 0 and h > 0.0:
            ax0, ax1 = _cell_range(tx - TRUNK_SIDE_M * 0.5,
                                   tx + TRUNK_SIDE_M * 0.5, ox, cell_m, n)
            ay0, ay1 = _cell_range(ty - TRUNK_SIDE_M * 0.5,
                                   ty + TRUNK_SIDE_M * 0.5, oy, cell_m, n)
            az0, az1 = _cell_range(tz, tz + TRUNK_TOP_FRAC * h, oz, cell_m, n)
            if ax0 < ax1 and ay0 < ay1 and az0 < az1:
                box = albedo_occ[ax0:ax1, ay0:ay1, az0:az1]
                raised = box[..., 3] < trunk_byte
                box[..., :3][raised] = np.clip(
                    occluders.bark_rgb[i] * 255.0, 0.0, 255.0
                ).astype(np.uint8)
                box[..., 3][raised] = trunk_byte

        # --- canopy ellipsoid ----------------------------------------------
        if canopy_occ > 0.0 and cr > 0.0 and cv > 0.0:
            cz = tz + CANOPY_CENTER_FRAC * h
            ell_vol = (4.0 / 3.0) * np.pi * cr * cr * cv
            occ_byte = np.uint8(round(
                255.0 * min(1.0, canopy_occ * min(1.0, ell_vol / cell_vol))))
            if occ_byte > 0:
                ax0, ax1 = _cell_range(tx - cr, tx + cr, ox, cell_m, n)
                ay0, ay1 = _cell_range(ty - cr, ty + cr, oy, cell_m, n)
                az0, az1 = _cell_range(cz - cv, cz + cv, oz, cell_m, n)
                if ax0 < ax1 and ay0 < ay1 and az0 < az1:
                    # Normalised squared distance from the ellipsoid centre,
                    # evaluated at cell centres (broadcasted 1-D axes).
                    xs = (ox + np.arange(ax0, ax1) + 0.5) * cell_m
                    ys = (oy + np.arange(ay0, ay1) + 0.5) * cell_m
                    zs = (oz + np.arange(az0, az1) + 0.5) * cell_m
                    d2 = ((xs[:, None, None] - tx) / cr) ** 2 \
                       + ((ys[None, :, None] - ty) / cr) ** 2 \
                       + ((zs[None, None, :] - cz) / cv) ** 2
                    inside = d2 <= 1.0
                    if not inside.any():
                        # Sub-cell canopy: occupy the centre-containing cell.
                        cxi = int(np.floor(tx / cell_m)) - ox
                        cyi = int(np.floor(ty / cell_m)) - oy
                        czi = int(np.floor(cz / cell_m)) - oz
                        if 0 <= cxi < n and 0 <= cyi < n and 0 <= czi < n:
                            cell = albedo_occ[cxi, cyi, czi]
                            if cell[3] < occ_byte:
                                cell[:3] = np.clip(
                                    occluders.leaf_rgb[i] * 255.0,
                                    0.0, 255.0).astype(np.uint8)
                                cell[3] = occ_byte
                        continue
                    box = albedo_occ[ax0:ax1, ay0:ay1, az0:az1]
                    raised = inside & (box[..., 3] < occ_byte)
                    box[..., :3][raised] = np.clip(
                        occluders.leaf_rgb[i] * 255.0, 0.0, 255.0
                    ).astype(np.uint8)
                    box[..., 3][raised] = occ_byte
