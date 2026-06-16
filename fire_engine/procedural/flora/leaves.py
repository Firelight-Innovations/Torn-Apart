"""
procedural/flora/leaves.py — cellular-automaton leaf placement on skeletons.

Leaves follow the *Dynamic Trees* idea taken one step further: instead of a
few big foliage-blob billboards, the canopy is **hundreds of individual
leaves**, grown by a small cellular automaton seeded at the branch tips.
Each tip injects "hydration" into a coarse 3-D cell grid; hydration spreads
to neighbouring cells losing one level per step, and every surviving cell
sprouts one or two leaf cards.  The canopy SHAPE therefore emerges from the
branch structure (a one-limb snag grows one tuft, a tiered oak grows a
ragged dome) and can never float free of the wood.

The result is a :class:`Leaves` struct-of-arrays — one row per **individual
leaf** — that the mesher turns into small oriented alpha-cutout quads merged
into the variant mesh (one draw per variant; the GPU batches the whole
canopy), and the impostor rasterizer turns into posterised dots.

Species scripts normally only call :func:`leaves_at_tips`; leafless or
near-leafless species (dead trees) pass fewer ids / lower ``rounds`` and
``density``, or return ``Leaves.empty()``.

Units: meters, Z-up, tree-local space (trunk base at the origin).

Docs: docs/systems/procedural.flora.md
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fire_engine.procedural.flora.skeleton import TreeSkeleton

__all__ = ["Leaves", "leaves_at_tips"]

# CA hydration propagates one cell per round along the 6 axis neighbours.
_NEIGHBOR_SHIFTS = ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1))

# Hard cap on grid cells (covers any sane canopy; a runaway-extent backstop).
_MAX_GRID_CELLS = 200_000


@dataclass
class Leaves:
    """
    Individual leaves attached to a skeleton — struct-of-arrays, one row
    per leaf card.

    Attributes
    ----------
    center : numpy.ndarray
        ``float32 (L, 3)`` — leaf centers, tree-local meters.
    radius : numpy.ndarray
        ``float32 (L,)`` — leaf half-size (m); the mesher's card is
        ``2 × radius`` across.
    sway : numpy.ndarray
        ``float32 (L,)`` — wind-sway weight in ``[0, 1]`` (≈0.85–1.0:
        leaves ride gusts harder than the wood they grow on).

    Docs: docs/systems/procedural.flora.md
    """

    center: np.ndarray
    radius: np.ndarray
    sway: np.ndarray

    @property
    def n_leaves(self) -> int:
        """Number of leaves ``L``.

        Docs: docs/systems/procedural.flora.md
        """
        return int(self.radius.shape[0])

    @staticmethod
    def empty() -> Leaves:
        """A zero-leaf instance (leafless species).

        Docs: docs/systems/procedural.flora.md
        """
        return Leaves(
            center=np.empty((0, 3), dtype=np.float32),
            radius=np.empty(0, dtype=np.float32),
            sway=np.empty(0, dtype=np.float32),
        )


def leaves_at_tips(
    sk: TreeSkeleton,
    ids: np.ndarray,
    rng: np.random.Generator,
    *,
    cell_m: float = 0.25,
    rounds: int = 3,
    density: float = 0.6,
    per_cell: tuple[int, int] = (1, 2),
    leaf_size_m: tuple[float, float] = (0.09, 0.14),
    sway_min: float = 0.85,
    max_leaves: int = 600,
) -> Leaves:
    """
    Grow individual leaves around the **tip segments** among *ids* with a
    cellular automaton.

    Tips (segments nothing else grows from — ``TreeSkeleton.tip_ids``) seed
    a coarse cell grid with hydration ``rounds``; each CA round every cell
    takes ``max(self, max(6-neighbours) − 1)``, so hydration radiates
    outward losing one level per cell.  Cells that end hydrated sprout
    leaves with a probability that falls toward the canopy rim — the blob
    interior fills in, the silhouette stays ragged.

    Parameters
    ----------
    sk : TreeSkeleton
        The finalized skeleton (``SkeletonBuilder.skeleton()``).
    ids : numpy.ndarray
        Candidate segment ids — typically the concatenated returns of the
        outer ``branches()`` calls.  Non-tips among them are ignored.
    rng : numpy.random.Generator
        Deterministic generator (consume the species def's rng).
    cell_m : float
        CA cell edge (m) — the canopy's "leaf voxel" size.  Default 0.25.
    rounds : int
        Hydration radius in cells.  1 = a tuft hugging the tip, 3 = a
        ~0.75 m foliage dome per tip.  Default 3.
    density : float
        Base leaf probability per hydrated cell (scaled down toward the
        rim).  Default 0.6.
    per_cell : tuple[int, int]
        Leaves per surviving cell, uniform inclusive.  Default (1, 2).
    leaf_size_m : tuple[float, float]
        Per-leaf half-size range (m), uniform.  Default (0.09, 0.14) —
        chunky 18–28 cm pixel-art leaves.
    sway_min : float
        Floor for the leaf sway weight; actual sway is
        ``max(seed-tip sway, uniform(sway_min, 1))``.  Default 0.85.
    max_leaves : int
        Deterministic thinning cap (vertex budget: 4 verts/leaf).
        Default 600.

    Returns
    -------
    Leaves
        May be empty (no tips in *ids*, ``rounds=0``, ``density=0``).

    Example
    -------
    ::

        leaves = leaves_at_tips(sk, np.concatenate([limbs, twigs]), rng,
                                cell_m=0.28, rounds=3, density=0.6)

    Docs: docs/systems/procedural.flora.md
    """
    tips = sk.tip_ids(ids)
    if tips.size == 0 or rounds <= 0 or density <= 0.0:
        return Leaves.empty()

    seeds = sk.end[tips].astype(np.float32)  # (T, 3)
    seed_sway = sk.sway[tips].astype(np.float32)  # (T,)

    # --- grid covering the seeds plus the CA growth reach -----------------
    pad = (rounds + 0.5) * cell_m
    lo = seeds.min(axis=0) - pad
    hi = seeds.max(axis=0) + pad
    dims = np.maximum(np.ceil((hi - lo) / cell_m).astype(np.int64), 1)
    if int(dims.prod()) > _MAX_GRID_CELLS:
        raise ValueError(
            f"leaves_at_tips: CA grid {tuple(dims)} exceeds "
            f"{_MAX_GRID_CELLS} cells — canopy reach or cell_m is off "
            f"(seeds span {hi - lo} m at cell_m={cell_m})"
        )

    # --- seed + propagate hydration (and the sway field alongside) --------
    hyd = np.zeros(tuple(dims), dtype=np.int16)
    swf = np.zeros(tuple(dims), dtype=np.float32)
    cell_idx = np.clip(((seeds - lo) / cell_m).astype(np.int64), 0, dims - 1)
    hyd[cell_idx[:, 0], cell_idx[:, 1], cell_idx[:, 2]] = rounds
    # np.maximum.at: two seeds in one cell keep the higher sway.
    np.maximum.at(swf, (cell_idx[:, 0], cell_idx[:, 1], cell_idx[:, 2]), seed_sway)

    for _ in range(rounds):  # fixed small loop (≤ a few)
        spread = hyd
        sw_spread = swf
        for axis, step in _NEIGHBOR_SHIFTS:
            n_h = np.roll(hyd, step, axis=axis)
            n_s = np.roll(swf, step, axis=axis)
            # roll wraps — zero the wrapped border slice.
            sl = [slice(None)] * 3
            sl[axis] = slice(0, 1) if step == 1 else slice(-1, None)
            n_h[tuple(sl)] = 0
            n_s[tuple(sl)] = 0.0
            grew = (n_h - 1) > spread
            spread = np.where(grew, n_h - 1, spread)
            sw_spread = np.where(grew, n_s, sw_spread)
        hyd = spread
        swf = sw_spread

    # --- hydrated cells → leaf cards ---------------------------------------
    ix, iy, iz = np.nonzero(hyd > 0)
    if ix.size == 0:
        return Leaves.empty()
    h = hyd[ix, iy, iz].astype(np.float32)  # 1 … rounds
    cell_sway = swf[ix, iy, iz]

    # Interior cells (high hydration) nearly always leaf; rim cells thin out.
    keep_p = np.clip(density * (0.45 + 0.55 * h / float(rounds)), 0.0, 1.0)
    keep = rng.random(ix.size) < keep_p
    ix, iy, iz = ix[keep], iy[keep], iz[keep]
    cell_sway = cell_sway[keep]
    if ix.size == 0:
        return Leaves.empty()

    counts = rng.integers(per_cell[0], per_cell[1] + 1, size=ix.size)
    n = int(counts.sum())
    if n == 0:
        return Leaves.empty()
    cell_center = (lo + (np.stack([ix, iy, iz], axis=1) + 0.5) * cell_m).astype(np.float32)
    centers = np.repeat(cell_center, counts, axis=0)
    centers = centers + rng.uniform(-0.45 * cell_m, 0.45 * cell_m, (n, 3)).astype(np.float32)

    sway = np.maximum(np.repeat(cell_sway, counts), rng.uniform(sway_min, 1.0, n)).astype(
        np.float32
    )
    radius = rng.uniform(leaf_size_m[0], leaf_size_m[1], n).astype(np.float32)

    if n > max_leaves:  # deterministic unbiased thinning
        pick = np.sort(rng.permutation(n)[:max_leaves])
        centers, radius, sway = centers[pick], radius[pick], sway[pick]

    return Leaves(
        center=np.ascontiguousarray(centers, np.float32),
        radius=radius,
        sway=np.clip(sway, 0.0, 1.0),
    )
