"""
terrain/lod/downsample.py — Whole-array downsample of a tiled materials block.

:func:`downsample_block` reduces a **tiled** ``(32·k, 32·k, Zspan·k)`` ``uint8``
materials block (``k = 1 << rank`` native chunks stacked per axis) to a coarse
``(32, 32, Zspan)`` materials array for one coarse LOD node — purely with numpy
reshape / stride views and whole-array reductions (Hard Rule 4: no per-voxel
Python loops).

Reduction policy
----------------
- **Solidity is always ANY** — a coarse cell is solid if *any* of the ``k³``
  merged native voxels is solid.  This deliberately preserves thin walls and the
  horizon silhouette (a 1-voxel wall survives any downsample factor) rather than
  eroding them, which a majority/min reduce would do.
- **Material id** of a solid coarse cell:

  - ``"any"`` mode (default): **max id over the solid voxels** — the largest
    material id present wins (grass id 2 beats dirt id 1 on the surface skin),
    so the coarse skin keeps the topmost cover material.
  - ``"majority"`` mode: the most-common **nonzero** id over the ``k³`` voxels
    (vectorised one-hot count + argmax — still whole-array, no voxel loop).

  Air cells (no solid voxel) are ``0`` in both modes; solidity stays ANY.

Determinism
-----------
Pure function of ``(tile, rank, mode)`` — no RNG, no global state.  The same
input always yields a byte-identical output.

Units & conventions
--------------------
``materials[x, y, z]`` ``uint8``; ``0`` = air, ``≥1`` = solid material id.  The
input block must already be assembled (tiled) so axis 0 is X, axis 1 is Y, axis
2 is Z, each spanning ``32·k`` (X/Y) or ``Zspan·k`` (Z) native voxels.

No panda3d import — fully headless-testable (Hard Rule 1).

Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-core
"""

from __future__ import annotations

import numpy as np

__all__ = ["downsample_block"]


def _reshape_6d(tile: np.ndarray, k: int) -> np.ndarray:
    """
    View ``tile`` as the 6-D ``(32, k, 32, k, Z, k)`` stride array (no copy).

    Splits each axis into ``(coarse, fine)`` pairs so the ``k³`` merged voxels of
    coarse cell ``(i, j, l)`` are ``r[i, :, j, :, l, :]``.  Requires
    C-contiguous input (callers pass :func:`numpy.ascontiguousarray`).
    Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-core
    """
    sx, sy, sz = tile.shape
    cx, cy, cz = sx // k, sy // k, sz // k
    return tile.reshape(cx, k, cy, k, cz, k)


def downsample_block(tile: np.ndarray, rank: int, mode: str = "any") -> np.ndarray:
    """
    Reduce a tiled ``(32·k, 32·k, Z·k)`` materials block to ``(32, 32, Z)``.

    Whole-array reduce by factor ``k = 1 << rank`` on every axis: solidity via
    ANY over the merged voxels, material id via max-id (``mode="any"``) or
    majority (``mode="majority"``).  No per-voxel Python loops — a single 6-D
    reshape + axis reductions (Hard Rule 4).

    Parameters
    ----------
    tile : numpy.ndarray
        ``uint8`` block of shape ``(32·k, 32·k, Z·k)`` indexed ``[x, y, z]``
        (``k = 1 << rank``), the ``k³`` native chunks assembled per axis.  Each
        axis length must be divisible by ``k``.  Copied to C-contiguous once if
        needed.
    rank : int
        Coarse LOD rank ``L`` (``0`` = identity, ``1/2/3`` = 2×/4×/8×).
    mode : str, default ``"any"``
        Material reduce: ``"any"`` → max-id over solid voxels (grass 2 beats
        dirt 1); ``"majority"`` → most-common nonzero id.  Solidity is ANY in
        both.

    Returns
    -------
    numpy.ndarray
        ``uint8`` coarse materials of shape ``(32, 32, Z)`` — ``0`` where no
        merged voxel was solid, else the reduced material id.

    Raises
    ------
    ValueError
        If an axis length is not divisible by ``k``.

    Example
    -------
    >>> import numpy as np
    >>> tile = np.zeros((64, 64, 64), dtype=np.uint8)   # rank 1 → k=2
    >>> tile[10, 10, 10] = 2                             # one solid (grass) voxel
    >>> coarse = downsample_block(tile, rank=1)
    >>> coarse.shape
    (32, 32, 32)
    >>> int(coarse[5, 5, 5])                             # 10//2 == 5 cell is solid
    2

    Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-core
    """
    k = 1 << int(rank)
    if k == 1:
        # Rank 0 is identity (the near path never calls this, but stay total).
        return np.ascontiguousarray(tile, dtype=np.uint8)

    for axis_len in tile.shape:
        if axis_len % k != 0:
            raise ValueError(
                f"downsample_block: axis length {axis_len} not divisible by k={k} (rank {rank})"
            )

    tile = np.ascontiguousarray(tile, dtype=np.uint8)
    r = _reshape_6d(tile, k)  # (32, k, 32, k, Z, k) — no copy
    merge_axes = (1, 3, 5)

    # Solidity: ANY merged voxel solid → coarse cell solid (preserves thin walls).
    solid = (r > 0).any(axis=merge_axes)  # (32, 32, Z) bool

    if mode == "majority":
        mat = _majority_material(r, merge_axes)
    else:
        # max-id over solid voxels only (air contributes 0, so plain max works).
        mat = r.max(axis=merge_axes).astype(np.uint8)  # (32, 32, Z)

    # Force air cells to 0 (mat is already 0 there, but make the contract explicit).
    return np.where(solid, mat, np.uint8(0)).astype(np.uint8)


def _majority_material(r: np.ndarray, merge_axes: tuple[int, int, int]) -> np.ndarray:
    """
    Most-common **nonzero** material id over the merged axes (vectorised mode).

    One-hot counts each candidate id ``1..max_id`` across the ``k³`` merged
    voxels and takes the argmax — whole-array, no per-voxel Python loop (the
    only loop is over the small distinct-id alphabet, ``≤ max_id`` values, which
    is a material-id loop, not a voxel loop; Hard Rule 4).  Ties go to the lower
    id (``argmax`` returns the first max), and an all-air cell stays ``0``.
    Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-core
    """
    max_id = int(r.max())
    if max_id == 0:
        # Fully air — collapse the merge axes to the coarse shape, all zeros.
        return np.zeros(
            tuple(r.shape[a] for a in (0, 2, 4)),
            dtype=np.uint8,
        )
    # counts[..., m-1] = number of merged voxels equal to id m (m in 1..max_id).
    ids = np.arange(1, max_id + 1, dtype=np.uint8)
    # (32, 32, Z, max_id): broadcast-compare then sum over the merged axes.
    counts = (r[..., None] == ids).sum(axis=merge_axes)  # sums the 3 merge axes
    winner = counts.argmax(axis=-1).astype(np.uint8) + np.uint8(1)  # id = index+1
    any_solid = counts.sum(axis=-1) > 0
    return np.where(any_solid, winner, np.uint8(0)).astype(np.uint8)
