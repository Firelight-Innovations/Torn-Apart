"""
terrain/lod/coarse_assembly.py — Tile + downsample a coarse node's chunk block.

:func:`assemble_coarse_materials` gathers the ``(2**rank)³`` native ``L0``
chunk-materials covered by one :class:`~fire_engine.world.terrain.lod.node.LodNode`,
tiles them into a single ``(32·k, 32·k, 32·k)`` ``uint8`` block (axis 0 = X,
axis 1 = Y, axis 2 = Z, C-order — the tiling
:func:`~fire_engine.world.terrain.lod.downsample.downsample_block` expects), and
reduces it to the ``(32, 32, 32)`` coarse materials the
:class:`~fire_engine.world.terrain.lod.coarse_chunk._CoarseChunk` shim meshes.

A **materials provider** decouples this from the chunk store: it maps an ``L0``
chunk coord to that chunk's ``uint8 (32, 32, 32)`` materials.  ``LodStreamer``
passes a provider that returns a loaded chunk's live materials when present,
else the deterministic ``generate_chunk`` baseline — so a coarse node is
byte-identical whether or not its chunks happen to be loaded (the same
determinism guarantee the near faceted path relies on for seam-correctness).

The only Python iteration is over the ``k³`` chunks of one node (``≤ 8³ = 512``
at rank 3) — a chunk-level gather, NOT a per-voxel loop (Hard Rule 4; the actual
reduce is the whole-array :func:`downsample_block`).

No panda3d import — fully headless-testable (Hard Rule 1).

Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-render
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from fire_engine.world.terrain.lod.downsample import downsample_block
from fire_engine.world.terrain.lod.node import LodNode

__all__ = ["assemble_coarse_materials"]

# Type of the L0 chunk-materials provider passed by the streamer.
MaterialsProvider = Callable[[tuple[int, int, int]], np.ndarray]


def assemble_coarse_materials(
    node: LodNode,
    materials_for: MaterialsProvider,
    *,
    chunk_size: int = 32,
    mode: str = "any",
) -> np.ndarray:
    """
    Tile ``node``'s covered chunks and downsample to ``(32, 32, 32)`` coarse cells.

    Builds the ``(32·k, 32·k, 32·k)`` ``uint8`` block (``k = 2**node.rank``) by
    writing each covered ``L0`` chunk's ``materials`` into its tile slot, then
    runs :func:`~fire_engine.world.terrain.lod.downsample.downsample_block` to
    reduce it to one ``(32, 32, 32)`` coarse block (solidity ANY; material id by
    ``mode``).  Deterministic: a pure function of the node, the provider's
    outputs and ``mode`` — no RNG, no global state (Hard Rule 2).

    Parameters
    ----------
    node : LodNode
        The coarse node ``(rank, nx, ny, nz)`` to assemble (rank ≥ 1).
    materials_for : Callable[[tuple[int, int, int]], numpy.ndarray]
        Provider mapping an ``L0`` chunk coord ``(cx, cy, cz)`` to that chunk's
        ``uint8 (chunk_size,)*3`` materials ``[x, y, z]`` (loaded-or-generated).
    chunk_size : int, default 32
        Voxels per ``L0`` chunk edge (``Config.chunk_size``).
    mode : str, default ``"any"``
        Material reduce forwarded to ``downsample_block`` (``"any"`` = max-id,
        ``"majority"`` = vectorized mode).  Solidity is ANY in both.

    Returns
    -------
    numpy.ndarray
        ``uint8 (chunk_size, chunk_size, chunk_size)`` coarse materials
        ``[x, y, z]`` for the node.

    Example
    -------
    >>> import numpy as np
    >>> from fire_engine.world.terrain.lod.node import LodNode
    >>> def flat(_coord):                       # every chunk: flat grass floor
    ...     m = np.zeros((32, 32, 32), np.uint8)
    ...     m[:, :, 0] = 2
    ...     return m
    >>> node = LodNode(rank=1, nx=0, ny=0, nz=0)
    >>> coarse = assemble_coarse_materials(node, flat)
    >>> coarse.shape
    (32, 32, 32)
    >>> int(coarse[5, 5, 0])                     # floor survives the downsample
    2

    Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-render
    """
    k = node.factor
    n = int(chunk_size)
    tile = np.zeros((n * k, n * k, n * k), dtype=np.uint8)
    ox, oy, oz = node.chunk_origin()
    for dx in range(k):
        for dy in range(k):
            for dz in range(k):
                mats = np.asarray(materials_for((ox + dx, oy + dy, oz + dz)), dtype=np.uint8)
                tile[
                    dx * n : (dx + 1) * n,
                    dy * n : (dy + 1) * n,
                    dz * n : (dz + 1) * n,
                ] = mats
    return downsample_block(tile, node.rank, mode)
