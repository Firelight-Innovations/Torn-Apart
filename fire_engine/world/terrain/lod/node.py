"""
terrain/lod/node.py — Coarse-LOD node addressing (`LodNode`).

A **coarse node** is one downsampled terrain tile at rank ``L``: it covers a
cube of ``2**L`` × ``2**L`` × ``2**L`` native ``L0`` chunk columns and is meshed
as a single 32³ block whose voxels are ``2**L`` times larger than an ``L0``
voxel.  :class:`LodNode` is the immutable key + addressing helper for one such
node; it owns the **snap rule** that maps an ``L0`` chunk coord to the node that
covers it (``node = chunk >> L`` on every axis) and the inverse enumeration of
the chunk block a node covers.

Why a cube (all three axes ``>> L``)
------------------------------------
The existing meshers (`build_mesh_faceted`, `build_mesh`) read
``n = chunk.materials.shape[0]`` and use that single ``n`` for all three axes —
they only mesh **cubes**.  A coarse node therefore downsamples a
``(32·2**L, 32·2**L, 32·2**L)`` tiled block to a ``32³`` cube (see
:func:`~fire_engine.world.terrain.lod.downsample.downsample_block`).  So the Z
axis is snapped by ``2**L`` exactly like X and Y — a deviation from the literal
scout-contract text that snaps only X/Y; the cube interpretation is what makes
the unchanged mesher run on the shim and is documented here as the authority.

Units & conventions
--------------------
- rank ``L`` : 0 = native (no downsample), 1 = 2×, 2 = 4×, 3 = 8×.
- factor ``k = 1 << L`` (1 / 2 / 4 / 8 voxels merged per axis).
- node voxel size = ``base_voxel_size * k`` meters (e.g. 0.5 / 1 / 2 / 4 m).
- node world span = ``base_chunk_meters * k`` meters per axis (16·k m).
- node key ``(rank, nx, ny, nz)`` — integer node coords in node space; the
  covered ``L0`` chunk block is ``[nx·k, (nx+1)·k) × …`` in chunk coords.
- Z-up (Panda3D native): +Z is world up.

No panda3d import — fully headless-testable (Hard Rule 1).

Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-core
"""

from __future__ import annotations

from dataclasses import dataclass

from fire_engine.core.math3d import Vec3

__all__ = ["LodNode", "rank_factor"]


def rank_factor(rank: int) -> int:
    """
    Downsample factor ``k = 1 << rank`` (voxels merged per axis at this rank).

    Parameters
    ----------
    rank : int
        Coarse LOD rank ``L`` (``0`` = native; ``1/2/3`` = 2×/4×/8×).

    Returns
    -------
    int
        ``2 ** rank`` (1, 2, 4, 8, …).

    Example
    -------
    >>> rank_factor(0), rank_factor(1), rank_factor(3)
    (1, 2, 8)

    Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-core
    """
    return 1 << int(rank)


@dataclass(frozen=True)
class LodNode:
    """
    Immutable coarse-LOD node key + addressing helper (one public dataclass).

    A node at ``(rank, nx, ny, nz)`` covers the cube of ``k³`` native ``L0``
    chunk columns ``[nx·k .. nx·k+k) × [ny·k .. ny·k+k) × [nz·k .. nz·k+k)``
    where ``k = 1 << rank``, and is meshed as a single 32³ block of
    ``base_voxel_size * k`` metre voxels.

    Snap rule
    ---------
    The node covering an ``L0`` chunk ``(cx, cy, cz)`` at rank ``L`` is
    ``(L, cx >> L, cy >> L, cz >> L)`` — floor-division by ``2**L`` on every
    axis (works for negative coords because ``>>`` floors).  Use
    :meth:`for_chunk` to build it.

    Attributes
    ----------
    rank : int
        Coarse LOD rank ``L`` (``0`` = native).
    nx, ny, nz : int
        Integer node coordinates in node space (chunk coord ``>> rank``).

    Example
    -------
    >>> n = LodNode.for_chunk((5, -3, 1), rank=1)   # 5>>1=2, -3>>1=-2, 1>>1=0
    >>> (n.rank, n.nx, n.ny, n.nz)
    (1, 2, -2, 0)
    >>> n.factor
    2
    >>> n.voxel_size(base_voxel_size=0.5)
    1.0
    >>> n.world_origin(base_chunk_meters=16.0)
    Vec3(64.0, -64.0, 0.0)
    >>> sorted(n.covered_chunks())[0]               # min-corner L0 chunk
    (4, -4, 0)

    Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-core
    """

    rank: int
    nx: int
    ny: int
    nz: int

    @classmethod
    def for_chunk(cls, chunk_coord: tuple[int, int, int], rank: int) -> LodNode:
        """
        Build the node that covers ``chunk_coord`` at ``rank`` (the snap rule).

        ``node = (rank, cx >> rank, cy >> rank, cz >> rank)``.  Right-shift
        floors toward negative infinity, so the snap is correct for negative
        chunk coords (e.g. ``-3 >> 1 == -2``).

        Parameters
        ----------
        chunk_coord : tuple[int, int, int]
            Native ``L0`` chunk coordinate ``(cx, cy, cz)``.
        rank : int
            Coarse LOD rank ``L``.

        Returns
        -------
        LodNode

        Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-core
        """
        cx, cy, cz = chunk_coord
        return cls(int(rank), cx >> rank, cy >> rank, cz >> rank)

    @property
    def key(self) -> tuple[int, int, int, int]:
        """The hashable node key ``(rank, nx, ny, nz)``.

        Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-core
        """
        return (self.rank, self.nx, self.ny, self.nz)

    @property
    def factor(self) -> int:
        """Downsample factor ``k = 1 << rank`` (voxels merged per axis).

        Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-core
        """
        return 1 << self.rank

    def voxel_size(self, base_voxel_size: float) -> float:
        """
        Node voxel edge in meters: ``base_voxel_size * 2**rank``.

        Parameters
        ----------
        base_voxel_size : float
            Native ``L0`` voxel edge in meters (``Config.voxel_size``, 0.5 m).

        Returns
        -------
        float
            Coarse voxel edge in meters (0.5/1/2/4 m for ranks 0..3).

        Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-core
        """
        return float(base_voxel_size) * self.factor

    def world_origin(self, base_chunk_meters: float) -> Vec3:
        """
        World-space minimum corner of this node, in meters (Vec3, Z-up).

        ``world_origin = (nx, ny, nz) * (base_chunk_meters * 2**rank)``.  This is
        identical to the min corner of the covered ``L0`` chunk block:
        ``(nx·k)·base_chunk_meters`` along each axis — so a coarse node lands
        exactly on the world position of the chunks it replaces.

        Parameters
        ----------
        base_chunk_meters : float
            Native ``L0`` chunk edge in meters (``Config.chunk_meters``, 16 m).

        Returns
        -------
        Vec3
            Node min-corner in world meters.

        Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-core
        """
        m = float(base_chunk_meters) * self.factor
        return Vec3(self.nx * m, self.ny * m, self.nz * m)

    def chunk_origin(self) -> tuple[int, int, int]:
        """
        Min-corner ``L0`` chunk coord of the covered block: ``(nx·k, ny·k, nz·k)``.

        Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-core
        """
        k = self.factor
        return (self.nx * k, self.ny * k, self.nz * k)

    def covered_chunks(self) -> list[tuple[int, int, int]]:
        """
        Enumerate the ``k³`` native ``L0`` chunk coords this node covers.

        The block ``[nx·k .. nx·k+k) × [ny·k .. ny·k+k) × [nz·k .. nz·k+k)`` in
        chunk-coordinate space, in C order (x outer, z inner) so it matches the
        downsample tiling order used by the coarse job.  ``k³`` is small
        (≤ 8³ = 512 at rank 3), so this is a chunk-level loop — NOT a per-voxel
        loop, and not a hot path (Hard Rule 4 is about voxel/vertex loops).

        Returns
        -------
        list[tuple[int, int, int]]
            ``k³`` chunk coords, ``(cx, cy, cz)``.

        Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-core
        """
        k = self.factor
        ox, oy, oz = self.chunk_origin()
        return [
            (ox + dx, oy + dy, oz + dz) for dx in range(k) for dy in range(k) for dz in range(k)
        ]
