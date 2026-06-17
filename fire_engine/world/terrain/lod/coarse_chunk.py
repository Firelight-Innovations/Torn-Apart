"""
terrain/lod/coarse_chunk.py — Duck-typed `Chunk` shim for coarse LOD nodes.

:class:`_CoarseChunk` lets the existing meshers
(:func:`~fire_engine.world.terrain.surface_nets.build_mesh_faceted`,
:func:`~fire_engine.world.terrain.meshing.build_mesh`) run **unchanged** on a
downsampled coarse node.  The meshers read exactly three ``Chunk`` attributes
plus one method:

- ``materials`` — ``uint8 (32, 32, 32)`` ``[x, y, z]`` (here the *downsampled*
  coarse block, NOT a native chunk).
- ``_voxel_size`` — meters per coarse voxel = ``base_voxel_size * 2**rank``
  (0.5 / 1 / 2 / 4 m for ranks 0..3).
- ``world_origin`` — ``Vec3`` node min-corner in meters.
- ``is_solid_mask()`` — ``materials > 0`` (the blocky path reads it).

Implementation
--------------
The shim wraps a **real** :class:`~fire_engine.world.terrain.chunk.Chunk`: a
``Chunk`` already derives ``world_origin = coord * (chunk_size * voxel_size)``,
so passing ``coord = (nx, ny, nz)`` (node coords) and
``voxel_size = base_voxel_size * 2**rank`` lands the node at the correct world
metres, because ``coord * (32 · base_vs · 2**L) == (nx · 2**L) · (32 · base_vs)``
— the min corner of the ``L0`` chunk block the node covers.  ``_CoarseChunk``
exists as the named, validating wrapper that documents this coord/voxel mapping
(one-public-symbol-per-module, Hard Rule 9); it is **read-only**, not
``Saveable``, and never enters ``ChunkManager.chunks``.

No panda3d import — fully headless-testable (Hard Rule 1).

Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-core
"""

from __future__ import annotations

import numpy as np

from fire_engine.core.math3d import Vec3
from fire_engine.world.terrain.chunk import Chunk
from fire_engine.world.terrain.lod.node import LodNode

__all__ = ["_CoarseChunk"]


class _CoarseChunk:
    """
    Read-only `Chunk`-compatible view of one downsampled coarse LOD node.

    Exposes exactly the four members the meshers read (``materials``,
    ``_voxel_size``, ``world_origin``, ``is_solid_mask()``) by wrapping a real
    :class:`~fire_engine.world.terrain.chunk.Chunk` built at the node's coord +
    scaled voxel size, so ``build_mesh_faceted`` / ``build_mesh`` mesh it
    unchanged and emit world-correct geometry.

    Parameters
    ----------
    node : LodNode
        The coarse node key ``(rank, nx, ny, nz)``.
    materials : numpy.ndarray
        ``uint8 (32, 32, 32)`` downsampled coarse materials ``[x, y, z]``
        (the output of
        :func:`~fire_engine.world.terrain.lod.downsample.downsample_block`).
    base_voxel_size : float, default 0.5
        Native ``L0`` voxel edge in meters (``Config.voxel_size``).  The coarse
        voxel edge is ``base_voxel_size * 2**rank``.
    chunk_size : int, default 32
        Voxels per coarse-node edge (always 32 — the meshers assume a cube).

    Example
    -------
    >>> import numpy as np
    >>> from fire_engine.world.terrain.lod.node import LodNode
    >>> mats = np.zeros((32, 32, 32), dtype=np.uint8)
    >>> mats[:, :, 0] = 2                              # flat grass floor
    >>> node = LodNode(rank=1, nx=2, ny=-2, nz=0)
    >>> cc = _CoarseChunk(node, mats, base_voxel_size=0.5)
    >>> cc._voxel_size                                 # 0.5 * 2**1
    1.0
    >>> cc.world_origin                                # (2,-2,0)*(16*2)
    Vec3(64.0, -64.0, 0.0)
    >>> from fire_engine.world.terrain.surface_nets import build_mesh_faceted
    >>> mesh = build_mesh_faceted(cc)                  # runs unchanged on the shim
    >>> mesh.is_empty
    False

    Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-core
    """

    __slots__ = ("_chunk", "node")

    def __init__(
        self,
        node: LodNode,
        materials: np.ndarray,
        *,
        base_voxel_size: float = 0.5,
        chunk_size: int = 32,
    ) -> None:
        self.node = node
        self._chunk = Chunk(
            (node.nx, node.ny, node.nz),
            materials,
            chunk_size=int(chunk_size),
            voxel_size=node.voxel_size(base_voxel_size),
        )

    @property
    def materials(self) -> np.ndarray:
        """Downsampled coarse materials, ``uint8 (32, 32, 32)`` ``[x, y, z]``.

        Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-core
        """
        return self._chunk.materials

    @property
    def _voxel_size(self) -> float:
        """Coarse voxel edge in meters (``base_voxel_size * 2**rank``).

        Named with the leading underscore to match the private attribute the
        meshers read (``surface_nets.py``/``meshing.py`` use ``chunk._voxel_size``).
        Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-core
        """
        return self._chunk._voxel_size

    @property
    def world_origin(self) -> Vec3:
        """Node min-corner in world meters (Vec3, Z-up) — see module docstring.

        Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-core
        """
        return self._chunk.world_origin

    def is_solid_mask(self) -> np.ndarray:
        """Boolean solidity mask ``materials > 0``, ``(32, 32, 32)`` ``[x, y, z]``.

        Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-core
        """
        return self._chunk.is_solid_mask()
