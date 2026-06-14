"""
terrain/chunk.py — The voxel ``Chunk``: a 32³ block of material ids.

A chunk is the atomic unit of terrain storage, generation, meshing, streaming
and saving.  It holds a dense ``uint8`` material array plus two flags
(``dirty``, ``edited``) and knows its own world-space origin.

Units & conventions
--------------------
- Voxel edge      = 0.5 m  (``Config.voxel_size``)
- Chunk edge      = 32 voxels = 16 m  (``Config.chunk_size`` × voxel_size)
- Chunk coord     = integer ``(cx, cy, cz)``; world origin = ``coord * 16.0 m``
- Z-up (Panda3D native): +Z is world up.

Material array index convention
-------------------------------
``materials[x, y, z]`` where ``x, y, z`` are **local voxel indices 0..31**:

    x → local +X (world east),   spans world X ∈ [origin_x, origin_x + 16)
    y → local +Y (world north),  spans world Y ∈ [origin_y, origin_y + 16)
    z → local +Z (world up),     spans world Z ∈ [origin_z, origin_z + 16)

The world-space centre of voxel ``(x, y, z)`` is::

    world_xyz = world_origin + (np.array([x, y, z]) + 0.5) * voxel_size

A value of ``0`` is **air**; any value ``>= 1`` is a **solid material id**
(1 = default ground in Session 1).  Stored as ``uint8`` (256 material ids max).
"""

from __future__ import annotations

import numpy as np

from fire_engine.core.math3d import Vec3


class Chunk:
    """
    One 32³ voxel chunk of terrain.

    Attributes
    ----------
    coord : tuple[int, int, int]
        Integer chunk coordinate ``(cx, cy, cz)``.  World origin (the
        minimum-corner of the chunk in meters) is ``coord * chunk_meters``.
    materials : numpy.ndarray
        ``uint8`` array of shape ``(32, 32, 32)`` indexed ``[x, y, z]``.
        ``0`` = air, ``>= 1`` = solid material id.
    dirty : bool
        ``True`` when the voxel data has changed since the last mesh build,
        i.e. the chunk needs to be re-meshed.  Set by generation, brush edits,
        and ``apply_delta``.  Cleared by the chunk manager after meshing.
    edited : bool
        ``True`` when the chunk deviates from its procedurally generated
        baseline (a brush touched it, or a save delta was applied).  Only
        ``edited`` chunks are written to the save delta.  Never auto-cleared.

    Example
    -------
    >>> import numpy as np
    >>> c = Chunk((1, 0, -1))
    >>> c.world_origin
    Vec3(16.0, 0.0, -16.0)
    >>> c.materials[0, 0, 0] = 1          # make one voxel solid
    >>> c.is_solid_mask()[0, 0, 0]
    np.True_
    """

    __slots__ = ("_chunk_size", "_voxel_size", "coord", "dirty", "edited", "materials")

    def __init__(
        self,
        coord: tuple[int, int, int],
        materials: np.ndarray | None = None,
        *,
        chunk_size: int = 32,
        voxel_size: float = 0.5,
    ) -> None:
        """
        Create a chunk.

        Parameters
        ----------
        coord : tuple[int, int, int]
            Integer chunk coordinate ``(cx, cy, cz)``.
        materials : numpy.ndarray, optional
            A ``uint8`` ``(chunk_size,)*3`` array.  If ``None``, an all-air
            (all-zero) array is allocated.  The array is taken by reference
            (not copied) so generation can hand its result straight in.
        chunk_size : int, default 32
            Voxels per chunk edge (``Config.chunk_size``).
        voxel_size : float, default 0.5
            Meters per voxel edge (``Config.voxel_size``).
        """
        self.coord: tuple[int, int, int] = (int(coord[0]), int(coord[1]), int(coord[2]))
        self._chunk_size = int(chunk_size)
        self._voxel_size = float(voxel_size)
        if materials is None:
            self.materials = np.zeros((chunk_size, chunk_size, chunk_size), dtype=np.uint8)
        else:
            if materials.shape != (chunk_size, chunk_size, chunk_size):
                raise ValueError(f"materials must be {(chunk_size,) * 3}, got {materials.shape}")
            self.materials = np.ascontiguousarray(materials, dtype=np.uint8)
        self.dirty: bool = True
        self.edited: bool = False

    @property
    def world_origin(self) -> Vec3:
        """
        World-space minimum-corner of this chunk, in meters (Vec3, Z-up).

        ``world_origin = coord * chunk_meters`` where
        ``chunk_meters = chunk_size * voxel_size`` (16.0 m with the defaults).
        """
        m = self._chunk_size * self._voxel_size
        return Vec3(self.coord[0] * m, self.coord[1] * m, self.coord[2] * m)

    @property
    def chunk_meters(self) -> float:
        """World-space side length of the chunk in meters (16.0 m default)."""
        return self._chunk_size * self._voxel_size

    def is_solid_mask(self) -> np.ndarray:
        """
        Boolean solidity mask, shape ``(32, 32, 32)`` indexed ``[x, y, z]``.

        ``True`` where ``materials > 0`` (solid), ``False`` where air.  This is
        the mask the mesher and lighting consume.

        Returns
        -------
        numpy.ndarray
            ``bool`` array, ``materials > 0``.
        """
        return self.materials > 0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        n_solid = int((self.materials > 0).sum())
        return (
            f"Chunk(coord={self.coord}, solid={n_solid}/{self.materials.size}, "
            f"dirty={self.dirty}, edited={self.edited})"
        )
