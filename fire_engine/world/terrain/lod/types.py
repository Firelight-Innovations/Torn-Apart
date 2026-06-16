"""
terrain/lod/types.py — Immutable LOD mesh job/result hand-off dataclasses.

Two trivial frozen support types (grouped here per Hard Rule 9): :class:`LodJob`
(the immutable snapshot a chunk's threaded mesh build reads) and :class:`LodResult`
(the produced mesh + originating coord/seq).  They carry the data across the
worker-thread boundary for :func:`~fire_engine.world.terrain.lod.job.build_lod_mesh`
and :class:`~fire_engine.world.terrain.lod.pool.TerrainLodPool`.

No panda3d import — fully headless-testable.

Docs: docs/systems/world.terrain.lod.md
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fire_engine.world.terrain.meshing import MeshArrays

__all__ = ["LodJob", "LodResult"]


@dataclass(frozen=True)
class LodJob:
    """
    Immutable hand-off snapshot for one chunk's threaded mesh build.

    The caller (``ChunkManager``, integrated separately) takes a **copy** of the
    chunk's ``materials`` and assembles the ``neighbors`` dict the *same way*
    ``ChunkManager._neighbor_materials`` / ``_neighbor_solids`` do, copying each
    neighbour array.  Because every array is a private, immutable snapshot, the
    worker thread only ever reads them — there is no cross-thread data race even
    while the live chunk is being edited on the main thread (Hard Rule 12).

    No ``light_sampler`` field: the threaded path bakes **no** per-face light.
    The live GPU game already passes ``light_sampler=None`` to the mesher (sun
    light is applied on the GPU), so the threaded path matches it exactly.
    Baked-light callers (if any are added) must stay on the synchronous
    ``mesh_chunk`` path in this phase.

    Attributes
    ----------
    coord : tuple[int, int, int]
        Integer chunk coordinate ``(cx, cy, cz)`` this job meshes.
    materials : numpy.ndarray
        ``uint8`` ``(chunk_size,)*3`` — an immutable snapshot COPY of the
        chunk's materials, indexed ``[x, y, z]``.
    neighbors : dict[tuple[int, int, int], numpy.ndarray | str]
        Neighbour data, dispatched by ``mesh_style``:

        - ``"faceted"`` — the 26 offsets in
          :data:`~fire_engine.world.terrain.surface_nets.NEIGHBOR_OFFSETS_26`
          mapped to a ``uint8 (32,32,32)`` materials COPY (or
          :data:`~fire_engine.world.terrain.meshing.WORLD_FLOOR_SOLID`).
        - ``"blocky"`` — the 6 face dirs mapped to a ``bool (32,32,32)``
          solidity COPY (or ``WORLD_FLOOR_SOLID``).

        Held by reference; whatever the caller passes is forwarded verbatim to
        the mesher.
    chunk_size : int
        Voxels per chunk edge (``Config.chunk_size``, 32).
    voxel_size : float
        Meters per voxel edge (``Config.voxel_size``, 0.5).
    shade_strength : float
        Faceted normal-accent strength (``Config.facet_shade_strength``);
        ignored for the blocky mesher.
    mesh_style : str
        ``"faceted"`` (default mesher) or ``"blocky"``.
    seq : int
        Monotonic submit sequence number for staleness discipline.  Carried
        unchanged into :class:`LodResult` so the consumer can drop results for
        a coord whose chunk was re-submitted with a newer ``seq`` while this one
        was in flight (mirrors ``VenturiWorker`` / ``CascadeAssemblyWorker``).

    Docs: docs/systems/world.terrain.lod.md
    """

    coord: tuple[int, int, int]
    materials: np.ndarray
    neighbors: dict[tuple[int, int, int], np.ndarray | str]
    chunk_size: int
    voxel_size: float
    shade_strength: float
    mesh_style: str
    seq: int


@dataclass(frozen=True)
class LodResult:
    """
    Result of one threaded mesh build, drained by the consumer.

    Attributes
    ----------
    coord : tuple[int, int, int]
        Chunk coordinate this mesh belongs to (echoes :attr:`LodJob.coord`).
    mesh : MeshArrays
        The produced mesh (possibly empty — see
        :class:`~fire_engine.world.terrain.meshing.MeshArrays.is_empty`).
    seq : int
        The originating :attr:`LodJob.seq`, for per-coord staleness checks.

    Docs: docs/systems/world.terrain.lod.md
    """

    coord: tuple[int, int, int]
    mesh: MeshArrays
    seq: int
