"""
terrain/lod/job.py — The pure LOD mesh-build transform.

Holds :func:`build_lod_mesh`, the pure function that turns a
:class:`~fire_engine.world.terrain.lod.types.LodJob` into a
:class:`~fire_engine.world.terrain.lod.types.LodResult`.  It is exactly what
runs on each ``TerrainLodPool`` worker thread, and is also synchronously
callable so its output can be asserted byte-identical to
``ChunkManager.mesh_chunk`` (Hard Rule 12: terrain meshing moves off the main
thread, but stays deterministic).

The job/result dataclasses live in ``terrain/lod/types.py``.

No panda3d import — fully headless-testable.

Docs: docs/systems/world.terrain.lod.md
"""

from __future__ import annotations

from typing import Any

from fire_engine.world.terrain.chunk import Chunk
from fire_engine.world.terrain.lod.coarse_chunk import _CoarseChunk
from fire_engine.world.terrain.lod.node import LodNode
from fire_engine.world.terrain.lod.types import LodJob, LodResult
from fire_engine.world.terrain.meshing import build_mesh
from fire_engine.world.terrain.surface_nets import build_mesh_faceted

__all__ = ["build_lod_mesh"]


def _chunk_for(job: LodJob) -> Any:
    """
    Reconstruct the meshable ``Chunk``-like object for ``job`` (rank-aware).

    ``rank == 0`` → a real :class:`~fire_engine.world.terrain.chunk.Chunk` at
    ``job.coord`` (the native L0 path; byte-identical to P1).  ``rank > 0`` → a
    :class:`~fire_engine.world.terrain.lod.coarse_chunk._CoarseChunk` whose
    ``materials`` are the job's already-downsampled ``(32, 32, 32)`` coarse
    block and whose coarse voxel edge is ``job.voxel_size`` (= ``base * 2**L``).
    The shim duck-types ``Chunk`` so the unchanged meshers run on it.
    Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-core
    """
    if job.rank <= 0:
        return Chunk(
            job.coord,
            job.materials,
            chunk_size=job.chunk_size,
            voxel_size=job.voxel_size,
        )
    # Coarse node: job.coord is the node coord, job.voxel_size is the scaled
    # coarse voxel edge. Recover base_voxel_size so _CoarseChunk lands the node
    # at the correct world metres (voxel_size == base * 2**rank).
    base_vs = job.voxel_size / float(1 << job.rank)
    node = LodNode(job.rank, job.coord[0], job.coord[1], job.coord[2])
    return _CoarseChunk(
        node,
        job.materials,
        base_voxel_size=base_vs,
        chunk_size=job.chunk_size,
    )


def build_lod_mesh(job: LodJob) -> LodResult:
    """
    Pure transform: build ``job``'s mesh and wrap it in a ``LodResult``.

    Rank-aware reconstruction (see :func:`_chunk_for`): a real
    :class:`~fire_engine.world.terrain.chunk.Chunk` for ``rank == 0`` (native
    L0) or a
    :class:`~fire_engine.world.terrain.lod.coarse_chunk._CoarseChunk` for
    ``rank > 0`` (a downsampled coarse node).  Then runs the mesher selected by
    ``job.mesh_style``:

    - ``"blocky"`` → :func:`~fire_engine.world.terrain.meshing.build_mesh`
      with the 6-dir ``neighbors`` solidity dict.
    - otherwise (``"faceted"``) →
      :func:`~fire_engine.world.terrain.surface_nets.build_mesh_faceted` with
      the 26-offset ``neighbors`` materials dict and ``job.shade_strength``.

    Always passes ``light_sampler=None`` (see
    :class:`~fire_engine.world.terrain.lod.types.LodJob`).  This is what runs on
    each worker thread; it is pure and deterministic — the same job always
    yields a byte-identical mesh, and for ``rank == 0`` it reproduces
    ``ChunkManager.mesh_chunk``'s output exactly for the same chunk + neighbour
    state.

    Parameters
    ----------
    job : LodJob
        The immutable hand-off snapshot to mesh.

    Returns
    -------
    LodResult
        ``(job.coord, mesh, job.seq, job.rank)``.

    Docs: docs/systems/world.terrain.lod.md
    """
    chunk = _chunk_for(job)
    if job.mesh_style == "blocky":
        mesh = build_mesh(chunk, job.neighbors, None)
    else:
        mesh = build_mesh_faceted(
            chunk,
            job.neighbors,
            None,
            shade_strength=job.shade_strength,
        )
    return LodResult(job.coord, mesh, job.seq, job.rank)
