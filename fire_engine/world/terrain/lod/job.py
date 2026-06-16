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

from fire_engine.world.terrain.chunk import Chunk
from fire_engine.world.terrain.lod.types import LodJob, LodResult
from fire_engine.world.terrain.meshing import build_mesh
from fire_engine.world.terrain.surface_nets import build_mesh_faceted

__all__ = ["build_lod_mesh"]


def build_lod_mesh(job: LodJob) -> LodResult:
    """
    Pure transform: build ``job``'s mesh and wrap it in a ``LodResult``.

    Reconstructs a :class:`~fire_engine.world.terrain.chunk.Chunk` from the
    job's snapshot materials, then runs the mesher selected by
    ``job.mesh_style``:

    - ``"blocky"`` → :func:`~fire_engine.world.terrain.meshing.build_mesh`
      with the 6-dir ``neighbors`` solidity dict.
    - otherwise (``"faceted"``) →
      :func:`~fire_engine.world.terrain.surface_nets.build_mesh_faceted` with
      the 26-offset ``neighbors`` materials dict and ``job.shade_strength``.

    Always passes ``light_sampler=None`` (see
    :class:`~fire_engine.world.terrain.lod.types.LodJob`).  This is what runs on
    each worker thread; it is pure and deterministic — the same job always
    yields a byte-identical mesh, and it reproduces ``ChunkManager.mesh_chunk``'s
    output exactly for the same chunk + neighbour state.

    Parameters
    ----------
    job : LodJob
        The immutable hand-off snapshot to mesh.

    Returns
    -------
    LodResult
        ``(job.coord, mesh, job.seq)``.

    Docs: docs/systems/world.terrain.lod.md
    """
    chunk = Chunk(
        job.coord,
        job.materials,
        chunk_size=job.chunk_size,
        voxel_size=job.voxel_size,
    )
    if job.mesh_style == "blocky":
        mesh = build_mesh(chunk, job.neighbors, None)
    else:
        mesh = build_mesh_faceted(
            chunk,
            job.neighbors,
            None,
            shade_strength=job.shade_strength,
        )
    return LodResult(job.coord, mesh, job.seq)
