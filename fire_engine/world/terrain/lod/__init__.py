"""
terrain/lod/ — Threaded terrain meshing / LOD job pipeline.

This sub-package holds the **pure, headless** pieces of the off-main-thread
terrain mesher (Hard Rule 12: the main thread orchestrates only — chunk
meshing runs on a worker pool).  It owns:

- :class:`LodJob` — an immutable hand-off snapshot (a *copy* of one chunk's
  materials plus its neighbour materials/solidity) carried across the thread
  boundary, tagged with a monotonic ``seq`` for staleness discipline.
- :class:`LodResult` — the produced :class:`~fire_engine.world.terrain.meshing.MeshArrays`
  plus the originating ``coord`` and ``seq``.
- :func:`build_lod_mesh` — the PURE transform that reconstructs a
  :class:`~fire_engine.world.terrain.chunk.Chunk` from the snapshot and runs the
  configured mesher.  Byte-identical to ``ChunkManager.mesh_chunk`` for the
  same chunk + neighbour state; runs both on the worker thread and (for tests)
  synchronously.
- :class:`TerrainLodPool` — the :class:`~fire_engine.core._impl.worker_pool.WorkerPool`
  subclass that fans :class:`LodJob`\\ s across N worker threads.

The snapshot/seq plumbing in ``chunk_manager.py`` (which copies the arrays and
assigns ``seq``) is integration and lives elsewhere; this package is the
independently-testable core.

Docs: docs/systems/world.terrain.lod.md
"""

from __future__ import annotations

from fire_engine.world.terrain.lod.job import build_lod_mesh
from fire_engine.world.terrain.lod.pool import TerrainLodPool
from fire_engine.world.terrain.lod.types import LodJob, LodResult

__all__ = ["LodJob", "LodResult", "TerrainLodPool", "build_lod_mesh"]
