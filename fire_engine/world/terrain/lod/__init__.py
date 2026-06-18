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
- :class:`LodStreamer` — the async streaming driver: each frame it drains finished
  meshes from a :class:`TerrainLodPool` into the ``ChunkManager``'s
  ``pending_meshes`` and submits a bounded batch of fresh jobs (dirty-first, then
  nearest missing), owning the per-coord ``seq`` staleness authority.  It is the
  off-thread counterpart of ``ChunkManager.stream_frame``.

Coarse ranks (P2) add the distant horizon:

- :class:`LodNode` / :func:`rank_factor` — coarse-node addressing (chunk ``>> L``).
- :func:`downsample_block` — whole-array reduce of a tiled chunk block to a 32³ node.
- :func:`assemble_coarse_materials` — gather a node's ``(2**L)³`` chunk block and
  downsample it (loaded-or-generated, deterministic).
- :class:`~fire_engine.world.terrain.lod.coarse_chunk._CoarseChunk` — the
  duck-typed ``Chunk`` shim that lets the unchanged mesher run on a coarse node.
- :func:`desired_node_set` / :class:`NodePlan` — the vectorised planner that
  partitions the camera window into near (L0) chunks + per-rank coarse nodes.
- :class:`CoarseLodStreamer` — the async coarse-node streaming driver: plans the
  desired nodes, submits ``rank > 0`` :class:`LodJob`\\ s to a separate
  :class:`TerrainLodPool`, and drains finished coarse meshes into the
  ``ChunkManager``'s ``pending_coarse_meshes`` (with ``unloaded_coarse_this_frame``
  for the hard band cut).  ``build_lod_mesh`` is rank-aware: ``rank == 0`` meshes a
  native chunk (byte-identical to P1), ``rank > 0`` meshes a ``_CoarseChunk``.

The neighbour-snapshotting still lives in ``chunk_manager.py``
(``_neighbor_materials`` / ``_neighbor_solids``); :class:`LodStreamer` copies what
those return and assigns the ``seq``.  ``build_lod_mesh`` + the pool remain the
independently-testable, panda3d-free core.

Docs: docs/systems/world.terrain.lod.md
"""

from __future__ import annotations

from fire_engine.world.terrain.lod.coarse_assembly import assemble_coarse_materials
from fire_engine.world.terrain.lod.coarse_streamer import CoarseLodStreamer
from fire_engine.world.terrain.lod.desired import NodePlan, desired_node_set
from fire_engine.world.terrain.lod.downsample import downsample_block
from fire_engine.world.terrain.lod.job import build_lod_mesh
from fire_engine.world.terrain.lod.node import LodNode, rank_factor
from fire_engine.world.terrain.lod.pool import TerrainLodPool
from fire_engine.world.terrain.lod.streamer import LodStreamer
from fire_engine.world.terrain.lod.types import LodJob, LodResult

__all__ = [
    "CoarseLodStreamer",
    "LodJob",
    "LodNode",
    "LodResult",
    "LodStreamer",
    "NodePlan",
    "TerrainLodPool",
    "assemble_coarse_materials",
    "build_lod_mesh",
    "desired_node_set",
    "downsample_block",
    "rank_factor",
]
