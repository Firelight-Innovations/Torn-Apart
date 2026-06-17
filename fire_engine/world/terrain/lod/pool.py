"""
terrain/lod/pool.py — N-thread worker pool for threaded terrain meshing.

:class:`TerrainLodPool` is the :class:`~fire_engine.core._impl.worker_pool.WorkerPool`
subclass that fans :class:`~fire_engine.world.terrain.lod.job.LodJob`\\ s across
``n_workers`` daemon threads, each running the pure
:func:`~fire_engine.world.terrain.lod.job.build_lod_mesh` transform (Hard Rule
12: chunk meshing runs off the main thread).  Mesh builds are independent and
numpy releases the GIL during the heavy array ops, so the workers genuinely
overlap.

No panda3d import — fully headless-testable.

Docs: docs/systems/world.terrain.lod.md
"""

from __future__ import annotations

import numpy as np

from fire_engine.core import get_logger
from fire_engine.core._impl.worker_pool import WorkerPool
from fire_engine.world.terrain.lod.job import build_lod_mesh
from fire_engine.world.terrain.lod.types import LodJob, LodResult
from fire_engine.world.terrain.meshing import MeshArrays

__all__ = ["TerrainLodPool"]

_log = get_logger("terrain.lod.pool")


def _empty_mesh() -> MeshArrays:
    """A zero-face :class:`MeshArrays` — the failure sentinel's payload.

    Identical in shape/dtype to the empty arrays the mesher itself returns for
    a fully-buried chunk, so a consumer that uploads it gets an empty (harmless)
    geometry rather than wedging.
    """
    return MeshArrays(
        positions=np.zeros((0, 3), np.float32),
        normals=np.zeros((0, 3), np.float32),
        uvs=np.zeros((0, 2), np.float32),
        colors=np.zeros((0, 4), np.float32),
        indices=np.zeros((0,), np.uint32),
    )


class TerrainLodPool(WorkerPool[LodJob, LodResult]):
    """
    N-background-thread pool that builds chunk meshes off the main thread.

    Lifecycle: :meth:`start` once after construction, :meth:`stop` at shutdown.
    The threads are daemons, so a missed :meth:`stop` never blocks process exit.

    Producer/consumer
    -----------------
    - :meth:`submit` — main thread enqueues a
      :class:`~fire_engine.world.terrain.lod.job.LodJob` (non-blocking).
    - :meth:`drain_results` — main thread pops all finished
      :class:`~fire_engine.world.terrain.lod.job.LodResult`\\ s (non-blocking).
      Drain order is NOT submit order — match results to coords via
      :attr:`LodResult.coord`, and drop stale ones via :attr:`LodResult.seq`.
    - :meth:`pending` — jobs submitted but not yet drained.

    Each :class:`~fire_engine.world.terrain.lod.job.LodJob` carries its own
    immutable input snapshot, so there is no shared mutable state — every
    :meth:`_process` call is independent (Hard Rule 12).

    Parameters
    ----------
    n_workers : int
        Number of worker threads (clamped to at least 1 by the base class).

    Example
    -------
    >>> pool = TerrainLodPool(n_workers=2)
    >>> pool.start()
    >>> # pool.submit(job); results = pool.drain_results()
    >>> pool.stop()

    Docs: docs/systems/world.terrain.lod.md
    """

    def __init__(self, n_workers: int) -> None:
        super().__init__("TerrainLodPool", n_workers)

    def _process(self, job: LodJob) -> LodResult:
        """Build ``job``'s mesh on a worker thread.

        Docs: docs/systems/world.terrain.lod.md
        """
        return build_lod_mesh(job)

    def _on_error(self, job: LodJob) -> None:
        """Post an empty-mesh sentinel so a failed job never wedges the consumer.

        A raised mesh build must still produce a result for ``job.coord`` with
        the originating ``job.seq``, or the consumer's per-coord seq tracking
        stalls forever (mirrors ``VenturiWorker`` / ``CascadeAssemblyWorker``).

        Docs: docs/systems/world.terrain.lod.md
        """
        _log.exception(
            "LOD mesh build failed (coord %r, seq %d, rank %d)",
            job.coord,
            job.seq,
            job.rank,
        )
        self._out.put(LodResult(job.coord, _empty_mesh(), job.seq, job.rank))
