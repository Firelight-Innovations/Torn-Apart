"""
wind/worker.py — Background terrain-venturi solver thread.

The wind field's terrain-funneling correction (``wind/venturi.py``) folds the
loaded chunk material arrays into a per-cell speed-up + deflection grid: a
column-occupancy reshape-fold over an 8×8 voxel footprint per wind cell, then a
handful of Jacobi flux-relaxation sweeps.  That is pure numpy and, like the
lighting cascade gather it mirrors, is heavy enough that running it on the
**main thread** would hitch a fly-around when the wind region recenters or
terrain is edited near the player.

This module moves that work onto a single background thread — a line-for-line
structural mirror of ``lighting/assembly_worker.py::CascadeAssemblyWorker``.
numpy releases the GIL during the heavy array ops, so the worker genuinely
overlaps the render thread.  The main thread (``wind/field.py``) keeps only the
cheap parts: snapshotting which chunk arrays to read into a :class:`VenturiJob`,
and applying the drained :class:`VenturiResult` to the gust field.

No panda3d import — fully headless-testable (same as ``assembly_worker.py``).

Threading contract
------------------
- The main thread builds a :class:`VenturiJob` with a *snapshot* of the chunk
  material arrays the solve will read (references, not copies — like the lighting
  worker), submits it on recenter / terrain-dirty, and keeps applying the
  previously-committed result until a fresh one drains.
- The worker solves and posts a :class:`VenturiResult`.
- The main thread drains results each frame, keeps the highest ``seq``, and
  applies only results whose ``origin_cell`` matches the region's *current*
  origin (a result for a stale origin is discarded — the field re-submits on
  recenter; identity correction holds in the meantime).  See
  ``wind/field.py`` and the system doc's origin-match Gotcha.

Determinism: a job's result depends only on its snapshot, origin and config
constants, so worker output is byte-identical to a synchronous
:func:`~fire_engine.world.wind.venturi.solve_venturi` for the same inputs (asserted in
``tests/test_wind_venturi.py``).  No RNG anywhere — venturi is a pure terrain
fold, not a stochastic process.

Docs: docs/systems/world.wind.md
"""

from __future__ import annotations

import numpy as np

from fire_engine.core import get_logger
from fire_engine.core._impl.worker import QueueWorker
from fire_engine.world.wind.types import VenturiJob, VenturiResult
from fire_engine.world.wind.venturi import solve_venturi

_log = get_logger("wind.worker")

__all__ = [
    "VenturiJob",
    "VenturiResult",
    "VenturiWorker",
]


def _identity_result(job: VenturiJob) -> VenturiResult:
    """
    The no-funneling result for ``job`` (speed-up 1, zero deflection).

    Posted when a solve raises so the consumer never starves — identical in
    spirit to ``CascadeAssemblyWorker``'s empty-bytes failure sentinel, but a
    *valid* identity grid the field can apply harmlessly.
    """
    n = int(job.cells)
    return VenturiResult(
        origin_cell=job.origin_cell,
        speedup=np.ones((n, n), dtype=np.float32),
        deflect=np.zeros((n, n, 2), dtype=np.float32),
        seq=job.seq,
    )


class VenturiWorker(QueueWorker[VenturiJob, VenturiResult]):
    """
    Single background thread that solves terrain-venturi correction grids.

    Lifecycle: :meth:`start` once after construction, :meth:`stop` at shutdown.
    The thread is a daemon, so a missed ``stop`` never blocks process exit.
    Structural mirror of
    :class:`fire_engine.lighting.assembly_worker.CascadeAssemblyWorker`.

    Producer/consumer
    -----------------
    - :meth:`submit` — main thread enqueues a :class:`VenturiJob` (non-blocking).
    - :meth:`drain_results` — main thread pops all finished
      :class:`VenturiResult`\\ s (non-blocking).
    - :meth:`pending` — number of jobs submitted but not yet returned.

    Both queues cross the thread boundary lock-free.  The job carries its own
    immutable input snapshot, so there is no shared mutable state at all (the
    lighting worker shares a block cache; venturi has none to share).

    Example
    -------
    >>> worker = VenturiWorker()
    >>> worker.start()
    >>> # worker.submit(job); results = worker.drain_results()
    >>> worker.stop()

    Docs: docs/systems/world.wind.md
    """

    def __init__(self) -> None:
        super().__init__("WindVenturiWorker")

    def _process(self, job: VenturiJob) -> VenturiResult:
        return solve_venturi(job)

    def _on_error(self, job: VenturiJob) -> None:
        _log.exception("Venturi solve failed (origin %r, seq %d)", job.origin_cell, job.seq)
        # Post a valid IDENTITY result so the consumer never starves —
        # a raised job must not leave the field stuck without a
        # correction grid forever (mirrors the assembly worker's
        # failure-sentinel discipline, but with a harmless identity).
        self._out.put(_identity_result(job))
