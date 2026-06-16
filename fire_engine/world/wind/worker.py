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
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

import numpy as np

from fire_engine.core import get_logger
from fire_engine.world.wind.venturi import solve_venturi

_log = get_logger("wind.worker")

__all__ = [
    "VenturiJob",
    "VenturiResult",
    "VenturiWorker",
]


@dataclass(frozen=True)
class VenturiJob:
    """
    One terrain-venturi solve request.

    Attributes
    ----------
    origin_cell : tuple[int, int]
        Wind-region origin (cell ``(0, 0)``'s corner, in wind cells) the solve
        is for.  Committed to the field only when a result with this origin
        drains — a result whose ``origin_cell`` no longer matches the region's
        current origin is discarded (the field re-submits on recenter).
    cells : int
        Wind cells per axis (e.g. 64).
    cell_m : float
        Wind cell edge in meters (e.g. 4.0).
    chunk_size : int
        Voxels per chunk edge (``config.chunk_size``, 32).
    voxel_size : float
        Meters per voxel (``config.voxel_size``, 0.5).  Each wind cell of
        ``cell_m`` therefore covers ``cell_m / voxel_size`` voxels per axis
        (8 at the defaults).
    ground_band : tuple[float, float]
        World Z band ``(z_lo, z_hi)`` over which column occupancy is folded
        (``[ground, ground + wind_layer_m]``).
    materials : dict[tuple[int, int, int], numpy.ndarray]
        Snapshot of ``uint8 (S, S, S)`` material arrays for the chunks the solve
        will read (coord → array).  Built on the main thread; referenced, not
        copied (like ``AssemblyJob``).  Cells with no loaded chunk data are
        treated as fully open.
    venturi_iters : int
        Jacobi flux-relaxation sweeps (``config.wind_venturi_iters``, ~8).
    venturi_max : float
        Clamp on the speed-up multiplier (``config.wind_venturi_max``, 2.2).
    deflect_gain : float
        Sideways-deflection gain (``config.wind_deflect_gain``, 0.15).
    seq : int
        Monotonic id; lets the consumer drop a superseded result.
    """

    origin_cell: tuple[int, int]
    cells: int
    cell_m: float
    chunk_size: int
    voxel_size: float
    ground_band: tuple[float, float]
    materials: dict[tuple[int, int, int], np.ndarray]
    venturi_iters: int
    venturi_max: float
    deflect_gain: float
    seq: int


@dataclass(frozen=True)
class VenturiResult:
    """
    A finished terrain-venturi correction grid.

    Attributes
    ----------
    origin_cell : tuple[int, int]
        The wind-region origin the solve was for (commit to the field only when
        it still matches the region's current origin).
    speedup : numpy.ndarray
        ``float32 (cells, cells)`` indexed ``[x, y]`` — multiplicative wind
        speed-up per cell, in ``[1.0, venturi_max]`` (1.0 = no funneling).
    deflect : numpy.ndarray
        ``float32 (cells, cells, 2)`` indexed ``[x, y, :]`` — additive sideways
        deflection (openness gradient × ``deflect_gain``), pushing flow around
        walls.  Scaled by the mean-wind magnitude when applied.
    seq : int
        Echoes the job's ``seq``.
    """

    origin_cell: tuple[int, int]
    speedup: np.ndarray
    deflect: np.ndarray
    seq: int


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


class VenturiWorker:
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
    """

    def __init__(self) -> None:
        self._in: queue.Queue[VenturiJob | None] = queue.Queue()
        self._out: queue.Queue[VenturiResult] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._pending = 0

    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the worker thread (idempotent)."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="WindVenturiWorker", daemon=True)
        self._thread.start()

    def submit(self, job: VenturiJob) -> None:
        """Enqueue a venturi-solve job (main thread)."""
        self._pending += 1
        self._in.put(job)

    def drain_results(self) -> list[VenturiResult]:
        """Pop and return all finished results (main thread, non-blocking)."""
        out: list[VenturiResult] = []
        while True:
            try:
                res = self._out.get_nowait()
            except queue.Empty:
                break
            self._pending -= 1
            out.append(res)
        return out

    def pending(self) -> int:
        """Jobs submitted but not yet drained."""
        return self._pending

    def stop(self, *, join: bool = True, timeout: float = 2.0) -> None:
        """Signal the worker to exit and (optionally) join it."""
        if self._thread is None:
            return
        self._in.put(None)  # sentinel
        if join:
            self._thread.join(timeout=timeout)
        self._thread = None

    # ------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            job = self._in.get()
            if job is None:  # sentinel → shutdown
                break
            try:
                self._out.put(solve_venturi(job))
            except Exception:
                _log.exception("Venturi solve failed (origin %r, seq %d)", job.origin_cell, job.seq)
                # Post a valid IDENTITY result so the consumer never starves —
                # a raised job must not leave the field stuck without a
                # correction grid forever (mirrors the assembly worker's
                # failure-sentinel discipline, but with a harmless identity).
                self._out.put(_identity_result(job))
