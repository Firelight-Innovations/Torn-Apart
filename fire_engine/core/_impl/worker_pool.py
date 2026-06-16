"""
core/_impl/worker_pool.py — Generic N-thread queue-based worker pool.

Provides :class:`WorkerPool`, the multi-thread sibling of
:class:`~fire_engine.core._impl.worker.QueueWorker`.  It shares the same
job/result queue lifecycle but fans a single shared input queue out across
``n_workers`` daemon threads, all running the same ``_run`` loop, and funnels
their outputs into a single shared result queue.  This is the variant used by
the terrain LOD system, where the per-job work (mesh build / decimation) is
independent and embarrassingly parallel — numpy releases the GIL during the
heavy array ops, so multiple workers genuinely overlap.

* :meth:`~WorkerPool.start` — spawn ``n_workers`` daemon threads (idempotent).
* :meth:`~WorkerPool.stop` — send one ``None`` sentinel per thread and join.
* ``_run`` loop — ``queue.get()`` → ``None`` → break; otherwise call
  :meth:`~WorkerPool._process` and ``put`` the result.
* :meth:`~WorkerPool.drain_results` — pop all finished results (non-blocking).
* :meth:`~WorkerPool.pending` — in-flight job count.

No panda3d import — fully headless-testable.  Uses :mod:`queue` and
:mod:`threading` with identical semantics to :class:`QueueWorker` (daemon
threads, no timeout on ``get``, ``get_nowait`` drain, 2-second default join
timeout).

Subclasses must implement :meth:`~WorkerPool._process` and may override
:meth:`~WorkerPool._on_error` to post a failure sentinel.

Docs: docs/systems/core._impl.md
"""

from __future__ import annotations

import queue
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Generic, TypeVar

__all__ = ["WorkerPool"]

_Job = TypeVar("_Job")
_Result = TypeVar("_Result")


class WorkerPool(ABC, Generic[_Job, _Result]):
    """
    Generic N-background-thread worker pool with queue-based job/result I/O.

    The multi-thread sibling of
    :class:`~fire_engine.core._impl.worker.QueueWorker`: one shared input queue
    feeds ``n_workers`` identical daemon threads, and one shared output queue
    collects their results.  Used by the terrain LOD system to parallelise
    independent mesh-build / decimation jobs.

    Type Parameters
    ---------------
    _Job
        The job type submitted via :meth:`submit`.
    _Result
        The result type returned via :meth:`drain_results`.

    Lifecycle
    ---------
    Call :meth:`start` once after construction; call :meth:`stop` at shutdown.
    The threads are daemons, so a missed :meth:`stop` never blocks process exit.

    Producer/consumer
    -----------------
    - :meth:`submit` — main thread enqueues a job (non-blocking).
    - :meth:`drain_results` — main thread pops all finished results (non-blocking).
    - :meth:`pending` — number of jobs submitted but not yet drained.

    Both queues cross the thread boundary lock-free; ``queue.Queue`` is itself
    thread-safe, so multiple workers may ``get``/``put`` concurrently.
    ``_pending`` is touched only on the main thread (in :meth:`submit` and
    :meth:`drain_results`) and therefore needs no lock.

    Subclasses
    ----------
    Implement :meth:`_process` to transform a single job into a result.
    **Because up to ``n_workers`` calls to :meth:`_process` may run
    concurrently, it must be pure** — operate only on its job's own immutable
    input snapshot and not touch shared mutable state without external
    synchronization.  Optionally override :meth:`_on_error` to post a failure
    sentinel when :meth:`_process` raises; the default does nothing.

    Parameters
    ----------
    thread_name : str
        Base ``name`` for the worker threads; each thread is named
        ``f"{thread_name}-{i}"`` for ``i`` in ``range(n_workers)``.
    n_workers : int
        Number of worker threads to spawn; clamped to at least ``1``.

    Example
    -------
    >>> class MyPool(WorkerPool[int, int]):
    ...     def _process(self, job: int) -> int:
    ...         return job * 2
    >>> pool = MyPool("MyPool", n_workers=4)
    >>> pool.start()
    >>> # pool.submit(job); results = pool.drain_results()
    >>> pool.stop()

    Docs: docs/systems/core._impl.md
    """

    def __init__(self, thread_name: str, n_workers: int) -> None:
        self._thread_name = thread_name
        self._n_workers = max(1, int(n_workers))
        self._in: queue.Queue[_Job | None] = queue.Queue()
        self._out: queue.Queue[_Result] = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._pending = 0

    # ------------------------------------------------------------------
    # Public lifecycle API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the ``n_workers`` worker threads (idempotent).

        Docs: docs/systems/core._impl.md
        """
        if self._threads:
            return
        for i in range(self._n_workers):
            thread = threading.Thread(
                target=self._run, name=f"{self._thread_name}-{i}", daemon=True
            )
            self._threads.append(thread)
            thread.start()

    def submit(self, job: _Job) -> None:
        """Enqueue a job (main thread, non-blocking).

        ``_pending`` is incremented here and only ever touched on the main
        thread (see :meth:`drain_results`), so it needs no lock.

        Docs: docs/systems/core._impl.md
        """
        self._pending += 1
        self._in.put(job)

    def drain_results(self) -> list[_Result]:
        """Pop and return all finished results (main thread, non-blocking).

        Docs: docs/systems/core._impl.md
        """
        out: list[_Result] = []
        for res in self._drain_iter():
            self._pending -= 1
            out.append(res)
        return out

    def _drain_iter(self) -> Iterator[_Result]:
        """Yield each currently-available result, stopping when the queue is empty."""
        while True:
            try:
                yield self._out.get_nowait()
            except queue.Empty:
                return

    def pending(self) -> int:
        """Jobs submitted but not yet drained.

        Docs: docs/systems/core._impl.md
        """
        return self._pending

    def stop(self, *, join: bool = True, timeout: float = 2.0) -> None:
        """Signal every worker to exit and (optionally) join them.

        Enqueues one ``None`` sentinel per worker thread so each loop exits,
        joins each thread with ``timeout``, then clears the thread list.
        Idempotent — a no-op if the pool was never started.

        Docs: docs/systems/core._impl.md
        """
        if not self._threads:
            return
        for _ in self._threads:
            self._in.put(None)  # one sentinel per thread
        if join:
            for thread in self._threads:
                thread.join(timeout=timeout)
        self._threads = []

    # ------------------------------------------------------------------
    # Extension points for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _process(self, job: _Job) -> _Result:
        """
        Transform *job* into a result.  Called on a worker thread.

        May run **concurrently** with other workers' :meth:`_process` calls,
        so it must be pure: read only the job's own immutable inputs and avoid
        unsynchronized writes to shared mutable state.

        Raised exceptions are caught by ``_run`` and forwarded to
        :meth:`_on_error`; the subclass decides whether to post a sentinel.
        """

    def _on_error(self, job: _Job) -> None:
        """
        Called on a worker thread after :meth:`_process` raises.

        Default: do nothing (the error has already been swallowed by ``_run``).
        Override to post a failure sentinel onto ``self._out`` so the consumer
        never starves on a job that raised.
        """

    # ------------------------------------------------------------------
    # Internal thread loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            job = self._in.get()
            if job is None:  # sentinel → shutdown this thread
                break
            try:
                self._out.put(self._process(job))
            except Exception:
                self._on_error(job)
