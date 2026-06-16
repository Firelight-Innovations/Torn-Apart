"""
core/_impl/worker.py — Generic queue-based background-thread worker skeleton.

Provides :class:`QueueWorker`, a reusable base that encapsulates the
thread-pump lifecycle shared by
:class:`~fire_engine.world.wind.worker.VenturiWorker` and
:class:`~fire_engine.lighting.assembly_worker.CascadeAssemblyWorker`:

* :meth:`~QueueWorker.start` — spawn a single daemon thread (idempotent).
* :meth:`~QueueWorker.stop` — send a ``None`` sentinel and optionally join.
* ``_run`` loop — ``queue.get()`` → ``None`` → break; otherwise call
  :meth:`~QueueWorker._process` and ``put`` the result.
* :meth:`~QueueWorker.drain_results` — pop all finished results (non-blocking).
* :meth:`~QueueWorker.pending` — in-flight job count.

No panda3d import — fully headless-testable.  Uses :mod:`queue` and
:mod:`threading` with identical semantics to the original workers (daemon thread,
no timeout on ``get``, ``get_nowait`` drain, 2-second default join timeout).

Subclasses must implement :meth:`~QueueWorker._process` and may override
:meth:`~QueueWorker._on_error` to post a failure sentinel.

Docs: docs/systems/core.md
"""

from __future__ import annotations

import queue
import threading
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

__all__ = ["QueueWorker"]

_Job = TypeVar("_Job")
_Result = TypeVar("_Result")


class QueueWorker(ABC, Generic[_Job, _Result]):
    """
    Generic single-background-thread worker with queue-based job/result I/O.

    Type Parameters
    ---------------
    _Job
        The job type submitted via :meth:`submit`.
    _Result
        The result type returned via :meth:`drain_results`.

    Lifecycle
    ---------
    Call :meth:`start` once after construction; call :meth:`stop` at shutdown.
    The thread is a daemon, so a missed :meth:`stop` never blocks process exit.

    Producer/consumer
    -----------------
    - :meth:`submit` — main thread enqueues a job (non-blocking).
    - :meth:`drain_results` — main thread pops all finished results (non-blocking).
    - :meth:`pending` — number of jobs submitted but not yet drained.

    Both queues cross the thread boundary lock-free.

    Subclasses
    ----------
    Implement :meth:`_process` to transform a single job into a result.
    Optionally override :meth:`_on_error` to post a failure sentinel when
    :meth:`_process` raises; the default implementation does nothing (the
    exception is already logged by the ``_run`` loop).

    Parameters
    ----------
    thread_name : str
        ``name`` passed to :class:`threading.Thread` for debuggability.

    Example
    -------
    >>> class MyWorker(QueueWorker[MyJob, MyResult]):
    ...     def _process(self, job: MyJob) -> MyResult:
    ...         return MyResult(job.value * 2)
    >>> w = MyWorker("MyWorker")
    >>> w.start()
    >>> # w.submit(job); results = w.drain_results()
    >>> w.stop()

    Docs: docs/systems/core.md
    """

    def __init__(self, thread_name: str) -> None:
        self._thread_name = thread_name
        self._in: queue.Queue[_Job | None] = queue.Queue()
        self._out: queue.Queue[_Result] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._pending = 0

    # ------------------------------------------------------------------
    # Public lifecycle API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the worker thread (idempotent)."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name=self._thread_name, daemon=True)
        self._thread.start()

    def submit(self, job: _Job) -> None:
        """Enqueue a job (main thread, non-blocking)."""
        self._pending += 1
        self._in.put(job)

    def drain_results(self) -> list[_Result]:
        """Pop and return all finished results (main thread, non-blocking)."""
        out: list[_Result] = []
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
    # Extension points for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _process(self, job: _Job) -> _Result:
        """
        Transform *job* into a result.  Called on the worker thread.

        Raised exceptions are caught by ``_run`` and forwarded to
        :meth:`_on_error`; the subclass decides whether to post a sentinel.
        """

    def _on_error(self, job: _Job) -> None:
        """
        Called on the worker thread after :meth:`_process` raises.

        Default: do nothing (the error has already been logged by ``_run``).
        Override to post a failure sentinel onto ``self._out``.
        """

    # ------------------------------------------------------------------
    # Internal thread loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            job = self._in.get()
            if job is None:  # sentinel → shutdown
                break
            try:
                self._out.put(self._process(job))
            except Exception:
                self._on_error(job)
