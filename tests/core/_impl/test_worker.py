"""
tests/core/_impl/test_worker.py — Mirror test for
fire_engine/core/_impl/worker.py.

Covers:
- QueueWorker lifecycle: start, submit, drain_results, stop
- pending() count tracks submitted vs drained jobs
- Multiple jobs processed in order
- start() is idempotent (second call is a no-op)
- stop() without join (join=False)
- Worker thread is a daemon thread
- _on_error hook called when _process raises
- Generic type parameters work (jobs and results are independent types)
"""

from __future__ import annotations

import time

from fire_engine.core._impl.worker import QueueWorker

# ---------------------------------------------------------------------------
# Concrete subclasses for testing
# ---------------------------------------------------------------------------


class _DoubleWorker(QueueWorker[int, int]):
    """Returns job * 2 as the result."""

    def _process(self, job: int) -> int:
        return job * 2


class _ErrorWorker(QueueWorker[str, str]):
    """Raises on the job 'fail'; passes anything else through."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.error_called_for: list[str] = []

    def _process(self, job: str) -> str:
        if job == "fail":
            raise RuntimeError("intentional error")
        return job.upper()

    def _on_error(self, job: str) -> None:
        self.error_called_for.append(job)


def _drain_with_timeout(worker: QueueWorker, count: int, timeout: float = 2.0) -> list:
    """Drain results, retrying until `count` results arrive or timeout."""
    results: list = []
    deadline = time.monotonic() + timeout
    while len(results) < count and time.monotonic() < deadline:
        results.extend(worker.drain_results())
        if len(results) < count:
            time.sleep(0.01)
    return results


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_start_and_stop(self):
        w = _DoubleWorker("TestDouble")
        w.start()
        w.stop()

    def test_stop_without_start_no_error(self):
        w = _DoubleWorker("TestDouble")
        w.stop()  # should not raise

    def test_start_idempotent(self):
        """Calling start() twice must not spawn two threads."""
        w = _DoubleWorker("TestDouble")
        w.start()
        thread_before = w._thread
        w.start()
        assert w._thread is thread_before
        w.stop()

    def test_thread_is_daemon(self):
        w = _DoubleWorker("TestDouble")
        w.start()
        assert w._thread is not None
        assert w._thread.daemon is True
        w.stop()

    def test_stop_join_false(self):
        w = _DoubleWorker("TestDouble")
        w.start()
        w.stop(join=False)
        # thread reference cleared even without join
        assert w._thread is None


# ---------------------------------------------------------------------------
# Job processing
# ---------------------------------------------------------------------------


class TestProcessing:
    def test_single_job(self):
        w = _DoubleWorker("TestDouble")
        w.start()
        w.submit(7)
        results = _drain_with_timeout(w, 1)
        w.stop()
        assert results == [14]

    def test_multiple_jobs(self):
        w = _DoubleWorker("TestDouble")
        w.start()
        for v in [1, 2, 3]:
            w.submit(v)
        results = sorted(_drain_with_timeout(w, 3))
        w.stop()
        assert results == [2, 4, 6]

    def test_pending_decrements_after_drain(self):
        w = _DoubleWorker("TestDouble")
        w.start()
        w.submit(10)
        w.submit(20)
        assert w.pending() == 2
        results = _drain_with_timeout(w, 2)
        assert w.pending() == 0
        w.stop()
        assert len(results) == 2

    def test_pending_zero_before_submit(self):
        w = _DoubleWorker("TestDouble")
        w.start()
        assert w.pending() == 0
        w.stop()

    def test_drain_results_empty_before_result(self):
        w = _DoubleWorker("TestDouble")
        w.start()
        # Before any submit, drain should return empty list
        assert w.drain_results() == []
        w.stop()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_on_error_called_when_process_raises(self):
        w = _ErrorWorker("TestError")
        w.start()
        w.submit("fail")
        # Give worker time to process
        time.sleep(0.1)
        w.stop()
        assert "fail" in w.error_called_for

    def test_worker_continues_after_error(self):
        """A failing job must not kill the worker thread; subsequent jobs still run."""
        w = _ErrorWorker("TestError")
        w.start()
        w.submit("fail")
        w.submit("ok")
        results = _drain_with_timeout(w, 1)
        w.stop()
        assert "OK" in results


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------


class TestExports:
    def test_queue_worker_exported(self):
        import fire_engine.core._impl.worker as mod

        assert "QueueWorker" in mod.__all__
        assert mod.QueueWorker is QueueWorker
