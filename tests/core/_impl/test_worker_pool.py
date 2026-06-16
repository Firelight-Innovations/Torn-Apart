"""
tests/core/_impl/test_worker_pool.py — Mirror test for
fire_engine/core/_impl/worker_pool.py.

Covers:
- WorkerPool processes every submitted job exactly once across N>1 threads
  (multiset of results == expected, no ordering assumption)
- start() is idempotent (second call spawns no extra threads)
- stop() joins all N threads, is idempotent, and is a no-op before start()
- n_workers is clamped to at least 1
- All N worker threads are daemons
- _on_error invoked on failure; an error-sentinel subclass never wedges the
  pool (a raising job still lets later jobs complete)
- pending() accounting across submit/drain
- Result completeness for a fixed input set is independent of interleaving
"""

from __future__ import annotations

import time

from fire_engine.core._impl.worker_pool import WorkerPool

# ---------------------------------------------------------------------------
# Concrete subclasses for testing
# ---------------------------------------------------------------------------


class _DoublePool(WorkerPool[int, int]):
    """Returns job * 2 as the result."""

    def _process(self, job: int) -> int:
        return job * 2


class _ErrorPool(WorkerPool[str, str]):
    """Raises on the job 'fail'; uppercases anything else.

    On error, posts a sentinel result so the consumer never starves — this
    also exercises the 'a raising job must not wedge the pool' guarantee.
    """

    def __init__(self, name: str, n_workers: int) -> None:
        super().__init__(name, n_workers)
        self.error_called_for: list[str] = []

    def _process(self, job: str) -> str:
        if job == "fail":
            raise RuntimeError("intentional error")
        return job.upper()

    def _on_error(self, job: str) -> None:
        self.error_called_for.append(job)
        self._out.put("ERR")


def _drain_with_timeout(pool: WorkerPool, count: int, timeout: float = 2.0) -> list:
    """Drain results, retrying until `count` results arrive or timeout.

    Bounded wait: short sleeps capped by a wall-clock deadline (never an
    unbounded block), so a hung worker fails the test instead of hanging it.
    """
    results: list = []
    deadline = time.monotonic() + timeout
    while len(results) < count and time.monotonic() < deadline:
        results.extend(pool.drain_results())
        if len(results) < count:
            time.sleep(0.001)
    return results


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_start_and_stop(self):
        pool = _DoublePool("TestDouble", n_workers=4)
        pool.start()
        pool.stop()

    def test_stop_without_start_no_error(self):
        pool = _DoublePool("TestDouble", n_workers=4)
        pool.stop()  # should not raise / no-op

    def test_start_spawns_n_threads(self):
        pool = _DoublePool("TestDouble", n_workers=3)
        pool.start()
        assert len(pool._threads) == 3
        pool.stop()

    def test_start_idempotent(self):
        """Calling start() twice must not spawn extra threads."""
        pool = _DoublePool("TestDouble", n_workers=3)
        pool.start()
        threads_before = list(pool._threads)
        pool.start()
        assert pool._threads == threads_before
        assert len(pool._threads) == 3
        pool.stop()

    def test_all_threads_are_daemons(self):
        pool = _DoublePool("TestDouble", n_workers=4)
        pool.start()
        assert all(t.daemon for t in pool._threads)
        pool.stop()

    def test_stop_joins_all_threads_and_is_idempotent(self):
        pool = _DoublePool("TestDouble", n_workers=4)
        pool.start()
        threads = list(pool._threads)
        pool.stop()
        # All threads have actually exited (joined within the timeout).
        assert all(not t.is_alive() for t in threads)
        assert pool._threads == []
        # Second stop() is a harmless no-op.
        pool.stop()
        assert pool._threads == []

    def test_n_workers_clamped_to_minimum_one(self):
        pool = _DoublePool("TestDouble", n_workers=0)
        assert pool._n_workers == 1
        pool.start()
        assert len(pool._threads) == 1
        pool.stop()

        neg = _DoublePool("TestDouble", n_workers=-5)
        assert neg._n_workers == 1


# ---------------------------------------------------------------------------
# Job processing
# ---------------------------------------------------------------------------


class TestProcessing:
    def test_every_job_processed_exactly_once(self):
        """Submit many jobs across N>1 threads; multiset of results matches."""
        pool = _DoublePool("TestDouble", n_workers=4)
        pool.start()
        jobs = list(range(50))
        for v in jobs:
            pool.submit(v)
        results = _drain_with_timeout(pool, len(jobs))
        pool.stop()
        expected = sorted(v * 2 for v in jobs)
        assert sorted(results) == expected
        assert pool.pending() == 0

    def test_results_complete_regardless_of_interleaving(self):
        """A fixed input set yields the complete result multiset every time."""
        for _ in range(5):
            pool = _DoublePool("TestDouble", n_workers=8)
            pool.start()
            jobs = list(range(20))
            for v in jobs:
                pool.submit(v)
            results = _drain_with_timeout(pool, len(jobs))
            pool.stop()
            assert sorted(results) == sorted(v * 2 for v in jobs)

    def test_pending_decrements_after_drain(self):
        pool = _DoublePool("TestDouble", n_workers=2)
        pool.start()
        pool.submit(10)
        pool.submit(20)
        assert pool.pending() == 2
        results = _drain_with_timeout(pool, 2)
        assert pool.pending() == 0
        pool.stop()
        assert len(results) == 2

    def test_pending_zero_before_submit(self):
        pool = _DoublePool("TestDouble", n_workers=2)
        pool.start()
        assert pool.pending() == 0
        pool.stop()

    def test_drain_results_empty_before_result(self):
        pool = _DoublePool("TestDouble", n_workers=2)
        pool.start()
        assert pool.drain_results() == []
        pool.stop()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_on_error_called_when_process_raises(self):
        pool = _ErrorPool("TestError", n_workers=2)
        pool.start()
        pool.submit("fail")
        # _on_error posts an 'ERR' sentinel; wait for it deterministically.
        results = _drain_with_timeout(pool, 1)
        pool.stop()
        assert "ERR" in results
        assert "fail" in pool.error_called_for

    def test_raising_job_does_not_wedge_the_pool(self):
        """A failing job must not kill its thread; later jobs still complete."""
        pool = _ErrorPool("TestError", n_workers=3)
        pool.start()
        pool.submit("fail")
        pool.submit("ok")
        pool.submit("good")
        # 3 jobs -> 3 results ('ERR' sentinel + two uppercased strings).
        results = _drain_with_timeout(pool, 3)
        pool.stop()
        assert sorted(results) == ["ERR", "GOOD", "OK"]

    def test_many_failures_across_threads(self):
        """Interleaved failing/passing jobs across N threads all account for."""
        pool = _ErrorPool("TestError", n_workers=4)
        pool.start()
        jobs = ["fail" if i % 2 == 0 else f"j{i}" for i in range(20)]
        for j in jobs:
            pool.submit(j)
        results = _drain_with_timeout(pool, len(jobs))
        pool.stop()
        assert len(results) == len(jobs)
        assert results.count("ERR") == sum(1 for j in jobs if j == "fail")


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------


class TestExports:
    def test_worker_pool_exported(self):
        import fire_engine.core._impl.worker_pool as mod

        assert "WorkerPool" in mod.__all__
        assert mod.WorkerPool is WorkerPool
