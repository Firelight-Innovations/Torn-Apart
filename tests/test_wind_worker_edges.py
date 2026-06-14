"""
tests/test_wind_worker_edges.py — Characterisation / golden-master tests for
VenturiWorker edge-cases and ordering behaviour (WP2).

Covers error-resilience, lifecycle idempotency, submit ordering, stop-with-
pending, submit-while-stopped, and on-thread equivalence for a job NOT already
exercised in test_wind_venturi.py.

Headless only — no window, no GPU.  All waits are bounded; no sleep-forever.

NOTE: This file PINS current behaviour.  Do NOT fix bugs found here — mark
suspicions in comments.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from fire_engine.core.config import Config
from fire_engine.world.wind import (
    VenturiJob,
    VenturiResult,
    VenturiWorker,
    solve_venturi,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared constants / helpers (mirror test_wind_venturi.py without duplicating)
# ─────────────────────────────────────────────────────────────────────────────
VOXEL = 0.5
CHUNK = 32
_DRAIN_TIMEOUT = 5.0  # seconds — short enough to fail fast in CI


def _cfg() -> Config:
    return Config()


def _empty_job(cfg: Config, *, seq: int = 1, cells: int = 8) -> VenturiJob:
    """Minimal open-terrain job (no obstacles) — solves fast."""
    ground = float(cfg.ground_height_m)
    return VenturiJob(
        origin_cell=(0, 0),
        cells=cells,
        cell_m=float(cfg.wind_cell_m),
        chunk_size=CHUNK,
        voxel_size=VOXEL,
        ground_band=(ground, ground + float(cfg.wind_layer_m)),
        materials={},
        venturi_iters=int(cfg.wind_venturi_iters),
        venturi_max=float(cfg.wind_venturi_max),
        deflect_gain=float(cfg.wind_deflect_gain),
        seq=seq,
    )


def _solid_job(cfg: Config, *, seq: int = 1, cells: int = 8) -> VenturiJob:
    """
    A small fully-solid terrain job (all voxels solid) — different from the
    wall-with-gap scenario in test_wind_venturi.py.  Produces speedup ≈ 1
    everywhere (no open gap to funnel through) but exercises a different code
    path through the occupancy fold.
    """
    ground = float(cfg.ground_height_m)
    cell_m = float(cfg.wind_cell_m)
    vpc = int(round(cell_m / VOXEL))
    region_v = cells * vpc
    vz_lo = int(np.floor(ground / VOXEL))
    vz_hi = int(np.ceil((ground + float(cfg.wind_layer_m)) / VOXEL))

    # Build one chunk covering the whole region at ground level.
    # Use a single large material slab: fill the relevant voxels.
    materials: dict = {}
    for gx in range(region_v):
        for gy in range(region_v):
            ccx, ccy = gx // CHUNK, gy // CHUNK
            for ccz in range(vz_lo // CHUNK, (vz_hi - 1) // CHUNK + 1):
                key = (ccx, ccy, ccz)
                if key not in materials:
                    materials[key] = np.zeros((CHUNK, CHUNK, CHUNK), dtype=np.uint8)
                az = max(ccz * CHUNK, vz_lo) - ccz * CHUNK
                bz = min(ccz * CHUNK + CHUNK, vz_hi) - ccz * CHUNK
                materials[key][
                    gx - ccx * CHUNK,
                    gy - ccy * CHUNK,
                    az:bz,
                ] = 1

    return VenturiJob(
        origin_cell=(0, 0),
        cells=cells,
        cell_m=cell_m,
        chunk_size=CHUNK,
        voxel_size=VOXEL,
        ground_band=(ground, ground + float(cfg.wind_layer_m)),
        materials=materials,
        venturi_iters=int(cfg.wind_venturi_iters),
        venturi_max=float(cfg.wind_venturi_max),
        deflect_gain=float(cfg.wind_deflect_gain),
        seq=seq,
    )


class _BoomArray(np.ndarray):
    """
    NumPy array subclass that raises RuntimeError on comparison (``> 0``).

    venturi.py does ``(sub > 0).sum(...)`` on material arrays.  This makes
    that call raise, exercising the worker's exception-recovery path without
    having to patch internal iteration.
    """

    def __gt__(self, other):
        raise RuntimeError("boom — intentional comparison fault")


def _boom_array() -> np.ndarray:
    """Return a (CHUNK, CHUNK, CHUNK) uint8 array that raises on ``> 0``."""
    base = np.ones((CHUNK, CHUNK, CHUNK), dtype=np.uint8)
    return base.view(_BoomArray)


def _raising_job(cfg: Config, *, seq: int = 99) -> VenturiJob:
    """
    Job whose material array raises during the venturi solid-fraction fold.

    venturi.py calls ``(sub > 0)`` on each chunk's material slice.  The
    _BoomArray overrides __gt__ to raise, so solve_venturi() raises and the
    worker must post an identity VenturiResult.
    """
    ground = float(cfg.ground_height_m)
    cell_m = float(cfg.wind_cell_m)
    # origin_cell=(1, 2) so the result echoes a non-trivial origin.
    return VenturiJob(
        origin_cell=(1, 2),
        cells=8,
        cell_m=cell_m,
        chunk_size=CHUNK,
        voxel_size=VOXEL,
        ground_band=(ground, ground + float(cfg.wind_layer_m)),
        # Provide a real chunk key so venturi's XY loop visits the array.
        # origin_cell=(1,2), cell_m=4, voxel=0.5 → vx0=8, vy0=16;
        # chunk (0,0,*) is NOT in the region footprint, so use ccx=0, ccy=0,
        # which maps to voxel 0..31 — the region voxels start at 8..72 for
        # cells=8, vpc=8.  ccx=0 intersects voxels 0..31 ∩ 8..72 → ax=8.
        # This is within the region so the array IS accessed.
        materials={(0, 0, int(np.floor(ground / (CHUNK * VOXEL)))): _boom_array()},
        venturi_iters=int(cfg.wind_venturi_iters),
        venturi_max=float(cfg.wind_venturi_max),
        deflect_gain=float(cfg.wind_deflect_gain),
        seq=seq,
    )


def _drain_until(
    worker: VenturiWorker, want: int, timeout_s: float = _DRAIN_TIMEOUT
) -> list[VenturiResult]:
    """Poll drain_results() until at least *want* results arrive or timeout."""
    out: list[VenturiResult] = []
    deadline = time.monotonic() + timeout_s
    while len(out) < want and time.monotonic() < deadline:
        out += worker.drain_results()
        if len(out) < want:
            time.sleep(0.001)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle idempotency
# ─────────────────────────────────────────────────────────────────────────────


class TestLifecycleIdempotency:
    def test_start_twice_no_crash(self):
        """start() called twice must not raise and must not spawn two threads."""
        worker = VenturiWorker()
        try:
            worker.start()
            t1 = worker._thread
            worker.start()  # second call must be a no-op
            t2 = worker._thread
            assert t1 is t2, "second start() spawned a new thread"
        finally:
            worker.stop()

    def test_stop_twice_no_crash(self):
        """stop() called twice must not raise."""
        worker = VenturiWorker()
        worker.start()
        worker.stop()
        worker.stop()  # idempotent — must not raise

    def test_is_running_transitions(self):
        """
        Pin the is_running / _thread state machine.

        Before start:  _thread is None.
        After start:   _thread is not None and is alive.
        After stop:    _thread is None (stop() sets it to None).
        """
        worker = VenturiWorker()
        assert worker._thread is None, "newly constructed worker should have no thread"

        worker.start()
        t = worker._thread
        assert t is not None
        assert t.is_alive(), "thread not alive after start()"

        worker.stop(join=True, timeout=2.0)
        assert worker._thread is None, "stop() should set _thread to None"

    def test_stop_before_start_no_crash(self):
        """stop() on a never-started worker must not raise."""
        worker = VenturiWorker()
        worker.stop()  # must be benign

    def test_stop_joins_within_timeout(self):
        """Thread is no longer alive within stop()'s join timeout."""
        worker = VenturiWorker()
        worker.start()
        t = worker._thread
        worker.stop(join=True, timeout=2.0)
        assert not t.is_alive(), "thread still alive after stop(join=True)"


# ─────────────────────────────────────────────────────────────────────────────
# Error resilience
# ─────────────────────────────────────────────────────────────────────────────


class TestErrorResilience:
    def test_raising_job_posts_identity_result(self):
        """A solve that raises must post a VenturiResult, not swallow it."""
        cfg = _cfg()
        worker = VenturiWorker()
        worker.start()
        try:
            bad = _raising_job(cfg, seq=99)
            worker.submit(bad)
            results = _drain_until(worker, 1)
            assert len(results) == 1, "worker silently dropped the errored job (no result posted)"
            r = results[0]
            assert isinstance(r, VenturiResult)
        finally:
            worker.stop()

    def test_raising_job_identity_arrays(self):
        """The identity result must have speedup==1 everywhere, deflect==0."""
        cfg = _cfg()
        worker = VenturiWorker()
        worker.start()
        try:
            bad = _raising_job(cfg, seq=42)
            worker.submit(bad)
            results = _drain_until(worker, 1)
            r = results[0]
            assert np.array_equal(r.speedup, np.ones((8, 8), dtype=np.float32)), (
                f"identity speedup not ones: {r.speedup}"
            )
            assert np.array_equal(r.deflect, np.zeros((8, 8, 2), dtype=np.float32)), (
                f"identity deflect not zeros: {r.deflect}"
            )
        finally:
            worker.stop()

    def test_raising_job_echoes_seq(self):
        """The identity result's seq must equal the failing job's seq."""
        cfg = _cfg()
        worker = VenturiWorker()
        worker.start()
        try:
            bad = _raising_job(cfg, seq=77)
            worker.submit(bad)
            results = _drain_until(worker, 1)
            assert results[0].seq == 77, f"seq not echoed on error: got {results[0].seq}"
        finally:
            worker.stop()

    def test_raising_job_echoes_origin_cell(self):
        """The identity result's origin_cell must match the failing job's."""
        cfg = _cfg()
        worker = VenturiWorker()
        worker.start()
        try:
            bad = _raising_job(cfg, seq=55)
            assert bad.origin_cell == (1, 2)
            worker.submit(bad)
            results = _drain_until(worker, 1)
            assert results[0].origin_cell == (1, 2), (
                f"origin_cell not echoed on error: {results[0].origin_cell}"
            )
        finally:
            worker.stop()

    def test_worker_survives_error_and_processes_next_job(self):
        """
        After a solve raises, the thread must still be alive and process a
        subsequent valid job correctly.

        SUSPICION: If _pending is decremented incorrectly after an error, the
        counter could go negative.  Pin both the counter value and the result.
        """
        cfg = _cfg()
        worker = VenturiWorker()
        worker.start()
        try:
            bad = _raising_job(cfg, seq=1)
            worker.submit(bad)
            _drain_until(worker, 1)  # consume the identity result

            # Verify thread survived.
            assert worker._thread is not None and worker._thread.is_alive(), (
                "thread died after a solve exception"
            )

            # Submit a valid job and verify it produces a real (non-identity) result.
            good = _empty_job(cfg, seq=2, cells=8)
            inline = solve_venturi(good)
            worker.submit(good)
            more = _drain_until(worker, 1)
            assert len(more) == 1
            assert more[0].seq == 2
            assert np.array_equal(more[0].speedup, inline.speedup)
        finally:
            worker.stop()

    def test_pending_counter_correct_after_error(self):
        """
        pending() must reach 0 after draining both the error result and a
        follow-up valid result.

        SUSPICION: The _pending counter is incremented in submit() and
        decremented in drain_results().  If the worker posted TWO results for
        one failing job (e.g., both the exception path AND a normal path ran),
        _pending could become negative.  This test pins the count.
        """
        cfg = _cfg()
        worker = VenturiWorker()
        worker.start()
        try:
            worker.submit(_raising_job(cfg, seq=10))
            worker.submit(_empty_job(cfg, seq=11, cells=8))
            results = _drain_until(worker, 2)
            assert len(results) == 2
            assert worker.pending() == 0, (
                f"pending not 0 after draining 2 results: {worker.pending()}"
            )
        finally:
            worker.stop()


# ─────────────────────────────────────────────────────────────────────────────
# submit / drain ordering
# ─────────────────────────────────────────────────────────────────────────────


class TestOrdering:
    def test_all_submitted_jobs_eventually_drain(self):
        """
        Every submitted valid job produces exactly one result, no matter the
        order in which they complete.

        Pins the INVARIANT: set of drained seqs == set of submitted seqs.
        (The worker is single-threaded so FIFO is expected, but we assert the
        set to avoid a brittle ordering dependency.)
        """
        cfg = _cfg()
        worker = VenturiWorker()
        worker.start()
        try:
            seqs = list(range(5, 10))
            for s in seqs:
                worker.submit(_empty_job(cfg, seq=s, cells=8))
            results = _drain_until(worker, len(seqs))
            assert len(results) == len(seqs), f"only {len(results)} of {len(seqs)} results drained"
            drained_seqs = {r.seq for r in results}
            assert drained_seqs == set(seqs), f"seq set mismatch: {drained_seqs} vs {set(seqs)}"
        finally:
            worker.stop()

    def test_single_thread_fifo_order(self):
        """
        Because VenturiWorker uses a single daemon thread, jobs must complete
        in submission order (FIFO queue).

        SUSPICION: If this ever fails it means either a second thread was
        somehow introduced or the queue is not strictly FIFO.  Pin current
        behaviour (FIFO expected).
        """
        cfg = _cfg()
        worker = VenturiWorker()
        worker.start()
        try:
            n = 6
            for i in range(n):
                worker.submit(_empty_job(cfg, seq=i, cells=8))
            results = _drain_until(worker, n)
            assert len(results) == n
            drained_seqs = [r.seq for r in results]
            assert drained_seqs == list(range(n)), (
                f"results not in submission order: {drained_seqs}"
            )
        finally:
            worker.stop()

    def test_mixed_error_and_valid_ordering_preserved(self):
        """
        A bad job followed by a valid job — both drain in submission order
        (error-identity first, then real result), and the valid job's data is
        correct.
        """
        cfg = _cfg()
        worker = VenturiWorker()
        worker.start()
        try:
            worker.submit(_raising_job(cfg, seq=20))
            good = _empty_job(cfg, seq=21, cells=8)
            inline = solve_venturi(good)
            worker.submit(good)
            results = _drain_until(worker, 2)
            assert len(results) == 2
            assert results[0].seq == 20, f"error result not first: seq={results[0].seq}"
            assert results[1].seq == 21, f"valid result not second: seq={results[1].seq}"
            assert np.array_equal(results[1].speedup, inline.speedup)
        finally:
            worker.stop()


# ─────────────────────────────────────────────────────────────────────────────
# On-thread equivalence (different job from test_wind_venturi.py)
# ─────────────────────────────────────────────────────────────────────────────


class TestOnThreadEquivalence:
    def test_fully_solid_job_matches_inline(self):
        """
        A fully-solid terrain job solved by the worker must be byte-identical
        to solve_venturi() called on the same job inline.

        This job is DIFFERENT from the wall-with-gap used in
        test_wind_venturi.py::TestVenturiWorker::test_worker_matches_on_thread.
        """
        cfg = _cfg()
        job = _solid_job(cfg, seq=33, cells=8)
        inline = solve_venturi(job)

        worker = VenturiWorker()
        worker.start()
        try:
            worker.submit(job)
            results = _drain_until(worker, 1)
            assert len(results) == 1
            r = results[0]
            assert r.seq == 33
            assert np.array_equal(r.speedup, inline.speedup), (
                "speedup differs between worker and inline solve"
            )
            assert np.array_equal(r.deflect, inline.deflect), (
                "deflect differs between worker and inline solve"
            )
        finally:
            worker.stop()

    def test_empty_job_matches_inline(self):
        """
        An open-terrain (no obstacles) job solved off-thread must equal the
        inline result byte-for-byte.
        """
        cfg = _cfg()
        job = _empty_job(cfg, seq=44, cells=12)
        inline = solve_venturi(job)

        worker = VenturiWorker()
        worker.start()
        try:
            worker.submit(job)
            results = _drain_until(worker, 1)
            assert len(results) == 1
            r = results[0]
            assert np.array_equal(r.speedup, inline.speedup)
            assert np.array_equal(r.deflect, inline.deflect)
        finally:
            worker.stop()


# ─────────────────────────────────────────────────────────────────────────────
# stop() with a pending queue
# ─────────────────────────────────────────────────────────────────────────────


class TestStopWithPendingQueue:
    def test_stop_does_not_hang_with_pending_jobs(self):
        """
        stop(join=True, timeout=2.0) must return within the timeout even when
        there are jobs still enqueued.

        SUSPICION: If the worker loop does not drain the in-queue before
        exiting (it just breaks on sentinel), any in-flight jobs that arrive
        BEFORE the sentinel will be processed, but jobs enqueued AFTER the
        sentinel will silently be dropped.  This test pins that stop() at
        minimum does not hang — not that it drains all pending.
        """
        cfg = _cfg()
        worker = VenturiWorker()
        worker.start()
        # Flood the queue with cheap jobs, then immediately stop.
        for i in range(20):
            worker.submit(_empty_job(cfg, seq=i, cells=4))
        t0 = time.monotonic()
        worker.stop(join=True, timeout=3.0)
        elapsed = time.monotonic() - t0
        assert elapsed < 3.5, f"stop() hung for {elapsed:.1f}s with pending jobs in queue"

    def test_stop_without_join_does_not_raise(self):
        """stop(join=False) is a fire-and-forget — must not raise."""
        cfg = _cfg()
        worker = VenturiWorker()
        worker.start()
        worker.submit(_empty_job(cfg, seq=1, cells=4))
        worker.stop(join=False)  # should not raise

    def test_thread_is_set_to_none_after_stop_join(self):
        """
        After stop(join=True), _thread is None regardless of whether there
        were pending jobs.
        """
        cfg = _cfg()
        worker = VenturiWorker()
        worker.start()
        for i in range(5):
            worker.submit(_empty_job(cfg, seq=i, cells=4))
        worker.stop(join=True, timeout=2.0)
        assert worker._thread is None


# ─────────────────────────────────────────────────────────────────────────────
# submit() when not started / after stop
# ─────────────────────────────────────────────────────────────────────────────


class TestSubmitEdgeCases:
    def test_submit_before_start_does_not_raise(self):
        """
        submit() before start() must not raise (the job is enqueued in the
        in-queue; no thread is running to process it, but the call itself is
        safe).

        SUSPICION: submit() blindly increments _pending and calls _in.put().
        A caller who submits before start() and then calls start() later should
        still see their result drain.  This test pins the no-raise behaviour.
        """
        cfg = _cfg()
        worker = VenturiWorker()
        # Do NOT call start().
        worker.submit(_empty_job(cfg, seq=1, cells=4))  # must not raise
        # Clean up: start so the sentinel can be delivered properly.
        worker.start()
        worker.stop()

    def test_submit_before_start_result_drains_after_start(self):
        """
        A job submitted before start() must eventually drain after start() is
        called — the job sits in the queue and the thread picks it up on boot.
        """
        cfg = _cfg()
        worker = VenturiWorker()
        job = _empty_job(cfg, seq=5, cells=4)
        inline = solve_venturi(job)
        worker.submit(job)
        worker.start()
        try:
            results = _drain_until(worker, 1)
            assert len(results) == 1
            assert results[0].seq == 5
            assert np.array_equal(results[0].speedup, inline.speedup)
        finally:
            worker.stop()

    def test_submit_after_stop_does_not_raise(self):
        """
        submit() after stop() must not raise.

        SUSPICION: _pending is an unsynchronised int.  Calling submit() on a
        stopped worker increments _pending but no thread will ever decrement it
        (no result will drain).  This test pins the no-raise behaviour and
        documents that pending() will be inflated after this sequence.
        """
        cfg = _cfg()
        worker = VenturiWorker()
        worker.start()
        worker.stop()
        # _thread is now None; submitting puts to _in but no thread reads it.
        worker.submit(_empty_job(cfg, seq=9, cells=4))  # must not raise

    def test_drain_results_when_not_started_returns_empty(self):
        """
        drain_results() on a never-started worker must return [] (the out-queue
        is empty).
        """
        worker = VenturiWorker()
        result = worker.drain_results()
        assert result == [], f"expected empty list, got {result}"
