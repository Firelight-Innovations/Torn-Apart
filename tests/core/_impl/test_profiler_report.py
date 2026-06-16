"""
tests/core/_impl/test_profiler_report.py — Mirror test for
fire_engine/core/_impl/profiler_report.py.

Covers:
- frame_time_stats: empty array, single element, known statistics
- build_snapshot: schema_version key, scopes/counters structure, JSON round-trip
- commit_frame: ring-buffer advances write index
- Determinism: same frame data → same stats
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from fire_engine.core._impl.profiler_report import frame_time_stats
from fire_engine.core.profiler import Profiler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_profiler(enabled: bool = True) -> Profiler:
    return Profiler(
        enabled=enabled,
        history_frames=64,
        frame_budget_ms=5.0,
    )


class FakeClock:
    def __init__(self) -> None:
        self.t = 0

    def __call__(self) -> int:
        return self.t

    def advance_ms(self, ms: float) -> None:
        self.t += round(ms * 1_000_000)


def _run_frame(prof: Profiler, clk: FakeClock, ms: float) -> None:
    """Simulate one frame.  Commit happens at the start of the NEXT begin_frame."""
    begin_t = clk.t
    prof.begin_frame()
    prof.end_frame()
    # Advance clock so next begin_frame commits this frame with duration ms.
    clk.t = begin_t + round(ms * 1_000_000)


# ---------------------------------------------------------------------------
# frame_time_stats
# ---------------------------------------------------------------------------


class TestFrameTimeStats:
    def test_empty_array_returns_zeros(self):
        s = frame_time_stats(np.array([], dtype=np.float64), budget_ms=5.0)
        for key in ("mean", "median", "min", "max", "p99", "p999", "fps_mean", "over_budget_pct"):
            assert s[key] == 0.0, f"key {key!r} should be 0.0 for empty input"

    def test_single_element(self):
        s = frame_time_stats(np.array([10.0]), budget_ms=5.0)
        assert s["mean"] == pytest.approx(10.0)
        assert s["min"] == pytest.approx(10.0)
        assert s["max"] == pytest.approx(10.0)
        assert s["fps_mean"] == pytest.approx(1000.0 / 10.0)
        assert s["over_budget_pct"] == pytest.approx(100.0)

    def test_known_statistics(self):
        frames = np.array([4.0, 5.0, 6.0, 40.0], dtype=np.float64)
        s = frame_time_stats(frames, budget_ms=5.0)
        assert s["max"] == pytest.approx(40.0)
        assert s["min"] == pytest.approx(4.0)
        # 2 out of 4 frames exceed 5ms budget
        assert s["over_budget_pct"] == pytest.approx(50.0)
        assert s["mean"] == pytest.approx(frames.mean())

    def test_fps_mean_formula(self):
        """fps_mean = 1000 / mean_ms."""
        frames = np.array([8.0, 12.0], dtype=np.float64)
        s = frame_time_stats(frames, budget_ms=5.0)
        assert s["fps_mean"] == pytest.approx(1000.0 / 10.0)

    def test_all_within_budget(self):
        frames = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        s = frame_time_stats(frames, budget_ms=5.0)
        assert s["over_budget_pct"] == 0.0

    def test_all_over_budget(self):
        frames = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        s = frame_time_stats(frames, budget_ms=5.0)
        assert s["over_budget_pct"] == pytest.approx(100.0)

    def test_determinism(self):
        """Same input produces identical output."""
        frames = np.array([3.0, 7.0, 2.0, 15.0], dtype=np.float64)
        a = frame_time_stats(frames.copy(), budget_ms=5.0)
        b = frame_time_stats(frames.copy(), budget_ms=5.0)
        assert a == b

    def test_result_is_json_serializable(self):
        """All values in the stats dict must be plain floats (JSON-safe)."""
        s = frame_time_stats(np.array([5.0, 10.0]), budget_ms=5.0)
        json.dumps(s)  # must not raise


# ---------------------------------------------------------------------------
# build_snapshot via Profiler.snapshot()
# ---------------------------------------------------------------------------


class TestBuildSnapshot:
    def test_snapshot_has_schema_version(self):
        from fire_engine.core.profiler import SCHEMA_VERSION

        prof = _make_profiler()
        clk = FakeClock()
        prof._time = clk
        snap = prof.snapshot()
        assert snap["schema_version"] == SCHEMA_VERSION

    def test_snapshot_json_round_trip(self):
        """build_snapshot output must survive json.dumps → json.loads."""
        prof = _make_profiler()
        clk = FakeClock()
        prof._time = clk
        # Run 2 frames; the first is committed when the second's begin_frame fires.
        _run_frame(prof, clk, 4.0)
        _run_frame(prof, clk, 4.0)
        snap = prof.snapshot()
        serialized = json.dumps(snap)
        loaded = json.loads(serialized)
        assert loaded["frames_measured"] == snap["frames_measured"]

    def test_snapshot_frames_measured_increments(self):
        """frames_measured counts committed frames (commit on next begin_frame)."""
        prof = _make_profiler()
        clk = FakeClock()
        prof._time = clk
        # Run N+1 frames so N are committed.
        for _ in range(6):
            _run_frame(prof, clk, 3.0)
        snap = prof.snapshot()
        # At least 5 frames committed (the last one is still pending).
        assert snap["frames_measured"] >= 5

    def test_snapshot_has_frame_ms_keys(self):
        prof = _make_profiler()
        clk = FakeClock()
        prof._time = clk
        _run_frame(prof, clk, 5.0)
        snap = prof.snapshot()
        for key in ("mean", "median", "min", "max", "p99", "p999", "fps_mean"):
            assert key in snap["frame_ms"], f"missing key {key!r} in frame_ms"

    def test_snapshot_scopes_list(self):
        """After named scopes, snapshot['scopes'] is a list of scope dicts."""
        prof = _make_profiler()
        clk = FakeClock()
        prof._time = clk
        # Frame 1: record a scope named "Physics"
        begin_t = clk.t
        prof.begin_frame()
        with prof.scope("Physics"):
            clk.advance_ms(2.0)
        prof.end_frame()
        clk.t = begin_t + round(5.0 * 1_000_000)
        # Frame 2: begin_frame commits frame 1 into the ring buffer
        prof.begin_frame()
        prof.end_frame()
        snap = prof.snapshot()
        assert isinstance(snap["scopes"], list)
        names = [s["name"] for s in snap["scopes"]]
        assert "Physics" in names

    def test_disabled_profiler_snapshot_empty_frames(self):
        """A disabled Profiler's snapshot has frames_measured == 0."""
        prof = _make_profiler(enabled=False)
        clk = FakeClock()
        prof._time = clk
        snap = prof.snapshot()
        assert snap["frames_measured"] == 0


# ---------------------------------------------------------------------------
# commit_frame integration (via normal profiler lifecycle)
# ---------------------------------------------------------------------------


class TestCommitFrame:
    def test_write_index_advances(self):
        """Each committed frame advances the ring-buffer write index.

        commit_frame is called at the START of the next begin_frame, so after
        running N frames the write_index equals N-1 (the last frame is pending).
        A second _run_frame causes the first to be committed.
        """
        prof = _make_profiler()
        clk = FakeClock()
        prof._time = clk
        assert prof._write_index == 0
        _run_frame(prof, clk, 1.0)
        # First frame is pending — no commit yet (write_index still 0).
        assert prof._write_index == 0
        # Second begin_frame commits the first frame.
        _run_frame(prof, clk, 1.0)
        assert prof._write_index == 1

    def test_frame_ms_recorded(self):
        """Frame duration is stored in the ring buffer after commit."""
        prof = _make_profiler()
        clk = FakeClock()
        prof._time = clk
        # Run frame 1 (8 ms), then frame 2 to commit frame 1.
        _run_frame(prof, clk, 8.0)
        _run_frame(prof, clk, 1.0)
        # Frame 1 is now at index 0; write_index advanced to 1.
        assert prof._frame_ms[0] == pytest.approx(8.0, abs=1.0)
