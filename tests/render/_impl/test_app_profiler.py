"""
tests/render/_impl/test_app_profiler.py — Headless tests for render/_impl/app_profiler.py.

Tests the pure time-throttle logic of maybe_write_snapshot using a
SimpleNamespace stand-in for the App instance.  No panda3d required.
"""

from __future__ import annotations

import time
import types

from fire_engine.render._impl.app_profiler import maybe_write_snapshot


def _make_fake_app(
    *,
    snapshot_path: str | None = "profiling/latest.json",
    last_snapshot_t: float | None = None,
    snapshot_interval_s: float = 10.0,
) -> types.SimpleNamespace:
    """Build a minimal fake App namespace for maybe_write_snapshot."""

    class _FakeProfiler:
        def __init__(self) -> None:
            self.snapshot_calls: list[str] = []

        def write_snapshot(self, path: str) -> None:
            self.snapshot_calls.append(path)

    fake = types.SimpleNamespace()
    fake._snapshot_path = snapshot_path
    fake._last_snapshot_t = last_snapshot_t if last_snapshot_t is not None else time.perf_counter()
    fake._snapshot_interval_s = snapshot_interval_s
    fake._profiler = _FakeProfiler()
    return fake


class TestMaybeWriteSnapshot:
    """maybe_write_snapshot is a pure time-throttle — no panda3d needed."""

    def test_does_nothing_when_snapshot_path_is_none(self) -> None:
        fake = _make_fake_app(snapshot_path=None)
        maybe_write_snapshot(fake)
        assert fake._profiler.snapshot_calls == []

    def test_does_not_write_before_interval_elapsed(self) -> None:
        """If last snapshot was < interval ago, skip the write."""
        now = time.perf_counter()
        fake = _make_fake_app(
            last_snapshot_t=now,  # just wrote
            snapshot_interval_s=100.0,  # very long interval
        )
        maybe_write_snapshot(fake)
        assert fake._profiler.snapshot_calls == []

    def test_writes_when_interval_has_elapsed(self) -> None:
        """If last snapshot was > interval ago, write should be called once."""
        fake = _make_fake_app(
            last_snapshot_t=time.perf_counter() - 999.0,  # very old snapshot
            snapshot_interval_s=1.0,
        )
        maybe_write_snapshot(fake)
        assert len(fake._profiler.snapshot_calls) == 1
        assert fake._profiler.snapshot_calls[0] == "profiling/latest.json"

    def test_updates_last_snapshot_t_after_write(self) -> None:
        before = time.perf_counter()
        fake = _make_fake_app(
            last_snapshot_t=before - 999.0,
            snapshot_interval_s=1.0,
        )
        maybe_write_snapshot(fake)
        # last_snapshot_t should have been updated to approximately now
        assert fake._last_snapshot_t >= before

    def test_does_not_write_twice_in_a_row_without_interval(self) -> None:
        fake = _make_fake_app(
            last_snapshot_t=time.perf_counter() - 999.0,
            snapshot_interval_s=1.0,
        )
        maybe_write_snapshot(fake)  # should write
        maybe_write_snapshot(fake)  # interval not elapsed again → skip
        assert len(fake._profiler.snapshot_calls) == 1

    def test_custom_path_is_passed_to_profiler(self) -> None:
        fake = _make_fake_app(
            snapshot_path="custom/path.json",
            last_snapshot_t=time.perf_counter() - 999.0,
            snapshot_interval_s=1.0,
        )
        maybe_write_snapshot(fake)
        assert fake._profiler.snapshot_calls == ["custom/path.json"]
