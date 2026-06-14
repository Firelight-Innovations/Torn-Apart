"""
tests/test_profiler.py — headless tests for the core performance profiler.

No window / GPU.  Covers the CLAUDE.md import rule (panda3d must NOT be
importable into core.profiler), the stats math against known arrays, scope
nesting, start/stop mismatch, the no-op-when-disabled guarantee, snapshot JSON
round-trip + schema, hitch detection + prime-suspect attribution, counters, and
determinism (profiler on vs off yields identical sim output).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fire_engine.core.config import Config
from fire_engine.core.profiler import (
    SCHEMA_VERSION,
    NullScope,
    Profiler,
    frame_time_stats,
    get_profiler,
    init_profiler,
)

_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Deterministic fake clock (nanoseconds)
# ---------------------------------------------------------------------------

class FakeClock:
    """Controllable monotonic ns clock for exact timing tests."""

    def __init__(self) -> None:
        self.t = 0

    def __call__(self) -> int:
        return self.t

    def advance_ms(self, ms: float) -> None:
        self.t += int(round(ms * 1e6))


def _run_frame(prof: Profiler, clk: FakeClock, total_ms: float,
               scopes: dict | None = None, counters: dict | None = None) -> None:
    """
    Simulate one frame: begin → (timed scopes) → end, then advance the clock so
    the NEXT begin_frame commits this frame with a full duration of *total_ms*.
    """
    begin_t = clk.t
    prof.begin_frame()
    if scopes:
        for name, ms in scopes.items():
            with prof.scope(name):
                clk.advance_ms(ms)
    if counters:
        for name, val in counters.items():
            prof.set_counter(name, val)
    prof.end_frame()
    clk.t = begin_t + int(round(total_ms * 1e6))


# ---------------------------------------------------------------------------
# Import rule: core.profiler must never pull in panda3d
# ---------------------------------------------------------------------------

def test_core_profiler_imports_without_panda3d():
    probe = (
        "import sys\n"
        "import fire_engine.core.profiler\n"
        "import fire_engine.core\n"
        "leaked = [m for m in sys.modules if m == 'panda3d' "
        "or m.startswith('panda3d.')]\n"
        "print('LEAK' if leaked else 'clean', leaked)\n"
        "sys.exit(1 if leaked else 0)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe], cwd=str(_ROOT),
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "clean" in proc.stdout


# ---------------------------------------------------------------------------
# Stats math against known arrays
# ---------------------------------------------------------------------------

def test_frame_time_stats_known_array():
    # 1..100 ms.  numpy is the oracle for the percentiles.
    arr = np.arange(1.0, 101.0)
    s = frame_time_stats(arr, budget_ms=5.0)
    assert s["min"] == 1.0
    assert s["max"] == 100.0
    assert s["mean"] == pytest.approx(50.5)
    assert s["median"] == pytest.approx(np.percentile(arr, 50.0))
    assert s["p99"] == pytest.approx(np.percentile(arr, 99.0))
    assert s["p999"] == pytest.approx(np.percentile(arr, 99.9))
    assert s["fps_mean"] == pytest.approx(1000.0 / 50.5)
    # 95 of 100 frames are > 5 ms budget.
    assert s["over_budget_pct"] == pytest.approx(95.0)


def test_frame_time_stats_empty():
    s = frame_time_stats(np.zeros(0), budget_ms=5.0)
    assert s["mean"] == 0.0 and s["fps_mean"] == 0.0 and s["max"] == 0.0


# ---------------------------------------------------------------------------
# Scope timing + nesting
# ---------------------------------------------------------------------------

def test_nested_scopes_accumulate_to_parent():
    clk = FakeClock()
    prof = Profiler(enabled=True, time_source=clk, history_frames=8,
                    hitch_window=4)
    prof.begin_frame()
    with prof.scope("A"):
        clk.advance_ms(5.0)
        with prof.scope("A:B"):
            clk.advance_ms(3.0)
    prof.end_frame()
    # Commit by starting the next frame 20 ms after this one began.
    clk.t = 20_000_000
    prof.begin_frame()

    snap = prof.snapshot()
    scopes = {s["name"]: s for s in snap["scopes"]}
    # A is inclusive: 5 ms own + 3 ms in B = 8 ms total.
    assert scopes["A"]["mean_ms"] == pytest.approx(8.0)
    assert scopes["A:B"]["mean_ms"] == pytest.approx(3.0)
    # Parent >= child (inclusive timing).
    assert scopes["A"]["mean_ms"] >= scopes["A:B"]["mean_ms"]
    assert snap["frame_ms"]["max"] == pytest.approx(20.0)


def test_reentrant_scope_not_double_counted():
    clk = FakeClock()
    prof = Profiler(enabled=True, time_source=clk, history_frames=4,
                    hitch_window=2)
    prof.begin_frame()
    with prof.scope("R"):           # outer start @0
        clk.advance_ms(2.0)
        with prof.scope("R"):       # re-entry: must NOT restart the timer
            clk.advance_ms(3.0)
    # outer stop @5 → R == 5 ms (not 5+3)
    prof.end_frame()
    clk.t = 10_000_000
    prof.begin_frame()
    scopes = {s["name"]: s for s in prof.snapshot()["scopes"]}
    assert scopes["R"]["mean_ms"] == pytest.approx(5.0)
    assert scopes["R"]["calls_per_frame"] == pytest.approx(2.0)


def test_stop_without_matching_start_raises():
    prof = Profiler(enabled=True)
    prof.begin_frame()
    prof.start("X")
    with pytest.raises(ValueError):
        prof.stop("Y")            # not the innermost open scope


def test_unbalanced_stack_logged_not_silent(caplog):
    clk = FakeClock()
    prof = Profiler(enabled=True, time_source=clk)
    prof.begin_frame()
    prof.start("leaky")           # never stopped
    prof.end_frame()              # should log an error and reset the stack
    assert any("never stopped" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# No-op when disabled
# ---------------------------------------------------------------------------

def test_disabled_profiler_is_noop():
    prof = Profiler(enabled=False)
    # No buffers allocated.
    assert prof._frame_ms.size == 0
    assert prof._scope_ms.size == 0
    # scope() returns the shared no-op (no allocation, no timing).
    s = prof.scope("anything")
    assert s is get_noop()
    with prof.scope("X"):
        pass
    prof.begin_frame()
    prof.end_frame()
    snap = prof.snapshot()
    assert snap["frames_measured"] == 0
    assert snap["scopes"] == []


def get_noop():
    # The disabled profiler always returns the same NullScope instance.
    from fire_engine.core.profiler import _NULL_SCOPE
    return _NULL_SCOPE


def test_disabled_scope_is_nullscope_type():
    prof = Profiler(enabled=False)
    assert isinstance(prof.scope("x"), NullScope)


# ---------------------------------------------------------------------------
# Snapshot round-trip + schema
# ---------------------------------------------------------------------------

def test_snapshot_json_roundtrip_and_schema():
    clk = FakeClock()
    prof = Profiler(enabled=True, time_source=clk, history_frames=16,
                    hitch_window=8)
    for _ in range(6):
        _run_frame(prof, clk, total_ms=6.0,
                   scopes={"Update:Weather": 3.9, "Update:Terrain:Mesh": 1.1},
                   counters={"draw_calls": 412, "triangles": 1_830_000})
    # Flush the last frame: _run_frame already set the clock to its end marker,
    # so an extra begin_frame (no advance) commits it with the right duration.
    prof.begin_frame()

    snap = prof.snapshot()
    # Lossless JSON round-trip.
    again = json.loads(json.dumps(snap))
    assert again == snap

    # Schema keys present.
    assert snap["schema_version"] == SCHEMA_VERSION
    assert snap["budget_ms"] == 5.0
    for k in ("mean", "median", "min", "max", "p99", "p999", "fps_mean"):
        assert k in snap["frame_ms"]
    for k in ("count", "per_second", "threshold_ms", "recent"):
        assert k in snap["hitches"]
    assert isinstance(snap["scopes"], list)
    # Scopes sorted by mean_ms descending; Weather > Terrain.
    names = [s["name"] for s in snap["scopes"]]
    assert names[0] == "Update:Weather"
    # Counters carry the _mean suffix.
    assert snap["counters"]["draw_calls_mean"] == pytest.approx(412.0)


def test_write_snapshot_atomic(tmp_path):
    clk = FakeClock()
    prof = Profiler(enabled=True, time_source=clk, history_frames=8,
                    hitch_window=4)
    for _ in range(4):
        _run_frame(prof, clk, total_ms=5.0, scopes={"Draw": 4.0})
    prof.begin_frame()            # flush the last frame (no advance)

    out = tmp_path / "deep" / "latest.json"
    prof.write_snapshot(str(out))
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["schema_version"] == SCHEMA_VERSION
    # No leftover temp files in the directory.
    assert [p.name for p in out.parent.iterdir()] == ["latest.json"]


# ---------------------------------------------------------------------------
# Hitch detection + prime-suspect attribution
# ---------------------------------------------------------------------------

def test_hitch_fires_on_spike_with_correct_suspect():
    clk = FakeClock()
    prof = Profiler(enabled=True, time_source=clk, history_frames=64,
                    hitch_window=10, hitch_abs_ms=8.0, hitch_rel_mult=1.5)
    # 20 smooth ~5 ms frames: Weather steady at ~1 ms, Terrain ~1 ms.
    for _ in range(20):
        _run_frame(prof, clk, total_ms=5.0,
                   scopes={"Update:Weather": 1.0, "Update:Terrain:Mesh": 1.0})
    assert prof.hitch_count == 0          # smooth input → no hitches

    # One spike frame: Weather balloons to 35 ms; frame total 41.7 ms.
    _run_frame(prof, clk, total_ms=41.7,
               scopes={"Update:Weather": 35.0, "Update:Terrain:Mesh": 1.0})
    prof.begin_frame()            # commit the spike frame (no advance)

    assert prof.hitch_count == 1
    h = prof.recent_hitch
    assert h is not None
    assert h["ms"] == pytest.approx(41.7, abs=0.1)
    assert h["prime_suspect"] == "Update:Weather"


def test_smooth_input_no_hitches():
    clk = FakeClock()
    prof = Profiler(enabled=True, time_source=clk, history_frames=64,
                    hitch_window=10, hitch_abs_ms=8.0, hitch_rel_mult=1.5)
    for _ in range(40):
        _run_frame(prof, clk, total_ms=4.5, scopes={"Draw": 3.0})
    prof.begin_frame()            # flush the last frame (no advance)
    assert prof.hitch_count == 0


# ---------------------------------------------------------------------------
# Capacity guard (dropped scope is warned, not silent)
# ---------------------------------------------------------------------------

def test_scope_capacity_warns(caplog):
    prof = Profiler(enabled=True, max_scopes=2)
    prof.begin_frame()
    with prof.scope("A"):
        pass
    with prof.scope("B"):
        pass
    with prof.scope("C"):         # over capacity → dropped + warned
        pass
    assert any("capacity reached" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Singleton wiring from Config
# ---------------------------------------------------------------------------

def test_singleton_disabled_by_default():
    prof = get_profiler()
    # Default Config has the profiler off.
    init_profiler(Config())
    assert prof.enabled is False
    assert isinstance(prof.scope("x"), NullScope)


def test_init_profiler_from_config_enables_in_place():
    prof_before = get_profiler()
    cfg = Config(profiler_enabled=True, profiler_history_frames=32,
                 profiler_frame_budget_ms=5.0)
    prof_after = init_profiler(cfg)
    # Mutated in place — same object, now enabled.
    assert prof_after is prof_before
    assert prof_after.enabled is True
    assert prof_after.history_frames == 32
    # Reset the singleton so other tests see the default-off state.
    init_profiler(Config())


# ---------------------------------------------------------------------------
# Determinism: enabling the profiler must not change sim output
# ---------------------------------------------------------------------------

def test_profiler_does_not_change_sim_output():
    """
    The SkySystem (headless weather+sky) is instrumented with a profiler scope.
    Running it with the profiler enabled vs disabled must produce byte-identical
    SkyState output — timing is observational only.
    """
    from fire_engine.core import Clock, EventBus, set_world_seed
    from fire_engine.world.sky import SkySystem

    def run(profiler_enabled: bool):
        init_profiler(Config(profiler_enabled=profiler_enabled,
                             profiler_history_frames=64))
        set_world_seed(1337)
        bus = EventBus()
        clock = Clock(fixed_dt=0.02, bus=bus)
        clock.game_time_of_day = 10.0 * 3600.0
        sky = SkySystem(Config(), clock, bus)
        out = []
        prof = get_profiler()
        for _ in range(30):
            prof.begin_frame()
            clock.update(0.016)
            st = sky.update((12.0, 34.0))
            prof.end_frame()
            out.append((st.sun_dir.x, st.sun_dir.y, st.sun_dir.z,
                        st.cloud_coverage, st.fog_density, st.rain_intensity))
        return out

    on = run(True)
    off = run(False)
    init_profiler(Config())       # reset singleton
    assert on == off
