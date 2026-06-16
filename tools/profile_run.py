"""
tools/profile_run.py — deterministic, scripted performance benchmark harness.

This is the headline AI-agent tool: a *repeatable* profiling run an agent can
invoke, then read the result of, to see exactly which engine stage is slow or
stuttering — fix it, re-run, and prove the fix against a saved baseline.

What it does
------------
- Boots the engine with a **fixed ``--seed``**, steps a **fixed ``--frames``**
  count along a **scripted camera path** (no human input) so runs are
  comparable, with the profiler force-enabled.
- Two modes:
    * **windowed** (default): the real boot path (``main.build_demo``) with the
      window/GPU, so render-stage cost is included in ``frame_ms``.  Needs a GPU
      (same requirement as ``tools/screenshot.py``).
    * **``--headless-sim``**: a GPU-free loop over the panda3d-free CPU stages
      (weather/sky sim + terrain streaming) — for catching CPU regressions (like
      the weather one) on a box with no GL.  It times a *subset* of the frame.
- Writes a **report JSON** (``profiling/report.json`` by default) and prints a
  human summary: overall p50/p99/p999, FPS, hitch count, and a **per-scope
  table sorted by total time**.
- ``--save-baseline`` writes ``profiling/baseline.json``; otherwise the run
  **diffs against the baseline** and flags any scope whose mean ms (or the
  overall p99) regressed beyond ``--regress-pct`` (default 15%).
- Exits **non-zero** if it cannot run (so CI / agents catch breakage) or if a
  regression is detected in diff mode (``--fail-on-regress``).

Run
---
    python tools/profile_run.py --seed 1 --frames 2000
    python tools/profile_run.py --frames 1500 --save-baseline
    python tools/profile_run.py --frames 1500 --fail-on-regress
    python tools/profile_run.py --headless-sim --frames 3000   # no GPU needed
    python tools/profile_run.py --frames 1500 --pstats         # also feed PStats

Then an agent reads ``profiling/report.json`` (stable, versioned schema — see
``core/profiler.py`` / ``docs/systems/profiler.md``).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
import sys
from pathlib import Path

# Repo root on sys.path so ``import main`` / ``fire_engine`` resolve.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_REPORT = "profiling/report.json"
_DEFAULT_BASELINE = "profiling/baseline.json"


# ---------------------------------------------------------------------------
# Scripted camera path (deterministic — no randomness)
# ---------------------------------------------------------------------------


def _flight_path(i: int):
    """
    Scripted camera position for frame ``i``: a fast sprint out from spawn with
    gentle XY turning + climb, crossing many fresh chunks and recentering every
    cascade window repeatedly.  Pure function of ``i`` -> fully reproducible.
    """
    from fire_engine.core.math3d import Vec3

    t = i * 0.8  # ~50 m/s feel
    x = math.sin(i * 0.01) * t * 0.5
    y = t
    z = 12.0 + math.sin(i * 0.02) * 6.0
    return Vec3(float(x), float(y), float(z))


def _flight_xy(i: int) -> tuple[float, float]:
    p = _flight_path(i)
    return (p.x, p.y)


# ---------------------------------------------------------------------------
# Profiler configuration override (force ON regardless of config.toml)
# ---------------------------------------------------------------------------


def _enable_profiler(base_config, frames: int):
    """Re-init the singleton profiler ON, sized to hold the whole run."""
    from fire_engine.core.profiler import init_profiler

    cfg = dataclasses.replace(
        base_config,
        profiler_enabled=True,
        profiler_overlay_enabled=False,
        profiler_snapshot_enabled=False,  # the harness writes its own report
        profiler_history_frames=max(
            int(getattr(base_config, "profiler_history_frames", 1024)), frames + 64
        ),
    )
    return init_profiler(cfg)


# ---------------------------------------------------------------------------
# Windowed run (real boot path, render included)
# ---------------------------------------------------------------------------


def run_windowed(seed: int, frames: int, warmup: int, pstats: bool) -> dict:
    """Boot the full demo, fly the scripted path, return the profiler snapshot."""
    import main as demo

    # Force the world seed before boot (build_demo reads config.world_seed).
    os.environ.setdefault("TA_PROFILE_RUN", "1")
    app = demo.build_demo()
    # Override the seed on the loaded config if it differs (re-seed RNG too).
    if seed is not None and int(getattr(app._config, "world_seed", seed)) != seed:
        from fire_engine.core.rng import set_world_seed

        app._config = dataclasses.replace(app._config, world_seed=int(seed))
        set_world_seed(int(seed))

    prof = _enable_profiler(app._config, frames)
    if pstats:
        from fire_engine.render.profiler_bridge import PStatsBridge

        PStatsBridge(prof, connect=True)

    # Don't let physical mouse / keys perturb the scripted path.
    app.input_state.mouse_captured = False
    app._set_mouse_capture(False)
    cam = app.camera_go.transform

    # Warmup (discarded): let prewarm settle + driver caches warm.
    for i in range(warmup):
        cam.local_position = _flight_path(i)
        app.taskMgr.step()
    # Reset the profiler so warmup frames aren't in the stats.
    prof.configure_from_config(
        dataclasses.replace(
            app._config,
            profiler_enabled=True,
            profiler_overlay_enabled=False,
            profiler_snapshot_enabled=False,
            profiler_history_frames=max(
                int(getattr(app._config, "profiler_history_frames", 1024)), frames + 64
            ),
        )
    )

    for i in range(warmup, warmup + frames):
        cam.local_position = _flight_path(i)
        app.taskMgr.step()
    # One extra step to commit the final frame (begin-to-begin model).
    app.taskMgr.step()

    snap = prof.snapshot()
    snap["meta"] = {
        "mode": "windowed",
        "seed": seed,
        "frames": frames,
        "warmup": warmup,
        "camera_path": "sprint_v1",
    }
    # Clean shutdown (don't enter the blocking loop).
    pipeline = getattr(app, "lighting_pipeline", None)
    if pipeline is not None:
        with contextlib.suppress(Exception):
            pipeline.shutdown()
    return snap


# ---------------------------------------------------------------------------
# Headless-sim run (GPU-free CPU-stage subset)
# ---------------------------------------------------------------------------


def run_headless_sim(seed: int, frames: int, warmup: int) -> dict:
    """
    Time the panda3d-free per-frame CPU stages without a window: the weather/sky
    sim (``Weather:Update`` lives inside ``SkySystem.update``) and terrain
    streaming.  This is a SUBSET of the real frame (no render), enough to catch
    CPU-side regressions on a GPU-less box.
    """
    from fire_engine.core import Clock, EventBus, load_config, set_world_seed
    from fire_engine.world.sky import SkySystem
    from fire_engine.world.terrain import ChunkManager

    cfg = load_config()
    if seed is not None:
        cfg = dataclasses.replace(cfg, world_seed=int(seed))
    set_world_seed(int(cfg.world_seed))

    bus = EventBus()
    clock = Clock(fixed_dt=cfg.fixed_dt, bus=bus)
    clock.game_time_of_day = 10.0 * 3600.0
    sky = SkySystem(cfg, clock, bus)
    chunks = ChunkManager(cfg, bus)
    prof = _enable_profiler(cfg, frames)

    def step(i: int) -> None:
        prof.begin_frame()
        with prof.scope("Clock"):
            clock.update(0.016)
        with prof.scope("Update"), prof.scope("Update:Sky"):
            sky.update(_flight_xy(i))  # Weather:Update nests inside
        with prof.scope("ChunkStream"):
            chunks.stream_frame(_flight_path(i), None)
        prof.end_frame()

    for i in range(warmup):
        step(i)
    prof.configure_from_config(
        dataclasses.replace(
            cfg,
            profiler_enabled=True,
            profiler_overlay_enabled=False,
            profiler_snapshot_enabled=False,
            profiler_history_frames=max(int(cfg.profiler_history_frames), frames + 64),
        )
    )
    for i in range(warmup, warmup + frames):
        step(i)
    prof.begin_frame()  # commit the last frame

    snap = prof.snapshot()
    snap["meta"] = {
        "mode": "headless-sim",
        "seed": seed,
        "frames": frames,
        "warmup": warmup,
        "camera_path": "sprint_v1",
        "note": "CPU-stage subset only -- no render stages",
    }
    return snap


# ---------------------------------------------------------------------------
# Reporting + baseline diff
# ---------------------------------------------------------------------------


def _scope_total_ms(scope: dict, frames_measured: int) -> float:
    return scope["mean_ms"] * frames_measured


def print_summary(snap: dict) -> None:
    fm = snap["frame_ms"]
    meta = snap.get("meta", {})
    n = snap["frames_measured"]
    print(f"\n=== profile_run [{meta.get('mode', '?')}] seed={meta.get('seed')} frames={n} ===")
    print(
        f"frame_ms:  p50 {fm['median']:.2f}   p99 {fm['p99']:.2f}   "
        f"p99.9 {fm['p999']:.2f}   mean {fm['mean']:.2f}   "
        f"max {fm['max']:.2f}   (~{fm['fps_mean']:.0f} FPS mean)"
    )
    print(
        f"budget {snap['budget_ms']:.1f} ms  ->  over-budget "
        f"{snap['over_budget_pct']:.1f}% of frames"
    )
    h = snap["hitches"]
    print(
        f"hitches:   {h['count']} ({h['per_second']:.2f}/s, threshold {h['threshold_ms']:.1f} ms)"
    )
    if h["recent"]:
        worst = max(h["recent"], key=lambda r: r["ms"])
        print(
            f"  worst recent: {worst['ms']:.1f} ms @ frame "
            f"{worst['frame']} -> {worst['prime_suspect']}"
        )

    print(f"\n{'scope':<34}{'total ms':>11}{'mean ms':>10}{'max ms':>10}{'% frame':>9}{'calls':>7}")
    print("-" * 81)
    scopes = sorted(snap["scopes"], key=lambda s: _scope_total_ms(s, n), reverse=True)
    for s in scopes:
        print(
            f"{s['name']:<34}{_scope_total_ms(s, n):>11.1f}"
            f"{s['mean_ms']:>10.2f}{s['max_ms']:>10.2f}"
            f"{s['pct_of_frame']:>8.1f}%{s['calls_per_frame']:>7.1f}"
        )
    if snap["counters"]:
        print("\ncounters (mean/frame):")
        for k, v in sorted(snap["counters"].items()):
            print(f"  {k:<34}{v:>14.1f}")


def diff_baseline(snap: dict, baseline_path: str, regress_pct: float) -> bool:
    """
    Compare *snap* against the saved baseline; print regressions.

    Returns True if any regression beyond ``regress_pct`` was found.  A scope
    regresses when its mean ms grew by more than the threshold; the overall p99
    is also checked.  New/removed scopes are reported but not treated as
    regressions.
    """
    p = Path(baseline_path)
    if not p.exists():
        print(
            f"\n[baseline] none at {baseline_path} — run with --save-baseline "
            f"first to enable regression checks."
        )
        return False
    base = json.loads(p.read_text(encoding="utf-8"))
    print(f"\n=== regression check vs {baseline_path} (threshold +{regress_pct:.0f}%) ===")
    factor = 1.0 + regress_pct / 100.0
    regressed = False

    # Overall frame p99.
    b99 = base["frame_ms"]["p99"]
    c99 = snap["frame_ms"]["p99"]
    if b99 > 0 and c99 > b99 * factor:
        regressed = True
        print(f"  REGRESS  frame_ms.p99  {b99:.2f} -> {c99:.2f} ms (+{(c99 / b99 - 1) * 100:.0f}%)")
    else:
        print(f"  ok       frame_ms.p99  {b99:.2f} -> {c99:.2f} ms")

    base_scopes = {s["name"]: s for s in base.get("scopes", [])}
    cur_scopes = {s["name"]: s for s in snap.get("scopes", [])}
    for name, cs in sorted(cur_scopes.items()):
        bs = base_scopes.get(name)
        if bs is None:
            print(f"  new      {name}  ({cs['mean_ms']:.2f} ms mean)")
            continue
        bm, cm = bs["mean_ms"], cs["mean_ms"]
        if bm > 0 and cm > bm * factor:
            regressed = True
            print(f"  REGRESS  {name}  {bm:.2f} -> {cm:.2f} ms mean (+{(cm / bm - 1) * 100:.0f}%)")
    for name in sorted(set(base_scopes) - set(cur_scopes)):
        print(f"  gone     {name}  (was {base_scopes[name]['mean_ms']:.2f} ms)")

    if not regressed:
        print("  -> no regressions beyond threshold.")
    return regressed


def write_report(snap: dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(snap, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\n[report] wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Deterministic perf benchmark.")
    ap.add_argument("--seed", type=int, default=1, help="world seed (default 1)")
    ap.add_argument("--frames", type=int, default=2000, help="timed frames (default 2000)")
    ap.add_argument("--warmup", type=int, default=60, help="discarded warmup frames (default 60)")
    ap.add_argument(
        "--headless-sim", action="store_true", help="GPU-free CPU-stage subset (no window)"
    )
    ap.add_argument(
        "--pstats", action="store_true", help="also connect to a PStats server (windowed mode)"
    )
    ap.add_argument(
        "--out", default=_DEFAULT_REPORT, help=f"report JSON path (default {_DEFAULT_REPORT})"
    )
    ap.add_argument(
        "--save-baseline",
        action="store_true",
        help=f"write the run to {_DEFAULT_BASELINE} as the baseline",
    )
    ap.add_argument("--baseline", default=_DEFAULT_BASELINE, help="baseline JSON to diff against")
    ap.add_argument(
        "--regress-pct", type=float, default=15.0, help="regression threshold percent (default 15)"
    )
    ap.add_argument(
        "--fail-on-regress", action="store_true", help="exit non-zero if a regression is detected"
    )
    args = ap.parse_args()

    try:
        if args.headless_sim:
            snap = run_headless_sim(args.seed, args.frames, args.warmup)
        else:
            snap = run_windowed(args.seed, args.frames, args.warmup, args.pstats)
    except Exception as exc:
        print(f"profile_run FAILED to run: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 2

    if snap["frames_measured"] == 0:
        print("profile_run produced 0 frames — boot/step failure.", file=sys.stderr)
        return 2

    print_summary(snap)

    if args.save_baseline:
        write_report(snap, _DEFAULT_BASELINE)
        print(f"[baseline] saved {_DEFAULT_BASELINE}")
        # Also write the normal report so the run is inspectable.
        write_report(snap, args.out)
        return 0

    write_report(snap, args.out)
    regressed = diff_baseline(snap, args.baseline, args.regress_pct)

    if regressed and args.fail_on_regress:
        return 1
    return 0


if __name__ == "__main__":
    # Force-exit so a lingering Panda3D window / OpenAL device doesn't hang us.
    code = main()
    sys.stdout.flush()
    os._exit(code)
