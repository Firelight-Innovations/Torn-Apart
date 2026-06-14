# profiler — System Doc
keywords: profiler, profiling, perf, performance, frame time, frametime, fps, budget, stutter, hitch, spike, pstats, py-spy, pyspy, speedscope, benchmark, regression, baseline, overlay, snapshot, scope, counter, prime suspect

> Documents `fire_engine/core/profiler.py` (the headless core) plus its render-side
> bridges `fire_engine/render/profiler_bridge.py` (PStats) and
> `fire_engine/render/profiler_overlay.py` (the F3 overlay), the `[profiler]` config
> section, and `tools/profile_run.py` (the benchmark harness).

## Role
Measures where each frame's time goes so **two audiences** can diagnose performance
against the **5 ms / 200 FPS total-frame budget**: a **human** live in-game (the F3
overlay + the PStats GUI) and an **AI coding agent** headlessly (a stable-schema JSON
snapshot + `tools/profile_run.py` with baseline regression checks). Stutter is
first-class: every hitch is counted, timestamped, and attributed to the scope that
spiked. The core is engine-agnostic (numpy + `core` only, no panda3d) so it is fully
headless-testable; it is **observational only** — it never calls `core.rng`, never
touches simulation state, and is excluded from saves, so enabling it cannot change the
world. It deliberately does NOT replace PStats' render-stage detail (Cull/Draw/Flip
come from PStats) and it does not auto-fix anything.

## Public API
From `fire_engine.core` (and `fire_engine.core.profiler`):
- `get_profiler() -> Profiler` — the process-wide singleton (a no-op until configured).
- `init_profiler(config) -> Profiler` — configure the singleton from `Config` at boot (mutates in place; existing references stay valid).
- `Profiler(...)` — construct directly for headless tests.
- `Profiler.scope(name)` — context manager timing a compound-named scope (`with prof.scope("Update:Weather"):`). No-op object when disabled.
- `Profiler.profiled(name)` — decorator form for a whole function/method.
- `Profiler.start(name)` / `Profiler.stop(name)` — manual pair (a mismatched `stop` raises `ValueError`).
- `Profiler.set_counter(name, value)` / `Profiler.add_counter(name, delta)` — non-time per-frame metrics (mirror PStats `set_level`/`add_level`).
- `Profiler.begin_frame()` / `Profiler.end_frame()` — call once each at the top/bottom of the main-loop body.
- `Profiler.snapshot() -> dict` — plain-dict summary (JSON-serializable; the agent contract).
- `Profiler.write_snapshot(path)` — atomic JSON write (tmp → `os.replace`).
- `Profiler.recent_frame_ms(n)`, `Profiler.last_frame_ms`, `Profiler.hitch_count`, `Profiler.recent_hitch` — overlay helpers.
- `Profiler.add_observer(on_start, on_stop)` / `add_counter_observer(cb)` — the panda3d-free seam the PStats bridge uses.
- `frame_time_stats(frames_ms, budget_ms) -> dict` — pure stats helper (mean/median/min/max/p99/p999/fps_mean/over_budget_pct).
- `SCHEMA_VERSION` — bump on any breaking snapshot-shape change.

Render-side (panda3d, `render/`): `PStatsBridge(profiler, connect=False)` mirrors scopes/counters into `PStatCollector`s; `ProfilerOverlay(base, profiler, config)` is the F3 HUD (`.toggle()`, `.update()`).

Tool: `python tools/profile_run.py [--seed N --frames N --headless-sim --save-baseline --fail-on-regress --pstats]`.

## Imports Allowed
- `core/profiler.py`: `core` siblings + `numpy` ONLY. **Never** panda3d (a test asserts this in a fresh subprocess). Foundation layer — callable from anywhere.
- `render/profiler_bridge.py`, `render/profiler_overlay.py`: panda3d (allowed in `render/` per ARCHITECTURE §3) + `core`.
- The registry (`render/registry.py`) and `world/sky/sky_state.py` import `core.profiler.get_profiler` to add scopes; both stay panda3d-free.

## Events
Published: none. **A frame hitch is per-frame data and is recorded in the profiler's own ring buffer — never published as an event** (CLAUDE.md Hard Rule 5). A one-off "profiling session started/stopped" event would be acceptable but is not currently emitted.
Subscribed: none.

## Units & Invariants
- All stored times are **milliseconds** (float64); timing is taken in integer nanoseconds (`time.perf_counter_ns`) and converted once per frame.
- `frame_ms` = **wall-clock between two successive `begin_frame()` calls** — the true full frame, *including* the GPU render/flip/vsync that runs after the loop body returns. This is the honest number vs a *total*-frame budget. `frame_ms − Σ(top-level scopes)` ≈ render + overhead (also visible in PStats). `end_frame()` records the loop-body (CPU) time as the `frame_cpu_ms` counter, so the CPU-vs-total split is always available.
- Scope times are **inclusive** (a parent includes its children), matching PStats collectors. Re-entering an already-active scope does not double-count (only the outermost span is measured); the call count still increments.
- A frame is a **hitch** when `frame_ms > max(profiler_hitch_abs_ms, profiler_hitch_rel_mult × rolling_median)` over the last `profiler_hitch_window` frames. The threshold is computed *before* the current frame is added, so a spike never inflates its own threshold.
- **Prime suspect** = the scope with the largest delta *above its own rolling mean* on the spiking frame (not the largest absolute scope — else a heavy-but-steady stage like Draw always wins). Falls back to the largest absolute scope if nothing rose above its mean.
- p99 / p999 are the **99th / 99.9th percentile frame times** (the "1% low" / "0.1% low" — high = bad).
- Determinism: identical sim output with the profiler on vs off (a test asserts this on the headless `SkySystem`).
- Cost: nearly free enabled (integer-ns adds into preallocated numpy arrays; pooled scope objects → no steady-state allocation), truly free disabled (no buffers/overlay/PStats constructed; `scope()` returns a shared `NullScope`).
- Capacity: `profiler_max_scopes` / `profiler_max_counters` cap the preallocated columns; a name past the cap is **dropped with a one-time logged warning** (never silently — a dropped scope would read as "free").

### Config (`[profiler]` table → flat `profiler_*` Config fields)
`profiler_enabled` (master switch, default false), `profiler_overlay_enabled`,
`profiler_frame_budget_ms` (5.0), `profiler_history_frames` (1024),
`profiler_hitch_abs_ms` (8.0), `profiler_hitch_rel_mult` (1.5), `profiler_hitch_window` (120),
`profiler_max_scopes` (64), `profiler_max_counters` (32), `profiler_recent_hitches` (16),
`profiler_overlay_graph_frames` (240), `profiler_overlay_hz` (8.0),
`profiler_snapshot_enabled` (false), `profiler_snapshot_path` (`profiling/latest.json`),
`profiler_snapshot_interval_s` (1.0), `profiler_pstats` (false).

### Instrumented stages (main loop, ARCHITECTURE §4a.1)
`Input`, `Clock`, `Update` (with per-component-type children `Update:<Type>` /
`LateUpdate:<Type>` / `FixedUpdate:<Type>` added by the registry — the weather render
component shows up as `Update:WeatherMapComponent`), `Weather:Update` (explicit, inside
`SkySystem.update`), `ChunkStream`, `Lighting`, `PostProcess`, `EventDrain`,
`CameraSync`. Render stages (`Cull`/`Draw`/`Flip`) come from PStats.

### JSON snapshot schema (v1 — keep stable; agents + the baseline-diff tool parse it)
```json
{
  "schema_version": 1, "timestamp": "2026-06-13T19:40:00Z",
  "frames_measured": 2000, "budget_ms": 5.0,
  "frame_ms": { "mean": 6.8, "median": 6.1, "min": 4.2, "max": 41.7,
                "p99": 18.3, "p999": 33.0, "fps_mean": 147.0 },
  "over_budget_pct": 71.2,
  "hitches": { "count": 23, "per_second": 1.4, "threshold_ms": 9.2,
    "recent": [ { "frame": 4310, "ms": 41.7, "prime_suspect": "Weather:Update" } ] },
  "scopes": [ { "name": "Update:WeatherMapComponent", "mean_ms": 3.9, "max_ms": 22.1,
                "pct_of_frame": 57.0, "calls_per_frame": 1.0 } ],
  "counters": { "draw_calls_mean": 412.0, "frame_cpu_ms_mean": 2.1 }
}
```

## Examples
**Add a scope** (anywhere, panda3d-free or not):
```python
from fire_engine.core.profiler import get_profiler
prof = get_profiler()
with prof.scope("Update:Terrain:Mesh"):
    mesh_dirty_chunks()
# or a whole function:
@prof.profiled("Weather:Update")
def update(self, dt): ...
# counters:
prof.set_counter("chunks_meshed", n)
```

**Read the JSON snapshot** (the live file or a report):
```python
import json
snap = json.load(open("profiling/latest.json"))
print(snap["frame_ms"]["p99"], "ms p99")
for s in snap["scopes"][:5]:
    print(s["name"], s["mean_ms"], "ms", s["pct_of_frame"], "%")
```

**Run the benchmark** (windowed; needs a GPU):
```
python tools/profile_run.py --seed 1 --frames 2000
```

**Save a baseline, then prove a fix recovered the budget:**
```
python tools/profile_run.py --frames 1500 --save-baseline   # before
# ... make your fix ...
python tools/profile_run.py --frames 1500 --fail-on-regress # after (diffs baseline)
```

**Catch a CPU regression with no GPU:**
```
python tools/profile_run.py --headless-sim --frames 3000
```

**Attach PStats (the human's flame graph):**
```
pstats                                   # launch the GUI server (ships with Panda3D)
# then, in another shell:
python tools/profile_run.py --frames 1500 --pstats
# (or set profiler_pstats = true in [profiler] and run `python main.py`)
```
In the GUI open the **Flame Graph** view; our scopes (`Update:Weather`, `ChunkStream`, …)
appear next to the built-in App/Cull/Draw/Flip bars.

**py-spy** (Python hot-spots our named scopes don't cover):
```
pip install py-spy
# attach to the running game (find the PID), record a speedscope profile:
py-spy record -o profiling/pyspy.json --format speedscope --pid <PID>
# snapshot a stuck/stuttering process right now:
py-spy dump --pid <PID>
```
`profiling/pyspy.json` opens in https://speedscope.app and in the Perfetto UI, and is
plain JSON an agent can parse. `py-spy` is a dev/optional dependency
(`requirements-dev.txt`), not a runtime dep.

## How to diagnose a slowdown (runbook for an AI agent)
1. **Run the benchmark**: `python tools/profile_run.py --seed 1 --frames 2000`.
2. **Read `profiling/report.json`** (or `profiling/latest.json` from a live run). Look at
   `frame_ms.p99`/`p999` vs `budget_ms`, `over_budget_pct`, and `hitches`.
3. **Find the top scope / largest regression**: the printed per-scope table is sorted by
   total time; `hitches.recent[*].prime_suspect` names the stage that spiked. If you have a
   baseline, the regression check flags the scope whose mean grew most.
4. **Drill in if the scope is coarse**: open **PStats** (`--pstats`) for the
   render-stage split, or **py-spy** for Python line-level hot-spots inside that scope.
5. **Fix**, then **re-run and confirm against the baseline**:
   `python tools/profile_run.py --frames 2000 --fail-on-regress` (exit 0 = recovered).
> First use of this runbook: diagnosing the current weather-system regression — it
> surfaces as `Weather:Update` and/or `Update:WeatherMapComponent` dominating the
> per-scope table and as the prime suspect on hitches.

## Gotchas
- **Don't let the profiler distort what it measures.** The overlay refreshes at
  `profiler_overlay_hz` (~8 Hz) and reuses its text nodes; the JSON snapshot writes on an
  interval, not every frame. The per-frame hot path is only integer-ns adds into a
  preallocated array. Don't add per-frame `snapshot()`/string building.
- **`frame_ms` is begin-to-begin**, so the **last frame of a run is never committed**
  (no following `begin_frame()`); `profile_run.py` steps one extra frame to flush it. Over
  thousands of frames this is noise; in tests, drive one extra `begin_frame()` to flush.
- **PStats needs the separate GUI app running** and a connection; the overlay works with
  or without it — never make the overlay depend on PStats.
- **Frame-time graph, not FPS graph.** FPS averaging hides spikes; the overlay plots
  per-frame milliseconds with a budget line and a hitch line.
- **`perf_counter_ns`, never `clock.dt`** — `clock.dt` is the sim's view, not wall-clock of
  stages, and the profiler must stay independent of the game timescale.
- **Vsync masks headroom.** A windowed run with vsync caps `frame_ms` at the refresh
  interval; the budget headroom won't show until vsync is off. (The honest displayed-frame
  time is still what the player feels.)
- **Capacity caps drop silently-looking scopes** — a dropped scope is logged once but
  reads as zero cost. Raise `profiler_max_scopes` if you instrument many stages.
- **Layout:** the render bridge package is `render/` (panda3d-only, formerly `world/`);
  the headless grouping package `world/` holds `terrain/weather/wind/sky`. The profiler
  core is `core/profiler.py`; its render mirrors are `render/profiler_bridge.py` +
  `render/profiler_overlay.py`; the explicit weather scope is in
  `world/sky/sky_state.py`. (This doc keeps the flat name `profiler.md` because the
  profiler spans `core/` + `render/` rather than mapping to one package.)
```
