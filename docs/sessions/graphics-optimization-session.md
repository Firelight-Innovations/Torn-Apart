# Session — Graphics Optimization (FPS / stutter)

**Date:** 2026-06-16
**Branch / worktree:** `perf/graphics-optimization` → `../torn-apart-graphics-perf` (off `master` @ `11fa2fe`)
**Goal:** Find why the game runs ~5 FPS for the first ~30 s then jumps to ~150 FPS with
constant stutters, root-cause it in code, fix the low-hanging fruit, and record the rest.

---

## TL;DR

- **Root-caused and FIXED the steady-state stutter / low FPS:** `LightningRendererComponent`
  rebuilt its *entire* rain-cover heightmap (`rebuild_all`, folding every loaded chunk) on
  **every** `ChunkLoadedEvent`, on the main thread, every frame terrain streamed. Its twin
  `RainRendererComponent` does the identical job correctly with a budgeted incremental refold.
  Fixed lightning to mirror rain.
- **Measured (benchmark, scripted sprint flight, seed 1):**
  `LateUpdate:LightningRendererComponent` **22.88 ms → 0.90 ms** mean; whole-frame mean
  **49.63 ms → 25.66 ms**; **~20 → ~39 FPS** on the stress path; p50 49.25 → 23.95 ms.
- **Two bigger items remain (NOT bugs, NOT yet touched):** the multi-second **startup stall**
  (shader compile + synchronous chunk gen) and **ChunkStream** (legitimate terrain streaming
  cost, now the dominant steady scope). These plus a **terrain LOD / distant-horizon system**
  and **universal LOD wiring** are recorded in `TODO.md` for a fresh agent.

---

## How it was diagnosed

Tool: `tools/profile_run.py` (windowed benchmark, profiler force-on, scripted camera sprint).
See `docs/systems/profiler.md` for the runbook.

```
# startup capture (no warmup, keeps the slow early frames)
python tools/profile_run.py --seed 1 --frames 600 --warmup 0   --out profiling/startup.json
# steady-state capture (discard 120 warmup frames)
python tools/profile_run.py --seed 1 --frames 800 --warmup 120 --out profiling/steady.json
```

### Startup capture (warmup 0) — the 5-FPS-for-30 s phase
```
frame_ms: p50 52.23  p99 74.12  p99.9 2910.55  mean 63.97  max 6857.72  (~16 FPS mean)
worst recent: 6857.7 ms @ frame 0 -> Update
Update                                40.53 mean  5798.09 max   (frame-0 mega-stall)
LateUpdate:LightningRendererComponent 24.59 mean    46.73 max   38.4% of frame, EVERY frame
ChunkStream                           12.72 mean    25.56 max
Lighting                               6.90 mean   501.88 max
```

### Steady-state capture (warmup 120) — the "150 FPS with stutters" phase
```
frame_ms: p50 49.25  p99 77.68  p99.9 98.87  mean 49.63  (~20 FPS on the stress flight)
worst recent: 103.3 ms @ frame 146 -> LateUpdate:LightningRendererComponent
LateUpdate:LightningRendererComponent 22.88 mean  59.81 max   46.1% of frame, EVERY frame
ChunkStream                           13.54 mean
Lighting                               5.05 mean  15.91 max
LateUpdate:RainRendererComponent       1.54 mean             (the correct twin — 15x cheaper)
```

> The benchmark flies a fast scripted path that crosses fresh chunks every frame, so it
> reproduces the *moving* case (constant `ChunkLoadedEvent`s). In the real game the bug is
> dormant while you stand still (no chunk loads → no rebuilds → 150 FPS) and fires while you
> move and during the initial streaming storm — exactly the reported symptom.

---

## Root cause (the fix that shipped this session)

`fire_engine/render/sky/lightning_renderer.py` + `_impl/lightning_bolt.py`.

Both rain and lightning own a `RainCoverField` (256×256 top-down heightmap of the highest
solid voxel per 1 m column) so rain doesn't fall through roofs and bolts strike roofs. The
field exposes two update paths:
- `rebuild_all(chunks)` — clear + re-fold **every** in-window chunk (each a 32³ argmax). Heavy.
- `rebuild_columns(chunks, dirty_cols)` — refold only changed columns. Cheap. The field's
  docstring calls this *"the incremental path the component amortises a budget of columns over."*

`RainRendererComponent` does it right: keeps `_dirty_columns: set`, refolds at most
`rain_cover_budget_columns` (default 4) per frame, full `rebuild_all` only on a recenter.

`LightningRendererComponent` did it wrong: `_on_chunk_loaded` set `_cover_committed = False`
and `late_update` → `refresh_cover` → `rebuild_all(all chunks)`. So **one full heightmap
rebuild from all loaded chunks per chunk-load, per frame, on the main thread** — O(chunks)
work that *grows* as more chunks load (the startup storm) and refires every chunk boundary
crossing (the movement stutter). This matches the prior profiler-session note:
*"live run flagged LightningRendererComponent ~20ms"*.

### The change
- `lightning_renderer.py`: added `_dirty_columns: set[tuple[int,int]]`; `_on_chunk_loaded`
  / `_on_terrain_edited` now mark columns dirty (mirroring rain) instead of invalidating the
  whole cover.
- `_impl/lightning_bolt.py::refresh_cover`: full `rebuild_all` only on recenter / first commit;
  otherwise drain a `rain_cover_budget_columns` budget of dirty columns via `rebuild_columns`.
- De-duplicated the shared `TerrainEditedEvent` payload parsing into a new panda3d-free
  `_impl/cover_events.py::edited_chunk_columns`, used by **both** rain and lightning (this also
  cleared the R0801 duplicate-code couple my first copy-paste introduced).

### Verification
- Full headless suite before changes: 4683 passed (3 pre-existing standards reds).
- After: `tests/standards` ruff-format + pylint-duplication + repo-structure gates **green**;
  cover/rain tests 49 passed; lightning tests 74 passed; `mypy` clean on the 3 changed files;
  new `tests/render/sky/_impl/test_cover_events.py` 5 tests passed.
- Remaining standards red is **pre-existing and unrelated**: `test_git_hygiene` flags a stale
  merged local branch `feature/procedural-buildings` (not deleted — owner's call).

### Files touched
```
fire_engine/render/sky/lightning_renderer.py          (dirty-column tracking)
fire_engine/render/sky/_impl/lightning_bolt.py         (budgeted refresh_cover)
fire_engine/render/sky/rain_renderer.py                (use shared cover_events helper)
fire_engine/render/sky/_impl/cover_events.py           (NEW — shared event parsing)
tests/render/sky/_impl/test_cover_events.py            (NEW — headless test mirror)
docs/systems/render.sky._impl.md                       (doc: refresh_cover + cover_events)
```

---

## Open items handed off (see `TODO.md` → Optimization)

1. **Startup stall (the rest of the 5-FPS-for-30 s).** Frame 0 = 6,857 ms; early frames in the
   seconds. Almost certainly lazy **GPU shader compilation** (cloud raymarch, GI/lighting, sky,
   post-FX all compile on first draw) + **synchronous initial chunk gen/mesh/upload**. The
   lightning fix removes the per-frame rebuilds *during* the storm but not the compile stalls.
   Next step: confirm with PStats (`--pstats`) on frame 0 and `py-spy dump` during the stall;
   then shader pre-warm and/or threaded chunk meshing (terrain meshing threading is noted
   "still pending" in prior sessions).
2. **ChunkStream (~13.5 ms/frame while moving).** Now the dominant steady scope after the
   lightning fix. Legitimate streaming work on the main thread — candidate for threading /
   budgeting, and the natural place a **terrain LOD** system pays off.
3. **`Lighting` spikes to ~500 ms** occasionally (cascade assembly). Prior session offloaded
   this to a worker thread; the residual spike suggests something still lands on the main thread.
4. **HDR float-buffer warning** on boot ("RGBA16F → fixed-point, HDR will clip") — this dev
   machine's iGPU; not a perf issue but degrades HDR. Known caveat.

## Bigger initiative recorded for planning

- **Terrain LOD / distant-horizon rendering** + a **universal LOD communication system for all
  world objects.** The primitive already exists — `core/lod.py::LODPolicy` (`band_for(distance)`,
  default bands 32/96/192/512 m) — and ARCHITECTURE §6 specifies that terrain and the render
  object model both read the *same* policy. But **nothing consumes it yet** (grep: only
  `core/lod.py` + `core/__init__.py` reference it). Terrain meshes at full detail to the
  streaming radius and there is no horizon/imposter tier. This is the "super-optimize terrain"
  track and is recorded as a multi-phase effort in `TODO.md`.

## Reproduce / continue
```
cd ../torn-apart-graphics-perf            # the worktree for this branch
.venv lives in the main checkout — invoke its python by absolute path
python tools/profile_run.py --seed 1 --frames 800 --warmup 120 --save-baseline   # set a baseline
# ...make a change...
python tools/profile_run.py --seed 1 --frames 800 --warmup 120 --fail-on-regress # prove it
```
