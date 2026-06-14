# Test Storm — Summary

Characterization (golden-master / pin-down) test expansion to lock in the **current** behavior of
`fire_engine` before a whole-codebase refactor. Built by an orchestrated swarm of Sonnet sub-agents.

- **Worktree (isolated):** `.claude/worktrees/test-storm`, branch `test-storm`, off HEAD `8586bb3`.
- **NOT committed** — working tree left for human review (per the brief).
- Companion docs: `test-storm-matrix.md` (coverage matrix + slice plan), `test-storm-findings.md`
  (suspected-bug log).

## Numbers

| | Tests |
|---|---|
| Baseline at start (`pytest -q`) | **993 passed**, 1 skipped |
| Added by the storm | **+1646** across **43 new files** |
| Final suite | **2639 passed**, 1 skipped — identical across two consecutive runs (deterministic) |

Verified green via a **full `pytest -q` run after every batch** (1286 → 1649 → 2004 → 2360 → 2639),
plus a final double-run to certify no nondeterminism. Every new file also passed twice in isolation
when written.

## What was added, by batch

- **Batch 1 (293):** `test_clock`, `test_lod`, `test_log`, `test_raycast`, `test_meshing_seams`,
  `test_surface_nets_correctness`, `test_chunk_delta`, `test_chunk_manager_streaming`.
- **Batch 2 (363):** `test_save_manager_edges`, `test_resources_loaders`, `test_grass_placement`,
  `test_zones_store_edges`, `editor/test_scene_components_coerce`, `editor/test_scene_runtime_edges`,
  `test_weather_classify`, `test_weather_lightning_resume`, `test_event_bus_edges`.
- **Batch 3 (355):** `test_weather_humidity_edges`, `test_weather_synoptic_integral`,
  `test_weather_clouds_edges`, `test_weather_map_grid`, `test_weather_cells_extra`,
  `test_sky_celestial_edges`, `test_sky_atmosphere_edges`, `test_wind_gusts`, `test_wind_modifiers`.
- **Batch 4 (356):** `test_wind_region`, `test_wind_venturi_edges`, `test_wind_worker_edges`,
  `test_procedural_maps`, `test_flora_leaves`, `test_flora_atlas`, `test_flora_impostor`,
  `test_buildings_triangulate`, `test_lighting_exposure`, `test_devtools_gizmo`.
- **Batch 5 (279):** `test_simulation_stubs`, `test_fly_controller`, `test_flora_mesher_determinism`,
  `test_buildings_defs`, `test_config_extra`, `test_math3d_boundaries`, `test_rain_cover_edges`.

## Coverage by package (after the storm)

| Package | State now | Notable new pins |
|---|---|---|
| `core` | strong | clock fixed-step/calendar/state round-trip; lod boundaries; log idempotency; event-bus error/edge policy; math3d NaN/inf/slerp boundaries; config derived props + invariants |
| `terrain` | strong | raycast (was untested); meshing seams + floor sentinel; surface-nets dual-cell; chunk delta round-trip; chunk_manager streaming + Saveable; rain-cover sentinel/roof/dirty-parity |
| `save` | strong | register order, atomic-write, backward-compat, apply_delta error path, NaN/non-contiguous array encoding |
| `resources` | strong | loader register/dispatch/suffix, case handling, refcount lifecycle, module-level fns |
| `zones` | strong | grass/leaf placement + GLSL-mirror hash, height-field bake; store delta/version edges; volume NaN/inf/boundary inclusivity |
| `scene` | strong | component coerce/defaults; runtime rebuild/spawn/round-trip (headless) |
| `weather` | strong | classify thresholds; lightning load-resume (verified safe); humidity/synoptic/clouds/cells/map depth |
| `sky` | strong | celestial day-wrap/continuity/ramps; atmosphere vectorization + Rayleigh dominance |
| `wind` | strong | gusts determinism; modifiers; region snap/hysteresis; venturi edges; worker error-resilience/ordering |
| `procedural` | strong | normal/emission maps; flora leaves/atlas/impostor/mesher determinism |
| `buildings` | strong | triangulate; DemoHouseDef structure + registry round-trip |
| `lighting` | strong | exposure meter determinism/adaptation (the last thin spot) |
| `devtools` | strong | gizmo ray-plane/axis math + pick |
| `ai`/`economy`/`politics` | pinned stubs | exact `NotImplementedError` messages + `__all__` |
| `player` | strong | FlyController mouse-look/movement/clamps (headless-importable) |

## Suspected bugs (full detail in `test-storm-findings.md`)

All pinned to **current** behavior — none "fixed". Highest-signal items:

1. **`wind/modifiers.py::GustFront._phase_m` uses process-salted `hash()`** — not cross-process
   deterministic despite a code comment claiming otherwise. Violates the determinism Hard Rule for
   co-op/replay/save-repro. Suggested fix: route through `core.rng.for_domain`/blake2b.
2. **`core/math3d.py` `normalized()` lets NaN through the zero-guard** (Vec3 & Quat) → silent NaN
   poison instead of an error.
3. **`core/clock.py::fixed_steps`** drops a sim tick at exact `N×fixed_dt` boundaries (float drift);
   `update()` accepts negative dt unguarded.
4. **`core/event_bus.py`** — one raising handler aborts all later handlers, and in `drain()` the
   remaining deferred events are permanently lost.
5. **`save/save_manager.py::load`** is non-atomic if a system's `apply_delta` raises (clock/earlier
   systems already mutated); transposed/non-contiguous arrays round-trip to a different layout.
6. **`weather`/`lighting` "coincidental" behaviors** — weather sampling depends on the *global* RNG at
   call time (not captured at construction); exposure "noon ≈ 1.0" holds only by constant coincidence
   and adapts as a dark transition. A refactor could perturb either.
7. **`wind/venturi.py`** — the `wind_venturi_max` config knob is effectively unreachable (dead).
8. Smaller: LOD has no input validation; `zones` `get_delta` before `mark_baseline` emits a full
   snapshot; `flora/leaves` can place leaf centers below ground; `triangulate` emits a zero-area
   triangle for collinear input; gizmo `pick` drops an axis handle when the ray origin sits on it.

We also found an **existing-test gap**: `test_wind_venturi.py`'s error-recovery case never actually
triggers a raising solve (its `.items()` override is inert) — worth tightening.

## Modules that resisted headless testing (refactor risk surface)

**Good news: essentially none.** Every targeted module was fully headless-testable. The only paths
that need the renderer are the by-design GPU/visual bridges that the headless rule already excludes
(`lighting/gpu.py`, the real `SceneVisualFactory`, `world/*_renderer.py`). The headless boundary is
clean — `player/FlyController` even imports `fire_engine.world.component` safely because the panda3d
import is `TYPE_CHECKING`-gated.

## ⚠️ Read this before merging the refactor

This branch pins the **pre-reorg flat layout** (`fire_engine.terrain`, `.weather`, `.wind`, `.sky`, …).
A parallel master-side refactor is regrouping packages (`world/`→`render/`; new `world/{terrain,
weather,wind,sky}` and `simulation/{ai,economy,politics,player}`). When that lands, **these test files
need mechanical import-path updates only** — the behavioral assertions are unchanged and are exactly
what proves the refactor preserved behavior. Recommended merge flow: rebase this branch onto the
refactor, run a search-replace on the moved module paths, and require `pytest -q` green. Any assertion
that then fails is a genuine behavior change to investigate (or one of the pinned suspected-bugs above
being "fixed" — reconcile against the findings log).

## Quarantine / flakiness

One assertion (`test_wind_modifiers.py::TestBandTravel::test_max_delta_shifts_with_time`) was
cross-process flaky — it depended on the salted-hash phase (#1 above) via a discrete argmax. Caught by
the full-suite gate (passed in isolation), fixed in-place to a phase-independent assertion. No other
flakiness; final suite deterministic across two runs.
