# Test Storm — Import Fix Brief (for the fix agent)

**TL;DR:** 43 new characterization-test files were merged to `master` (commit `b8695c2`, merge
`441675b`). They were written against the **pre-reorg flat package layout**, but `master` has since
regrouped packages (`world/`→`render/`; new `world/` and `simulation/` groups). So **22 of the 43 new
files fail at collection** with `ModuleNotFoundError`. Fix = mechanical import remap. Nothing is wrong
with the test logic; the modules just live at new paths.

## The problem, exactly

`pytest --collect-only -q` on `master` currently reports:

```
1853/1857 tests collected, 22 errors during collection
```

All 22 errors are new test files importing packages that the refactor **moved**. The ~21 new files
that test *un-moved* packages (core, save, resources, zones, scene, procedural, buildings, devtools,
lighting) collect and run fine already.

## Remap table (old import → new import)

⚠️ **`world/` was REUSED — do not blind find/replace `fire_engine.world`.** The old `world/`
(renderers) became `render/`; the *new* `world/` is the sim-world grouping. Apply these prefix
rewrites:

| Old path used in tests | New path on master |
|---|---|
| `fire_engine.terrain` (+ `.chunk`, `.brush`, `.generation`, `.meshing`, `.surface_nets`, `.raycast`, `.chunk_manager`, `.rain_cover`) | `fire_engine.world.terrain…` |
| `fire_engine.weather` (+ submodules `.system`, `.cells`, `.classify`, `.clouds`, `.humidity`, `.synoptic`, `.weather_map`, `.lightning`, `.bolt`) | `fire_engine.world.weather…` |
| `fire_engine.wind` (+ `.field`, `.gusts`, `.modifiers`, `.region`, `.venturi`, `.worker`, `.debug`) | `fire_engine.world.wind…` |
| `fire_engine.sky` (+ `.celestial`, `.atmosphere`, …) | `fire_engine.world.sky…` |
| `fire_engine.ai` | `fire_engine.simulation.ai` |
| `fire_engine.economy` | `fire_engine.simulation.economy` |
| `fire_engine.politics` | `fire_engine.simulation.politics` |
| `fire_engine.player` (+ `.fly_controller`) | `fire_engine.simulation.player…` |
| `fire_engine.world.component` / `.gameobject` / `.app` / `.transform` (old renderer layer) | `fire_engine.render.component` / `.gameobject` / `.app` / `.transform` |

**No change needed** (these packages did NOT move): `fire_engine.core`, `.save`, `.resources`,
`.zones`, `.scene`, `.procedural`, `.buildings`, `.devtools`, `.lighting`.

### Specific gotcha — `test_fly_controller.py`
It does `from fire_engine.world.component import Component` → must become
`from fire_engine.render.component import Component`. Its `InputState` import is under
`TYPE_CHECKING` (`fire_engine.world.app`) → update to `fire_engine.render.app` for cleanliness (won't
break collection, but keep it correct).

### Also check: hard-coded shader/asset paths
A couple of GLSL-mirror tests read a `.vert` file by path to pin shader constants
(`test_grass_placement.py`, the flora tests). If shaders moved with the renderers
(`world/shaders/` → `render/shaders/`), update those file paths too, or the mirror assertion will
fail on a missing file rather than on content.

## Files that need remapping (the 22 erroring on collection)

terrain: `test_raycast`, `test_meshing_seams`, `test_surface_nets_correctness`, `test_chunk_delta`,
`test_chunk_manager_streaming`, `test_rain_cover_edges` ·
weather: `test_weather_classify`, `test_weather_lightning_resume`, `test_weather_humidity_edges`,
`test_weather_synoptic_integral`, `test_weather_clouds_edges`, `test_weather_map_grid`,
`test_weather_cells_extra` ·
wind: `test_wind_gusts`, `test_wind_modifiers`, `test_wind_region`, `test_wind_venturi_edges`,
`test_wind_worker_edges` ·
sky: `test_sky_celestial_edges`, `test_sky_atmosphere_edges` ·
simulation: `test_simulation_stubs`, `test_fly_controller`

(Exact set: run the verify command below — don't trust this list blindly if the layout shifted again.)

## Workflow for the fix agent

1. **Remap imports** in the affected files per the table above. Verify after each pass:
   ```
   python -m pytest --collect-only -q
   ```
   Goal: `0 errors during collection` — every new file imports.
2. **Run the full suite** and read the real signal:
   ```
   python -m pytest -q
   ```
   - **green** = the refactor preserved that behavior. ✅
   - **red** = a genuine behavior difference between pre- and post-reorg code. This is the to-fix
     list.
3. **Fix the codebase** (not the tests) to turn red → green — UNLESS the failing test is pinning one
   of the **known pre-existing bugs** in `docs/sessions/test-storm-findings.md`. For those, the test
   encodes *current (buggy) behavior on purpose*; decide per item whether to (a) fix the bug and
   update that one test to the corrected behavior, or (b) leave it pinned. Don't silently "fix" a test
   to make red go away — that defeats the safety net.

## Reference docs (all under docs/sessions/)
- `test-storm-summary.md` — overview, per-package coverage, merge guidance.
- `test-storm-findings.md` — ~30 suspected bugs, each pinned to current behavior (read before "fixing"
  any failing test).
- `test-storm-matrix.md` — coverage matrix / slice plan.

## Important framing
These are **characterization (golden-master) tests** — they capture what the code *did* at `8586bb3`,
not what it *should* do. After the import remap, a failing assertion means post-reorg behavior diverged
from pre-reorg behavior. Treat each as a real regression to investigate, cross-checking the findings
log so you don't mistake an intentionally-pinned pre-existing bug for a new break.
