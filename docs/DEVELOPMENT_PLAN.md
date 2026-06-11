# Torn Apart — Session 1 Development Plan (Coding-Agent Handoff)
*For: Claude Code | Date: June 9, 2026 | Rev 2 — Unity-clone API, quaternion transforms, brush editing, locked slice scope*
*Read ARCHITECTURE.md and CLAUDE.md first. ARCHITECTURE.md is authoritative on design; this file on sequencing. Conflicts → ARCHITECTURE.md wins; log the conflict in DECISIONS.md.*

## Mission for Today — the Vertical Slice (owner-approved scope)
By end of session, `python main.py` delivers this demo loop:

> Fly a free camera (WASD + mouse-look) over hard-edged voxel terrain with a procedurally generated ground texture and sun lighting (dark under overhangs), chunks streaming at ≥60 fps. **Left-click fires an explosion** — a sphere brush carves a crater, the chunk remeshes and relights in-frame. **F5 saves, F9 loads**: craters survive the round-trip because only deltas are stored. Headless `pytest -q` is green for everything below the render layer.

In scope: core, Unity-clone object model with quaternion `Transform`, procedural registry + 1 texture, terrain gen/mesh/stream, brush editing, sunlight v0, **full** Resource Manager, minimal SaveManager. Out of scope (do not build): walking/collision, place-mode UI, point lights, biomes/ZoneVolumes, NPCs, economy/politics beyond stubs, greedy meshing, octree LOD.

**Note:** the owner is setting up the repo (venv, git, possibly the folder skeleton). At start of work, *detect and verify* rather than recreate: if `.venv`/`git` exist, use them; if packages are missing, create only what's absent. Never overwrite owner-created files without reading them first.

If behind schedule: cut Phase 7 stretch items, then shrink Phase 5 to cache+refcount with the real loader behind a TODO — never cut tests, the demo loop, or Final Verification.

---

## Phase 0 — Scaffold, Core, Math (~1h)

**Setup (skip whatever the owner already did)**
```
python -m venv .venv && .venv\Scripts\activate
pip install panda3d numpy msgpack pytest
git init
```
`requirements.txt`, `.gitignore` (`.venv/`, `__pycache__/`, `saves/`, `*.png` under `tools/out/`), package tree:

```
fire_engine/
  main.py
  core/        # event_bus.py, config.py, rng.py, clock.py, log.py, lod.py, math3d.py
  procedural/  # registry.py, defs.py, textures/
  resources/   # manager.py, loaders.py
  world/       # app.py, gameobject.py, component.py, transform.py, registry.py,
               # texture_bridge.py, geometry_bridge.py, camera.py
  terrain/     # chunk.py, chunk_manager.py, meshing.py, generation.py, brush.py, raycast.py
  lighting/    # light_grid.py, sunlight.py
  buildings/ ai/ economy/ politics/   # stubs: docstrings + NotImplementedError pointing at ARCHITECTURE.md §
  player/      # fly_controller.py
  save/        # save_manager.py, saveable.py
  tools/       # preview_texture.py, dump_save.py
  tests/
```

**Implement (real):**
1. `core/event_bus.py` — `subscribe/unsubscribe/publish` (sync) + `publish_deferred/drain`. Events `@dataclass(frozen=True)`: `ChunkLoadedEvent`, `ChunkUnloadedEvent`, `TerrainEditedEvent(chunk_coords, brush)`, `GameDayTickEvent`.
2. `core/rng.py` — `set_world_seed(int)`; `for_domain(*keys) -> np.random.Generator` via `SeedSequence` + stable (non-`hash()`! Python salts it) digest of keys, e.g. blake2 of repr. Same keys → same stream, across processes.
3. `core/config.py` — frozen dataclass: `world_seed`, `voxel_size=0.5`, `chunk_size=32`, `light_grid_scale=2`, `view_distance_chunks=6`, `fixed_dt=0.02`, debug flags; `config.toml` overrides.
4. `core/clock.py` — frame dt, fixed-step accumulator for `fixed_update`, game calendar, `GameDayTickEvent`.
5. `core/lod.py` — `LODPolicy` dataclass with distance bands.
6. **`core/math3d.py` — the foundation of the Unity-clone API.** `Vec3` (float32 numpy-backed; `+ - * dot cross length normalized lerp`; constants `ZERO/ONE/UP/FORWARD/RIGHT` — Z-up, forward=+Y) and `Quat` (`identity / from_axis_angle / from_euler(h,p,r) / as_euler / mul / rotate(vec) / slerp / normalized / inverse`). Pure numpy, no panda3d. Docstrings with units (radians) and examples — this file is permanent agent prompt-context.

**Acceptance:** `pytest tests/test_event_bus.py tests/test_rng.py tests/test_math3d.py -q` green. Math tests must include: quat axis-angle 90° about Z rotates +Y→−X (or document your handedness and assert consistently); `slerp(a,b,0)==a`, `slerp(a,b,1)==b`; euler→quat→euler round-trip; rotate∘inverse = identity; determinism of `for_domain` across two interpreter runs (subprocess test).
**Commit:** `phase 0: scaffold + core + math3d`

---

## Phase 1 — World API: Window, Unity Object Model, Fly Camera (~1.5h)

Follow ARCHITECTURE.md §5.4 *exactly* — it is the API contract.

1. `world/transform.py` — `Transform` per §5.4: parent/children, local TRS state, cached world matrix with dirty propagation to children, `forward/right/up` derived from rotation, `translate/rotate/look_at/transform_point/inverse_transform_point`, `Space.SELF|WORLD`. Pure Python/numpy (headless-testable); `world/` syncs it to the NodePath, not the other way round.
2. `world/gameobject.py` + `world/component.py` — `GameObject` (name, tag, layer, active, `add_component(type, **kw)`, `get_component(s)`, `get_component_in_children`, `remove_component`, `set_active`, `compare_tag`) and `Component` with the full Unity lifecycle: `awake → on_enable → start → update → late_update → fixed_update → on_disable → on_destroy`.
3. `world/registry.py` — the batched executor: per-type component buckets; pending-awake/pending-start queues flushed in Unity order (all awakes before any start, same frame); `update`/`late_update` per bucket; `fixed_update` driven by the Clock accumulator; deferred `destroy` executed end-of-frame. `instantiate/destroy/find_with_tag/find_objects_with_tag` module functions.
4. `world/app.py` — Panda3D `ShowBase` wrapper (1280×720, vsync, `setFrameRateMeter(True)`). Frame order: input → `clock.update` → registry.run_frame (awake/start/update/fixed/late) → chunk streaming → lighting dirty work → `event_bus.drain()` → render. GameObjects with renderable components own a NodePath created via World API only; Transform→NodePath sync converts `math3d` types to Panda3D types here and only here.
5. `player/fly_controller.py` — a `Component` (uses `update` + `transform.rotation = Quat...`): WASD move along `transform.forward/right`, mouse-look via quaternion yaw (world Z) × pitch (local X) with pitch clamp, Shift = 5× speed, ESC toggles mouse capture.

**Acceptance:** headless `pytest tests/test_gameobject.py tests/test_transform.py -q`: lifecycle order fixture (record call order across two frames: all awakes → enables → starts → updates → lates; destroy fires on_disable+on_destroy at frame end); parent/child world-position composition; `look_at` points `forward` at target; set_active cascade. Visual: window opens, camera flies, FPS meter on. The object-model tests must import nothing from panda3d.
**Commit:** `phase 1: unity-clone object model + quaternion transform + fly camera`

---

## Phase 2 — Procedural API: Registry + First Texture (~45 min)

1. `procedural/defs.py` — `ProceduralDef` (name, `generate(rng, **params)`), registration decorator.
2. `procedural/registry.py` — `register/get(name, **params)` with `(name, world_seed, frozen params)` cache; rng injected via `core.rng.for_domain("procedural", name, params_digest)`.
3. `procedural/textures/base.py` — `ProceduralTextureDef` → `np.ndarray (H,W,4) uint8`; shared helper: layered value noise = random coarse grids bilinearly upsampled and summed by octave weight (pure numpy, no per-pixel loops).
4. `procedural/textures/wasteland_ground.py` — 256×256 dirt + dead-grass patches via two noise fields + color ramp; registered `"wasteland_ground"`.
5. `world/texture_bridge.py` — numpy RGBA → `panda3d.core.Texture`, **nearest-neighbor min/mag filters**.
6. `tools/preview_texture.py <def_name>` — writes `tools/out/<def_name>.png` headlessly (PNG via Pillow-free path: write with Panda3D? No — keep headless: use `zlib`+manual PNG chunk writer or just `pip install pillow` and add to requirements; pillow is fine).

**Acceptance:** `pytest tests/test_procedural.py -q`: registry round-trip; byte-identical regeneration across fresh registries with same seed; different seed → different bytes; shape/dtype. `python tools/preview_texture.py wasteland_ground` produces a sane PNG (commit it under `tools/out/` as a visual fixture).
**Commit:** `phase 2: procedural registry + wasteland_ground`

---

## Phase 3 — Terrain: Chunks, Generation, Meshing, Streaming, Brushes (~2h — the centerpiece)

1. `terrain/chunk.py` — `Chunk`: `coord (cx,cy,cz)`, `materials: uint8[32,32,32]` (0=air), `dirty`, `edited` flags; world origin = coord·16 m.
2. `terrain/generation.py` — heightmap: layered value noise from `rng.for_domain("terrain", "height")`, amplitude ≈24 m + low-frequency ridges; **plus a 3D-noise carve pass** producing at least occasional overhangs/shallow caves (needed to *see* lighting). Fill below surface with material 1. Must be a pure function of (seed, chunk coord).
3. `terrain/meshing.py` — culled-face mesher, fully vectorized: per axis/direction face mask = `solid & ~neighbor_solid` using padded arrays (pad from neighbor chunks; world edge pads as air **except** the bottom of the world, which pads as solid so the map has no see-through floor). Output: positions/normals/uvs/colors numpy arrays + index array; flat per-face normals (duplicate verts, no smoothing — the retro look). UVs planar from world coords mod tile size. `world/geometry_bridge.py` turns those arrays into a `Geom` with **one bulk memoryview write per array** — no `GeomVertexWriter` per-vertex loops.
4. `terrain/chunk_manager.py` — desired-set = chunks within `view_distance_chunks` XY radius of camera, Z −2..+4; load/generate/mesh ≤2 per frame (budgeted queue, nearest-first); unload at radius+1 (hysteresis); publishes `ChunkLoadedEvent/ChunkUnloadedEvent`. Implements `Saveable` (`save_key="terrain"`): `get_delta() -> {coord: materials_array}` for `edited` chunks; `apply_delta` overlays after generation (and marks them `edited` so they re-save).
5. `terrain/brush.py` — **the only mutation path** (ARCHITECTURE.md §5.5): `SphereBrush(radius)`, `BoxBrush(half_extents)`, `CylinderBrush(radius, height)`; `apply_brush(brush, center: Vec3, mode: ADD|REMOVE, material=1)` rasterizes the shape into a boolean mask per intersected chunk (one vectorized expression — e.g. sphere: `(X-cx)²+(Y-cy)²+(Z-cz)² ≤ r²` on `np.indices` grids), applies it, sets `dirty|edited`, publishes `TerrainEditedEvent`.
6. `terrain/raycast.py` — voxel DDA raycast (camera ray → first solid voxel hit, max 100 m). A short numpy-assisted loop over ≤200 steps is acceptable here (it's once per click, not per voxel).
7. Demo binding in `main.py`: left-click = `apply_brush(SphereBrush(2.5m), hit_point, REMOVE)` — "explosion" (this is a dev binding, not a player verb — see §5.5).

**Acceptance:** headless `pytest tests/test_terrain.py tests/test_brush.py -q`: generation determinism (hash equality across runs); mesher fixtures (1 voxel→6 faces / 12 tris; two adjacent→10 faces; buried→0; chunk-boundary pair produces no interior faces); brush correctness (sphere voxel count within analytic ±tolerance; ADD then REMOVE round-trips; multi-chunk brush touches correct chunk set); desired-set math pure-function tests. Visual: terrain streams while flying; craters appear on click; ≥60 fps at view distance 6.
**Commit:** `phase 3: voxel terrain + brush editing + streaming`

---

## Phase 4 — Lighting v0: Sunlight (~45 min)

1. `lighting/light_grid.py` — per-chunk 16³ `uint8` light arrays (scale 2 → 1 m cells); occupancy = `materials.reshape(16,2,16,2,16,2).max(axis=(1,3,5)) > 0`.
2. `lighting/sunlight.py` — vectorized column pass over a chunk *column stack* (needs chunks above): light = 255 where no occupancy above (cumulative-OR down −Z), else ambient 40; then one 3×3×3 box-blur diffusion for soft penumbra. Subscribe to `TerrainEditedEvent` + chunk loads → recompute affected columns, mark touched chunks dirty for remesh.
3. Bake: mesher samples light grid at face centers (nearest cell is fine v0) → vertex color; default Panda3D shader modulates texture by vertex color. No custom GLSL today.

**Acceptance:** headless `pytest tests/test_lighting.py -q`: empty column → all 255; occupancy at k → 255 above / ambient below; blur conserves range; edit → correct cells invalidated; determinism. Visual: overhang undersides and crater interiors visibly darker; blasting a hole into a cave roof lets a soft light shaft in (the money shot — screenshot it).
**Commit:** `phase 4: sunlight light grid baked to vertex colors`

---

## Phase 5 — Resource Manager, in full (~1h, owner-requested 100%)

1. `resources/loaders.py` — format loaders keyed by suffix: `.egg`/`.bam`/`.gltf|.glb` (Panda3D loader via a thin adapter that `world/` registers at boot — keeps `resources/` free of panda3d imports), `.ogg/.wav` (audio), `.png/.jpg` (static textures for hand-crafted assets only).
2. `resources/manager.py` — `load(path) -> Handle` (cache hit returns same handle), `acquire/release` refcounting, `unload_unreferenced()`, `stats()`. Procedural environment textures explicitly do NOT route here (README states this); landmark models, player hands, audio do.
3. Test fixtures: a hand-written minimal `.egg` (it's a plain-text format — a single textured triangle, ~15 lines, committed under `tests/fixtures/`) exercises the *real* Panda3D loader in one non-headless-marked test (`@pytest.mark.window`, excluded from default run); headless tests use a fake loader to verify cache/refcount/unload logic.
4. Wire in: `world/app.py` registers the real loader adapter at boot; `main.py` loads the fixture triangle and parents it at spawn as proof (remove later).

**Acceptance:** `pytest tests/test_resources.py -q` headless-green (cache identity, refcount lifecycle, unload only at zero refs, suffix dispatch, unknown-suffix error); the `window`-marked loader test passes when run explicitly.
**Commit:** `phase 5: resource manager (full)`

---

## Phase 6 — SaveManager: F5/F9 Delta Saves (~45 min)

1. `save/saveable.py` — `Saveable` protocol exactly per ARCHITECTURE.md §5.12. **No pickle imports anywhere in the repo — add a test that greps the source tree for `import pickle`/`cPickle` and fails if found.**
2. `save/save_manager.py` — `register(saveable)`; `save(path)`: header (format version 1, world_seed, config digest, game clock) + per-system msgpack+zlib blobs (numpy arrays encoded as `(dtype, shape, bytes)` triples); atomic write (tmp → `os.replace`). `load(path)`: validate header (mismatch → `SaveIncompatibleError`, no partial load) → reset clock → `apply_delta` per system in registration order. Saveables today: clock, terrain.
3. Keybinds: F5 → `saves/quick.ta`, F9 → load it (despawn/respawn loaded chunks after delta application).
4. `tools/dump_save.py <file>` — header, per-system keys, compressed/raw sizes.

**Acceptance:** `pytest tests/test_save.py -q`: round-trip (blast 3 craters in headless chunk data → save → fresh world same seed → load → voxel arrays identical); wrong-seed load raises; un-edited world saves to < 1 KB of deltas; no-pickle test green. Visual: blast craters → F5 → blast more → F9 → only the first craters exist.
**Commit:** `phase 6: delta save manager + F5/F9`

---

## Phase 7 — Stretch (priority order; cut freely)
1. Crosshair + brush-mode debug HUD line (current brush, position, chunk coord).
2. `BoxBrush` ADD bound to right-click (place material 1) — completes the demo loop both directions.
3. Second texture `cracked_rock` by height band (multi-material meshing: one Geom per material per chunk).
4. Sim-layer stubs with real interfaces + docstrings: `ai/tiers.py`, `NPCArchetype`, `FactionDef`, `GoodDef`, `BuildingDef` registered in the procedural registry.
5. `tools/screenshot.py` offscreen-buffer capture for CI smoke tests.

---

## Final Verification (mandatory, ~15 min)
1. `pytest -q` green, no skips in the headless suite; `pytest -m window` green when a display is available.
2. Fresh-state run: stash untracked, `python main.py`, confirm boot (catches untracked deps); pop.
3. Demo loop end-to-end: fly 2+ min (no leak: chunk count plateaus, FPS ≥60, no >100 ms hitches), blast craters, F5/F9 persistence, cave-roof light shaft screenshot saved to `tools/out/`.
4. Determinism: same seed twice → identical `preview_texture` PNG bytes and identical chunk hashes; different seed → different.
5. README.md: setup, run, controls (WASD/mouse/Shift/ESC/click/F5/F9), screenshot. Final commit: `session 1 complete: explorable procedural voxel world with brush edits + delta saves`.

## Known Traps
- **Panda3D is Z-up, Y-forward** — matches our `math3d` constants; never convert ad hoc. Unity muscle-memory will tempt you toward Y-up: resist; §5.4 is explicit.
- **Python's `hash()` is salted per process** — RNG key digests must use a stable hash (blake2/sha256 of canonical repr) or determinism silently breaks across runs.
- **All `GeomVertexData` writes bulk** via memoryview/`modify_array`; per-vertex Python writers will blow the frame budget by Phase 3.
- **Chunk-edge meshing needs neighbor padding** (sample adjacent chunks; world bottom pads solid, other edges air) or you get face leaks and z-fighting between chunks.
- **`np.roll` wraps** — use padded arrays, not roll, for face masks.
- **Lifecycle ordering bugs are subtle:** all `awake`s flush before any `start` in the same frame; components added during iteration go to next flush — copy Unity's queue discipline, don't improvise.
- **Mouse-look with quaternions:** compose yaw about *world* Z with pitch about *local* X and clamp pitch before composing; accumulating a single quaternion from raw mouse deltas drifts into roll.
- **`fixed_update` accumulator:** clamp spiral-of-death (max 5 fixed steps/frame).

## Sessions 2+ (orientation only)
S2: ZoneVolumes/biomes, multi-material terrain, point-light flood fill, greedy meshing + LOD bands. S3: BuildingDef generation, first settlement. S4: AI active tier, NPC rendering, walking player with collision. S5: economy/politics world-map tick. Ongoing: editors, landmark pipeline.
