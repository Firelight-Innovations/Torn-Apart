# Session 01 — Handoff Note

*Date: 2026-06-09 · Scope: the owner-approved vertical slice (DEVELOPMENT_PLAN.md "Mission for Today").*

## What shipped

`python main.py` boots an explorable, deterministic procedural voxel world: fly a free camera
(WASD + mouse-look) over hard-edged voxel terrain textured with the procedural `wasteland_ground`
material and lit by baked sunlight (dark under overhangs); chunks stream in around the camera;
**left-click** carves a `SphereBrush(REMOVE)` crater that remeshes and relights in a frame or two;
**F5/F9** save and revert via deltas. The headless test suite is green for everything below the
render layer.

### Per-phase summary (8 commits)

| Commit | What landed | Headless tests |
|---|---|---|
| `phase 0: scaffold + core + math3d` | Package tree; `core/` event bus, deterministic RNG (`for_domain`, blake2b), frozen `Config`, `Clock` (fixed-step + game calendar), `LODPolicy`, logging; `Vec3`/`Quat` (numpy float32, Z-up, scalar-first quats). | 78 (math3d 52, event_bus 17, rng 9) |
| `phase 1: unity-clone object model + quaternion transform + fly camera` | `Transform` (quaternion-only, cached world matrix + dirty propagation), `GameObject`/`Component` (full Unity lifecycle), batched `ComponentRegistry`, `App` ShowBase shell, `CameraComponent`, `FlyController`. | 57 (gameobject 26, transform 31) |
| `phase 2: procedural registry + wasteland_ground` | `ProceduralDef`/`ProceduralTextureDef`, registry with `(name, seed, params)` cache, `value_noise` helper, `wasteland_ground` texture, `texture_bridge`, `preview_texture.py`. | 19 |
| `phase 5: resource manager (full)` | Refcounted `ResourceManager` + `Handle`, suffix-dispatch loaders, panda3d-free via `resource_adapter` IoC, hand-written `.egg` fixture + `@pytest.mark.window` real-loader test. | 27 |
| `phase 3: voxel terrain + brush editing + streaming` | `Chunk`, seamless world-coord `generate_chunk` (heightmap + 3-D carve), vectorised culled-face `build_mesh`, `apply_brush` (sphere/box/cylinder), `raycast_voxel`, streaming `ChunkManager` (`Saveable`), `geometry_bridge`. | 30 (terrain 22, brush 8) |
| `phase 4: sunlight light grid baked to vertex colors` | `LightGrid`, `occupancy_from_materials`, `SunlightComputer` (column pass + box blur, event-subscribed), `make_light_sampler` baked into mesh vertex colours. | 32 |
| `phase 6: delta save manager + F5/F9` | `Saveable` protocol, `SaveManager` (msgpack+zlib envelope, atomic write, header validation), numpy/tuple-key encoding, no-pickle source-tree test, `dump_save.py`. | 14 |
| `integration: main.py demo loop + terrain render path` | `main.py` boot wiring; `App` terrain-render injection (`chunk_manager`/`light_sampler`/`terrain_root`/`setup_terrain_rendering`/`_stream_and_upload_terrain`); `ChunkManager.pending_meshes`/`unloaded_this_frame`/`reset_to_baseline`; `tools/screenshot.py`; `tools/out/spawn.png`. | — |

**Totals:** 257 headless tests passing, 1 deselected (`@pytest.mark.window`, the real-`.egg`
loader — passes under `pytest -m window`).

## The demo loop (how it hangs together)

1. Boot follows ARCHITECTURE §4a.1: config → seed → bus/clock → procedural registration →
   App (window + camera) → resource loaders → ChunkManager → SunlightComputer → SaveManager → player.
2. `main.py` injects `chunk_manager` + `light_sampler` into the App and calls
   `setup_terrain_rendering(ground_tex)`, then pre-warms the spawn area.
3. Each frame: input → clock → `ComponentRegistry.run_frame` → `_stream_and_upload_terrain`
   (stream ≤2 chunks, upload `MeshArrays` as bulk `Geom`s under `terrain_root`) → `event_bus.drain`
   → camera sync.
4. Left-click → `raycast_voxel` → `apply_brush(SphereBrush, REMOVE)` → `TerrainEditedEvent` →
   `SunlightComputer` relights the column + marks chunks dirty → next stream remeshes with fresh light.
5. F5 → `SaveManager.save` (delta = edited chunks only). F9 → `reset_to_baseline()` then
   `SaveManager.load` (re-applies only the saved craters, so post-save craters are undone).

## How it was built

Implementation was delegated phase-by-phase per DEVELOPMENT_PLAN.md, one commit per phase
(prefixed `phase N:`), plus a final integration commit. Each phase commit shipped its system doc
under `docs/systems/`. This doc-review pass reconciled every system doc with the shipped code and
authored the top-level docs (README, DECISIONS, this note).

## Determinism guarantee

The entire world is a pure function of `world_seed` (in `config.toml`). Verified: the same seed
yields **byte-identical** `wasteland_ground` texture bytes and `generate_chunk` output; a different
seed differs. All randomness flows through `core.rng.for_domain` (blake2b digest, never `hash()`),
which makes delta saves and reproducible bug repro possible.

See `tools/out/spawn.png` (committed) for the spawn view — lit tops, shadowed undersides, the retro
hard-edged look.

## Known limitations / deferred items

- **Phase 7 stretch was mostly NOT built.** Specifically deferred: the **crosshair / brush-mode HUD**,
  **right-click `BoxBrush` ADD** (place mode — the demo only removes), the second texture
  **`cracked_rock`** / multi-material meshing, and **richer sim-layer stubs** (`ai/tiers.py`,
  `FactionDef`, `GoodDef`, `BuildingDef` registered in the procedural registry). The simulation
  packages (`buildings/ai/economy/politics`) remain minimal stubs that raise `NotImplementedError`
  pointing at the relevant ARCHITECTURE.md section. **`tools/screenshot.py` WAS built** (offscreen
  capture, `--explode` flag).
- **No player physics:** the camera is a free-fly controller (no collision, gravity, or inventory) —
  it passes through terrain. Embodied player deferred to Session 4.
- **Lighting is v0:** CPU sunlight column pass only (no point lights, no bounce, no GPU compute).
  Tall structures whose top is beyond the view distance are treated as air above (acceptable for v0).
- **Terrain is single-material, no octree LOD, no greedy meshing** (all explicitly out of scope this session).

## Orientation for Session 2+ (from DEVELOPMENT_PLAN "Sessions 2+")

- **S2:** ZoneVolumes / biomes, multi-material terrain, point-light flood fill, greedy meshing + LOD bands.
- **S3:** `BuildingDef` generation, the first settlement.
- **S4:** AI active tier, NPC rendering, walking player with collision.
- **S5:** economy / politics world-map tick.
- **Ongoing:** in-engine editors, the landmark asset pipeline.

The module boundaries (only `world/` + `lighting/` import panda3d; `save/` depends on nothing above
`core`; sim layers register *into* SaveManager) are set up so each future system slots in without
redesign. Start any Session 2 work by grepping `docs/systems/` (the AI search index) and reading the
matching package doc.
