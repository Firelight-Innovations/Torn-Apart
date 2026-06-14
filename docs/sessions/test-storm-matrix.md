# Test Storm — Coverage Matrix & Slice Plan

**Goal:** characterization (golden-master / pin-down) tests that capture the *current* behavior of
`fire_engine` at HEAD `8586bb3`, as a safety net for an imminent whole-codebase refactor.
Capture what the code DOES, not what it SHOULD do. Suspected bugs are logged to
`test-storm-findings.md`, never "fixed" in the test.

**Worktree (isolated from master):** `.claude/worktrees/test-storm`, branch `test-storm`.
**Baseline at start:** `993 passed, 1 skipped` (`pytest -q`).

**Layout note (frozen HEAD):** headless packages are flat — `fire_engine.terrain`, `.weather`,
`.wind`, `.sky`, `.core`, `.save`, `.resources`, `.zones`, `.scene`, `.procedural`, `.buildings`,
`.devtools`, `.lighting`. `fire_engine.world` = panda3d render bridge (OUT OF SCOPE). `lighting/gpu.py`
imports panda3d (OUT OF SCOPE; not exported by `__init__`). A master-side refactor is renaming
`world/`→`render/` and regrouping; this worktree predates it on purpose.

## Hard rules for every slice (from CLAUDE.md)
- Headless only — no `panda3d`, no `fire_engine.world`/`lighting.gpu` imports.
- Determinism via `core.rng.for_domain(*keys)` + `set_world_seed` with FIXED seeds.
- No pickle. Save round-trips assert delta equality (dicts of primitives/numpy).
- No Python loops over voxels/vertices — `np.array_equal` / `np.allclose`.
- Pull constants from `core.config`, don't hard-code source magic numbers.
- Tests ADD files under `tests/` only. Never edit production code. Suspected bugs → report to
  orchestrator (who writes the findings log) — do NOT edit the findings log from a worker (avoids
  parallel write collisions).

## Coverage status legend
WELL = solid existing coverage · THIN = exists but key paths unpinned · NONE = no tests.

## Slice plan (one file per agent → disjoint paths, no write collisions)

Ranked roughly by (coverage_gap × blast_radius). Batches below.

### Batch 1 — foundation + highest-blast terrain (8)
| # | New file | Module(s) | Status | Focus |
|---|----------|-----------|--------|-------|
| 1 | tests/test_clock.py | core/clock.py | THIN | fixed_steps accumulator, calendar day/time, GameDayTickEvent, get_state/set_state round-trip, spiral-of-death cap (MAX_FIXED_STEPS), zero/neg/large dt |
| 2 | tests/test_lod.py | core/lod.py | NONE | band_for boundaries (== threshold, <, > max, inf), custom/empty/unsorted bands |
| 3 | tests/test_log.py | core/log.py | NONE | get_logger idempotency (no dup handler), format, root level, distinct names |
| 4 | tests/test_raycast.py | terrain/raycast.py | NONE | hits/misses, ±X/±Y/±Z normals, max_distance 0/short/long, origin-inside, Hit fields, determinism |
| 5 | tests/test_meshing_seams.py | terrain/meshing.py | THIN | empty→0 faces, full-solid interior culling, neighbor padding present/absent, WORLD_FLOOR_SOLID −Z floor, MeshArrays counts, verts_per_face=4, determinism |
| 6 | tests/test_surface_nets_correctness.py | terrain/surface_nets.py | THIN | grass/dirt face_materials, verts_per_face=6, NEIGHBOR_OFFSETS_26 == 26, empty/solid, dual-cell border, determinism |
| 7 | tests/test_chunk_delta.py | terrain/chunk.py | THIN | is_solid_mask (air/solid/sparse), save_delta/apply_delta cycle, dirty/edited flags, world_origin, coord bounds |
| 8 | tests/test_chunk_manager_streaming.py | terrain/chunk_manager.py | THIN | desired_set bounds, stream_frame load-rate cap, unload hysteresis, get_delta/apply_delta round-trip on fresh instance |

### Batch 2 — save/resources/zones/scene + weather core (9)
| # | New file | Module(s) | Status | Focus |
|---|----------|-----------|--------|-------|
| 9 | tests/test_save_manager_edges.py | save/save_manager.py, save/saveable.py | THIN | register order, atomic write (no .tmp left), missing-key backward compat, apply_delta error behavior, config_digest mismatch, NaN/inf arrays, nested dict, SaveIncompatibleError ctor, Saveable runtime check |
| 10 | tests/test_resources_loaders.py | resources/loaders.py + manager module-level fns | NONE/THIN | register_loader, dispatch, registered_suffixes, UnknownResourceFormatError, case/dot handling, re-register; module-level load/acquire/release/unload, path normalization, refcount never negative |
| 11 | tests/test_grass_placement.py | zones/grass_placement.py | THIN | hash_lowbias32 GLSL mirror, grass/leaf hash_seed determinism+bounds, instance_count math+caps, bake_grass_height_field encode + HEIGHT_SENTINEL |
| 12 | tests/test_zones_store_edges.py | zones/store.py, zones/volume.py | WELL→edges | duplicate add ids, delta version mismatch, get_delta w/o mark_baseline, ZoneVolume NaN/inf/zero-size |
| 13 | tests/editor/test_scene_components_coerce.py | scene/components.py | THIN | coerce_params, type coercion, extra keys, default_params deep-copy (no aliasing), make_component, default_components_for_kind |
| 14 | tests/editor/test_scene_runtime_edges.py | scene/runtime.py | WELL→edges | on_rebuilt callback, rebuild idempotency, visual_factory=None headless, spawn_position with no spawn / parented spawn |
| 15 | tests/test_weather_classify.py | weather/classify.py | NONE | classify priority (fog>storm>rain>overcast>cloudy>clear), each threshold boundary, WeatherType enum members |
| 16 | tests/test_weather_lightning_resume.py | weather/lightning.py | THIN | scheduled_strikes load-resume (partition==concat), cell_id_int hash determinism, THUNDERSTORM-only, thinning monotonic w/ intensity |
| 17 | tests/test_event_bus_edges.py | core/event_bus.py | WELL→edges | handler exception propagation, double-subscribe/unsubscribe-once, unsubscribe during publish, re-entrant publish_deferred ordering |

### Batch 3 — weather/sky/wind depth (9)
| # | New file | Module(s) | Status | Focus |
|---|----------|-----------|--------|-------|
| 18 | tests/test_weather_humidity_edges.py | weather/humidity.py | THIN | saturation vs T, RH/condense/wind_gate/emergent_fog boundaries, vectorization shape parity, T extremes |
| 19 | tests/test_weather_synoptic_integral.py | weather/synoptic.py | THIN | dD/dt ≈ wind (finite-diff), scalar vs vector equivalence, speed band [v_min,v_max] |
| 20 | tests/test_weather_clouds_edges.py | weather/clouds.py | THIN | classify_genus per regime, cloud_layers altitude ordering + finiteness, vectorization, coverage/density 0 & 1 |
| 21 | tests/test_weather_map_grid.py | weather/weather_map.py | THIN | rasterize == grid of sample_local, time-invariance, channels in range, texel_centers shape |
| 22 | tests/test_weather_cells_extra.py | weather/cells.py | WELL→edges | regime_ambient, contribution Gaussian → 0 beyond radius, day_regime determinism, active()/intensity envelope boundaries |
| 23 | tests/test_sky_celestial_edges.py | sky/celestial.py | WELL→edges | day-boundary + 86400 wrap continuity, noon continuity, smoothstep/color_ramp/lerp_color boundary inputs |
| 24 | tests/test_sky_atmosphere_edges.py | sky/atmosphere.py | THIN | sky_radiance [N,3] vectorization, transmittance monotonic+finite+nonneg, Rayleigh blue-dominant |
| 25 | tests/test_wind_gusts.py | wind/gusts.py | NONE | build_modes determinism, eval_gusts determinism + output shapes/ranges |
| 26 | tests/test_wind_modifiers.py | wind/modifiers.py | NONE | GustFront determinism (seed_key), apply() in-place mutation, shape/speed/strength effect |

### Batch 4 — wind/procedural/buildings/lighting/devtools/sim (10)
| # | New file | Module(s) | Status | Focus |
|---|----------|-----------|--------|-------|
| 27 | tests/test_wind_region.py | wind/region.py | NONE | maybe_recenter snap granularity, margin hysteresis, X/Y meshgrid, recenter back-and-forth restores origin |
| 28 | tests/test_wind_venturi_edges.py | wind/venturi.py | THIN | fully-solid/fully-open/zero-chunks, deflection, venturi_max clamp, idempotency |
| 29 | tests/test_wind_worker_edges.py | wind/worker.py | THIN | error resilience (job raises → identity result, worker survives), submit/drain order, stop timeout, lifecycle idempotency |
| 30 | tests/test_procedural_maps.py | procedural/maps.py | NONE | derive_normal_map shape/dtype/wrap, flat_normal_map=(128,128,255,255), black_emission_map=(0,0,0,255), determinism |
| 31 | tests/test_flora_leaves.py | procedural/flora/leaves.py | NONE | leaves_at_tips determinism, leaf-count bounds, Leaves.empty(), array shapes, rounds=0 |
| 32 | tests/test_flora_atlas.py | procedural/flora/atlas.py | NONE | bark/leaf_texture + compose_atlas determinism, shapes, bark alpha=255, leaf binary alpha, AtlasLayout |
| 33 | tests/test_flora_impostor.py | procedural/flora/impostor.py | NONE | rasterize_impostor + impostor_atlas shape/dtype/binary-alpha/determinism |
| 34 | tests/test_buildings_triangulate.py | buildings/triangulate.py | THIN | tri count = N−2, area preserved, winding, convex/concave, collinear/degenerate |
| 35 | tests/test_lighting_exposure.py | lighting/exposure.py | THIN | ExposureMeter determinism (fixed ray set), multiplier bounds, dark-slow/bright-fast adaptation |
| 36 | tests/test_devtools_gizmo.py | devtools/gizmo.py | NONE | ray_plane_intersect, closest_on_axis, Gizmo handles, update_drag deltas (TRANSLATE) |

### Batch 5 — remaining gaps (6)
| # | New file | Module(s) | Status | Focus |
|---|----------|-----------|--------|-------|
| 37 | tests/test_simulation_stubs.py | ai, economy, politics | NONE | exact NotImplementedError type + verbatim message for NPCArchetype/GoodDef/FactionDef; __all__ exports |
| 38 | tests/test_fly_controller.py | player/fly_controller.py | NONE | construction/float-coercion, awake reads euler, set_input_state, update mouse-look (yaw free, pitch ±_PITCH_LIMIT, no roll), keyboard WASD/sprint/vertical, _horizontal fallback, dt/speed=0 edges. MIRROR test_gameobject.py imports for headless Component; flag if not headless |
| 39 | tests/test_flora_mesher_determinism.py | procedural/flora/mesher.py | THIN | mesh_leaves RNG determinism, mesh_branches counts, merge_parts, mesh_leaf_area_m2 |
| 40 | tests/test_buildings_defs.py | buildings/defs.py | THIN | DemoHouseDef determinism, generated Building structure (storeys/walls/rooms), registry get |
| 41 | tests/test_config_extra.py | core/config.py | THIN | derived chunk_meters/light_cell_meters, resolve_graphics_preset determinism, field bounds, TOML error paths |
| 42 | tests/test_math3d_boundaries.py | core/math3d.py | WELL→edges | Vec3/Quat NaN/inf, slerp t=0/1 & out-of-range, large magnitudes, normalize near-zero |
| 43 | tests/test_rain_cover_edges.py | terrain/rain_cover.py | THIN | mark_dirty subset vs rebuild_all, OPEN_SKY_Z sentinel, recenter hysteresis, roof overhang raises column height |

## Progress
- [x] Batch 1 dispatched / integrated — 8 files, 293 tests, full suite 1286 passed
- [x] Batch 2 dispatched / integrated — 9 files, 363 tests, full suite 1649 passed
- [x] Batch 3 dispatched / integrated — 9 files, 355 tests, full suite 2004 passed
- [x] Batch 4 dispatched / integrated — 10 files, 356 tests; caught + fixed one cross-process-flaky assertion (test_wind_modifiers band-travel); full suite 2360 passed
- [x] Batch 5 dispatched / integrated — 7 files, 279 tests
- [x] Full-suite green re-run (×2) at end — 2639 passed, 1 skipped, identical both runs (deterministic). See test-storm-summary.md.
