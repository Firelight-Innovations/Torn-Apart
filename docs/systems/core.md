# core — System Doc
keywords: vec3, quat, quaternion, math3d, event bus, eventbus, publish, subscribe, drain, rng, seed, for_domain, config, clock, fixed_update, lod, lodpolicy, logging, math, rotation, euler, hpr, slerp, chunk loaded, game day, terrain edited, weather changed, world seed, determinism, blake2b, float32, z-up, forward, right, up, meters, radians, fixed_dt, spiral of death, saveable, get_state, set_state, game_time_scale, time scale, sky config, cloud altitude, star count, shader_source, load_glsl, glsl, shader file, vert, frag, comp, syntax highlighting, include, glsl include, shader include, lit_surface, graphics, graphics quality, preset, gfx, post process, postprocess, hdr, bloom, fxaa, lens flare, volumetric clouds, cloud quality, god rays, render scale, quality preset, resolve_graphics_preset, GRAPHICS_PRESETS, off low medium high, profiler, profiling, perf, frame time, fps, budget, stutter, hitch, scope, counter, snapshot, get_profiler, init_profiler, Profiler, frame_time_stats, ring buffer, p99, 1% low, prime suspect

> One doc per code package; filename matches the package exactly (`docs/systems/core.md` ↔ `fire_engine/core/`).

## Role

`core/` is the **foundation layer** — pure Python/numpy, zero panda3d imports, callable from every other layer in the engine.  It provides:

- **Math primitives** (`Vec3`, `Quat`) that the entire object model depends on.
- **Event Bus** for upward/sideways state-change notifications between layers.
- **Deterministic RNG service** (`for_domain`) — the single source of all randomness; same seed → same world, across processes.
- **Typed configuration** (`Config`, `load_config`) loaded from `config.toml`.
- **Frame Clock** with fixed-step accumulator and in-game calendar.
- **LOD Policy** (shared distance-band thresholds for World and Terrain).
- **Logging** (`get_logger`) with a consistent format.
- **Performance profiler** (`Profiler`, `get_profiler`, `init_profiler`) — the engine-agnostic frame timer + numpy ring buffer + hitch detection. Headless, observational only; the panda3d overlay/PStats bridge live in `render/`. **Full details: `docs/systems/profiler.md`.**

`core/` deliberately does NOT: render anything, touch the Panda3D scene graph, know about terrain chunks, or hold game-world state.  It is stateless except for the global world seed in `rng.py` (and the process-wide `Profiler` singleton, which holds only observational timing).

## Public API

All symbols below are re-exported from `fire_engine.core` (`__init__.py`).

### Math (`core/math3d.py`)

| Symbol | Description |
|---|---|
| `Vec3(x, y, z)` | Immutable float32 3-vector. Components `x`, `y`, `z`. |
| `Vec3.ZERO` | `(0, 0, 0)` |
| `Vec3.ONE` | `(1, 1, 1)` |
| `Vec3.UP` | `(0, 0, 1)` — world up (+Z) |
| `Vec3.FORWARD` | `(0, 1, 0)` — world forward (+Y) |
| `Vec3.RIGHT` | `(1, 0, 0)` — world right (+X) |
| `v + w`, `v - w` | Component-wise add/sub. Returns new `Vec3`. |
| `v * scalar`, `scalar * v` | Scalar multiply. |
| `-v` | Negate. |
| `v == w` | Exact float equality. Use `approx_eq` for numerical work. |
| `v.approx_eq(w, eps=1e-6)` | Component-wise absolute tolerance. |
| `v.dot(w)` | Dot product (float). |
| `v.cross(w)` | Cross product (right-handed, returns `Vec3`). |
| `v.length` | Euclidean length (float, meters). |
| `v.length_squared` | Squared length (avoids sqrt). |
| `v.normalized()` | Unit-length copy; raises `ValueError` on zero vector. |
| `v.lerp(w, t)` | Linear interpolate; `t=0→v`, `t=1→w`. |
| `v.to_numpy()` | float32 `(3,)` copy. |
| `Vec3.from_numpy(arr)` | Construct from any array-like shape `(3,)`. |
| `iter(v)`, `v[i]` | Iterate or index as `(x, y, z)`. |
| `Quat(w, x, y, z)` | Unit quaternion, scalar-first `[w,x,y,z]`. |
| `Quat.identity()` | No rotation. |
| `Quat.from_axis_angle(axis, radians)` | Right-handed rotation about `axis`. |
| `Quat.from_euler(h, p, r)` | HPR Euler in radians (H=yaw/Z, P=pitch/X, R=roll/Y). Composition: H then P then R. |
| `q.as_euler()` | `(h, p, r)` radians. Round-trip safe away from gimbal singularities. |
| `q1 * q2` | Hamilton product: applies `q2` first, then `q1`. |
| `q.rotate(v)` | Rotate `Vec3 v` by this quaternion. |
| `Quat.slerp(a, b, t)` | Spherical interpolation; `t=0→a`, `t=1→b`. Short-arc, normalised output. |
| `q.normalized()` | Unit-norm copy. |
| `q.inverse()` | Conjugate (= inverse for unit quaternions). |
| `q.approx_eq(other, eps)` | Handles the q ≡ −q double cover. |

### Events (`core/event_bus.py`)

| Symbol | Description |
|---|---|
| `EventBus()` | Pub/sub dispatcher (synchronous + deferred). |
| `bus.subscribe(EventType, handler)` | Register handler. |
| `bus.unsubscribe(EventType, handler)` | De-register (no-op if absent). |
| `bus.publish(event)` | Immediate synchronous delivery. |
| `bus.publish_deferred(event)` | Enqueue for next `drain()`. |
| `bus.drain()` | FIFO dispatch of deferred queue (snapshot before dispatch). |
| `ChunkLoadedEvent(coord)` | Terrain chunk finished generating and meshing. |
| `ChunkUnloadedEvent(coord)` | Terrain chunk evicted from memory. |
| `TerrainEditedEvent(chunk_coords, brush)` | Brush edit applied to terrain voxel data. |
| `GameDayTickEvent(day)` | In-game day counter incremented. |
| `WeatherChangedEvent(previous, current, day)` | Discrete weather state changed (`WeatherType.value` strings, e.g. `"clear"` → `"rain"`).  Published by `fire_engine.world.sky.WeatherSystem` via `publish_deferred`. |
| `BuildingChangedEvent(building_id, change, bounds_min, bounds_max)` | A building was added/modified/removed (`change` ∈ `"added"`/`"modified"`/`"removed"`); `bounds_*` are the world AABB.  Published by `fire_engine.buildings.BuildingManager`. |

### RNG (`core/rng.py`)

| Symbol | Description |
|---|---|
| `set_world_seed(seed: int)` | Set module-level world seed. Call once at boot. |
| `for_domain(*keys) -> np.random.Generator` | Deterministic child generator from (world_seed, keys). Cross-process stable. |

### Config (`core/config.py`)

| Symbol | Description |
|---|---|
| `Config` | Frozen dataclass with all engine settings. |
| `Config.world_seed` | `int` — RNG seed for procedural systems (textures, noise, NPC). **Not** used by terrain (flat/authored). |
| `Config.world_size_m` | `float` — square world footprint side in meters, centred on origin (1000 m = 1 km area). |
| `Config.ground_height_m` | `float` — flat baseline ground surface height (world Z, meters). |
| `Config.voxel_size` | `float` — meters per voxel edge (0.5 m) |
| `Config.chunk_size` | `int` — voxels per chunk edge (32) |
| `Config.light_grid_scale` | `int` — terrain voxels per light cell (2) |
| `Config.view_distance_chunks` | `int` — streaming radius in chunks |
| `Config.fixed_dt` | `float` — fixed timestep in seconds (0.02) |
| `Config.show_fps` | `bool` |
| `Config.show_chunk_borders` | `bool` |
| `Config.show_light_grid` | `bool` |
| `Config.sky_cloud_altitude_m` | `float` — cloud layer base altitude, world Z meters (96.0). From the `[sky]` TOML table (flattened like `[debug]`). |
| `Config.sky_cloud_thickness_m` | `float` — cloud layer vertical thickness in meters (8.0). |
| `Config.sky_cloud_cell_m` | `float` — horizontal edge of one cloud cell in meters (12.0); the renderer fills `cloud_coverage` fraction of cells. |
| `Config.sky_star_count` | `int` — star count for the `"night_sky"` procedural texture (2500). |
| `Config.mesh_style` | `str` — terrain mesher: `"faceted"` (flat-shaded surface nets, default) or `"blocky"` (culled-face cubes). From the `[terrain]` TOML table (flattened like `[debug]`). |
| `Config.facet_shade_strength` | `float` — [0,1] strength of the faceted mesher's normal-based facet accent shading (0.25; 0 = off). |
| `Config.chunk_meters` | property — `chunk_size * voxel_size` (16.0 m) |
| `Config.light_cell_meters` | property — `voxel_size * light_grid_scale` (1.0 m) |
| `Config.gfx_*` | Graphics-quality knobs for the HDR post-processing + volumetric cloud pipeline (from the `[graphics]` TOML table). Master switch `gfx_post_process`; bloom (`gfx_bloom`, `gfx_bloom_mips`, `gfx_bloom_threshold`, `gfx_bloom_knee`, `gfx_bloom_strength`); `gfx_fxaa`; `gfx_lens_flare`; clouds (`gfx_clouds`, `gfx_cloud_steps`, `gfx_cloud_light_steps`, `gfx_cloud_resolution_scale`, `gfx_cloud_max_dist_m`); `gfx_god_rays`, `gfx_god_ray_samples`; `gfx_hdr_format` (`rgba16f`/`rgba8`), `gfx_render_scale`. **Aesthetic look-knobs (preset-independent, tune freely):** `gfx_god_ray_strength`, `gfx_lens_flare_strength`, `gfx_lens_flare_threshold`, `gfx_tonemap_hue_preserve`, `gfx_sun_disc_intensity`, `gfx_sun_halo_intensity`, `gfx_sun_min_brightness`, `gfx_sky_inscatter_scale` — the presets intentionally don't carry these, so they survive a preset change. Dataclass defaults == the `"high"` preset. |
| `GRAPHICS_PRESETS` | `dict[str, dict]` — the `off`/`low`/`medium`/`high` quality presets, each mapping `gfx_*` knobs to values. `"high"` mirrors the dataclass defaults. |
| `resolve_graphics_preset(table) -> dict` | Expand a `[graphics]` table into flat `gfx_*` kwargs: the `preset` key picks the base set; any explicit `gfx_*` key overrides it. Invalid preset → `"high"` + warning (never raises). Deterministic. |
| `load_config(path="config.toml") -> Config` | Load from TOML; returns defaults if file missing.  Flattens the `[debug]`, `[sky]`, `[terrain]`, `[lighting]`, `[fog]`, `[grass]` and `[graphics]` tables (the last via preset expansion). |

### Clock (`core/clock.py`)

| Symbol | Description |
|---|---|
| `Clock(fixed_dt, bus, game_time_scale)` | Frame clock + fixed-step accumulator + game calendar. |
| `clock.update(real_dt: float)` | Advance by one real frame (seconds). |
| `clock.fixed_steps() -> Iterator[float]` | Yield up to 5 fixed-dt intervals (spiral-of-death guard). |
| `clock.dt` | Last real frame duration in seconds. |
| `clock.fixed_dt` | Fixed timestep in seconds (from Config). |
| `clock.game_time_scale` | Read/**write** property — real→game seconds multiplier (default 60.0).  Dev tooling (time-of-day scrubbers, fast-forward) may change it at runtime; takes effect from the next `update()`. |
| `clock.game_day` | Current in-game day number (int, starts at 0). |
| `clock.game_time_of_day` | Seconds elapsed within the current in-game day. |
| `clock.total_real_time` | Total real seconds since boot. |
| `clock.get_state() -> dict` | Saveable state (primitives only). |
| `clock.set_state(dict)` | Restore from saved state. |

### LOD (`core/lod.py`)

| Symbol | Description |
|---|---|
| `LODPolicy(bands=(32.0, 96.0, 192.0, 512.0))` | Frozen distance-band policy. |
| `policy.band_for(distance_m: float) -> int` | 0 = full detail; `len(bands)` = beyond last threshold. |

### Logging (`core/log.py`)

| Symbol | Description |
|---|---|
| `get_logger(name: str) -> logging.Logger` | Sane-formatted logger; handler installed once. |

### Shader source (`core/shader_source.py`)

| Symbol | Description |
|---|---|
| `load_glsl(anchor: str, name: str) -> str` | Read GLSL source `name` from the `shaders/` directory beside `anchor` (the caller's `__file__`). Used by `world/grass_shaders.py`, `world/sky_shaders.py`, `world/terrain_shader.py` and `lighting/glsl.py` to load their `.vert`/`.frag`/`.comp` sidecars (real shader files, for editor syntax highlighting + LSP) and re-export them under the original string constants. Expands include directives: a line of the exact shape `//#include "lit_surface.glsl"` is replaced by that file's text (looked up in the same `shaders/` dir), wrapped in `// --- begin/end include` marker comments. One level only — an included file may not itself include (`ValueError`); a missing included file raises `FileNotFoundError`. The directive is a valid GLSL comment so each sidecar still lints standalone; no `#line` directives are emitted (flaky on Intel drivers). Sources without the directive pass through byte-identical. Panda3d-free — reading a text file is callable from any layer. |

## Imports Allowed

`core/` may only import:
- Python standard library (`math`, `hashlib`, `logging`, `tomllib`, `collections`, `dataclasses`, ...)
- `numpy`

**No panda3d imports.** No imports from any other `fire_engine.*` package (core is the bottom of the dependency graph).

## Events

### Published
| Event | When | Publisher |
|---|---|---|
| `GameDayTickEvent(day)` | When the in-game day counter increments | `Clock.update()` via `bus.publish_deferred` |

### Subscribed
`core/` subscribes to nothing.  It is the foundation; other layers subscribe to its events.

## Units & Invariants

### Coordinate System
- **Z-up** (Panda3D native): `forward = +Y`, `right = +X`, `up = +Z`.
- **Not** Unity's Y-up.  The Unity API shape is used (same method names, snake_case), but the axes are ours.
- All distances in **meters**.  `Vec3` components are meters unless the caller documents otherwise.
- Rotations in **radians** throughout (no degrees anywhere in `math3d`).

### Quat Handedness
Right-handed rotation about `+Z` by `+π/2` (CCW from above) rotates `+Y → −X`:
```python
Quat.from_axis_angle(Vec3.UP, pi/2).rotate(Vec3.FORWARD) ≈ Vec3(-1, 0, 0)
```
Tests in `tests/test_math3d.py::TestQuatHandedness` assert this explicitly.

### Quat Multiplication Order
`q1 * q2` applies `q2` **first**, then `q1`.  Matches scipy `Rotation.__mul__` and Unity's `Quaternion.*`.  A composed camera rotation `yaw * pitch` therefore yaws in world space, then pitches in the yawed frame.

### RNG Determinism Guarantee
`for_domain(*keys)` is **cross-process deterministic**: the same `(world_seed, keys)` pair produces an identical stream on every Python process, on every machine, across interpreter restarts.  This guarantee holds because:
1. `np.random.SeedSequence` is numpy-stable.
2. Key hashing uses `hashlib.blake2b` (NOT Python's `hash()`, which is salted per-process since Python 3.3).

The subprocess test in `tests/test_rng.py::TestCrossProcessDeterminism` verifies this empirically.

### Config Invariants
- `Config` is frozen (`frozen=True` dataclass) — cannot be mutated after construction.
- `chunk_meters == 16.0` and `light_cell_meters == 1.0` with the locked default values.

### Clock Fixed-Step Guard
`clock.fixed_steps()` yields **at most 5** fixed-dt intervals per frame regardless of how long the real frame took.  Excess accumulation is silently dropped to prevent the spiral-of-death (see DEVELOPMENT_PLAN.md Known Traps).  Default `MAX_FIXED_STEPS = 5`.

### Game Time Scale
Default: 1 real second = 60 in-game seconds → 1 real minute = 1 game hour → 1 game day ≈ 24 real minutes.  Adjustable at runtime via the `clock.game_time_scale` read/write property (dev tooling only; gameplay systems should not change it).  Changing the scale never rewinds or jumps the calendar — only the rate of future accrual.

## Examples

### Math3d
```python
from math import pi
from fire_engine.core.math3d import Vec3, Quat

# Build a position and rotation
pos = Vec3(10.0, 5.0, 2.0)           # 10 m east, 5 m north, 2 m up
yaw = Quat.from_axis_angle(Vec3.UP, pi / 4)   # 45° heading

# Rotate the forward vector to find "where the entity is pointing"
facing = yaw.rotate(Vec3.FORWARD)    # ≈ Vec3(−0.707, 0.707, 0)

# Interpolate between two rotations
q0 = Quat.identity()
q1 = Quat.from_axis_angle(Vec3.UP, pi / 2)
mid = Quat.slerp(q0, q1, 0.5)       # 45° heading

# Compose pitch × yaw (yaw applied first in world space):
pitch = Quat.from_axis_angle(Vec3.RIGHT, -0.3)   # look down slightly
combined = yaw * pitch               # yaw, then pitch in yaw-frame
```

### Event Bus
```python
from fire_engine.core.event_bus import EventBus, ChunkLoadedEvent

bus = EventBus()

def on_chunk_loaded(evt: ChunkLoadedEvent) -> None:
    print(f"Chunk {evt.coord} ready ({evt.coord[0]*16} m east)")

bus.subscribe(ChunkLoadedEvent, on_chunk_loaded)
bus.publish(ChunkLoadedEvent(coord=(3, 0, 0)))   # immediate

# Deferred (dispatched once per frame tick)
bus.publish_deferred(ChunkLoadedEvent(coord=(4, 0, 0)))
# ... end of frame:
bus.drain()
```

### RNG
```python
from fire_engine.core.rng import set_world_seed, for_domain

set_world_seed(1337)   # once, at boot

# Get a deterministic generator for terrain chunk (4, 5, 0)
rng = for_domain("terrain", (4, 5, 0))
noise = rng.random((32, 32))        # float64 32×32

# Always identical for same seed + keys:
rng2 = for_domain("terrain", (4, 5, 0))
assert (rng2.random((32, 32)) == rng.random((32, 32))).all()  # True
```

### Config
```python
from fire_engine.core.config import load_config

cfg = load_config("config.toml")
print(cfg.chunk_meters)          # 16.0
print(cfg.light_cell_meters)     # 1.0
print(cfg.fixed_dt)              # 0.02
```

### Clock + Fixed Steps
```python
from fire_engine.core.clock import Clock
from fire_engine.core.event_bus import EventBus

bus = EventBus()
clock = Clock(fixed_dt=0.02, bus=bus)

def game_loop(real_dt: float) -> None:
    clock.update(real_dt)
    for _ in clock.fixed_steps():
        physics_tick(clock.fixed_dt)   # 50 Hz, max 5× per frame
    bus.drain()
```

## Gotchas

1. **Never use Python `hash()` for RNG keys** — it is salted per-process since Python 3.3 and will silently break cross-run determinism. `for_domain` uses `hashlib.blake2b` internally; no callers should re-implement this.

2. **`Vec3` is mutable via `_data` attribute** — although immutable by convention, the internal numpy array is not copy-on-write.  Callers must not store `v._data` references; always use `v.to_numpy()` for owned copies.

3. **Quat double-cover**: `q` and `-q` represent the same rotation.  Use `approx_eq` (which handles both cases) rather than `==` when comparing quaternions numerically.

4. **`as_euler()` near gimbal lock** (pitch ≈ ±π/2): the heading/roll split is ambiguous; the method assigns all rotation to heading and zeros roll.  The **rotation** represented is still correct; only the numeric H/R values differ.

5. **`Clock.fixed_steps()` is a generator** — call it exactly once per frame via `for _ in clock.fixed_steps()`.  The accumulator is consumed during iteration; calling it twice in one frame would yield an empty second iteration.

6. **`publish_deferred` during `drain()`** defers to the *next* drain, not the current one.  The `drain()` implementation snapshots the queue at entry so handlers cannot inject into the current sweep.

7. **`Config` is frozen** — attempting `cfg.world_seed = 9999` raises `FrozenInstanceError`.  Load once at boot, pass by reference to all systems.

8. **`set_world_seed` is global module state** — the world seed is module-level in `rng.py`.  Change it only at boot or during a world-load (save/load sets it from the save header before any generation).

9. **`[graphics]` precedence**: the `preset` key sets a base, then any explicit `gfx_*` key in the *same* table overrides just that field — so you can run `preset = "low"` but force `gfx_lens_flare = true`.  The `Config` dataclass defaults equal the `"high"` preset, so a missing `[graphics]` table (or a missing `gfx_*` key not covered by the chosen preset) yields high-quality values.  Set `preset = "off"` (or `gfx_post_process = false`) to fall back to the legacy in-shader tonemap path with no HDR buffer/bloom/flare — the escape hatch for weak GPUs.

## Config fields

Per-field reference for `fire_engine.core.config.Config` (moved here from the
class docstring to keep `config.py` under 500 lines).  All fields have a
dataclass default.  Instantiate via `load_config(path)`.

### Core world fields

| Field | Type | Default | Description |
|---|---|---|---|
| `world_seed` | int | 1337 | RNG seed for procedural systems (textures, ambient noise, NPC behaviour). Terrain is flat/authored and does NOT use the seed. |
| `world_size_m` | float | 1000.0 | Square world footprint side length in meters, centred on the origin (1000 m = a 1 km × 1 km area spanning [-500, +500] on X and Y). |
| `ground_height_m` | float | 8.0 | Flat baseline ground surface height (world Z, meters); solid below it, air above. |
| `voxel_size` | float | 0.5 | Meters per voxel edge (locked at 0.5 m). |
| `chunk_size` | int | 32 | Voxels per chunk edge (locked at 32). |
| `light_grid_scale` | int | 2 | Terrain voxels per light cell edge (2). |
| `view_distance_chunks` | int | 6 | Chunk-streaming XY radius in chunks. |
| `fixed_dt` | float | 0.02 | Fixed-update period in seconds (50 Hz = 0.02). |
| `msaa_samples` | int | 4 | Hardware MSAA sample count for the window framebuffer (0 = off). Anti-aliases geometry edges only (facet silhouettes, crater rims, the horizon) — surface interiors stay single-sample, so the pixel-art texel look is unaffected. |

### Debug flags (from `[debug]` table)

| Field | Type | Default | Description |
|---|---|---|---|
| `show_fps` | bool | True | Overlay FPS counter. |
| `show_chunk_borders` | bool | False | Debug overlay for chunk boundaries. |
| `show_light_grid` | bool | False | Debug overlay for the light grid. |
| `debug_wind_ball` | bool | False | Spawn the dev-only "wind ball": a bright procedural sphere on the ground near spawn that is pushed by `WindField.sample` each fixed step (a physics seam proof — it scoots on gusts, rolls in storms). Off by default. |
| `debug_demo_building` | bool | True | Spawn the feature-showcase demo house in front of spawn at boot (the building-system evaluation build; set false to hide). |

### Sky fields (from `[sky]` table)

| Field | Type | Default | Description |
|---|---|---|---|
| `sky_cloud_altitude_m` | float | 96.0 | Base altitude of the cloud layer (world Z, m). |
| `sky_cloud_thickness_m` | float | 8.0 | Vertical thickness of the cloud layer (m). |
| `sky_cloud_cell_m` | float | 12.0 | Horizontal edge of one cloud cell (m); the renderer fills coverage-fraction of cells. |
| `sky_star_count` | int | 2500 | Star count baked into the "night_sky" procedural texture. |

### Terrain/mesh fields (from `[terrain]` table)

| Field | Type | Default | Description |
|---|---|---|---|
| `mesh_style` | str | "faceted" | Terrain mesher: "faceted" (flat-shaded surface nets — the Daggerfall-ish semi-smooth look, default) or "blocky" (classic culled-face cubes). |
| `facet_shade_strength` | float | 0.25 | [0,1] strength of the faceted mesher's normal-based facet accent shading (0 = off). |
| `ground_texels_per_m` | float | 16.0 | Virtual texels per world meter for the GPU terrain shader's world-space procedural ground pattern (non-repeating pixel art); ~16 → 0.0625 m texels matching the voxel grid. |
| `lod_streaming_enabled` | bool | true | Use the off-thread `LodStreamer` path (`world/terrain/lod/streamer.py`) instead of the synchronous `ChunkManager.stream_frame`. Scheduling only — mesh output is byte-identical either way. |
| `lod_worker_threads` | int | 4 | Worker thread count for `TerrainLodPool` (off-main-thread chunk meshing, Hard Rule 12). |
| `lod_submit_per_frame` | int | 16 | Max chunk mesh jobs the `LodStreamer` submits per `stream_frame` (dirty-first, then nearest missing). |
| `lod_max_uploads_per_frame` | int | 8 | Max finished meshes uploaded to the scene graph per frame (nearest-first); leftovers wait for the next frame. |

### Lighting fields (from `[lighting]` table)

| Field | Type | Default | Description |
|---|---|---|---|
| `lighting_backend` | str | "gpu" | "gpu" (volumetric radiance cascades, GLSL compute) or "cpu" (legacy baked-vertex sunlight column pass). |
| `light_c0_cells` | int | 96 | Cascade-0 texels per axis (96 → 48 m box at 0.5 m cells). |
| `light_c0_cell_m` | float | 0.5 | Cascade-0 cell edge in meters (0.5 = one terrain voxel). |
| `light_c1_cells` | int | 96 | Cascade-1 texels per axis (96 → 96 m box). |
| `light_c1_cell_m` | float | 1.0 | Cascade-1 cell edge in meters (1.0). |
| `light_c2_cells` | int | 64 | Cascade-2 texels per axis (64 → 256 m box): the coarse FAR cascade that keeps distant terrain lit with low-resolution shadows + GI once it leaves cascade 1, instead of falling back to flat sky ambient. Assembled off-thread like the others. |
| `light_c2_cell_m` | float | 4.0 | Cascade-2 cell edge in meters (4.0). |
| `light_quant_m` | float | 0.0625 | Shading sample-grid quantisation in meters (0.0625 → 8×8×8 visible light pixels per 0.5 m voxel — the pixelated-light look). This is only the visible sample-snap grid; the underlying GI data resolution is the cascade-0 cell (`light_c0_cell_m`), so shrinking this past the cell size yields a finer-but-smoother grid, not more detail. |
| `light_gi_rays` | int | 16 | Ray-marched GI: sphere directions gathered per cell (fibonacci spiral; more = smoother ambient, linearly more inject-time cost). |
| `light_gi_steps` | int | 24 | Max one-cell march steps per GI ray (reach in meters = steps × the cascade cell size). |
| `light_gi_iters` | int | 2 | Gather iterations per inject; ≥2 lets the feedback term carry sky→wall→floor bounce and second-bounce colour bleed. |
| `light_gi_smooth_passes` | int | 1 | Air-masked 3³ box-filter passes applied to the ray-gathered GI component after the gather (0 disables). Completes the gather's 8-phase ray-fan tile (8× the effective ray count) — removes the blotchy patch / colour-confetti gather noise. Contact GI stays voxel-crisp; the filter never crosses solid cells (no leaks). |
| `light_penumbra_deg` | float | 2.5 | Celestial penumbra cone half-angle in degrees: the shadow-edge refinement march jitters its rays inside this cone, so soft shadow edges widen with occluder distance. |
| `light_bounce_strength` | float | 0.7 | [0,1] albedo-tinted bounce gain (first bounce at inject + gather feedback). |
| `light_tree_trunk_occ` | float | 0.85 | [0,1] occupancy a tree trunk splats into the cascade volumes (lighting/occluders.py) — near-opaque wood. |
| `light_tree_canopy_extinction_gain` | float | 1.0 | Multiplier on each tree's leaf-derived per-METER canopy extinction: transmittance through X m of crown centre = exp(-sigma·gain·X), the same at every cascade cell size. 1.0 = the species' real leaf density; raise for darker shade, lower for airier canopies; 0 disables canopy occlusion. |
| `light_ao_strength` | float | 0.6 | [0,1] strength of occupancy-based ambient occlusion at surfaces. |
| `light_max_point_lights` | int | 64 | Max simultaneous point/area lights uploaded to the GPU. |
| `light_exposure` | float | 0.9 | Tonemap exposure multiplier for the HDR lighting pipeline. |
| `exposure_adapt_enabled` | bool | True | Auto-exposure (eye adaptation) on/off. |
| `exposure_min` | float | 0.55 | Clamp minimum of the adaptation multiplier (× light_exposure). |
| `exposure_max` | float | 5.0 | Clamp maximum of the adaptation multiplier. |
| `exposure_key` | float | 0.18 | Metering key: target multiplier = key / scene luminance (0.18 ≈ photographic middle gray; noon open field ≈ 1.0×). |
| `exposure_tau_dark_s` | float | 4.0 | Adaptation time constant entering darkness in seconds (slow, like real eyes). |
| `exposure_tau_bright_s` | float | 0.7 | Adaptation time constant entering bright light in seconds (fast stop-down). |

### Fog fields (from `[fog]` table)

| Field | Type | Default | Description |
|---|---|---|---|
| `fog_enabled` | bool | True | Volumetric froxel fog + god rays on/off. |
| `fog_froxels_x` | int | 160 | Froxel grid resolution X (screen-aligned). |
| `fog_froxels_y` | int | 90 | Froxel grid resolution Y (screen-aligned). |
| `fog_froxels_z` | int | 64 | Froxel grid resolution Z (exponential depth slices). |
| `fog_far_m` | float | 192.0 | Far range of the froxel volume in meters. |
| `fog_anisotropy` | float | 0.55 | Henyey-Greenstein g for sun scattering ([0,1); higher = stronger forward god rays). |

### Grass fields (from `[grass]` table)

| Field | Type | Default | Description |
|---|---|---|---|
| `grass_density_per_m2` | float | 12.0 | Default blade tufts per square meter for grass volumes lacking a `density` param. |
| `grass_blade_height_m` | float | 0.6 | Unscaled tuft height in meters (per-blade jitter scales it 0.7–1.3×). |
| `grass_fade_start_m` | float | 60.0 | Camera distance where blades begin shrinking away (meters). |
| `grass_fade_end_m` | float | 90.0 | Camera distance where blades are fully gone (meters). |
| `grass_max_instances` | int | 200000 | Hard cap on instances per grass volume. |

### Flora fields (from `[flora]` table, prefix `flora_`)

GPU-instanced wildflower sprites (`world/flora_renderer.py`) inside `"flowers"` zone volumes. Density is overridable per volume via `params["density"]`. (Bushes and trees are 3-D meshes now — see the [trees] table below.)

| Field | Type | Default | Description |
|---|---|---|---|
| `flora_flower_density_per_m2` | float | 1.5 | Wildflowers per m² (1.5). |
| `flora_flower_height_m` | float | 0.45 | Flower sprite height (0.45 m). |
| `flora_flower_fade_start_m` | float | 60.0 | Flowers fade like grass (60 m). |
| `flora_flower_fade_end_m` | float | 90.0 | Fully gone (90 m). |
| `flora_flower_max_instances` | int | 50000 | Per-volume cap (50 000). |

### 3-D tree/bush fields (from `[trees]` table, prefixes `tree_`/`bush_`)

Instanced 3-D flora meshes (`world/tree_renderer.py`) inside `"trees"` / `"bushes"` zone volumes, placed CPU-side on a jittered grid (`zones/tree_placement.py`). Near distance draws the variant mesh; past the mesh fade window the renderer crossfades to an instanced billboard impostor, which itself fades out at the impostor window — billboards are LOD only.

| Field | Type | Default | Description |
|---|---|---|---|
| `tree_density_per_m2` | float | 0.02 | Trees per m² (0.02 = 1 per 50 m²; `params["density"]` overrides per volume). |
| `tree_min_spacing_m` | float | 3.0 | Placement grid floor: no two trunks closer than ≈0.3× this (3.0 m). |
| `tree_max_instances` | int | 2000 | Per-volume cap (2 000). |
| `tree_mesh_fade_start_m` | float | 110.0 | 3-D mesh shrink-away window start (110 m); the impostor fades IN here. |
| `tree_mesh_fade_end_m` | float | 140.0 | 3-D mesh shrink-away window end (140 m). |
| `tree_impostor_fade_start_m` | float | 300.0 | Impostor shrink-away window start (300 m, the old sprite landmark range). |
| `tree_impostor_fade_end_m` | float | 380.0 | Impostor shrink-away window end (380 m). |
| `tree_default_species` | str | "tree_gnarled_oak" | Species def when a volume names none. |
| `bush_density_per_m2` | float | 0.08 | Bushes per m² (0.08). |
| `bush_min_spacing_m` | float | 1.2 | Bush spacing floor (1.2 m). |
| `bush_max_instances` | int | 5000 | Per-volume cap (5 000). |
| `bush_mesh_fade_start_m` | float | 60.0 | Bush mesh window start (60 m). |
| `bush_mesh_fade_end_m` | float | 80.0 | Bush mesh window end (80 m). |
| `bush_impostor_fade_start_m` | float | 120.0 | Bush impostor window start (120 m). |
| `bush_impostor_fade_end_m` | float | 150.0 | Bush impostor window end (150 m). |
| `bush_default_species` | str | "bush_scrub" | Default bush species. |

### Building fields (from `[buildings]` table, prefix `building_`)

Free-form floorplan buildings (`fire_engine/buildings/`): per-storey 2-D plans of segment/arc walls with thickness and parametric openings. All distances in meters.

| Field | Type | Default | Description |
|---|---|---|---|
| `building_default_storey_height_m` | float | 3.0 | Floor-to-floor height when a storey gets no explicit value (3.0). |
| `building_default_wall_thickness_m` | float | 0.3 | Wall thickness when a wall gets no explicit value (0.3). |
| `building_slab_thickness_m` | float | 0.2 | Floor/ceiling/roof slab thickness (0.2). |
| `building_foundation_depth_m` | float | 0.5 | Foundation slab depth below building-local z=0 (0.5). |
| `building_arc_segments_per_quarter` | int | 8 | Chords per quarter circle when tessellating arc walls (8); used identically by meshing and room detection so polygons agree. |
| `building_snap_eps_m` | float | 0.01 | Endpoint-snap tolerance for room auto-detection (0.01 = 1 cm). |

### Wind-field fields (from `[wind]` table, prefix `wind_`)

These drive the spatially-varying wind field (`fire_engine/world/wind/`): a 64×64-cell × 4 m (256 m) player-centred grid of horizontal wind velocity, summed from ~12 seeded spectral gust modes that advect downwind, plus an analytic vertical boundary-layer profile. All distances meters, speeds m/s, frequencies rad/s, times seconds.

| Field | Type | Default | Description |
|---|---|---|---|
| `wind_time_scale` | float | 1.0 | Wind-clock rate in seconds per REAL second (1.0). Gust travel/oscillation are an aesthetic real-time effect, deliberately independent of the game-clock timescale (`Clock.game_time_scale`). |
| `wind_cells` | int | 64 | Grid cells per axis (64 → 256 m region at 4 m cells). |
| `wind_cell_m` | float | 4.0 | Cell edge in meters (4.0). |
| `wind_snap_cells` | int | 8 | Origin snap granularity in cells for the recenter window (8 → snaps to 32 m). |
| `wind_margin_cells` | int | 8 | Recenter hysteresis: re-snap only when the player drifts past this many cells from the region centre (8 → 32 m band). |
| `wind_gust_modes` | int | 12 | Number of spectral Brownian-band gust modes summed per cell (12). |
| `wind_gust_wavelen_min` | float | 20.0 | Gust spatial wavelength minimum (m). |
| `wind_gust_wavelen_max` | float | 120.0 | Gust spatial wavelength maximum (m; big slow gusts dominate). |
| `wind_gust_omega_min` | float | 0.15 | Intrinsic temporal frequency minimum (rad/s). |
| `wind_gust_omega_max` | float | 0.8 | Intrinsic temporal frequency maximum (rad/s) — the gust's own pulsing on top of downwind advection. |
| `wind_gust_base` | float | 0.6 | Base gust amplitude gain (calm air). |
| `wind_gust_storm_gain` | float | 1.4 | Extra gust amplitude per unit storminess: storms gust much harder. |
| `wind_storm_freq_gain` | float | 0.8 | Temporal-frequency boost per unit storminess: storms are choppier, not just stronger. |
| `wind_speed_ref` | float | 8.0 | Reference mean wind speed (m/s) at which the gust gain reaches full strength. |
| `wind_turb_base` | float | 0.2 | Base turbulence channel value, calm. |
| `wind_turb_storm_gain` | float | 1.0 | Turbulence increase per unit storminess. |
| `wind_shear` | float | 0.18 | Vertical-profile shear exponent (0.18): the power-law boundary-layer wind shear. |
| `wind_profile_z_ref` | float | 10.0 | Reference height (m) where the vertical profile reaches 1.0. |
| `wind_profile_floor` | float | 0.35 | Minimum profile multiplier at ground level: wind never fully stops at z=ground. |
| `wind_profile_cap` | float | 1.6 | Maximum profile multiplier high up. |
| `wind_layer_m` | float | 8.0 | Vertical band (m) above ground over which the venturi solver folds terrain occupancy (WP2). |
| `wind_venturi_iters` | int | 8 | Venturi flux-relaxation iterations (WP2). |
| `wind_venturi_max` | float | 2.2 | Clamp on venturi speed-up multiplier (WP2). |
| `wind_deflect_gain` | float | 0.15 | Venturi sideways-deflection gain (WP2). |
| `wind_updraft_gain` | float | 0.4 | Analytic obstacle-updraft gain (WP2). |
| `wind_mote_count` | int | 1500 | Dust/pollen mote instance count (WP4). |
| `wind_mote_box_m` | float | 24.0 | Camera-anchored mote lattice cell size in meters (WP4). |
| `wind_mote_size_m` | float | 0.04 | Mote billboard size in meters (WP4). |
| `wind_mote_life_s` | float | 6.0 | Mote looping lifetime in seconds (WP4). |
| `wind_leaf_density_per_m2` | float | 0.15 | Leaf-litter instances per m² of a "trees" zone volume (WP4). |
| `wind_leaf_size_m` | float | 0.12 | Leaf billboard size in meters (WP4). |
| `wind_leaf_max_instances` | int | 20000 | Hard cap on leaf instances per volume (WP4). |

### Rain-cover heightmap fields (from `[rain]` table, prefix `rain_`)

Drive the top-down cover heightmap (`terrain/rain_cover.py`) that the M6 volumetric rain renderer samples to discard rain under roofs/overhangs.

| Field | Type | Default | Description |
|---|---|---|---|
| `rain_cover_cells` | int | 256 | Columns per axis of the player-centred cover window (256 → a 256 m square at 1 m cells). |
| `rain_cover_cell_m` | float | 1.0 | Column edge in meters (1.0 = light-cell size). |
| `rain_cover_budget_columns` | int | 4 | Chunk-columns the renderer refolds per refresh so a full rebuild amortises over frames. |

### Weather simulation fields (from `[weather]` table)

| Field | Type | Default | Description |
|---|---|---|---|
| `weather_synoptic_components` | int | 4 | Vector sinusoids in the synoptic flow W(t). |
| `weather_synoptic_speed_min_ms` | float | 1.5 | Synoptic speed band minimum (m/s). |
| `weather_synoptic_speed_max_ms` | float | 11.0 | Synoptic speed band maximum (m/s). |
| `weather_synoptic_period_min_h` | float | 2.5 | Sinusoid period band minimum (game h). |
| `weather_synoptic_period_max_h` | float | 14.0 | Sinusoid period band maximum (game h). |
| `weather_domain_m` | float | 6000.0 | Half-extent of the spawn square (±m around origin). |
| `weather_spawn_slots_per_day` | int | 8 | Candidate cell spawn slots per day. |
| `weather_cell_radius_min_m` | float | 350.0 | Cell footprint radius minimum (m). |
| `weather_cell_radius_max_m` | float | 1200.0 | Cell footprint radius maximum (m). |
| `weather_cell_duration_min_s` | float | 2400.0 | Cell lifetime minimum (game s; 40 min). |
| `weather_cell_duration_max_s` | float | 10800.0 | Cell lifetime maximum (game s; 3 h). |
| `weather_storm_wind_max_ms` | float | 9.0 | Extra gust a THUNDERSTORM adds at its core (m/s). |
| `weather_fog_max_density` | float | 0.028 | Cap on FOG_BANK fog coefficient (1/m). |
| `weather_temp_mean_c` | float | 12.0 | Daily mean air temperature (°C). |
| `weather_temp_amp_c` | float | 8.0 | Daily temperature swing amplitude (°C). |
| `weather_map_cells` | int | 128 | Raster resolution of the weather map (square, N×N texels). |
| `weather_map_cell_m` | float | 24.0 | Raster texel size (m) → 128×24 ≈ 3 km span. |
| `weather_wetness_tau_s` | float | 3600.0 | Wetness decay time constant (game s). |
| `weather_wetness_step_s` | float | 600.0 | Quadrature step into the past (game s). |
| `weather_wetness_samples` | int | 12 | Number of past rain samples (window = step·samples). |
| `weather_humidity_base_min` | float | 0.35 | Seeded per-day calm-air humidity baseline band minimum. |
| `weather_humidity_base_max` | float | 0.65 | Seeded per-day calm-air humidity baseline band maximum. |
| `weather_humidity_rain_gain` | float | 1.00 | Humidity added per unit recent rain (0–1). |
| `weather_humidity_wetness_gain` | float | 0.30 | Humidity added per unit ground wetness (0–1). |
| `weather_humidity_recent_tau_s` | float | 18000.0 | Recent-rain decay time constant (game s; 5 h). |
| `weather_humidity_recent_step_s` | float | 1800.0 | Recent-rain quadrature step into past (game s). |
| `weather_humidity_recent_samples` | int | 12 | Recent-rain samples (window = step·samples = 6 h). |
| `weather_fog_emergent_max` | float | 0.022 | Max emergent fog coefficient at full condensation (1/m). |
| `weather_fog_sat_ref_c` | float | 5.0 | Reference temperature for saturation humidity (°C). |
| `weather_fog_sat_base` | float | 0.63 | Saturation humidity at the reference temperature (0–1). |
| `weather_fog_sat_slope_per_c` | float | 0.011 | Saturation humidity rise per °C above the reference. |
| `weather_fog_condense_band` | float | 0.10 | Humidity overshoot over saturation for full condensation. |
| `weather_fog_wind_full_ms` | float | 1.0 | Full fog at/below this wind speed (m/s). |
| `weather_fog_wind_none_ms` | float | 3.0 | No emergent fog at/above this wind speed (m/s). |
| `cloud_genera_high_alt_m` | float | 1400.0 | High (cirrus) band base altitude (m). |
| `cloud_genera_high_thick_m` | float | 120.0 | High band thickness (m; thin veil). |
| `cloud_genera_mid_alt_m` | float | 850.0 | Mid (alto-) band base altitude (m). |
| `cloud_genera_mid_thick_m` | float | 220.0 | Mid band thickness (m). |
| `cloud_genera_low_alt_m` | float | 500.0 | Low (cumulus/stratus) band base altitude (m). |
| `cloud_genera_low_thick_m` | float | 400.0 | Low band thickness (m; storms deepen it). |
| `cloud_genera_high_cov_floor` | float | 0.06 | Cirrus present even in fair weather (residual floor). |
| `cloud_genera_high_cov_weight` | float | 0.35 | Extra high-band coverage per unit sampled coverage. |
| `cloud_genera_high_density` | float | 0.30 | Ice cloud is thin: cap on high-band opacity. |
| `cloud_genera_mid_cov_weight` | float | 0.60 | Mid-band coverage per unit sampled coverage. |
| `cloud_genera_high_detail_scale` | float | 0.45 | High band: stretched, smooth streaks. |
| `cloud_genera_mid_detail_scale` | float | 0.85 | Mid band: moderate lumpiness. |
| `cloud_genera_low_detail_scale` | float | 1.30 | Low band: billowy cumulus detail. |
| `weather_summon_upwind_m` | float | 2500.0 | How far upwind a summoned cell spawns (m). |
| `weather_summon_rain_radius_m` | float | 700.0 | Summoned rainstorm footprint radius (m). |
| `weather_summon_rain_duration_s` | float | 5400.0 | Summoned rainstorm lifetime (game s; 90 min). |
| `weather_summon_rain_intensity` | float | 0.85 | Summoned rainstorm peak intensity (0–1). |
| `weather_summon_storm_radius_m` | float | 950.0 | Summoned thunderstorm footprint radius (m). |
| `weather_summon_storm_duration_s` | float | 6000.0 | Summoned thunderstorm lifetime (game s; 100 min). |
| `weather_summon_storm_intensity` | float | 1.0 | Summoned thunderstorm peak intensity (0–1). |
| `weather_summon_fog_radius_m` | float | 600.0 | Summoned fog-bank footprint radius (m). |
| `weather_summon_fog_duration_s` | float | 7200.0 | Summoned fog-bank lifetime (game s; 2 h). |
| `weather_summon_fog_intensity` | float | 0.9 | Summoned fog-bank peak intensity (0–1). |
| `weather_gustfront_range_m` | float | 600.0 | Register a gust-front modifier when a cell's leading edge is within this range of the player (m). |
| `weather_gustfront_strength_ms` | float | 7.0 | Peak added wind speed along the summoned gust front (m/s). |
| `weather_gustfront_width_m` | float | 80.0 | Gust-front band half-width (Gaussian sigma, m). |
| `weather_lightning_strikes_per_min` | float | 2.5 | Peak strike rate per cell at full intensity (thinned by cell intensity). |
| `weather_lightning_cloud_base_m` | float | 220.0 | Cloud-base height above ground the bolt starts at (m). |
| `weather_lightning_ground_z_m` | float | 8.0 | Fallback ground-plane world Z when no cover heightmap (m; == ground_height_m). |
| `bolt_step_len_min_m` | float | 5.0 | Stepped-leader step length minimum (m). |
| `bolt_step_len_max_m` | float | 15.0 | Stepped-leader step length maximum (m). |
| `bolt_cone_deg` | float | 38.0 | Half-angle of the downward candidate-direction fan (deg). |
| `bolt_candidates` | int | 7 | K candidate directions fanned each step. |
| `bolt_softmax_temp` | float | 0.35 | Softmax temperature for the seeded direction pick. |
| `bolt_branch_prob` | float | 0.12 | Per-step probability a side branch spawns. |
| `bolt_max_steps` | int | 400 | Hard cap on leader steps (the one bounded loop). |
| `bolt_noise_gain` | float | 0.6 | Weight of the seeded value-noise "air resistance" in the score. |
| `bolt_repulsion_gain` | float | 0.45 | Weight of repulsion from the existing channel in the score. |
| `bolt_branch_max_depth` | int | 3 | Branches stop spawning sub-branches past this depth. |

### Graphics-quality fields (from `[graphics]` table, prefix `gfx_`)

These drive the HDR post-processing pipeline and volumetric clouds so the look can be dialed down (or off) on weak GPUs. Pick a `preset` (off/low/medium/high) in `[graphics]`; any explicit `gfx_*` key in the same table overrides that preset's value (see `resolve_graphics_preset`). The dataclass defaults equal the `"high"` preset.

| Field | Type | Default | Description |
|---|---|---|---|
| `gfx_preset` | str | "high" | Which preset produced these values (off/low/medium/high; informational). |
| `gfx_post_process` | bool | True | Master switch for the offscreen HDR buffer + post chain. False ⇒ shaders tonemap internally (legacy path), no bloom/flare. |
| `gfx_hdr_format` | str | "rgba16f" | Scene buffer format: "rgba16f" (float HDR) or "rgba8" (LDR fallback for GPUs lacking float render targets). |
| `gfx_render_scale` | float | 1.0 | Internal render resolution scale (1.0 = full; 0.75 = render at 75% then upscale). |
| `gfx_bloom` | bool | True | Bloom on/off. |
| `gfx_bloom_mips` | int | 5 | Bloom downsample pyramid depth (more = wider, softer glow; costs more). |
| `gfx_bloom_threshold` | float | 1.0 | Luminance above which pixels bloom (HDR). |
| `gfx_bloom_knee` | float | 0.5 | Soft-knee width below the threshold. |
| `gfx_bloom_strength` | float | 0.06 | Bloom contribution added back at composite. |
| `gfx_fxaa` | bool | True | Cheap post anti-aliasing pass. |
| `gfx_lens_flare` | bool | True | Screen-space lens flare when looking near the (unoccluded) sun. |
| `gfx_clouds` | bool | True | Volumetric raymarched clouds on/off. |
| `gfx_cloud_steps` | int | 96 | Primary raymarch sample count (quality). |
| `gfx_cloud_light_steps` | int | 8 | Sun light-march steps per sample (self-shadow quality; dominant cost). |
| `gfx_cloud_resolution_scale` | float | 1.0 | Cloud pass resolution (0.5 = half-res, the biggest perf win on an iGPU). |
| `gfx_cloud_max_dist_m` | float | 6000.0 | Far raymarch distance for clouds (meters). |
| `gfx_weather_map` | bool | True | Upload the spatial weather-map texture and sample it in the cloud raymarch (spatial coverage/density/precip). Off ⇒ the cloud shader uses the flat ambient scalars (the pre-M4 look); the renderer skips the re-raster + upload entirely. Master kill switch for the M4 GPU weather contract. |
| `gfx_cloud_virga` | bool | True | Gray rain shafts hanging below storm-cloud bases (driven by the weather map's precip channel). Requires `gfx_weather_map`; off ⇒ storm bases still lower/darken but no virga streaks. |
| `gfx_cloud_genera` | bool | True | Render layered WMO cloud genera (M9). Requires `gfx_weather_map`; off ⇒ the single cloud slab (the pre-M9 look). |
| `gfx_god_rays` | bool | True | Screen-space crepuscular rays through clouds. |
| `gfx_god_ray_samples` | int | 32 | Radial sample count for god rays. |
| `gfx_rain_mode` | str | "particles" | Volumetric rain mode: "off", "cylinders" (cheap scrolled shells — the low preset), or "particles" (GPU-instanced falling streaks — medium+). Both rendered modes honour the rain-cover heightmap cull and the weather-map precip footprint. |
| `gfx_rain_particles` | int | 12000 | Instanced rain-streak count in "particles" mode. |
| `gfx_rain_occlusion` | bool | True | Sample the rain-cover heightmap to discard streaks under cover (all presets, all modes; false ⇒ rain everywhere, the old look). |
| `gfx_lightning_bolts` | bool | True | Render procedural lightning bolts (M7): camera-facing stepped-leader ribbons, a two-phase flash, a transient scene light and a sky/cloud flash pulse, on a strike. On for low+ presets, off for "off". The headless strike schedule + ThunderEvents still run when this is off (audio/gameplay only see no drawn bolt). |
| `gfx_foliage_shadow_refine` | bool | True | Per-fragment celestial-shadow refinement march on foliage (grass/flora/trees/impostors; the lit_surface.glsl `u_refine` gate). Terrain always refines; turning this off keeps foliage on the cheap trilinear cascade shadows (iGPU relief). |
| `gfx_god_ray_strength` | float | 0.4 | God-ray contribution added at composite (aesthetic, preset-independent). |
| `gfx_lens_flare_strength` | float | 0.055 | Lens-flare contribution at composite (aesthetic, preset-independent). |
| `gfx_lens_flare_threshold` | float | 4.0 | HDR luminance the flare isolates as "the sun" (aesthetic, preset-independent). |
| `gfx_tonemap_hue_preserve` | float | 0.8 | [0,1] blend toward the hue-preserving tonemap (aesthetic, preset-independent). |
| `gfx_sun_disc_intensity` | float | 45.0 | HDR gain on the sun disc (aesthetic, preset-independent). |
| `gfx_sun_halo_intensity` | float | 1.8 | HDR gain on the forward-Mie glow haloing the sun (aesthetic, preset-independent). |
| `gfx_sun_min_brightness` | float | 0.25 | Floor on the sun disc/halo transmittance so a low sun still reads bright instead of fading out (aesthetic, preset-independent). |
| `gfx_sky_inscatter_scale` | float | 0.9 | Multiplier on the scattered-sky radiance (aesthetic, preset-independent). |

### Profiler fields (from `[profiler]` table, prefix `profiler_`)

Drive the frame profiler (`fire_engine/core/profiler.py`) + its overlay / PStats bridge. Everything is observational — never affects the sim or saves.

| Field | Type | Default | Description |
|---|---|---|---|
| `profiler_enabled` | bool | False | Master switch. False ⇒ scopes are no-ops, no ring buffer / overlay / PStats objects are constructed (truly free). |
| `profiler_overlay_enabled` | bool | True | Build the in-game F3 overlay (only when `profiler_enabled`). |
| `profiler_frame_budget_ms` | float | 5.0 | Per-frame budget in ms (200 FPS = 5.0). |
| `profiler_history_frames` | int | 1024 | Ring-buffer length (percentile span). |
| `profiler_hitch_abs_ms` | float | 8.0 | Absolute hitch threshold floor (ms). |
| `profiler_hitch_rel_mult` | float | 1.5 | Hitch when ms > max(abs, mult × median). |
| `profiler_hitch_window` | int | 120 | Frames the rolling median spans. |
| `profiler_max_scopes` | int | 64 | Preallocated per-scope columns. |
| `profiler_max_counters` | int | 32 | Preallocated per-counter columns. |
| `profiler_recent_hitches` | int | 16 | Recent hitches kept in the snapshot. |
| `profiler_overlay_graph_frames` | int | 240 | Frames drawn in the overlay graph. |
| `profiler_overlay_hz` | float | 8.0 | Overlay refresh rate (Hz). |
| `profiler_snapshot_enabled` | bool | False | Periodically write snapshot JSON. |
| `profiler_snapshot_path` | str | "profiling/latest.json" | Snapshot JSON path (AI-agent contract). |
| `profiler_snapshot_interval_s` | float | 1.0 | Seconds between snapshot writes. |
| `profiler_pstats` | bool | False | Connect to a PStats server at boot. |
