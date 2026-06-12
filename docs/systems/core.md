# core — System Doc
keywords: vec3, quat, quaternion, math3d, event bus, eventbus, publish, subscribe, drain, rng, seed, for_domain, config, clock, fixed_update, lod, lodpolicy, logging, math, rotation, euler, hpr, slerp, chunk loaded, game day, terrain edited, weather changed, world seed, determinism, blake2b, float32, z-up, forward, right, up, meters, radians, fixed_dt, spiral of death, saveable, get_state, set_state, game_time_scale, time scale, sky config, cloud altitude, star count, shader_source, load_glsl, glsl, shader file, vert, frag, comp, syntax highlighting, graphics, graphics quality, preset, gfx, post process, postprocess, hdr, bloom, fxaa, lens flare, volumetric clouds, cloud quality, god rays, render scale, quality preset, resolve_graphics_preset, GRAPHICS_PRESETS, off low medium high

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

`core/` deliberately does NOT: render anything, touch the Panda3D scene graph, know about terrain chunks, or hold game-world state.  It is stateless except for the global world seed in `rng.py`.

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
| `WeatherChangedEvent(previous, current, day)` | Discrete weather state changed (`WeatherType.value` strings, e.g. `"clear"` → `"rain"`).  Published by `fire_engine.sky.WeatherSystem` via `publish_deferred`. |

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
| `load_glsl(anchor: str, name: str) -> str` | Read GLSL source `name` from the `shaders/` directory beside `anchor` (the caller's `__file__`). Used by `world/grass_shaders.py`, `world/sky_shaders.py`, `world/terrain_shader.py` and `lighting/glsl.py` to load their `.vert`/`.frag`/`.comp` sidecars (real shader files, for editor syntax highlighting + LSP) and re-export them under the original string constants. Panda3d-free — reading a text file is callable from any layer. |

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
