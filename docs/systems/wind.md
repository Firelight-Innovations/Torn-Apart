# wind — System Doc
keywords: wind, gust, breeze, storm, brownian, turbulence, venturi, motes, leaves, particles, flag, cloth, advection, spectral, gust front, wind field, vertical profile, boundary layer

> Status: **WP1+WP2+WP3 shipped** (headless wind core + config; terrain venturi/worker; world-side render component, GPU upload, grass rebind, wiring). WP4 (motes/leaves) is a seam. WP5 finalizes this doc. Do not delete the pending markers until the matching package lands.

## Role
`fire_engine/wind/` is the single source of truth for everything wind-driven: a player-centred, time-evolving **2.5-D wind velocity field**. A 2-D horizontal grid (64×64 cells × 4 m = 256 m region) of wind velocity is summed from ~12 seeded spectral "Brownian-band" gust modes whose phases advance with game time and **advect downwind** (so gust bands visibly travel across a field), plus an analytic vertical boundary-layer profile for height. It scales with weather (storms = stronger, choppier, gustier), is CPU-sampleable for physics and future audio, and exposes seams for terrain venturi funneling (WP2), GPU upload + grass/particle consumption (WP3/WP4), and localized gust fronts from a future volumetric-weather system (the modifier seam).

It is **headless** — numpy + `core` only, **no panda3d** (the upload/render half lives in `world/`). It deliberately does NOT: render anything, hold any Saveable state (the field is a pure function of seed/time/weather/player-cell, so it costs **zero save bytes**), integrate any per-frame random walk, or own event wiring (the renderer forwards terrain edits into `update()`).

## Public API
Exports from `fire_engine/wind/__init__.py`:

- `WindField(config, worker=None)` — the field. `update(dt, game_time, sky_state, player_pos, chunks=None)` once per frame (sub-ms, main thread); `sample(positions (N,3)) -> (N,3)` m/s, vectorized; `snapshot` property; `add_modifier(m)` / `remove_modifier(m)`.
- `WindSnapshot` — frozen atomically-published field state: `field` (`float32 (cells, cells, 4)` `[x, y]`: vx, vy, turb, reserved), `origin_m`, `cell_m`, `cells`, `game_time`.
- `WindModifier` — in-place modifier `Protocol`: `apply(X, Y, t, vx, vy, turb) -> None`. The volumetric-weather seam.
- `GustFront(seed_key, direction, speed, strength, width_m, period_m=400.0, turb_gain=0.6)` — a working moving-line-front modifier; pure function of `(seed_key, t)`.
- `pack_wind_field(snap) -> bytes` — pack a snapshot to Panda3D 2-D-texture RAM bytes (float16, row-major `(y, x)`, BGRA: B=turb, G=vy, R=vx, A=horizontal speed). Mirrors `lighting/volume.pack_volume`.
- `vertical_profile(z, z_ground, cfg) -> np.ndarray` — analytic boundary-layer wind-speed multiplier `clamp((max(z-z_ground,0)/z_ref)**shear, floor, cap)`.
- `VenturiWorker()` — off-thread terrain-funneling solver, a structural mirror of `lighting/assembly_worker.CascadeAssemblyWorker`: daemon thread `"WindVenturiWorker"`, in/out `queue.Queue`, idempotent `start()`, `submit(job)`, non-blocking `drain_results()`, `pending()`, `stop(join=True, timeout=2.0)` (None sentinel). A solve that raises logs + posts a valid **identity** result so the consumer never starves.
- `VenturiJob(origin_cell, cells, cell_m, chunk_size, voxel_size, ground_band, materials, venturi_iters, venturi_max, deflect_gain, seq)` / `VenturiResult(origin_cell, speedup (cells,cells), deflect (cells,cells,2), seq)` — frozen dataclasses; `materials` is a `coord -> uint8 (S,S,S)` snapshot (references, not copies).
- `solve_venturi(job) -> VenturiResult` — the **pure** on-thread solve (called by the worker and inline in tests). Folds the chunk terrain into a per-cell column solid fraction over the z-band `[ground, ground+wind_layer_m]`, relaxes it into a speed-up + sideways deflection. Deterministic, no RNG.

Internal modules: `gusts.py` (`build_modes(cfg)` + `eval_gusts(modes, X, Y, t_eff, mean)`), `region.py` (`WindRegion` recenter window), `venturi.py` (`column_solid_fraction` + `solve_venturi`), `worker.py` (`VenturiWorker`).

## Imports Allowed
`fire_engine.core` (config, rng) and numpy only. **No panda3d, no `direct`** (hard rule — a test in `tests/test_wind.py` AST-parses every `wind/*.py` and fails on a panda3d/direct import). Does NOT import `fire_engine.sky` — the weather input is duck-typed (see Units & Invariants).

## Events
Published: none. The wind field is a pure function and never emits events.
Subscribed: none. (WP3's render component in `world/` owns event wiring — it forwards `TerrainEditedEvent`/`ChunkLoadedEvent` by passing `chunks` + a dirty flag into `update()`, keeping `wind/` bus-free and pure.)

## Units & Invariants
- World space **meters**, Z-up. Velocities **m/s**. `turb` channel dimensionless (~0..3). Frequencies rad/s, wavevectors rad/m, time seconds.
- `field` array is indexed `[x, y]` (matching `WindRegion.X/Y` meshgrid `ij` order). Channels: `vx, vy, turb, reserved`.
- `origin_m` is the world XY of cell `(0,0)`'s **corner** (the texel-(0,0)-corner convention the GPU binds as `u_wind_origin`). Cell `(i,j)`'s **centre** is at `(origin_cell + (i,j) + 0.5) * cell_m`.
- **Determinism:** same `world_seed` + `game_time` + `sky_state` + player cell ⇒ bit-identical `WindSnapshot` (in-process and cross-process). All randomness via `for_domain("wind", "gusts")` (drawn once at construction). **No Saveable** — zero save bytes by construction.
- **sky_state is duck-typed:** `update()` reads `.wind_dir` (unit XY), `.wind_speed` (m/s), `.rain_intensity`, `.cloud_coverage`, `.cloud_density` (all 0..1). `sky_state=None` ⇒ calm defaults (light +X breeze) so headless tests need no sky package.
- Internal field is **float32**; **float16** only at `pack_wind_field` time.
- `vertical_profile` is monotone non-decreasing in z between `floor` and `cap`; never returns below `floor` (wind never fully dies at ground level) nor above `cap`.
- `sample()` clamps out-of-region points to the nearest edge value; never returns NaN; `vz` is the analytic **obstacle updraft** (`bilinear(updraft_gain_grid) × local horizontal speed / profile`, where `updraft_gain_grid = wind_updraft_gain · clip(speedup−1, 0, None)`), so motes/leaves rise over a windward constriction. `vz == 0` everywhere when there is no venturi worker / no committed result.
- **Venturi units/invariants:** `speedup` is a multiplier in `[1, wind_venturi_max]` (1 = no funneling); `deflect` is an additive m/s push scaled by `|mean wind|`; both are aligned to the field's `[x, y]` cell layout and a specific `origin_cell`. A wind cell of `cell_m` (4 m) covers `cell_m/voxel_size` (8) voxel columns; the solve folds occupancy over the z-band `[ground, ground+wind_layer_m]`. Cells over unloaded terrain are fully **open** (never fabricate a wall). `solve_venturi` is a pure function of its job — **no RNG**.

## Examples
Physics push (a ball on a plane gets shoved by a gust):
```python
import numpy as np
from fire_engine.core import load_config, set_world_seed
from fire_engine.wind import WindField

cfg = load_config(); set_world_seed(cfg.world_seed)
field = WindField(cfg)
field.update(dt, game_time, sky_system.state, camera_world_pos)   # once per frame
v = field.sample(ball_pos[None])[0]            # (vx, vy, 0) m/s at the ball
ball_vel += (v - ball_vel) * drag * dt
```
Future flag/cloth shader (4-liner, once WP3 uploads `u_wind_tex`):
```glsl
vec2 uv = (world_xy - u_wind_origin) / (u_wind_cell_m * u_wind_cells);
vec4 w  = texture(u_wind_tex, uv);             // R=vx G=vy B=turb A=speed
vec2 dir = (w.a > 1e-3) ? w.xy / w.a : u_wind_dir;
pos.xy  += dir * (lean_amount * w.a);
```
Future procedural wind audio: `speed = hypot(*field.sample(camera_pos[None])[0][:2])` drives a noise-band gain/pitch each frame.

Registering a localized gust front (the volumetric-weather seam):
```python
from fire_engine.wind import GustFront
field.add_modifier(GustFront(("storm", 1), direction=(1, 0), speed=14.0,
                             strength=8.0, width_m=24.0))
```

## Gotchas
- **Committed-origin discipline (WP3):** the GPU's `u_wind_origin` must be refreshed only together with a texture upload — never on a bare recenter — or the texels and origin disagree for a frame. (Mirror `lighting/gpu.py::_commit_assembly_result`.)
- **fp16 pack layout is PINNED:** `pack_wind_field` transposes `[x,y]→[y,x]` (row-major) and swaps RGBA→BGRA (B=turb, G=vy, R=vx, A=speed). A test asserts it. Changing the transpose/channel map means updating the GPU uniform contract and the shader decode together.
- **`t_eff` frequency chirp:** `t_eff = game_time * (1 + storm_freq_gain*storminess)` slightly chirps gust frequency while storminess changes. This is intentional and sub-perceptual (storminess only moves over the sky's 20-game-minute blends). It is kept as a closed form rather than an accumulated integral precisely to preserve determinism / zero-byte saves.
- **No `np.roll` in the venturi solver:** roll wraps the grid edges and would leak crowding/flux across the world — `venturi.py` uses edge-replicate padded slicing for every neighbour mean and box blur.
- **Venturi origin-match discard rule:** a `VenturiResult` is committed **only** when its `origin_cell` still equals the region's current origin. A result solved for a window the player has since left is *discarded* (never index-shifted to the new window); the field re-submits a fresh job on every recenter and applies the **identity** correction in the meantime. This keeps the applied `speedup`/`deflect`/`updraft` grids and the cells they scale perfectly aligned with zero index math, at the cost of a 1–2-frame identity flash right after a recenter (sub-perceptual — the gust field itself is continuous).
- **Venturi model deviation:** the plan sketched `flux = passw·neighbour-mean + (1−passw)·flux` with `speedup = flux/passw`. That is a Laplace smoothing that relaxes an open gap *toward its zero-flux walls* → `speedup ≤ 1` (no acceleration), so it cannot meet the gap-`>1.3` acceptance. `venturi.py` instead diffuses the **solid** field outward (blockage crowding) and accelerates open cells sitting in a crowded neighbourhood — same ingredients (occupancy fold → bounded padded Jacobi → speed-up → 3×3 blur → openness-gradient deflect), genuine funneling, exact identity on open terrain. See `venturi.py`'s module docstring.
- The field is **analytic in position**, so recentering is free (just recompute coordinate meshes) and field values at a world point shared between two window placements are bit-identical at a fixed time.

## Venturi terrain funneling — `venturi.py` + `worker.py` (WP2, shipped)
Wind speeds up through gaps/canyons/tunnels and rises over windward obstacles, computed off-thread from the loaded voxel terrain and folded into the gust field.

- **`worker.py` — `VenturiWorker`**: a structural mirror of `lighting/assembly_worker.CascadeAssemblyWorker` (daemon thread `"WindVenturiWorker"`, in/out `queue.Queue`, idempotent `start`, `submit`/`drain_results`/`pending`, `stop(join, timeout)` with a `None` sentinel). numpy releases the GIL during the solve, so it genuinely overlaps the render thread. A solve that raises logs + posts a valid **identity** `VenturiResult` (speed-up 1, zero deflect) so the field never starves — and the thread survives to process the next job.
- **`venturi.py` — `solve_venturi(job)`** (pure): ① fold each intersecting chunk's `materials` over the cell's 8×8 voxel footprint and the z-band into `solid (cells,cells)` in 0..1 (reshape-fold idiom from `lighting/volume._downsample_chunk_block`; missing chunks = open); ② diffuse the solid field outward with `wind_venturi_iters` padded-slice Jacobi sweeps (blockage crowding — **no `np.roll`**); ③ `speedup = clip(1 + crowd_gain·crowd·passability, 1, wind_venturi_max)`, 3×3 box-blurred; `deflect = stack(np.gradient(openness)) · wind_deflect_gain`.
- **`field.py` orchestration**: `WindField.update()` submits a `VenturiJob` (snapshotting the intersecting chunk arrays by reference) when the region **recentered**, when `chunks` first becomes available, or whenever the renderer passes `chunks` (it does so *only* on a recenter / terrain-edit, so a non-`None` `chunks` is itself the recompute request — this keeps `wind/` bus-free). Each update it drains results, keeps the highest `seq`, and commits the newest one **whose `origin_cell` matches the current origin** (else identity — see the origin-match Gotcha). Applied as `vx *= speedup; vx += deflect_x·|mean|` (same for `y`) before modifiers; `sample()` adds the analytic `vz` updraft. Config knobs: `wind_layer_m`, `wind_venturi_iters`, `wind_venturi_max`, `wind_deflect_gain`, `wind_updraft_gain`.
- **Renderer contract (WP3/WP5):** pass `update(..., chunks=chunk_provider.chunks)` **only** on a recenter or `TerrainEditedEvent`/`ChunkLoadedEvent` (dirty) — and once at startup so the first solve initialises; pass `chunks=None` on every other frame (the committed correction holds). Tests live in `tests/test_wind_venturi.py`.

## Render integration — `world/wind_renderer.py` (WP3, shipped)
`WindSystemComponent` is the world-side half of the wind system (panda3d lives in `world/`, never in `wind/`). It owns the per-frame orchestration of the headless `WindField`, packs the published snapshot into a small 2-D float16 texture, and binds the wind uniform contract on `App.terrain_root` so grass — and later flags/cloth/motes/leaves — sample the field by scene-graph inheritance (the same mechanism that gives them the cascade/fog uniforms). It is wired in `main.py` next to the grass component and is **GPU lighting backend only** (it disables itself on the CPU backend / when no `WindField` was built, leaving the scalar grass sway fallback in place).

### Component (`add_component` kwargs)
- `base` — the `App` (provides `terrain_root`, `camera_go`, `lighting_pipeline`).
- `clock` — the shared `Clock`; monotonic absolute game time `= game_day*86400 + game_time_of_day` (the gust phases advect off this — it must not wrap, hence the day fold).
- `wind_field` — the headless `WindField`; `None` disables the component.
- `worker` — the `VenturiWorker` (or `None`); **the component owns it and stops it in `on_destroy`** (`main()` also stops it on the window-teardown exit path).
- `sky_system` — read-only weather source; its `.state` is passed straight into `WindField.update`.
- `chunk_provider` — anything with a `.chunks` dict (`ChunkManager`); forwarded to `update(chunks=...)` only when terrain is dirty (and on the first update, so the venturi worker initialises).
- `lighting_pipeline` — must be the live GPU pipeline; `None` disables.
- `bus` — subscribes `TerrainEditedEvent`/`ChunkLoadedEvent` → set a dirty flag (state-change events only; the heavy venturi solve is off-thread).

### GPU uniform contract (bound on `terrain_root`, inherited by every node under it)
| uniform | type | meaning |
| --- | --- | --- |
| `u_wind_tex` | `sampler2D` | RGBA16F: **R=vx, G=vy, B=turb, A=horizontal speed** (m/s). `FT_linear`, `WM_clamp`. |
| `u_wind_origin` | `vec2` | world XY (m) of texel (0,0)'s **corner** — refreshed ONLY together with an upload. |
| `u_wind_cell_m` | `float` | cell edge in meters (`wind_cell_m`, 4.0). |
| `u_wind_cells` | `float` | cells per axis (`wind_cells`, 64). |
| `u_wind_enabled` | `float` | `0.0` boot default (set in `main.py`) / `1.0` once the first upload lands. |
| `u_time_s` | `float` | already bound by the grass component (shared real-time clock for the in-shader gust oscillation). |

Decode in any shader: `vec2 uv = (world_xy - u_wind_origin) / (u_wind_cell_m * u_wind_cells); vec4 w = texture(u_wind_tex, uv);`

### Texture format decision
`Texture.T_half_float` + `Texture.F_rgba16` (true half-float, 2 bytes × 4 channels), uploaded with `set_ram_image(pack_wind_field(snap))`. `pack_wind_field` emits exactly that layout (little-endian fp16, row-major `(y,x)`, BGRA), so no repack is needed in `world/` and `wind/field.py` stays untouched. **This is the engine's first CPU fp16 `set_ram_image`** — the lighting pipeline's `rgba16` radiance textures are GPU-written only. Crucial gotcha discovered here: with `F_rgba16`, Panda3D's `T_float` component width is **4 bytes** (it expects a 64×64×4×4 = 64 KB fp32 buffer and asserts on a 32 KB fp16 one), while `T_half_float` is **2 bytes** (the 32 KB fp16 buffer `pack_wind_field` produces). Use `T_half_float`. A `test_wind.py` pack test pins the byte length and channel order.

### Filtering — deliberate deviation from grass
`FT_linear` min/mag (grass's height/field textures are *nearest*). Wind is a smooth physical field: linear filtering is what makes a gust **glide** across the grass instead of snapping cell-to-cell at the 4 m grid boundaries. `WM_clamp` u+v so blades outside the 256 m window read the nearest edge velocity (matches `WindField.sample`'s edge clamp).

### Committed-origin discipline (gotcha)
`u_wind_origin` is refreshed **only in the same `late_update` as a texture upload**, never on a bare recenter — exactly the discipline `lighting/gpu.py::_commit_assembly_result` follows for the radiance-cascade window origins. If the origin moved but the texels did not (or vice-versa), the shader would decode the wind UV against a mismatched origin for a frame and the field would visibly jump. Because the component packs + uploads + rebinds the origin every frame, they can never disagree.

### Grass consumption + fallback
`world/shaders/grass.vert` branches on `u_wind_enabled`: when `> 0.5` each blade samples its own local wind from `u_wind_tex` (advecting spectral crests in the field mean neighbouring blades read genuinely different velocities — the travelling-gust look comes from the field, not a fake per-blade phase); the `else` branch is the **verbatim scalar path** driven by `GrassRendererComponent`'s `u_wind_dir`/`u_sway_*`/`u_gust_freq` SkyState uniforms (CPU backend / wind off). `v_base_world` is untouched, so lighting is identical on both paths.

## WP3 binding for WP4 (motes/leaves)
WP4's `DustMoteComponent`/`LeafLitterComponent` need **no new uniforms**: parent their instanced nodes under `terrain_root` and they inherit `u_wind_tex`/`u_wind_origin`/`u_wind_cell_m`/`u_wind_cells`/`u_wind_enabled` (and `u_time_s`) automatically. In the mote/leaf vertex shaders, decode with the same two lines as grass (`uv = (world_xy - u_wind_origin) / (u_wind_cell_m * u_wind_cells); w = texture(u_wind_tex, uv)`), then advect each instance by `w.xy` (use `w.b` turbulence for jitter/rise, `w.a` speed for the leaf gust-kick gate). Guard with `if (u_wind_enabled > 0.5)` and fall back to a flat drift otherwise, mirroring grass.

## WP4 — Particles (pending)
`world/mote_renderer.py` + procedural `dust_mote`/`leaf_sprite` textures consume the wind texture for GPU-instanced dust motes and leaf litter (the latter on `ZoneStore` volumes tagged `"trees"`).
