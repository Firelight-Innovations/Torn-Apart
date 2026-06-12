# wind — System Doc
keywords: wind, gust, breeze, storm, brownian, turbulence, venturi, motes, leaves, particles, flag, cloth, advection, spectral, gust front, wind field, vertical profile, boundary layer

> Status: **WP1 shipped** (headless wind core + config). Sections marked WP2/WP3/WP4 are seams that later work packages fill; WP5 finalizes this doc. Do not delete the pending markers until the matching package lands.

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

Internal modules: `gusts.py` (`build_modes(cfg)` + `eval_gusts(modes, X, Y, t_eff, mean)`), `region.py` (`WindRegion` recenter window).

WP2 will add: `VenturiWorker`, `VenturiJob`, `VenturiResult`, `solve_venturi` (not yet exported).

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
- `sample()` clamps out-of-region points to the nearest edge value; never returns NaN; `vz == 0` in WP1 (WP2 adds analytic obstacle updraft).

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
- **No `np.roll` in the venturi solver (WP2):** roll wraps the grid edges — use padded arrays/slicing.
- The field is **analytic in position**, so recentering is free (just recompute coordinate meshes) and field values at a world point shared between two window placements are bit-identical at a fixed time.

## WP2 — Venturi terrain funneling (pending)
`venturi.py` (`solve_venturi`) + `worker.py` (`VenturiWorker`) fold terrain occupancy into a per-cell speed-up + deflection, off-thread. The seam is the clearly-marked identity block in `field.py::WindField.update` (step 4) and `vz` in `sample()`.

## WP3 — Render integration (pending)
`world/wind_renderer.py` uploads the packed snapshot as `u_wind_tex` on `terrain_root` and rebinds grass to sample it. See the GPU uniform contract in the plan.

## WP4 — Particles (pending)
`world/mote_renderer.py` + procedural `dust_mote`/`leaf_sprite` textures consume the wind texture for GPU-instanced dust motes and leaf litter (the latter on `ZoneStore` volumes tagged `"trees"`).
