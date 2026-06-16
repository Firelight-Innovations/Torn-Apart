# render.sky — System Doc
keywords: sky, sky renderer, SkyRendererComponent, sky dome, skydome, clouds, volumetric cloud, cloud slab, raymarch, atmosphere, rayleigh, mie, sun disc, moon, moon phase, star, stars, shooting star, twinkle, night sky, galaxy, fog, exponential fog, terrain light scale, day night, weather map, WeatherMapComponent, rain, rain renderer, RainRendererComponent, rain cover, rain heightmap, rain occlusion, rain particles, rain cylinders, gfx_rain_mode, gfx_rain_occlusion, precip gate, storm footprint, lightning, LightningRendererComponent, lightning bolt, bolt pool, leader, return stroke, afterglow, restrike, thunder, ThunderEvent, wind renderer, WindSystemComponent, wind field, wind texture, wind ball, WindBallDebugComponent, sky shaders, SKY_DOME_FRAGMENT, SKY_DOME_VERTEX, CLOUD_VOLUMETRIC_FRAGMENT, rain shaders, RAIN_PARTICLE_VERTEX, RAIN_PARTICLE_FRAGMENT, RAIN_CYLINDER_VERTEX, RAIN_CYLINDER_FRAGMENT, lightning shaders, LIGHTNING_VERTEX, LIGHTNING_FRAGMENT, u_weather_map, u_wmap_origin, u_wmap_cell_m, u_weather_map_enabled, u_wind_tex, u_wind_origin, u_wind_enabled, u_rain_height_tex, u_rain_height_origin, u_lightning_flash, committed-origin, cover heightmap, RainCoverField, pack_weather_map, pack_wind_field, external_lighting, gfx_clouds, gfx_weather_map, gfx_lightning_bolts, cloud_noise, moon_surface, WMO, cloud genera, virga, u_virga_enabled

> One doc per code package; filename matches the package exactly (`docs/systems/render.sky.md` <-> `fire_engine/render/sky/`).

## Role

`render/sky/` is the **atmospheric render layer** — the panda3d-facing half of sky, weather, rain, wind, and lightning. It owns every Component that reads the headless sky/weather/wind simulation state (in `fire_engine/world/sky/`, `world/weather/`, `world/wind/`) and translates it into Panda3D scene-graph writes, GLSL shader uniforms, and GPU texture uploads each frame.

Sub-systems housed here:

- **`SkyRendererComponent`** (`sky_renderer.py`) — draws the sky dome (physical single-scattering Rayleigh+Mie atmosphere, sun disc, moon, stars/galaxy, shooting stars) and the volumetric raymarched cloud slab (`sky_shaders.py`). Drives `sky_system.update()` once per frame (the single authoritative update caller). With `external_lighting=False` also applies exponential fog and day/night colour scale on `terrain_root`.
- **`WeatherMapComponent`** (`weather_renderer.py`) — packs the headless `WeatherMap` raster into an RGBA16F texture and binds the M4 weather-map uniform contract on `base.render` so cloud/rain shaders sample spatially-varying storm cells.
- **`RainRendererComponent`** (`rain_renderer.py`) — M6 volumetric rain in `"particles"` or `"cylinders"` mode, gated by a rain-cover heightmap (`RainCoverField`) and the weather-map precip channel (`rain_shaders.py`).
- **`LightningRendererComponent`** (`lightning_renderer.py`) — M7 procedural bolt rendering. Subscribes to `LightningStrikeEvent`, regrows the bolt geometry deterministically, plays a leader->return-stroke->afterglow->restrike envelope, pulses `u_lightning_flash` on `base.render`, adds a transient flash point-light, and re-publishes `ThunderEvent` (`lightning_shaders.py`).
- **`WindSystemComponent`** (`wind_renderer.py`) — packs the headless `WindField` snapshot into an RGBA16F texture each frame and binds the wind uniform contract on `terrain_root` (u_wind_tex, u_wind_origin, u_wind_cell_m, u_wind_cells, u_wind_enabled). GPU lighting backend only.
- **`WindBallDebugComponent`** (`wind_debug.py`) — developer-only diagnostic ball driven by `WindField.sample`; gates on `config.debug_wind_ball`.
- **Shader-source modules** (`sky_shaders.py`, `rain_shaders.py`, `lightning_shaders.py`) — headless-importable string constants re-exported from `.vert`/`.frag` sidecar files via `core.shader_source.load_glsl`.

Private implementation helpers live in `render/sky/_impl/` (see `docs/systems/render.sky._impl.md`).

This package deliberately does NOT: simulate sky physics (that is `world/sky/`), generate voxel terrain, or own the GPU lighting cascade assembly (that is `lighting/`).

## Public API

Exported from each module (callers import the class directly or through `fire_engine.render`):

### sky_renderer.py

| Symbol | Description |
|---|---|
| `SkyRendererComponent` | Render component for the procedural sky + weather. Add via `go.add_component(SkyRendererComponent, base=app, sky_system=sky_sys, terrain_root=app.terrain_root, clock=clock)`. |

### weather_renderer.py

| Symbol | Description |
|---|---|
| `WeatherMapComponent` | Render component that uploads the weather map and binds the M4 weather-map uniform contract on `base.render`. |

### rain_renderer.py

| Symbol | Description |
|---|---|
| `RainRendererComponent` | Render component for M6 volumetric rain (particles or cylinders), gated by rain-cover heightmap and weather-map precip channel. |

### lightning_renderer.py

| Symbol | Description |
|---|---|
| `LightningRendererComponent` | Render component for M7 procedural lightning bolts; subscribes to `LightningStrikeEvent`, publishes `ThunderEvent`. |

### wind_renderer.py

| Symbol | Description |
|---|---|
| `WindSystemComponent` | Render component that uploads the wind field texture and binds its uniform contract on `terrain_root`. |

### wind_debug.py

| Symbol | Description |
|---|---|
| `WindBallDebugComponent` | Dev-only ball pushed by the wind field — a physics-sampling seam proof (gate: `config.debug_wind_ball`). |

### sky_shaders.py

| Symbol | Description |
|---|---|
| `SKY_DOME_VERTEX` | GLSL vertex shader for the sky dome (inverted UV-sphere, camera-centred). |
| `SKY_DOME_FRAGMENT` | GLSL fragment shader for the sky dome (physical atmosphere, sun, moon, stars). |
| `CLOUD_VOLUMETRIC_VERTEX` | GLSL vertex shader for the volumetric cloud slab dome. |
| `CLOUD_VOLUMETRIC_FRAGMENT` | GLSL fragment shader: raymarches the cloud slab, Beer-Lambert self-shadow, HG forward scatter. |

### rain_shaders.py

| Symbol | Description |
|---|---|
| `RAIN_PARTICLE_VERTEX` | GLSL vertex shader for the GPU-instanced rain streak particles. |
| `RAIN_PARTICLE_FRAGMENT` | GLSL fragment shader for rain particles (heightmap cull + precip gate). |
| `RAIN_CYLINDER_VERTEX` | GLSL vertex shader for the low-preset rain cylinders. |
| `RAIN_CYLINDER_FRAGMENT` | GLSL fragment shader for rain cylinders (same gates as particles). |

### lightning_shaders.py

| Symbol | Description |
|---|---|
| `LIGHTNING_VERTEX` | GLSL vertex shader: expands bolt segments to camera-facing ribbons, applies `u_reveal` front. |
| `LIGHTNING_FRAGMENT` | GLSL fragment shader: emits hot HDR core + soft glow scaled by `u_flash` (additive blend). |

## Imports Allowed

Per ARCHITECTURE.md §4a.2 and Hard Rule 1:

- `panda3d.*` — all modules here are inside `render/` (the sole panda3d bridge); panda3d imports are required and expected.
- `fire_engine.core` — logging (`get_logger`), events (`LightningStrikeEvent`, `ThunderEvent`, `ChunkLoadedEvent`, `TerrainEditedEvent`), `shader_source.load_glsl`, `rng.for_domain`.
- `fire_engine.render.component` — `Component` base class.
- `fire_engine.render._impl.quad` — shared quad geometry builders (used by `rain_build`).
- `fire_engine.world.sky` — `pack_weather_map` (headless; no panda3d).
- `fire_engine.world.weather` — `WeatherMap`, `generate_bolt` (headless).
- `fire_engine.world.wind` — `WindField`, `BallParams`, `debug_ball_step`, `pack_wind_field` (headless).
- `fire_engine.world.terrain` — `RainCoverField` (headless).
- `fire_engine.render.sky._impl.*` — private implementation helpers.
- `fire_engine.render.sky.{sky_shaders,rain_shaders,lightning_shaders}` — for compiling GLSL at `start()`.
- Python standard library; `numpy`.

Must NOT import: any `fire_engine.*` layer not listed above; the `_impl` modules may not import the component classes that use them (only the reverse).

## Events

### Published
- `ThunderEvent` (deferred via `bus.publish_deferred`) — `LightningRendererComponent` emits one per `LightningStrikeEvent`, carrying `pos`, `distance_m`, `delay_s = distance / 343`, `time_abs`, and `intensity` for the delayed audio crack.

### Subscribed
- `LightningStrikeEvent` — `LightningRendererComponent` (triggers bolt geometry + flash light + ThunderEvent).
- `ChunkLoadedEvent` — `LightningRendererComponent` (marks cover heightmap dirty), `RainRendererComponent` (marks affected chunk columns dirty).
- `TerrainEditedEvent` — `LightningRendererComponent` (marks cover dirty), `RainRendererComponent` (marks touched chunk columns dirty), `WindSystemComponent` (flags venturi field dirty).

## Units & Invariants

- World space: **meters**, Z-up (Panda3D native).
- Sky dome radius: 800 m (`_DOME_RADIUS_M`). Camera far plane is extended to cover it.
- Volumetric cloud slab: altitude `_VCLOUD_ALT_M`, thickness `_VCLOUD_THICK_M` (meters; constants from `sky_geom.py`).
- Wind texture: RGBA16F fp16 BGRA layout (`pack_wind_field`); R=vx, G=vy, B=turb, A=speed (all m/s).
- Weather texture: RGBA16F fp16 BGRA; R=coverage, G=density, B=precip, A=fog.
- Rain-cover texture: single-channel float32 (`F_r32`), world Z meters, nearest-filtered.
- **Committed-origin discipline**: `u_wind_origin`, `u_wmap_origin`, `u_rain_height_origin` are refreshed ONLY in the same frame as a texel upload, never on a bare recenter.
- All GPU texture uploads use bulk `set_ram_image` (never per-element writes).
- `SkyRendererComponent` is the **sole** caller of `sky_system.update(player_pos)` per frame.
- `WindSystemComponent.late_update` advances the wind clock by **real** `dt x wind_time_scale` — deliberately independent of the game timescale.
- Shooting stars: deterministic 30-game-minute slots decided by `for_domain("sky", "shooting_stars", game_day, slot)`.

## Examples

```python
# Wire up the full atmospheric stack (mirrors main.py)
from fire_engine.render import instantiate
from fire_engine.render.sky.sky_renderer import SkyRendererComponent
from fire_engine.render.sky.weather_renderer import WeatherMapComponent
from fire_engine.render.sky.rain_renderer import RainRendererComponent
from fire_engine.render.sky.lightning_renderer import LightningRendererComponent
from fire_engine.render.sky.wind_renderer import WindSystemComponent

sky_go = instantiate()
sky_go.add_component(
    SkyRendererComponent,
    base=app,
    sky_system=sky_system,
    terrain_root=app.terrain_root,
    clock=clock,
    external_lighting=True,   # GPU pipeline owns fog/colour-scale
)

weather_go = instantiate()
weather_go.add_component(WeatherMapComponent, base=app, sky_system=sky_system)

rain_go = instantiate()
rain_go.add_component(
    RainRendererComponent,
    base=app, sky_system=sky_system, chunk_provider=chunk_manager,
    lighting_pipeline=pipeline, bus=bus,
)

lightning_go = instantiate()
lightning_go.add_component(
    LightningRendererComponent,
    base=app, sky_system=sky_system, chunk_provider=chunk_manager,
    lighting_pipeline=pipeline, bus=bus,
)

wind_go = instantiate()
wind_go.add_component(
    WindSystemComponent,
    base=app, clock=clock, wind_field=wind_field, worker=venturi_worker,
    sky_system=sky_system, chunk_provider=chunk_manager,
    lighting_pipeline=pipeline, bus=bus,
)
```

## Gotchas

- **`u_time_s` is NOT inherited** — `RainRendererComponent` and `WindBallDebugComponent` self-bind `u_time_s` on their own nodes each frame. Setting it on `terrain_root` or `render` does NOT reach them.
- **Committed-origin invariant**: if you upload texels but forget to refresh the origin (or vice-versa), the shader decodes against a stale tile, causing a one-frame jump or pop.
- **GPU lighting backend required** for `WindSystemComponent`, `RainRendererComponent`, and `LightningRendererComponent`. They each disable themselves (with a log warning) when `lighting_pipeline is None`.
- **Sample the weather map at raw world XY** — never `world_xy + u_wind`. The weather raster bakes in storm-cell drift; adding `u_wind` would double-advect the storm.
- **`SkyRendererComponent` owns the sole `sky_system.update()` call** — nothing else may call it. Double-advancing the sky/weather state within a frame is a hard bug.
- **`external_lighting=True`** disables `SkyRendererComponent`'s Panda3D `Fog`, `set_color_scale`, and clear-colour blend. Use this whenever the GPU lighting pipeline provides froxel volumetric fog; mixing both doubles the fog.
- Panda3D `T_half_float + F_rgba16` expects a true 16-bit float (fp16, 2 bytes/channel) buffer — NOT fp32. `pack_wind_field` and `pack_weather_map` both produce exactly the fp16 BGRA layout this expects.
