# sky — System Doc
keywords: sky, skybox, sky dome, day night cycle, daynight, sun, moon, stars, star field, galaxy, milky way, weather, rain, fog, storm, wind, clouds, cloud coverage, celestial, time of day, sunrise, sunset, dawn, dusk, twilight, daylight, moon phase, sky gradient, zenith, horizon, fog color, fog density, terrain light scale, SkyState, SkySystem, WeatherSystem, WeatherType, WeatherParams, sun_direction, moon_direction, force_weather, WeatherChangedEvent, markov, night_sky, rain_streak, atmosphere, rayleigh, mie, scattering, transmittance, sun_radiance, moon_radiance, sky_ambient, physical sky, single scattering, earth shadow, moon_surface, external_lighting, cloud_noise, bake_shape_noise, bake_detail_noise, worley, perlin-worley, volumetric clouds, 3d noise, tileable noise, to_panda_texture_3d, sampler3D, cloud density field

> One doc per code package; filename matches the package exactly (`docs/systems/sky.md` ↔ `fire_engine/sky/`).

## Role

`sky/` is the **headless half of the procedural sky + weather feature** — a Layer 1 Service, peer of `lighting/`.  Once per frame, `SkySystem.update()` reads the game clock and produces a frozen `SkyState` snapshot: sun/moon directions, sky gradient colors, blended weather parameters (clouds, fog, rain, wind), star visibility, the legacy `terrain_light_scale` multiplier, and the **HDR radiance contract** (`sun_radiance` / `moon_radiance` / `sky_ambient`) consumed by the GPU volumetric lighting pipeline.  Weather follows a **deterministic Markov chain** over 2-game-hour segments — the entire schedule is a pure function of `(world_seed, game_day, segment)`, so saves cost ~0 bytes unless a dev override is active.

**The sky is physically simulated** (`sky/atmosphere.py`): an Earth-like Rayleigh + Mie **single-scattering** model (spherical planet, exponential density, nested sun-ray transmittance with planet occlusion → the earth-shadow twilight arch).  It is evaluated twice from the same constants: in numpy here (a boot-time LUT over sun elevation feeds `SkySystem`'s per-frame scalar interpolation) and per pixel in the GLSL dome shader (`world/sky_shaders.py`) — so the *picture* of the sunset and the *light* it casts on terrain agree by construction.  Sunsets turn the actual scene light orange because `sun_radiance` follows the modelled transmittance.

`sky/` deliberately does NOT: import panda3d, issue render commands, draw the sky dome/clouds/rain (that is `world/`'s render half, which consumes `SkyState`), or own the procedural sky textures (`"night_sky"`, `"rain_streak"`, `"moon_surface"` live in `procedural/textures/`).

## Public API

All symbols below are re-exported from `fire_engine.sky` (`__init__.py`).

### SkyState (`sky/sky_state.py`)

| Symbol | Description |
|---|---|
| `SkyState` | Frozen per-frame snapshot.  Fields below. |
| `.sun_dir`, `.moon_dir` | Unit `Vec3`, Z-up, FROM the scene TOWARD the body. |
| `.sun_color` | Linear RGB 0–1; warm amber at horizon → near-white (1.0, 0.97, 0.90) at noon. |
| `.sun_intensity` | 0–1; exactly 0 below the horizon; dimmed by cloud cover. |
| `.moon_phase` | 0–1 from `game_day` (0 = new, 0.5 = full; cycle = `MOON_CYCLE_DAYS` = 30 days). |
| `.daylight` | Smooth 0–1 day factor (0 night, 1 midday; 0.5 at sunrise/sunset). |
| `.star_visibility` | 0–1 (1 = clear night); `(1 − daylight) · (1 − 0.85 · cloud_coverage)`. |
| `.zenith_color`, `.horizon_color` | Sky gradient, weather-graded (overcast/storm desaturate + darken toward gray). |
| `.cloud_coverage`, `.cloud_density` | 0–1 (fraction of cloud cells filled / cloud opacity), already blended. |
| `.fog_density` | Exponential fog coefficient, 1/m (0 = none; FOG weather ≈ 0.025). |
| `.fog_color` | Horizon color blended toward weather gray (dims at night). |
| `.rain_intensity` | 0–1. |
| `.wind_dir`, `.wind_speed` | Unit XY tuple; m/s. |
| `.terrain_light_scale` | RGB multiplier: clear day ≈ (1, 1, 1); night floor ≈ (0.16, 0.19, 0.30); warm (1.0, 0.82, 0.62)-tinted dawn/dusk; ×~0.75 overcast, ×~0.55 storm.  Smooth everywhere.  **CPU lighting backend only** — the GPU pipeline reads the three radiance fields below. |
| `.sun_radiance` | **Linear HDR RGB** direct sun at the ground (atmosphere-transmitted): clear noon ≈ (3.2, 3.0, 2.6); strongly orange + dimmer near sunset (R/B ratio rises monotonically as the sun drops); smooth twilight tail to exactly 0 at −4° elevation; ×(1 − 0.92·coverage·density) under cloud. |
| `.moon_radiance` | Linear HDR RGB moonlight: pale blue-white, full moon high ≈ (0.06, 0.07, 0.10); × phase illuminated fraction (`0.5·(1−cos 2π·phase)`), × elevation ramp, × cloud attenuation; 0 below the horizon. |
| `.sky_ambient` | Linear HDR RGB hemispheric skylight irradiance (the GI skylight injection): clear noon ≈ (0.21, 0.40, 0.71); warm-gray sunset; overcast = desaturated at similar luminance; clear moonless night floor ≈ (0.010, 0.012, 0.022) + a small moonlight bump. |

### SkySystem (`sky/sky_state.py`)

| Symbol | Description |
|---|---|
| `SkySystem(config, clock, bus)` | Composer.  Constructs its `WeatherSystem` internally. |
| `sky.weather` | The owned `WeatherSystem` (register it with SaveManager). |
| `sky.update() -> SkyState` | Reads `clock.game_day` / `clock.game_time_of_day`, computes + caches.  Call once per frame. |
| `sky.state` | Property: last computed `SkyState` (`update()` invoked lazily if never run). |

### Weather (`sky/weather.py`)

| Symbol | Description |
|---|---|
| `WeatherType` | `str` Enum: `CLEAR, CLOUDY, OVERCAST, FOG, RAIN, STORM` (values `"clear"`…`"storm"`). |
| `WeatherParams` | Frozen dataclass: `cloud_coverage, cloud_density, fog_density, rain_intensity, wind_dir, wind_speed` — same units as `SkyState`. |
| `WeatherSystem(config, bus=None)` | Saveable (`save_key = "weather"`). |
| `ws.current` | Property: discrete `WeatherType` as of the last `update()` (override wins). |
| `ws.update(game_day, game_time_of_day) -> WeatherParams` | Blended params; publishes `WeatherChangedEvent` (deferred) on discrete change. |
| `ws.force_weather(weather)` | Dev override; `None` clears it and blends back to the natural schedule. |
| `ws.get_delta() -> dict` | `{}` unless an override/release blend is active. |
| `ws.apply_delta(delta)` | Restore override state; subsequent behaviour identical. |
| `SEGMENT_SECONDS`, `SEGMENTS_PER_DAY`, `BLEND_SECONDS` | 7200 s, 12, 1200 s (20 game minutes). |

### Celestial (`sky/celestial.py`) — pure functions

| Symbol | Description |
|---|---|
| `sun_direction(time_of_day_s) -> Vec3` | Unit dir toward the sun.  Sunrise 06:00 in +X (east), noon `z ≥ 0.9`, sunset 18:00 in −X, midnight `z ≤ −0.9`.  Continuous. |
| `moon_direction(time_of_day_s) -> Vec3` | Roughly opposite the sun + ~1 h phase lead (both briefly visible at twilight). |
| `daylight_factor(time_of_day_s) -> float` | Smoothstep on sun elevation; fully 1 ~1 h after sunrise, fully 0 ~1 h after sunset. |

### Atmosphere (`sky/atmosphere.py`) — pure physical model

| Symbol | Description |
|---|---|
| `transmittance(view_dirs, samples=64) -> (N, 3)` | Rayleigh+Mie optical-depth transmittance from the ground observer toward each direction (quadratic step spacing — dense near the observer).  Sun/moon disc tint. |
| `sun_radiance(sun_z, samples=64) -> (3,) or (N, 3)` | `SUN_GROUND_SCALE × transmittance` with a smoothstep twilight fade to 0 at −4° (`SUN_FADE_LO_Z`).  Azimuth-symmetric: takes `sin(elevation)`. |
| `sky_radiance(view_dirs, sun_dir, steps=16, light_steps=8) -> (N, 3)` | Single-scattered sky radiance: per-sample nested sun-ray march with planet occlusion (earth-shadow arch).  **Quadratic view-step spacing** — linear steps skip the dense low atmosphere on grazing rays and render the day horizon black.  Mirrored exactly by the GLSL dome shader. |
| `sky_ambient(sun_z, samples=48, ...) -> (3,)` | Cosine-weighted hemispheric irradiance over a fixed Fibonacci direction set, × `AMBIENT_SCALE` (0.4, calibrated to the SkyState contract). |
| `BETA_RAYLEIGH, BETA_MIE, MIE_G, RAYLEIGH_SCALE_HEIGHT_M, MIE_SCALE_HEIGHT_M, PLANET_RADIUS_M, ATMOSPHERE_TOP_M, SUN_TOA_RADIANCE, ...` | The physical constants — **shared verbatim** with `world/sky_shaders.SKY_DOME_FRAGMENT`; change them in both places or the picture and the light disagree. |

`SkySystem` never calls these per frame: a module-level `_AtmosphereLUT` (56 sun elevations × {sun, ambient, zenith, horizon}) is built once per process (~0.2 s) and interpolated with `np.interp` per frame.

### Cloud noise (`sky/cloud_noise.py`) — pure deterministic bake

| Symbol | Description |
|---|---|
| `bake_shape_noise(size=64) -> (N,N,N,4) uint8` | Cloud SHAPE volume: R = Perlin-Worley billowy base (the cloud bulk); G/B/A = increasing-frequency inverted-Worley FBM octaves the raymarch uses to erode the base into wisps.  Deterministic (`for_domain("sky","cloud_shape")`), **tileable** (every octave's lattice period divides the texture, so `WM_repeat` never seams).  64³ ≈ 1.7 s; 128³ is sharper but ~30 s → disk-cache it (deterministic ⇒ the cache is always valid). |
| `bake_detail_noise(size=32) -> (N,N,N,4) uint8` | High-frequency Worley FBM packed across R/G/B for fine edge detail.  Same determinism/tileability. |

Headless numpy (no panda3d); `world/texture_bridge.to_panda_texture_3d` uploads the arrays as `sampler3D`s for `world/shaders/cloud_volumetric.frag`.  Arrays are page-major `[z,y,x,c]`.

### Related (owned elsewhere)

- `"night_sky"` / `"rain_streak"` / `"moon_surface"` procedural textures — see `docs/systems/procedural.md`.  The moon disc texture is seeded per world (`for_domain`): every world grows different craters/maria.
- The render half: dome single-scatter shader, 2.5×-sized limb-darkened sun disc + textured moon disc with phase terminator, cloud/rain renderers, and the `SkyRendererComponent(external_lighting=...)` flag — `world/sky_renderer.py` / `world/sky_shaders.py`.  Under HDR output (`u_hdr_output=1`) the sun disc/halo gains and the scattered-sky brightness are config-exposed `gfx_*` uniforms pushed once at build (`_build_dome`): `gfx_sun_disc_intensity`, `gfx_sun_halo_intensity`, `gfx_sun_min_brightness` (transmittance floor that keeps a grazing sunrise/sunset sun bright instead of fading, hue preserved), `gfx_sky_inscatter_scale` (sky-radiance multiplier — lower to cut the washed-out look when the sun is low, without dimming the disc).  See `docs/systems/world.md` "Aesthetic vs quality config".
- `[sky]` config table (`sky_cloud_altitude_m`, `sky_cloud_thickness_m`, `sky_cloud_cell_m`, `sky_star_count`) — see `docs/systems/core.md`.
- `clock.game_time_scale` (read/write, dev time scrubbing) — see `docs/systems/core.md`.

## Imports Allowed

`sky/` may only import:
- Python standard library (`math`, `dataclasses`, `enum`, ...)
- `numpy`
- `fire_engine.core` (Config, Clock, EventBus, `for_domain`, `math3d.Vec3`)

**No panda3d imports.** Never import from `world/`, `terrain/`, `lighting/`, or any higher layer.  The render half (`world/`) imports `fire_engine.sky` downward — never the reverse.

## Events

### Published
| Event | When | Publisher |
|---|---|---|
| `WeatherChangedEvent(previous, current, day)` | Discrete weather state changes (at most once per segment boundary, or on `force_weather` toggle).  `previous`/`current` are `WeatherType.value` strings. | `WeatherSystem.update()` via `bus.publish_deferred` |

Never per-frame events: blended parameter changes are returned from `update()`, not published.

### Subscribed
`sky/` subscribes to nothing.  It reads the `Clock` directly (downward call).

## Units & Invariants

- **Z-up, forward = +Y, east/right = +X.**  Directions are unit `Vec3` pointing FROM the scene TOWARD the body.  Distances meters, time seconds, angles radians.  Colors are linear RGB tuples, components 0–1.
- `time_of_day_s` is `clock.game_time_of_day`: seconds in `[0, 86400)`, 0 = midnight.  Fixed v0 schedule: sunrise 06:00, sunset 18:00; the sun arc tilts 20° toward −Y (southern sky), the moon 12°.
- `fog_density` is the exponential fog coefficient in **1/m** (transmittance `e^(−density·distance)`).
- **Determinism:** the weather schedule and all natural (un-forced) params are pure functions of `(world_seed, game_day, game_time_of_day)`.  Two fresh `SkySystem`s with the same seed and clock state produce bit-identical `SkyState`s.  All randomness flows through `for_domain("weather", ...)`.
- **Day anchoring:** each day's segment 0 is drawn from a fixed initial distribution (≈ stationary) instead of chaining from the previous day — this bounds recompute to ≤ 12 Markov steps per day.  The midnight hand-off is still parameter-blended.  STORM is rare and (almost) only reachable from RAIN; FOG probability is ×3 in segments 2–4 (04:00–10:00).
- **Blending:** every transition (natural, force, release) crossfades over `BLEND_SECONDS` = 20 game minutes with smoothstep — params never pop.  Per-state targets: CLEAR(coverage .12, density .35, fog .0008, rain 0) · CLOUDY(.45, .55, .0012, 0) · OVERCAST(.85, .80, .003, 0) · FOG(.55, .50, .025, 0, calm ×.30) · RAIN(.90, .85, .006, .7) · STORM(.98, .95, .008, 1.0, gusty ×1.9).  Wind comes from the closed-form **synoptic flow** (`fire_engine.weather.Synoptic`, exposed as `WeatherSystem.synoptic`): direction drifts smoothly over hours (no per-day snaps, continuous across midnight), speed = synoptic speed × the per-state multiplier above.  See `docs/systems/weather.md`.
- `terrain_light_scale` components are always in `[0, 1.05]`; exactly (1, 1, 1) × weather-dim at clear noon.
- Saves: `get_delta()` is `{}` on the natural schedule (baseline regenerates from seed).  Only `force_weather` overrides / in-flight release blends are snapshotted, as plain primitives.

## Examples

### Boot + per-frame use
```python
from fire_engine.core import Clock, EventBus, load_config, set_world_seed
from fire_engine.sky import SkySystem

cfg = load_config()
set_world_seed(cfg.world_seed)
bus = EventBus()
clock = Clock(fixed_dt=cfg.fixed_dt, bus=bus)

sky = SkySystem(cfg, clock, bus)
# save_manager.register(sky.weather)        # delta saves

def frame(real_dt: float) -> None:
    clock.update(real_dt)
    state = sky.update()                    # once per frame, before render
    # world render half consumes `state`:
    #   sun light dir/color, sky gradient, fog, cloud layer, rain, stars
    bus.drain()
```

### React to weather changes
```python
from fire_engine.core import WeatherChangedEvent

def on_weather(evt: WeatherChangedEvent) -> None:
    print(f"day {evt.day}: {evt.previous} -> {evt.current}")
    if evt.current == "storm":
        start_thunder_ambience()

bus.subscribe(WeatherChangedEvent, on_weather)
```

### Dev override (debug key binding)
```python
from fire_engine.sky import WeatherType

sky.weather.force_weather(WeatherType.STORM)   # blends in over 20 game min
sky.weather.force_weather(None)                # blends back to the schedule
```

### Pure celestial queries (no system needed)
```python
from fire_engine.sky import sun_direction, moon_direction

noon_sun = sun_direction(12 * 3600.0)     # Vec3, z ≈ 0.94
dusk_moon = moon_direction(17.5 * 3600.0) # already above the east horizon
```

## Gotchas

1. **Call `update()` once per frame, after `clock.update()`.**  `SkyState` is a snapshot of the clock at call time; the `state` property does NOT recompute (except a single lazy first call).

2. **`set_world_seed` is global** — `WeatherSystem` memoises its schedule per instance, so changing the seed mid-life leaves stale cached segments.  Seeds change only at boot/world-load; build a fresh `SkySystem` afterwards (tests: fully consume one instance before reseeding).

3. **`force_weather` anchors at the *next* `update()`** — the blend starts from the next frame's params, so forcing then asserting targets immediately will fail; advance game time past `BLEND_SECONDS` first.

4. **Register `sky.weather` (not `SkySystem`) with SaveManager.**  `SkySystem` itself holds no saveable state; everything else re-derives from clock + seed.

5. **Color ramps key on sun elevation (`sun_dir.z`), not time.**  Dawn and dusk share keyframes by construction; do not add time-keyed palettes or the two twilights will diverge.

6. **The event is deferred** — `WeatherChangedEvent` arrives on `bus.drain()`, not inside `update()`.  Tests must drain.

7. **`moon_phase` comes from `game_day` only** — it does not affect `moon_dir` geometry (v0).  Renderer uses it to pick the moon sprite/mask.

8. **Per-frame cost is scalar-only** (a handful of ramps and lerps) — no arrays are built in `update()`.  Keep it that way; anything per-pixel belongs in textures or shaders.

9. **`night_sky` / `rain_streak` arrive V-FLIPPED at the GPU.**  `world/texture_bridge.to_panda_texture` vertically flips every texture (OpenGL bottom-left UV origin).  The render side compensates (equirect V mapping in the dome shader; mirrored V axis on the rain cylinders) — see `docs/systems/world.md` gotchas 11–14.  Do not "fix" the orientation in the texture defs; you would re-introduce the upward-falling-rain bug.

10. **`_AtmosphereLUT` is module-global and built on first `SkySystem` construction** (~0.2 s).  It is seed-independent physics, so sharing across systems/tests is safe — but a test that monkeypatches atmosphere constants must reset `sky_state._ATMOSPHERE_LUT` to `None` or it will read stale tables.

11. **The GLSL dome shader duplicates the atmosphere math** (`world/sky_shaders.py` mirrors `sky/atmosphere.py` constants and the quadratic step spacing).  Any change to the model MUST be made in both files, or the rendered sky will disagree with `sun_radiance`/`sky_ambient` and terrain lighting drifts out of sync with the picture.

12. **`SkyRendererComponent(external_lighting=True)` is mandatory on the GPU lighting backend** — otherwise the sky renderer's `terrain_root.set_color_scale` and Panda3D `Fog` double-apply on top of the volumetric terrain shader.  main.py wires this from `config.lighting_backend`.

13. **Overcast attenuates `sun_radiance` by ×(1 − 0.92·coverage·density)** — under a storm the direct sun is nearly gone and the scene is lit almost entirely by the (desaturated) `sky_ambient`.  That is intended: overcast light IS diffuse.

14. **Dome fog/sun composite ORDER (sky_dome.frag).**  The froxel fog is composited over the BACKGROUND sky first (`col = col*fogA + fogRGB`); the sun/moon **discs + halos are added AFTER, attenuated by the fog transmittance `fogA` only** — never by the inscatter `fogRGB`.  This is what lets a bright HDR sun punch through fog (dimmed, not erased).  Reversing the order (disc into `col` before fog) re-introduces the "sun hidden behind a grey fog layer" bug.  The disc/halo brightness is `discGain`/`haloGain`, which jump far above 1.0 when `u_hdr_output` is on so bloom bleeds the disc into a soft blob; the legacy (post-off) path keeps the original 14.0/0.55 clamped-disc values.  The atmosphere reads as a flat gradient ONLY when its >1.0 radiance is clamped — with HDR post-processing the physical dark→red/orange→bright sunrise progression emerges from the existing scattering with no extra hand-painting.
