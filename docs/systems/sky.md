# sky — System Doc
keywords: sky, skybox, sky dome, day night cycle, daynight, sun, moon, stars, star field, galaxy, milky way, weather, rain, fog, storm, wind, clouds, cloud coverage, celestial, time of day, sunrise, sunset, dawn, dusk, twilight, daylight, moon phase, sky gradient, zenith, horizon, fog color, fog density, terrain light scale, SkyState, SkySystem, WeatherSystem, WeatherType, WeatherParams, sun_direction, moon_direction, force_weather, WeatherChangedEvent, markov, night_sky, rain_streak

> One doc per code package; filename matches the package exactly (`docs/systems/sky.md` ↔ `torn_apart/sky/`).

## Role

`sky/` is the **headless half of the procedural sky + weather feature** — a Layer 1 Service, peer of `lighting/`.  Once per frame, `SkySystem.update()` reads the game clock and produces a frozen `SkyState` snapshot: sun/moon directions, sky gradient colors, blended weather parameters (clouds, fog, rain, wind), star visibility, and the `terrain_light_scale` RGB multiplier the renderer applies to terrain vertex light.  Weather follows a **deterministic Markov chain** over 2-game-hour segments — the entire schedule is a pure function of `(world_seed, game_day, segment)`, so saves cost ~0 bytes unless a dev override is active.

`sky/` deliberately does NOT: import panda3d, issue render commands, draw the sky dome/clouds/rain (that is `world/`'s render half, which consumes `SkyState`), or own the procedural sky textures (`"night_sky"` and `"rain_streak"` live in `procedural/textures/`).

## Public API

All symbols below are re-exported from `torn_apart.sky` (`__init__.py`).

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
| `.terrain_light_scale` | RGB multiplier: clear day ≈ (1, 1, 1); night floor ≈ (0.16, 0.19, 0.30); warm (1.0, 0.82, 0.62)-tinted dawn/dusk; ×~0.75 overcast, ×~0.55 storm.  Smooth everywhere. |

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

### Related (owned elsewhere)

- `"night_sky"` / `"rain_streak"` procedural textures — see `docs/systems/procedural.md`.
- `[sky]` config table (`sky_cloud_altitude_m`, `sky_cloud_thickness_m`, `sky_cloud_cell_m`, `sky_star_count`) — see `docs/systems/core.md`.
- `clock.game_time_scale` (read/write, dev time scrubbing) — see `docs/systems/core.md`.

## Imports Allowed

`sky/` may only import:
- Python standard library (`math`, `dataclasses`, `enum`, ...)
- `numpy`
- `torn_apart.core` (Config, Clock, EventBus, `for_domain`, `math3d.Vec3`)

**No panda3d imports.** Never import from `world/`, `terrain/`, `lighting/`, or any higher layer.  The render half (`world/`) imports `torn_apart.sky` downward — never the reverse.

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
- **Blending:** every transition (natural, force, release) crossfades over `BLEND_SECONDS` = 20 game minutes with smoothstep — params never pop.  Per-state targets: CLEAR(coverage .12, density .35, fog .0008, rain 0) · CLOUDY(.45, .55, .0012, 0) · OVERCAST(.85, .80, .003, 0) · FOG(.55, .50, .025, 0, low wind) · RAIN(.90, .85, .006, .7) · STORM(.98, .95, .008, 1.0, high wind).  Wind direction is per-day from `for_domain("weather", "wind", day)`.
- `terrain_light_scale` components are always in `[0, 1.05]`; exactly (1, 1, 1) × weather-dim at clear noon.
- Saves: `get_delta()` is `{}` on the natural schedule (baseline regenerates from seed).  Only `force_weather` overrides / in-flight release blends are snapshotted, as plain primitives.

## Examples

### Boot + per-frame use
```python
from torn_apart.core import Clock, EventBus, load_config, set_world_seed
from torn_apart.sky import SkySystem

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
from torn_apart.core import WeatherChangedEvent

def on_weather(evt: WeatherChangedEvent) -> None:
    print(f"day {evt.day}: {evt.previous} -> {evt.current}")
    if evt.current == "storm":
        start_thunder_ambience()

bus.subscribe(WeatherChangedEvent, on_weather)
```

### Dev override (debug key binding)
```python
from torn_apart.sky import WeatherType

sky.weather.force_weather(WeatherType.STORM)   # blends in over 20 game min
sky.weather.force_weather(None)                # blends back to the schedule
```

### Pure celestial queries (no system needed)
```python
from torn_apart.sky import sun_direction, moon_direction

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
