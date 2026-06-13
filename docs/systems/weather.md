# weather — System Doc
keywords: weather, synoptic, wind direction, prevailing wind, storm, storm cell, front, air mass, displacement, D(t), drift, steering current, volumetric weather, regime, shower, thunderstorm, cloud bank, fog bank, classify, WeatherType, LocalWeather, sample_local, force_weather

> Status: under construction (volumetric-weather branch). M1 (synoptic flow)
> and M2 (storm cells + local sampling + classify) shipped; weather map,
> emergent fog, lightning land in M3–M8.  This doc grows with each milestone.

## Role
Headless spatial weather simulation — the layer between `sky/` (which
composes per-frame `SkyState`) and `wind/` (local gusts).  Owns the
**synoptic flow** (the slow steering current that sets base wind and carries
cells) and the **storm cells** that drift on it.  Weather is sampled *at a
world position*: stand under a passing shower and it rains; a kilometer away
it's dry.  Everything natural is a closed-form pure function of (world_seed,
game time, position) — no integrated state, zero save bytes for natural
weather.  Deliberately does NOT: render anything (bridges live in `world/`),
simulate local gusts (that's `wind/`), or own `SkyState` (that's `sky/`).

## Public API
| Export | Description |
|---|---|
| `Synoptic(config)` | Seeded closed-form synoptic wind. Built once from `for_domain("weather", "synoptic")`. |
| `Synoptic.wind(t_abs)` | `((ux, uy), speed_ms)` — unit direction + speed at absolute game time. Speed guaranteed inside `[weather_synoptic_speed_min_ms, ..._max_ms]`. |
| `Synoptic.wind_vec(t_abs)` | Vector form; accepts scalar or `(M,)` ndarray of times → `(2,)` / `(M, 2)`. |
| `Synoptic.displacement(t_abs)` | Air-mass displacement `D(t)` in meters since t=0; `dD/dt ≡ wind_vec` to machine precision. Storm-cell centers ride `spawn_pos + D(t) − D(spawn_time)`. |
| `CellKind` | `SHOWER` / `THUNDERSTORM` / `CLOUD_BANK` / `FOG_BANK` — what a cell does. |
| `Regime` | `HIGH_PRESSURE` / `MIXED` / `FRONTAL` — per-day air mass; sets ambient sky + spawn mix. |
| `StormCell` | Frozen analytic cell: `center(t, syn)`, `radius(t)`, `intensity(t)`, `active(t)`, `contribution(points_xy, t, syn) → (N,)` Gaussian footprint. |
| `day_regime(day)` / `regime_ambient(regime)` / `natural_cells(day, config)` | Pure-fn-of-(seed, day) spawn schedule (memoise per day). |
| `classify(local) → WeatherType` | Discrete label from a `LocalWeather` sample (fog→storm→rain→overcast→cloudy→clear, first match wins). |
| `WeatherType` | `clear`/`cloudy`/`overcast`/`fog`/`rain`/`storm` — exact legacy string values. |
| `LocalWeather` | Frozen local sample: cloud_coverage/density, fog_density, rain_intensity, wind_dir/speed, humidity, wetness, temperature_c. First six map 1:1 onto `SkyState`. |
| `WeatherSystem(config, bus=None)` | The system (`save_key="weather"`). `update(day, tod, player_pos=None) → LocalWeather`; `sample_local(pos_xy, t_abs) → LocalWeather`; `sample_fields(points_xy, t_abs) → (cov, den, rain, fog, gust)` vectorised core; `.cells` (active, nearest first); `.current` (label); `force_weather(type\|None)` dev override; `get_delta`/`apply_delta`. |
| `WeatherMap(config)` (M3) | Square `(cells, cells, 4)` float32 raster cache of the four spatial channels around a moving center (`weather_map_cells` × `weather_map_cell_m`). `rasterize(system, center_xy, t_abs) → (N,N,4)`; `texel_centers(center_xy) → (N*N, 2)`; `.cells`/`.cell_m`/`.span_m`. Layout `out[row=Y, col=X, channel]`. Pure derivation of the sim (never saved). |
| `MAP_CHANNELS` | `("coverage", "density", "precip", "fog")` — the raster's last-axis channel order. |

## Imports Allowed
`core` (config, rng), numpy, stdlib.  From M8: `wind` (modifier seam only —
`wind/` never imports `weather/`, no cycle).  **Never panda3d** (headless;
AST-guarded once the package has its leak test).

## Events
Published: `WeatherChangedEvent` (deferred) when the **committed** discrete
label changes — i.e. after the classification hysteresis (`HYSTERESIS_SECONDS`
= 60 game s) so the label never flickers at a threshold.  M7 adds
`LightningStrikeEvent` via the consumer.
Subscribed: none.

## Units & Invariants
- Meters, m/s, **game seconds** (1 game hour = 3600 game s; synoptic flow
  follows the game clock, unlike `wind/`'s real-time gust clock).
- `W(t) = C + Σ aᵢ sin(ωᵢ t + φᵢ)` per axis; `D(t)` is its exact analytic
  integral with `D(0) = (0, 0)`.
- Speed band is a hard guarantee (amplitude budget — see synoptic.py
  docstring); direction swings up to ±~50° around a per-world prevailing
  heading with periods of `weather_synoptic_period_min_h..max_h` game hours.
- Determinism: pure function of (world_seed, t). Two instances with the same
  seed are bit-identical (`tests/test_weather_synoptic.py`).
- **Cells** (M2): one `for_domain("weather", "regime", day)` draw picks the
  day regime; `weather_spawn_slots_per_day` draws of
  `for_domain("weather", "cell", day, slot)` accept against the regime spawn
  probability.  A cell's footprint is `intensity·exp(−(d/radius)²·ln 50)` (1/50
  of peak at one radius); envelope = smoothstep grow (first 20 %) · plateau ·
  smoothstep decay (last 30 %); `radius(t)` grows 0.55→1.0 of `radius_m`.
- **Sampling**: `sample_local` = regime ambient (cosine-blended across the
  midnight hand-off) + Σ cell contributions to coverage/density/rain, FOG_BANKs
  to fog; wind dir = synoptic, speed = `syn·(0.7+0.5·coverage) + Σ storm_gust`.
- Regime ambient coverage sits cleanly inside the `classify` buckets so a
  cell-free day reads as its regime (HIGH→clear, MIXED→cloudy, FRONTAL→overcast).

## Examples
```python
from fire_engine.core import EventBus, load_config, set_world_seed
from fire_engine.weather import Synoptic, WeatherSystem

set_world_seed(1337)
syn = Synoptic(load_config())
(ux, uy), v = syn.wind(2 * 86400.0 + 9.5 * 3600.0)   # day 2, 09:30
carry = syn.displacement(7200.0) - syn.displacement(3600.0)  # 1 h of drift (m)

ws = WeatherSystem(load_config(), EventBus())
lw = ws.update(game_day=2, game_time_of_day=9.5 * 3600.0, player_pos=(0.0, 0.0))
print(ws.current, lw.rain_intensity, [c.kind.value for c in ws.cells])
```

## Gotchas
- `WeatherSystem.force_weather` scales the synoptic speed by a per-state
  multiplier (STORM ×1.9, FOG ×0.30) before it reaches `SkyState.wind_speed`
  — gameplay wind ≠ raw synoptic wind.  Anything advected by `D(t)` (storm
  cells) must use the **raw** synoptic flow (`cell.center(t, synoptic)`), never
  the multiplied value.
- `sky/weather.py` is now a **compatibility shim** that re-exports
  `WeatherType`/`WeatherSystem`/`LocalWeather` from `fire_engine.weather`.  New
  code imports from `fire_engine.weather` directly.
- `update(..., player_pos)` defaults to the **origin** when `player_pos` is
  `None` (the renderer threads the camera position through from M4); until
  then natural weather is sampled at (0, 0).
- The discrete `current` label has 60-game-s hysteresis — it lags the raw
  `classify(sample)` by up to a minute.  Read `LocalWeather` fields, not the
  label, for continuous values.
- Don't sample `wind_vec`/`sample_local` with huge `(M,)` arrays per frame —
  meant for scalar/per-cell evaluation; the per-frame budget is scalar-only.
- `force_weather` is a dev shim over the legacy global states; the real spatial
  summon API lands in M8.  Legacy (old-Markov) save deltas still load.
- **M4 GPU contract**: `WeatherMap.rasterize` output is packed by
  `fire_engine.sky.pack_weather_map` (fp16 BGRA, row-major — no transpose) and
  uploaded by `world/weather_renderer.py::WeatherMapComponent`.  The cloud
  shader samples the resulting texture at the **RAW world XY** — never add the
  wind drift, because the raster already bakes in cell motion each re-raster
  (adding `u_wind` would double-advect the storm off its own rain).  Render
  bridges live in `world/` (Hard Rule 1); the packer is headless in `sky/`.
- The `WeatherMapComponent` only **reads** the weather system (`rasterize` is a
  pure fn of seed/center/`t_abs`); the `SkyRendererComponent` is the single
  driver of `sky_system.update(player_pos)`.  Don't add a second `update` caller
  or weather double-advances per frame.
