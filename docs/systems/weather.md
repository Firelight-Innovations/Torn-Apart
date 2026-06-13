# weather — System Doc
keywords: weather, synoptic, wind direction, prevailing wind, storm, storm cell, front, air mass, displacement, D(t), drift, steering current, volumetric weather, regime, shower, thunderstorm, cloud bank, fog bank, classify, WeatherType, LocalWeather, sample_local, force_weather, humidity, relative_humidity, saturation_humidity, emergent fog, condensation, condense, ground fog, rain_recent_at, wetness_at, dew, mist, cloud genera, WMO genera, CloudGenus, cloud_layers, classify_genus, CloudLayers, CloudBand, cirrus, cirrostratus, altocumulus, altostratus, stratus, stratocumulus, cumulus, cumulonimbus, nimbostratus, cloud band, cloud altitude band, anvil, cirrus residual

> Status: under construction (volumetric-weather branch). M1 (synoptic flow),
> M2 (storm cells + local sampling + classify), M3 (weather map + ground
> wetness) and M5 (emergent humidity + condensation fog) shipped; lightning and
> the spatial summon API land in M7–M8.  This doc grows with each milestone.

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
| `LocalWeather` | Frozen local sample: cloud_coverage/density, fog_density, rain_intensity, wind_dir/speed, humidity, wetness, temperature_c. First six map 1:1 onto `SkyState`. `humidity`/`wetness`/`temperature_c` are all live (M5). |
| `WeatherSystem(config, bus=None)` | The system (`save_key="weather"`). `update(day, tod, player_pos=None) → LocalWeather`; `sample_local(pos_xy, t_abs) → LocalWeather`; `sample_fields(points_xy, t_abs) → (cov, den, rain, fog, gust)` vectorised core (fog includes emergent condensation); `wetness_at(points, t)` / `rain_recent_at(points, t)` closed-form moisture quadratures; `.cells` (active, nearest first); `.current` (label); `force_weather(type\|None)` dev override; `get_delta`/`apply_delta`. |
| `humidity.py` (M5) | Emergent-fog formulas, all vectorised pure fns of resolved `Config`: `humidity_base(day, cfg)` (seeded per-day calm baseline); `relative_humidity(rain_recent, wetness, h_base, cfg)`; `saturation_humidity(T_c, cfg)` (rises with T); `condense_fraction(humidity, h_sat, cfg)`; `wind_gate(wind_speed, cfg)`; `emergent_fog(humidity, T_c, wind_speed, cfg) → (N,)` fog coefficient (1/m). |
| `WeatherMap(config)` (M3) | Square `(cells, cells, 4)` float32 raster cache of the four spatial channels around a moving center (`weather_map_cells` × `weather_map_cell_m`). `rasterize(system, center_xy, t_abs) → (N,N,4)`; `texel_centers(center_xy) → (N*N, 2)`; `.cells`/`.cell_m`/`.span_m`. Layout `out[row=Y, col=X, channel]`. Pure derivation of the sim (never saved). |
| `MAP_CHANNELS` | `("coverage", "density", "precip", "fog")` — the raster's last-axis channel order. |
| `CloudGenus` (M9) | The 8 WMO genera the appearance model expresses: `CIRRUS`/`CIRROSTRATUS` (high), `ALTOCUMULUS`/`ALTOSTRATUS` (mid), `STRATOCUMULUS`/`STRATUS`/`CUMULUS`/`CUMULONIMBUS` (low). `str` Enum (`.value` = `"cirrus"`…`"cumulonimbus"`). NIMBOSTRATUS (low thick rain layer) is folded into STRATUS at high precip. |
| `CloudBand` (M9) | `HIGH`/`MID`/`LOW` `int` Enum (0/1/2) — the altitude band a genus renders in; also the GPU band order. `BAND_HIGH`/`BAND_MID`/`BAND_LOW` are the bare-int aliases. |
| `classify_genus(coverage, density, precip, regime) → (high, mid, low)` (M9) | Dominant genus per band — pure, vectorised fn of the sampled fields + the day `Regime`. Scalar in → a 3-tuple of `CloudGenus`; array in → three object ndarrays. CLOUD_BANK→stratus family, SHOWER→STRATUS rain layer, THUNDERSTORM(precip)→CUMULONIMBUS, HIGH_PRESSURE residual→CIRRUS, FRONTAL overcast→cirrostratus/altostratus/stratocumulus stack. |
| `cloud_layers(coverage, density, precip, regime, config) → CloudLayers` (M9) | Continuous per-band layer params (the renderer's drive): `CloudLayers` bundles `genus_high/mid/low` + length-3 (high,mid,low) ndarrays `base_altitude_m` (strictly decreasing high→low), `thickness_m` (storms deepen the low slab), `coverage`, `density` (cirrus always thin), `detail_scale`. Pure, deterministic, continuous in the inputs (no jumps); zero save bytes. |

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
- **Emergent fog** (M5): fog is *not* a state — it condenses.
  `humidity = clamp(h_base(day) + rain_gain·rain_recent + wetness_gain·wetness,
  0, 1)` (`h_base` seeded per day, cosine-blended across midnight like the
  ambient); it condenses where it exceeds the **temperature-dependent**
  saturation `h_sat = clamp(sat_base + sat_slope·(T − sat_ref), 0.5, 1.0)`
  (rises with T → fog forms in the cold, not the heat). `condense =
  smoothstep(humidity − h_sat, 0, condense_band)`; wind gate `= 1 −
  smoothstep(wind_speed, fog_wind_full, fog_wind_none)` (full ≤1 m/s, none
  ≥3 m/s). `emergent_fog = fog_emergent_max·condense·gate`, **added** to the
  baseline + FOG_BANK fog and capped at `weather_fog_max_density`. So a calm
  humid night after evening rain grows ground fog through the cool pre-dawn,
  which burns off as the warming air's `h_sat` climbs back over the humidity.
- **WMO cloud genera** (M9): `cloud_layers` / `classify_genus` are a pure,
  closed-form, **vectorised** map from the *already-sampled* fields
  (coverage/density/precip + day `Regime`) onto layered altitude bands — so
  they cost **zero save bytes** and are **identical between the weather-map
  raster and a local sample** (same fields in → same layers out). The mapping
  is a strong function of the regime/cell hint: CLOUD_BANK (cover, no precip) →
  stratus/stratocumulus; SHOWER (moderate precip) → STRATUS rain layer
  (nimbostratus role); THUNDERSTORM (high precip) → CUMULONIMBUS tower (deeper,
  darker low slab); HIGH_PRESSURE residual cover → CIRRUS high (fair-weather
  "mares' tails", always present via a small cover floor); FRONTAL overcast
  stacks cirrostratus → altostratus → stratocumulus. **Band altitudes are
  strictly ordered** (`base_altitude_m[HIGH] > [MID] > [LOW]`) and the high band
  density is always capped low (`cloud_genera_high_density`) — ice cloud is
  thin. All `cloud_genera_*` tunables live in `[weather]`/`Config`.
- `rain_recent_at` is the same fixed-offset exponential quadrature as
  `wetness_at` but with a longer decay (`weather_humidity_recent_tau_s`, ~5 h):
  the air stays muggy for hours after a shower while the ground dries in ~1 h,
  so evening rain still feeds pre-dawn humidity. Both are pure fns of
  (seed, t, pos) — zero save bytes, recompute on load.

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
- **M9 genera reach the GPU with NO new texture data.** `cloud_layers` is the
  *canonical headless* mapping (CPU consumers / tests / future content), but the
  cloud shader does **not** read a genus code — it re-derives the same three
  altitude bands from the existing `coverage`/`density`/`precip` weather-map
  channels plus the `cloud_genera_*` band uniforms. This was deliberate: packing
  a genus into a spare sub-channel would touch the M3/M4 weather-map **4-channel
  packing contract**, which is shared/locked. `MAP_CHANNELS` is unchanged
  (`coverage, density, precip, fog`). If you ever *do* need genus on the GPU as
  data, add a *separate* small lookup — never repurpose a weather-map channel.
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
- **Emergent fog recursion guard**: `_sample_core` (no emergent fog) is the
  routine the rain-history quadratures (`wetness_at` / `rain_recent_at`) call;
  `sample_fields` adds the emergent term on top.  Emergent fog *depends* on the
  rain history, so the quadratures must NOT call `sample_fields` (that would
  recurse) — they call `_sample_core`. Keep that split.
- Emergent fog needs **calm air** (the wind gate shuts above ~3 m/s) and a
  point with rain in its recent past.  With the default synoptic band the flow
  rarely drops below ~3 m/s and cells race across the map, so *natural* fixed-
  point fog is uncommon — by design (fog is an occasional emergent treat, not a
  daily event). In-game demo / tests force it via a near-still synoptic band
  (which also keeps an injected cell roughly stationary so a point actually gets
  rained on); see `tests/test_weather_fog.py`.
