# weather — System Doc
keywords: weather, synoptic, wind direction, prevailing wind, storm, storm cell, front, air mass, displacement, D(t), drift, steering current, volumetric weather

> Status: under construction (volumetric-weather branch). M1 (synoptic flow)
> shipped; storm cells, weather map, emergent fog, lightning land in M2–M8.
> This doc grows with each milestone.

## Role
Headless spatial weather simulation — the layer between `sky/` (which
composes per-frame `SkyState`) and `wind/` (local gusts).  Owns the
**synoptic flow**: the slow, hours-scale steering current that sets the base
wind direction/speed and (from M2) carries storm cells across the world.
Everything is a closed-form pure function of (world_seed, game time) — no
integrated state, zero save bytes for natural weather.  Deliberately does
NOT: render anything (bridges live in `world/`), simulate local gusts
(that's `wind/`), or own `SkyState` (that's `sky/`).

## Public API
| Export | Description |
|---|---|
| `Synoptic(config)` | Seeded closed-form synoptic wind. Built once from `for_domain("weather", "synoptic")`. |
| `Synoptic.wind(t_abs)` | `((ux, uy), speed_ms)` — unit direction + speed at absolute game time. Speed guaranteed inside `[weather_synoptic_speed_min_ms, ..._max_ms]`. |
| `Synoptic.wind_vec(t_abs)` | Vector form; accepts scalar or `(M,)` ndarray of times → `(2,)` / `(M, 2)`. |
| `Synoptic.displacement(t_abs)` | Air-mass displacement `D(t)` in meters since t=0; `dD/dt ≡ wind_vec` to machine precision. Storm-cell centers ride `spawn_pos + D(t) − D(spawn_time)`. |

## Imports Allowed
`core` (config, rng), numpy, stdlib.  From M8: `wind` (modifier seam only —
`wind/` never imports `weather/`, no cycle).  **Never panda3d** (headless;
AST-guarded once the package has its leak test).

## Events
Published: none yet (M7 adds `LightningStrikeEvent` via the consumer).
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

## Examples
```python
from fire_engine.core import load_config, set_world_seed
from fire_engine.weather import Synoptic

set_world_seed(1337)
syn = Synoptic(load_config())
(ux, uy), v = syn.wind(2 * 86400.0 + 9.5 * 3600.0)   # day 2, 09:30
carry = syn.displacement(7200.0) - syn.displacement(3600.0)  # 1 h of drift (m)
```

## Gotchas
- `sky/weather.py::WeatherSystem` scales the synoptic speed by a per-state
  multiplier (STORM ×1.9, FOG ×0.30) before it reaches `SkyState.wind_speed`
  — gameplay wind ≠ raw synoptic wind.  Anything advected by `D(t)` (storm
  cells) must use the **raw** synoptic flow, never the multiplied value.
- Don't sample `wind_vec` with huge `(M,)` arrays per frame — it's meant for
  scalar/per-cell evaluation; the per-frame cost budget is scalar-only.
