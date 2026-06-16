# world.weather._impl — System Doc
keywords: weather impl, private impl, _impl, _save, _sampling, _summon, _update, WeatherSystem internals, method cluster, local_to_dict, local_from_dict, cell_to_dict, cell_from_dict, get_delta, apply_delta_summons, apply_delta_override, sample_core, sample_fields, sample_local, wetness_at, emergent_fog, local_wind_speed, temperature, summon_cell, suppress, clear_all, do_update, update_gust_fronts, emit_lightning, classified_state, _lerp_local, _smoothstep

## Role
Private implementation helpers for `fire_engine.world.weather.WeatherSystem` — not a
public API.  This sub-package exists solely to keep `system.py` under the 500-line
module limit (Hard Rule 8) by extracting coherent method clusters into dedicated
private modules.  Each module's functions receive the `WeatherSystem` instance as
their first argument (`ws`) and are called only from the matching stubs inside
`WeatherSystem`.  **Do NOT import from `fire_engine.world.weather._impl` outside
the `fire_engine.world.weather` package.**  The public surface of the weather system
lives entirely in `fire_engine.world.weather` (`docs/systems/world.weather.md`).

The four modules partition the system's responsibilities:

- `_sampling.py` — spatial field sampling (coverage, density, rain, fog, wetness,
  humidity, temperature, wind speed) at arbitrary world positions.
- `_summon.py` — M8 summon/suppress/clear API that creates and removes
  `StormCell` entries.
- `_save.py` — `Saveable` protocol cluster: serialise/deserialise `LocalWeather`
  and `StormCell` to plain-primitive dicts, produce and consume the system's
  save delta.
- `_update.py` — per-frame update loop: blending, hysteresis, gust-front
  coupling, lightning emission, weather-change event dispatch.

## Public API
This package exposes no public API.  All symbols are private (`_`-prefixed
modules; the `__all__` re-exports those module objects as implementation
handles only).  The canonical public API is `fire_engine.world.weather`.

## Imports Allowed
Same constraints as the parent `fire_engine.world.weather` package:
`core` (config, rng, event_bus), `numpy`, stdlib.  `_update.py` lazily imports
`fire_engine.world.wind.GustFront` (inside the `update_gust_fronts` function body)
to avoid a circular dependency — `wind` never imports `weather`.  **Never panda3d.**
Cross-`_impl` imports (e.g. `_update.py` importing `_sampling.py`) are allowed
because they are still within the private boundary.

## Events
None directly published here.  `_update.do_update` calls `ws._bus.publish_deferred`
with `WeatherChangedEvent` and `LightningStrikeEvent`; the event types are
defined in `fire_engine.core.event_bus`.  See `docs/systems/world.weather.md ##
Events` for the full contract.

## Units & Invariants
- All position arguments are world XY in **meters**; time arguments are absolute
  **game seconds** (1 game hour = 3600 s).
- Temperature returned by `temperature()` is in **degrees Celsius**.
- Wind speed arrays from `local_wind_speed()` are in **m/s**.
- Fog coefficient arrays from `emergent_fog()` are in **1/m** (extinction coeff).
- `sample_core` intentionally omits the emergent-fog condensation term so that
  `wetness_at` / `rain_recent_at` quadratures can call it without recursion;
  `sample_fields` is the only caller that adds the emergent term.
- `_lerp_local` renormalises `wind_dir` after lerp; short-circuits at t≤0 / t≥1
  so a completed blend is bit-exact equal to its endpoint.
- `get_delta()` returns `{}` for unmodified natural weather (no summons, no
  suppressions, no override): zero save bytes for the common case.
- `apply_delta_summons` / `apply_delta_override` are fault-tolerant: malformed or
  legacy dict entries are skipped silently, never fatal.

## Examples
```python
# These are internal entry points — external code uses WeatherSystem methods.
# Shown for maintainers tracing execution:

from fire_engine.world.weather._impl import _sampling, _save, _summon, _update

# _sampling: sample the spatial fields at two world positions
import numpy as np
pts = np.array([[0.0, 0.0], [500.0, 500.0]], dtype=np.float64)
cov, den, rain, fog_bank, gust = _sampling.sample_core(ws, pts, t_abs=3600.0)

# _save: round-trip a LocalWeather through plain primitives
d = _save.local_to_dict(local_weather)
restored = _save.local_from_dict(d)

# _summon: spawn a thunderstorm 2 km upwind of the player
cell_id = _summon.summon_cell(
    ws, CellKind.THUNDERSTORM,
    time_abs=3600.0, player_pos=(0.0, 0.0), upwind_m=2000.0
)

# _update: core update is called from WeatherSystem.update — never call directly
```

## Gotchas
- `sample_core` (no emergent fog) is the function the rain-history quadratures
  (`wetness_at` / rain-recent variants) call internally; calling `sample_fields`
  from those quadratures would recurse.  Keep the two-level split.
- `_update.do_update` imports `_sampling.sample_local` and
  `fire_engine.world.wind.GustFront` lazily (inside the function body) to break
  potential import cycles.  Do not hoist these to module-level imports.
- `suppress` in `_summon.py` differentiates summoned cells (id prefix `"s:"` —
  dropped outright) from natural cells (id prefix `"n:"` — added to the
  suppression set).  Mixing the strategies is intentional: a summoned cell has no
  underlying schedule entry to suppress.
- `apply_delta_summons` bumps `_summon_seq` past any restored summoned-cell id so
  fresh summons after a load never collide with restored ids.
- `_smoothstep` in `_update.py` returns a Hermite curve clamped to [0, 1];
  `_lerp_local` uses it to blend from the override snapshot (`_override_from`) to
  the forced-weather target over `blend_seconds` game seconds.
- All modules in this package are consumed exclusively via `WeatherSystem` — do
  not reach into `_impl` from tests or render bridges.
