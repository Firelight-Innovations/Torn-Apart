# world — System Doc
keywords: world, environment, natural environment, grouping package, terrain, weather, wind, sky, headless, simulation, layer 2, layer 3, chunk, storm, atmosphere, voxel, biome

> One doc per code package; filename matches the package exactly (`docs/systems/world.md` ↔ `fire_engine/world/`).

## Role

`world/` is the **grouping package** for the engine's natural-environment simulation systems (ARCHITECTURE.md §4a — Layer 2 Structure and Layer 3 Services).  It exports nothing of its own; all real APIs live in its sub-packages.

These are all **headless** packages (numpy + `fire_engine.core` only — **no panda3d imports**).  Their Panda3D upload/render bridges live in `fire_engine.render` (formerly `world/` before the package reorg), not here.

Sub-packages:

| Sub-package | Purpose |
|---|---|
| `world.terrain` | 32³-voxel chunk terrain: generation, dual-contouring mesher, brush editing, streaming, rain-cover heightmap. |
| `world.weather` | Spatial storm-cell weather simulation: day regimes, drifting cells, lightning, wetness, humidity, wind attachment, saveable summons. |
| `world.wind` | Spatially-varying wind field: spectral gust simulation, terrain venturi acceleration, worker-thread recenter, GPU texture pack. |
| `world.sky` | Physically-based sky and celestial: Rayleigh+Mie atmosphere, sun/moon geometry, cloud noise, `SkyState` snapshot, `SkySystem` composer. |

`world/` deliberately does NOT: export any symbol from `__init__.py`; import panda3d (fully headless-testable); implement any rendering, lighting, or scene-graph operations.

## Public API

`fire_engine.world` itself exports nothing (`__init__.py` only holds the module docstring).  Import sub-package APIs directly:

```python
from fire_engine.world.terrain import ChunkManager, apply_brush
from fire_engine.world.weather import WeatherSystem, WeatherType
from fire_engine.world.wind import WindField
from fire_engine.world.sky import SkySystem, SkyState
```

See the individual sub-package docs for complete API tables:
- `docs/systems/world.terrain.md`
- `docs/systems/world.weather.md`
- `docs/systems/world.wind.md`
- `docs/systems/world.sky.md`

## Imports Allowed

Per ARCHITECTURE.md §4a.2, `world/` as a grouping package has no direct imports in its own `__init__.py`.  Each sub-package observes its own import rules — see the individual sub-package docs.  No sub-package in `world/` may import panda3d.

General rule: sub-packages may import `core`, `numpy`, and sibling `world/` sub-packages where the dependency is downward or lateral (e.g. `sky` reads `weather`; `weather` reads `wind`; neither reverses).  No upward imports into `render`, `lighting`, or higher layers.

## Events

### Published
Sub-packages publish their own events.  None are published from `world/__init__.py` itself.

| Event | Source sub-package |
|---|---|
| `WeatherChangedEvent` | `world.weather` (via `WeatherSystem.update`) |
| `LightningStrikeEvent` | `world.weather` (via `WeatherSystem.update`) |
| `TerrainEditedEvent` | `world.terrain` (via `apply_brush`) |

### Subscribed
None at this level.

## Units & Invariants

- All sub-packages are headless — no panda3d imports anywhere in `world/`.
- World space is **Z-up, forward = +Y, east/right = +X** (Panda3D native convention).
- Distances in **meters**; time in **seconds** of in-game time; voxel = 0.5 m; chunk = 32³ voxels = 16 m.
- All randomness must use `core.rng.for_domain(*keys)` — never `random.*` or unseeded `np.random.*`.  Determinism is enforced: same `world_seed` + same inputs → identical outputs (tested in the characterization suite).
- No per-voxel/per-vertex Python loops; all bulk work uses numpy array expressions (Hard Rule 4).
- Each sub-package is `Saveable` (terrain, weather) or trivially re-derived from seed+clock (wind, sky).

## Examples

```python
# Typical engine boot wiring — one import per used sub-package.
from fire_engine.core import Clock, EventBus, load_config, set_world_seed
from fire_engine.world.terrain import ChunkManager
from fire_engine.world.sky import SkySystem

cfg = load_config()
set_world_seed(cfg.world_seed)
bus = EventBus()
clock = Clock(fixed_dt=cfg.fixed_dt, bus=bus)

terrain = ChunkManager(cfg, bus)    # streams and persists voxel chunks
sky = SkySystem(cfg, clock, bus)    # headless sky/weather composer

# Per-frame update order:
def frame(real_dt: float) -> None:
    clock.update(real_dt)
    state = sky.update()            # advances weather, returns SkyState snapshot
    terrain.stream_frame(...)       # load/unload chunks around camera
    bus.drain()
```

## Gotchas

1. **`world/` is a grouping package only** — it has no public API of its own.  Always import from a sub-package (`world.terrain`, `world.sky`, etc.).  Never `from fire_engine.world import ...`.

2. **No panda3d here.** All rendering bridges live in `fire_engine.render` (formerly `world/` before the 2026-06-13 reorg).  If you need to upload a mesh, texture, or bind a shader uniform, look in `render/`.

3. **Sub-package inter-dependencies are downward only**: `sky` reads `weather`; `weather` reads `wind` (lazily via `GustFront`); `wind` reads `terrain` (venturi).  Never introduce upward cycles.  `terrain` has no inbound `world/` deps.

4. **Register Saveables with SaveManager explicitly** — `world/` does not auto-register.  Typical wiring: `save_manager.register(terrain)` and `save_manager.register(sky.weather)`.

5. **Thread safety**: `wind` field recenter runs off the main thread (worker); `terrain` meshing runs off the main thread (worker).  `sky` and `weather` are single-threaded (call from the main loop only).  Do not share chunk or cell state across threads without the manager's documented API.
