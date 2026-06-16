# render.vegetation._impl — System Doc
keywords: render vegetation _impl, private vegetation implementation, tree_build, tree_occluders, tree_species_cache, zone_renderer, leaf_litter, mote_utils, build_volume, build_volumes, rebuild_volume, setup_zone_events, on_terrain_event, update_weather, TreeOccluderSet, TreeSpeciesCache, species cache, canopy extinction, leaf litter component, LeafLitterComponent, mote_utils, additive mote helpers

> One doc per code package; filename matches the package exactly (`docs/systems/render.vegetation._impl.md` <-> `fire_engine/render/vegetation/_impl/`).

## Role

`render/vegetation/_impl/` is the **private implementation helpers** package for
`fire_engine.render.vegetation`.

It exists to satisfy the <=500-line module limit (Hard Rule 8): method clusters extracted from the
large vegetation-renderer components live here as free functions taking the owning component
instance as their first argument, or as separate standalone helpers.  This package is **not a
public API** -- nothing outside `render/vegetation/` should import from it directly.

Modules:

- `tree_build.py` -- volume-build helpers for `TreeRendererComponent`: `build_volume`,
  `build_volumes`, `rebuild_volume`.  Creates GeomNodes, uploads RGBA32F instance data textures,
  sets BoundingBox + `set_final`, and fires `GpuLightingPipeline.set_static_occluders`.
- `tree_occluders.py` -- helpers for building `TreeOccluderSet` from placed tree instances
  (heights/canopy radii/bounce colours/per-meter canopy extinction sigma from species pool data).
- `tree_species_cache.py` -- `TreeSpeciesCache`: loads and caches per-species `TreeMesh` variant
  pools, computes shared impostor quad sizes, and maintains species-level extents for occluder
  building.
- `zone_renderer.py` -- shared lifecycle helpers for all zone-instanced vegetation renderers
  (`GrassRendererComponent`, `FloraRendererComponent`, `TreeRendererComponent`): `setup_zone_events`,
  `on_terrain_event`, `update_weather` and attribute-contract documentation.
- `leaf_litter.py` -- `LeafLitterComponent`: GPU-instanced leaf-litter particles underneath tree
  canopies.  Shares the mote instancing pattern but is driven by tree-placement data.
- `mote_utils.py` -- shared helpers for `DustMoteComponent` and `LeafLitterComponent`: unit-quad
  Geom setup, additive node configuration, per-frame hash-driven position/phase update.

`render/vegetation/_impl/` deliberately does NOT: export symbols to users of
`fire_engine.render.vegetation`, define the public component classes, or hold game logic.

## Public API

No public API -- all symbols are internal to `render/vegetation/`.

Notable free functions and classes (internal use only):

| Module | Symbol | Description |
|---|---|---|
| `tree_build` | `build_volume(self_obj, kind, vol)` | Build all GeomNodes for one ZoneVolume of trees/bushes (per-species/variant instanced meshes + impostor nodes). |
| `tree_build` | `build_volumes(self_obj, kind)` | Build all volumes for a given kind (trees or bushes). |
| `tree_build` | `rebuild_volume(self_obj, kind, vol)` | Detach old nodes, re-bake placements, rebuild the volume (called on TerrainEditedEvent). |
| `tree_occluders` | `make_tree_occluder_set(placements, species_pool)` | Build a `TreeOccluderSet` from placed-instance arrays and the species pool extents. |
| `tree_species_cache` | `TreeSpeciesCache(config)` | Loads and caches per-species TreeMesh variant pools. `get(species_name)` returns the variant list; `impostor_size(species_name)` returns the raster-pool quad dimensions (meters). |
| `zone_renderer` | `setup_zone_events(self_obj, bus)` | Subscribe to ChunkLoadedEvent and TerrainEditedEvent; mark dirty volume sets. |
| `zone_renderer` | `on_terrain_event(self_obj, event)` | Handle a terrain event: mark the chunk columns intersecting the event footprint as dirty. |
| `zone_renderer` | `update_weather(self_obj, state)` | Push per-frame wind/sway/rain shader inputs onto the renderer root. |
| `leaf_litter` | `LeafLitterComponent(base, sky_system, tree_renderer, config)` | GPU-instanced leaf-litter particles underneath tree canopies; driven by tree-placement data from the given `tree_renderer`. |
| `mote_utils` | `build_mote_node(name, base, count) -> NodePath` | Build a camera-local wrapping additive instanced particle node. |

## Imports Allowed

- `panda3d.*`, `direct.*` (Hard Rule 1: all modules here live in `render/`)
- `fire_engine.core` (math3d, rng, shader_source, event bus)
- `fire_engine.zones` (bake_tree_instances, instances_data_block, species_mix_from_params, etc.)
- `fire_engine.world.terrain` (ChunkLoadedEvent, TerrainEditedEvent)
- `fire_engine.lighting` (GpuLightingPipeline, TreeOccluderSet)
- `fire_engine.render.bridges` (texture/geometry bridges)
- `fire_engine.render._impl.quad` (shared quad geometry)
- Parent component types via TYPE_CHECKING guards only
- Python standard library, `numpy`

## Events

Published: none.

Subscribed: `zone_renderer.setup_zone_events` wires ChunkLoadedEvent and TerrainEditedEvent
subscriptions -- but the subscription call is initiated by the public component classes, not from
within `_impl/` directly.

## Units & Invariants

- All spatial values in world **meters**, Z-up.
- `build_volume` sets an infinite BoundingBox (+-1e6 m) + `set_final` on every instanced node.
- `tree_species_cache` caches mesh data for the session; `clear()` exists for test isolation.
- `zone_renderer.update_weather` pushes `u_sway_base`, `u_sway_gust`, `u_gust_freq`, and
  `u_rain_intensity` from the SkyState -- must be called once per `late_update`.
- `LeafLitterComponent` follows the same `u_time_s` self-bind discipline as other particle
  renderers (not inherited from `render`).

## Examples

```python
# Internal usage only -- do not import from render.vegetation._impl directly.
# From render/vegetation/tree_renderer.py:
from fire_engine.render.vegetation._impl.tree_build import build_volumes, rebuild_volume
from fire_engine.render.vegetation._impl.zone_renderer import setup_zone_events, update_weather

setup_zone_events(self, bus)    # self = TreeRendererComponent instance
build_volumes(self, "trees")
# ... per late_update:
update_weather(self, sky_state)
```

## Gotchas

- All functions take the **owning component as first arg** -- call as `func(self, ...)`, not
  `self.func(...)`.
- `TreeSpeciesCache` is a per-renderer instance (not a module-level singleton) -- two
  `TreeRendererComponent` instances each get their own cache.  In tests, call `cache.clear()`
  between cases to avoid cross-contamination.
- `zone_renderer.setup_zone_events` subscribes to events on the bus passed in; if `bus` is None
  (CPU backend or testing) it is a no-op -- the caller must handle the None case.
- `rebuild_volume` detaches old NodePaths synchronously; do not hold references to them after
  the call (Panda3D will garbage-collect the underlying nodes).
- `LeafLitterComponent` reads from the `tree_renderer` instance directly -- construct the tree
  renderer before the leaf litter component or you will get an attribute error on `_placements`.
