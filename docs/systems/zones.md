# zones — System Doc
keywords: zone, volume, region, grass, vegetation, foliage, blades, tuft, instancing, gl_InstanceID, biome, spawn area, box volume, height field, placement, density, ZoneVolume, ZoneStore, trees, leaves, leaf litter, wind particles, flora, flowers, flower, bushes, bush, shrub, scrub, tree sprites, flora_placement, flora_hash_seed, flora_instance_count, flora_instance_attribs, FLORA_KINDS, atlas variant

## Role
Tagged axis-aligned box volumes in world space, and the math behind systems that act on them. A `ZoneVolume` with `tag="grass"` tells the GPU grass renderer (`world/grass_renderer.py`) where blades grow; `tag="flowers"` / `tag="bushes"` / `tag="trees"` volumes tell the flora renderer (`world/flora_renderer.py::FloraRendererComponent`) where wildflowers, scrub bushes and tree sprites stand; `tag="trees"` volumes ALSO tell the wind system's leaf-litter renderer (`world/mote_renderer.py::LeafLitterComponent`) where fallen leaves scatter and stream on gusts (one forest volume → trees + their litter); `tag="biome"` volumes are reserved for surface-material regions (snow / bare dirt — planned, not yet consumed). The `ZoneStore` registry holds all volumes and participates in delta saves (`save_key="zones"`). `grass_placement.py` and `flora_placement.py` are the headless, testable half of the GPU-instanced vegetation: the python mirrors of the shaders' instance hash chains, the per-volume instance counts, and the height-field bake. This package deliberately does NOT touch panda3d, render anything, or own per-plant data — blades and plants exist only on the GPU.

## Public API
- `ZoneVolume(id, tag, min_corner, max_corner, biome=None, params={})` — frozen dataclass AABB (world meters). `contains_xy(wx, wy)`, `intersects_chunk(coord, chunk_meters)`, `area_xy_m2`, `size_m`, `to_dict()`/`from_dict()`.
- `ZoneStore` — registry + Saveable (`save_key="zones"`). `add(tag, min_corner, max_corner, *, biome=None, params=None) -> ZoneVolume` (assigns id), `remove(id)`, `volumes(tag=None)`, `get(id)`, `mark_baseline()`, `version` (monotonic change counter renderers watch), `get_delta()`/`apply_delta()`.
- `hash_lowbias32(x)` — vectorized uint32 hash; line-for-line mirror of `lowbias32` in `world/grass_shaders.py`.
- `instance_attribs(indices, seed, min_corner, max_corner)` — per-instance `x`, `y`, `rot`, `scale`, `phase` arrays; EXACTLY what the vertex shader derives from `gl_InstanceID`.
- `grass_hash_seed(volume)` — per-volume shader seed via `for_domain("zones", "grass", volume.id)`.
- `grass_instance_count(volume, config)` — `density × area_xy_m2`, density from `params["density"]` or `config.grass_density_per_m2`, clamped to `config.grass_max_instances`.
- `leaf_hash_seed(volume)` — per-`"trees"`-volume shader seed via `for_domain("wind", "leaves", volume.id)` (the wind system's leaf-litter instance chain; mirrors `grass_hash_seed`).
- `leaf_instance_count(volume, config)` — `density × area_xy_m2`, density from `params["leaf_density"]` or `config.wind_leaf_density_per_m2`, clamped to `config.wind_leaf_max_instances`. The pure-function mirror of what `LeafLitterComponent` instances per `"trees"` volume (it holds no per-leaf CPU state, exactly like grass).
- `bake_grass_height_field(volume, chunks, config)` — `(H, W, 4) uint8` field, 1 texel/voxel (0.5 m); R encodes surface height inside the volume's Z window, `HEIGHT_SENTINEL` (255) = no surface → shader culls the blade. Generic over any volume — the flora renderer bakes the SAME field for flower/bush/tree volumes.
- `FLORA_KINDS` — `("flowers", "bushes", "trees")`, the zone tags the flora renderer consumes (`flora_placement.py`).
- `flora_hash_seed(volume, kind)` — per-(volume, kind) shader seed via `for_domain("zones", "flora", kind, volume.id)`; a `"trees"` volume's tree seed never collides with its `leaf_hash_seed`.
- `flora_instance_count(volume, config, kind)` — `density × area_xy_m2`, density from `params["density"]` or `config.flora_<kind>_density_per_m2`, clamped to `config.flora_<kind>_max_instances`.
- `flora_instance_attribs(indices, seed, min_corner, max_corner, n_variants, scale_min, scale_span)` — per-instance `x`, `y`, `rot`, `scale`, `phase`, `variant` arrays; EXACTLY what `world/shaders/flora.vert` derives from `gl_InstanceID` — the grass chain plus one lowbias32 link (`h5 % n_variants` → sprite-atlas cell), scale range parameterised per kind.

## Imports Allowed
`numpy`, `fire_engine.core` only (foundation layer — headless, no panda3d, Hard Rule 1).

## Events
Published: none.
Subscribed: none (the *renderer* in `world/` subscribes to `TerrainEditedEvent` / `ChunkLoadedEvent` and calls `bake_grass_height_field` again; this package stays pure).

## Units & Invariants
- All corners/areas in world **meters**, Z-up; `min_corner < max_corner` per axis (validated).
- Height-field texels are `config.voxel_size` (0.5 m); row 0 = min-Y edge, column 0 = min-X edge, **no vertical flip** on upload (`texture_bridge.to_field_texture`).
- R-channel encoding: `surface_z = min_z + R/254 × (max_z − min_z)`; `R == 255` is the no-ground sentinel — never interpolate the field (nearest filtering only).
- `instance_attribs` must stay byte-equal to the GLSL hash chain in `world/grass_shaders.py` — both files carry mirror comments; tests pin the python side.
- Determinism: same world seed + volume id → identical `grass_hash_seed` → identical blade placement. Bakes are pure functions of (volume, chunks, config).
- Saves: delta is the full volume list when it deviates from the `mark_baseline()` snapshot, `{}` otherwise. Old saves without a `"zones"` key load fine (SaveManager skips absent keys).

## Examples
```python
from fire_engine.zones import ZoneStore, grass_instance_count, grass_hash_seed

zones = ZoneStore()
vol = zones.add("grass", (-12.0, -5.0, 6.0), (12.0, 25.0, 10.0),
                params={"density": 12.0})        # blades per m²
zones.mark_baseline()                            # boot defaults = baseline
save_manager.register(zones)                     # rides F5/F9 delta saves

count = grass_instance_count(vol, cfg)           # 8640 for the demo box
seed = grass_hash_seed(vol)                      # shader uniform u_hash_seed
```
```python
# Headless check of what the GPU will draw (no panda3d needed):
import numpy as np
from fire_engine.zones import instance_attribs, bake_grass_height_field

attrs = instance_attribs(np.arange(count), seed, vol.min_corner, vol.max_corner)
field = bake_grass_height_field(vol, chunk_manager.chunks, cfg)
```

## The `"trees"` tag (flora trees + wind leaf litter — one volume, two consumers)
A `"trees"` volume feeds two independent renderers, both keyed on the same box:

- **Tree sprites** — `FloraRendererComponent` (`world/flora_renderer.py`) instances `flora_instance_count(vol, cfg, "trees")` crossed-quad trees seeded by `flora_hash_seed(vol, "trees")`, standing on the baked height field. Density: `params["density"]` or `config.flora_tree_density_per_m2` (0.02/m² = one tree per 50 m²).
- **Leaf litter** — the wind system's `LeafLitterComponent` (`world/mote_renderer.py`) instances `leaf_instance_count(vol, cfg)` gust-driven leaves seeded by `leaf_hash_seed(vol)`. Density: `params["leaf_density"]` (default `config.wind_leaf_density_per_m2 = 0.15`); leaves spawn biased low, settle in calm air and stream out in gusts/storms. Full reference: `docs/systems/wind.md`.

Both rebuild when `ZoneStore.version` changes, so planting a forest is one `zones.add("trees", ...)` — trees and the leaves blowing under them arrive together. The two hash seeds come from different `for_domain` keys and never correlate. `"flowers"` and `"bushes"` volumes work the same way with their single flora consumer.

## Gotchas
- The volume's **Z window matters**: only surfaces whose top face lies in `[min_z, max_z]` grow grass. A window that doesn't straddle the terrain surface bakes all-sentinel and renders nothing.
- `bake_grass_height_field` reads only chunks present in the mapping — unloaded chunks bake to sentinel. The renderer re-bakes on `ChunkLoadedEvent`, but a headless caller must pre-load chunks itself.
- Editing the GLSL hash chain without `hash_lowbias32`/`instance_attribs` (or vice versa) silently desyncs tests from the screen. The mirror comment in each file names the other. The same discipline binds `flora_instance_attribs` ↔ `world/shaders/flora.vert` (the flora chain adds the `h5` variant link; `tests/test_flora.py` pins both the mirror and the chain constants).
- `ZoneVolume.params` must stay msgpack-primitive (floats/strings) — it goes straight into the save delta.
