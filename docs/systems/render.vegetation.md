# render.vegetation — System Doc
keywords: render vegetation, grass renderer, GrassRendererComponent, flora renderer, FloraRendererComponent, tree renderer, TreeRendererComponent, mote renderer, DustMoteComponent, building renderer, BuildingRendererComponent, grass_renderer, flora_renderer, tree_renderer, mote_renderer, building_renderer, grass_shaders, flora_shaders, tree_shaders, mote_shaders, building_shaders, GPU instanced grass, GPU instanced flora, GPU instanced trees, impostor, billboard LOD, tree mesh, tree impostor, grass height field, height field, instance count, gl_InstanceID, lowbias32 hash, sway, wind sway, canopy sway, sway_base, sway_gust, gust_freq, lit_surface, lit surface contract, ZoneVolume, zone_store, bake_grass_height_field, instances_data_block, species_mix, BoundingBox set_final, terrain_light_scale, dust mote, leaf litter, tree occluder, canopy extinction, BuildingManager, building mesh, plaster_wall, wood_floor, roof_shingle, stone_foundation, SurfaceMaterial, per-surface material, material_textures, face_materials, _load_material_textures, BuildingChangedEvent, fog inheritance, radiance cascades, froxel fog, gfx_foliage_shadow_refine, u_refine, GPU lighting backend

> One doc per code package; filename matches the package exactly (`docs/systems/render.vegetation.md` <-> `fire_engine/render/vegetation/`).

## Role

`render/vegetation/` is the **vegetation and particle render layer** -- GPU-instanced grass,
flowers, trees, dust motes, and free-form building geometry.  It also contains
`BuildingRendererComponent` because buildings, like vegetation, are instance-or-unique geometry
that inherits lighting from `render` rather than carrying its own.

Every module imports `panda3d` (Hard Rule 1).

Modules:

- `grass_renderer.py` -- `GrassRendererComponent`: GPU-instanced crossed-quad grass tufts for
  every `tag="grass"` ZoneVolume.
- `grass_shaders.py` -- GLSL source constants for the grass vertex/fragment shaders.
- `flora_renderer.py` -- `FloraRendererComponent`: GPU-instanced sprite flora (flowers) for every
  `tag="flowers"` ZoneVolume, generalising the grass idiom over a per-kind spec table.
- `flora_shaders.py` -- GLSL source constants for the flora vertex/fragment shaders (shared with
  tree impostors).
- `tree_renderer.py` -- `TreeRendererComponent`: instanced 3-D trees and bushes for every
  `tag="trees"` / `tag="bushes"` ZoneVolume, with billboard impostors past the mesh fade
  window; also pushes tree canopy occluders into `GpuLightingPipeline`.
- `tree_shaders.py` -- GLSL source constants for the tree mesh vertex/fragment shaders.
- `mote_renderer.py` -- `DustMoteComponent`: GPU-instanced dust-mote particles.
- `mote_shaders.py` -- GLSL source constants for the mote shaders.
- `building_renderer.py` -- `BuildingRendererComponent`: draws `BuildingManager` buildings as
  real lit geometry (one `GeomNode` per building, `building.vert/frag` with `lit_surface.glsl`).
- `building_shaders.py` -- GLSL source constants for the building vertex/fragment shaders.

`render/vegetation/` deliberately does NOT: generate placement data (that is the headless
`zones/` package), manage zone volumes, or compute lighting cascade values.

## Public API

### GrassRendererComponent (`grass_renderer.py`, shaders in `grass_shaders.py`)

| Symbol | Description |
|---|---|
| `GrassRendererComponent(base, sky_system, zone_store, chunk_provider, lighting_pipeline, bus)` | Add via `go.add_component`. GPU lighting backend only. Draws every `tag="grass"` ZoneVolume as instanced crossed-quad tufts; re-bakes height fields on TerrainEditedEvent/ChunkLoadedEvent. |

No per-blade CPU state. One shared 3-crossed-quad Geom per volume, `set_instance_count(n)`;
the vertex shader (`grass.vert`) derives each blade from `gl_InstanceID` via the lowbias32 hash
chain (mirrored in `zones.grass_placement`). Lighting and fog inherit from `App.terrain_root`.

### FloraRendererComponent (`flora_renderer.py`, shaders in `flora_shaders.py`)

| Symbol | Description |
|---|---|
| `FloraRendererComponent(base, sky_system, zone_store, chunk_provider, lighting_pipeline, bus)` | Add via `go.add_component`. GPU lighting backend only. Draws every `tag="flowers"` ZoneVolume as instanced sprite flora; identical wiring to GrassRendererComponent. |

The hash chain adds one link (`h5`) to select the atlas variant, mirrored in
`zones.flora_placement.flora_instance_attribs`.

### TreeRendererComponent (`tree_renderer.py`, shaders in `tree_shaders.py`)

| Symbol | Description |
|---|---|
| `TreeRendererComponent(base, sky_system, zone_store, chunk_provider, lighting_pipeline, bus)` | Add via `go.add_component`. GPU lighting backend only. Draws every `tag="trees"` / `tag="bushes"` ZoneVolume as instanced 3-D meshes with billboard impostor crossfade at distance. |

Per-instance data rides an RGBA32F data texture (`to_data_texture_f32`); `tree.vert` reads it
with `texelFetch(u_inst_tex, ivec2(col, gl_InstanceID), 0)`. After every (re)bake the component
pushes a `TreeOccluderSet` via `GpuLightingPipeline.set_static_occluders`.

### DustMoteComponent (`mote_renderer.py`, shaders in `mote_shaders.py`)

| Symbol | Description |
|---|---|
| `DustMoteComponent(base, sky_system, config)` | GPU-instanced atmospheric dust motes. Camera-local wrapping lattice; hash-chain XYZ/phase from `gl_InstanceID`. Additive blend, depth-write off. |

### BuildingRendererComponent (`building_renderer.py`, shaders in `building_shaders.py`)

| Symbol | Description |
|---|---|
| `BuildingRendererComponent(base, building_manager, lighting_pipeline, bus)` | Draws every BuildingManager building as a lit GeomNode. Subscribes to BuildingChangedEvent. GPU lighting backend only. |

Building meshes are emitted in building-local space; `building.vert` derives world position as
`(p3d_ModelMatrix * p3d_Vertex).xyz`. Uses the full `lit_surface.glsl` contract with `u_refine = 1.0`.

Per-surface materials: the mesh carries a `SurfaceMaterial` id per face (`MeshArrays.face_materials`),
so `to_geom_node(mesh, material_textures=...)` splits the building geom into one Geom per material,
each bound to its own procedural albedo — `WALL`→`plaster_wall`, `FLOOR`→`wood_floor`,
`ROOF`→`roof_shingle`, `FOUNDATION`→`stone_foundation` (loaded by `_load_material_textures`). The
node-level texture is the wall albedo fallback, so a missing content def degrades gracefully rather
than failing. No shader change — each sub-Geom's RenderState binds its albedo to `p3d_Texture0`.

## Imports Allowed

- `panda3d.*`, `direct.*` (Hard Rule 1)
- `fire_engine.core` (math3d, rng, shader_source, event bus)
- `fire_engine.zones` (ZoneStore, bake_grass_height_field, bake_tree_instances, instances_data_block, etc.)
- `fire_engine.world.terrain` (ChunkLoadedEvent, TerrainEditedEvent)
- `fire_engine.buildings` (BuildingManager, BuildingChangedEvent, meshing)
- `fire_engine.lighting` (GpuLightingPipeline, TreeOccluderSet)
- `fire_engine.render.bridges` (texture/geometry bridges)
- `fire_engine.render._impl.quad` (shared quad geometry)
- `fire_engine.render.vegetation._impl` (private helpers)
- Python standard library, `numpy`

## Events

### Published

None directly from `render/vegetation/`.

### Subscribed

- `TerrainEditedEvent` / `ChunkLoadedEvent` -- GrassRendererComponent, FloraRendererComponent,
  TreeRendererComponent (mark dirty volume height fields / placements for next `late_update` rebuild).
- `BuildingChangedEvent` -- BuildingRendererComponent (mark building dirty for `late_update` re-mesh).

## Units & Invariants

- All spatial values in world **meters**, Z-up.
- BoundingBox on every instanced node is set to effectively infinite (+-1e6 m) so Panda3D
  never frustum-culls using the base Geom bounds.
- `u_time_s` is self-bound on each renderer root node -- NOT inherited (wind-particles gotcha).
- Height-field textures use `to_field_texture` (no vertical flip); no-ground sentinel is 255 in R.
- Data textures (tree instances) use `to_data_texture_f32` (BGRA reorder; `texelFetch` only).
- `building.vert` MUST use `p3d_ModelMatrix` -- building geometry is building-local, not world-space.
- All vegetation components are GPU-lighting-backend-only.

## Examples

```python
from fire_engine.render.vegetation.grass_renderer import GrassRendererComponent

env_go = instantiate(name="Environment")
env_go.add_component(
    GrassRendererComponent,
    base=app, sky_system=sky_system, zone_store=zone_store,
    chunk_provider=chunk_manager, lighting_pipeline=gpu_pipeline, bus=event_bus,
)
```

```python
from fire_engine.render.vegetation.building_renderer import BuildingRendererComponent

env_go.add_component(
    BuildingRendererComponent,
    base=app, building_manager=building_manager, lighting_pipeline=gpu_pipeline, bus=event_bus,
)
```

## Gotchas

- **`building.vert` must use `p3d_ModelMatrix`** -- building meshes are local-space; terrain
  meshes are world-space. Mixing them up makes lighting sample at the wrong world position.
- **Data-texture channel order is BGRA** even at float32 -- use `texelFetch`, never `texture()`.
  A shader `.r` returns `block[i, col, 0]` after the channel swap.
- **`u_time_s` must be self-bound** on each renderer root (not inherited from `render`).
- **Hash chain must be mirrored** between the GPU vertex shader and the headless Python placement:
  `grass.vert` <-> `zones.grass_placement`; `flora.vert` <-> `zones.flora_placement.flora_instance_attribs`.
  Edit both or neither.
- `BuildingRendererComponent` re-syncs on `manager.version` bump -- test teardown with a version
  bump to catch leaked nodes.
