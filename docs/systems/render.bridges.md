# render.bridges — System Doc
keywords: render bridges, texture_bridge, geometry_bridge, post_process, profiler_bridge, profiler_overlay, resource_adapter, terrain_shader, to_panda_texture, to_field_texture, to_panda_texture_3d, to_data_texture_f32, to_panda_cubemap, to_geom, to_geom_node, make_material_state, make_vertex_format, register_panda_loaders, PostProcessPipeline, post processing, HDR, bloom, lens flare, god rays, FXAA, composite, u_hdr_output, apply_terrain_shader, terrain surface shader, GPU terrain, PStatsBridge, pstats, ProfilerOverlay, F3 overlay, frame time, fps, hitch, numpy to panda3d, BGRA, memoryview, bulk write, interleaved vertex buffer, V3N3T2C4, RenderState, TextureAttrib, FilterManager, renderQuadInto, RGBA16F, float buffer, resource manager, asset loaders, egg, bam, gltf

> One doc per code package; filename matches the package exactly (`docs/systems/render.bridges.md` ↔ `fire_engine/render/bridges/`).

## Role

`render/bridges/` is the **Panda3D bridge layer** — the set of modules that translate pure-Python /
numpy engine data into Panda3D scene-graph objects, and vice versa.  Every module here imports
`panda3d` (Hard Rule 1: `render/` is the sole package permitted to import Panda3D for non-GPU
work).  All layers *below* `render/` (terrain, lighting, procedural, resources, etc.) stay
headless-testable because they never import Panda3D directly — they produce numpy arrays and plain
Python objects that these bridges promote into Panda3D types.

Modules:

- `texture_bridge.py` — numpy RGBA arrays → Panda3D `Texture` objects
  (`to_panda_texture`, `to_field_texture`, `to_panda_texture_3d`, `to_data_texture_f32`,
  `to_panda_cubemap`).
- `geometry_bridge.py` — `MeshArrays` → Panda3D `Geom` / `GeomNode` with bulk writes
  (`to_geom`, `to_geom_node`, `make_material_state`, `make_vertex_format`).
- `terrain_shader.py` — GLSL surface shader for GPU-volumetric-lit terrain
  (`apply_terrain_shader`).
- `post_process.py` — HDR offscreen render target + post-processing chain
  (`PostProcessPipeline`: bloom, lens flare, god rays, FXAA, composite tonemap).
- `post_shaders.py` — GLSL source constants for all post-process passes (loaded from
  `render/shaders/`).
- `profiler_bridge.py` — mirrors the headless core profiler into Panda3D PStats
  (`PStatsBridge`).
- `profiler_overlay.py` — in-game F3 HUD: frame-time graph, 1%/0.1% lows, hitch counter,
  top scopes (`ProfilerOverlay`).
- `resource_adapter.py` — injects Panda3D-backed asset loaders into `ResourceManager`
  (`register_panda_loaders`).

`render/bridges/` deliberately does NOT: implement game logic, generate terrain or procedural
content, or subscribe to game events directly.  It is a pure translation layer.

## Public API

### texture_bridge

| Symbol | Description |
|---|---|
| `to_panda_texture(rgba) -> Texture` | `(H,W,4) uint8` RGBA numpy → Panda3D `Texture`. Vertically flips (OpenGL UV origin) and reorders RGBA→BGRA. Nearest-neighbour filters (retro pixel look). |
| `to_field_texture(rgba) -> Texture` | Data-field upload (grass height field, cover heightmap, etc.). **No vertical flip** — array row 0 → V=0, so `uv = (world_xy − min) / size` samples directly. Nearest filter, edge clamp. |
| `to_panda_texture_3d(volume, *, linear=True, repeat=True) -> Texture` | `(N,N,N,4) uint8` volume → 3-D `Texture`. Page-major `[z,y,x,channel]` order matches Panda3D's native layout; only RGBA→BGRA swap applied, no transpose. Used for baked cloud noise. |
| `to_data_texture_f32(block) -> Texture` | `(rows, cols, 4) float32` → RGBA32F 2-D texture. No flip; shader reads with `texelFetch(u_inst_tex, ivec2(col, gl_InstanceID), 0)`. Used for per-tree-instance transform data. BGRA reorder applies even at float type. |
| `to_panda_cubemap(faces) -> Texture` | `(6,S,S,4) uint8` → cube map. Follows OpenGL face order (+X,−X,+Y,−Y,+Z,−Z); no vertical flip (GL cube convention); RGBA→BGRA only. |

### geometry_bridge

| Symbol | Description |
|---|---|
| `to_geom(mesh) -> Geom` | `MeshArrays` → `Geom`. One bulk memoryview write for the interleaved V3N3T2C4 vertex buffer; one bulk write for the index buffer. No per-vertex Python loops (Hard Rule 7). |
| `to_geom_node(mesh, name, material_textures=None) -> GeomNode` | `to_geom` wrapped in a named `GeomNode`. When `mesh.face_materials` and `material_textures` are set, splits the mesh into one `Geom` per material id, each with a `RenderState` from `make_material_state`. |
| `make_material_state(entry) -> RenderState` | Build a `RenderState` from a single albedo `Texture` (legacy) or an `(albedo, normal, emission)` triple for the GPU terrain shader (`p3d_Texture0/1/2` by stage sort 0/1/2). `None` → empty state. |
| `make_vertex_format() -> GeomVertexFormat` | Interleaved V3N3T2C4 format: vertex(3), normal(3), texcoord(2), color(4) — all float32. |

### post_process

| Symbol | Description |
|---|---|
| `PostProcessPipeline(base, config)` | HDR offscreen target + bloom/flare/god-ray/FXAA chain. On buffer-allocation failure disables itself (`enabled=False`) — never fatal. |
| `pp.enabled` | `False` when `gfx_post_process` is off or GPU couldn't allocate the buffer. |
| `pp.update(lighting_pipeline)` | Per-frame: project sun to screen space for god rays. |
| `pp.hdr_color_tex / pp.depth_tex` | Linear-HDR scene colour + depth buffers. |

### profiler_bridge / profiler_overlay

| Symbol | Description |
|---|---|
| `PStatsBridge(profiler, connect=True)` | Mirrors core `Profiler` scopes/counters → Panda3D `PStatCollector`s. Connects to the `pstats` GUI server on port 5185. |
| `ProfilerOverlay(base, profiler, config)` | F3 in-game HUD: frame-time bar graph, 1%/0.1% lows, hitch counter, top-5 scopes. |

### terrain_shader

| Symbol | Description |
|---|---|
| `apply_terrain_shader(terrain_root, pipeline, *, seed=0.0, texels_per_m=16.0, extra_materials=None)` | Replace the fixed-function texture×vertex-colour pipeline on `terrain_root` with the GPU terrain GLSL shader. Samples radiance cascades, voxel-marched shadows, voxel AO, emission maps, froxel fog, and ACES tonemaps. World-space procedural ground (non-repeating palette LUT). Call once at boot after `lighting_pipeline` is ready. |

### resource_adapter

| Symbol | Description |
|---|---|
| `register_panda_loaders(resource_manager)` | Inject Panda3D-backed loaders for `.egg`/`.bam`/`.gltf`/`.glb` (models), `.ogg`/`.wav` (audio), `.png`/`.jpg` (textures) into a `ResourceManager`. Call once during boot after `ShowBase` is initialised. |

## Imports Allowed

- `panda3d.*`, `direct.*` (all modules here are in `render/bridges/` — Hard Rule 1)
- `fire_engine.core` (math3d, profiler, shader_source, config)
- `fire_engine.resources` (type annotation for `ResourceManager`)
- `fire_engine.world.terrain.meshing` (`MeshArrays` dataclass — numpy-only import)
- Python standard library, `numpy`

Not allowed: `fire_engine.simulation.*`, `fire_engine.world.*` (would create circular deps or
cross the render boundary).

## Events

Published: none — all bridges are stateless translators or one-way upload paths.

Subscribed: `PostProcessPipeline.update` is called by `App._frame_task`; it does not subscribe
directly to the event bus.

## Units & Invariants

- Spatial values in world **meters** (Z-up).
- All bulk texture uploads use the RGBA→BGRA channel reorder (Panda3D's native RAM image order).
- `to_panda_texture` flips rows (OpenGL UV origin bottom-left); `to_field_texture` and
  `to_data_texture_f32` do NOT flip.
- `make_vertex_format` layout: 12 float32 per vertex = 48 bytes per vertex in the interleaved
  array.
- `PostProcessPipeline` HDR buffer is RGBA16F; `to_data_texture_f32` produces RGBA32F.
- `apply_terrain_shader` is idempotent — safe to call once at boot only.

## Examples

```python
from fire_engine.render.bridges.texture_bridge import to_panda_texture
import numpy as np

arr = np.zeros((64, 64, 4), dtype=np.uint8)
arr[..., 0] = 200  # R
arr[..., 3] = 255  # A
tex = to_panda_texture(arr)
app.terrain_root.set_texture(tex)
```

```python
from fire_engine.render.bridges.geometry_bridge import to_geom_node
from fire_engine.render.bridges.terrain_shader import apply_terrain_shader

node = to_geom_node(mesh, "chunk_0_0_0", material_textures)
node_path = app.terrain_root.attach_new_node(node)

apply_terrain_shader(app.terrain_root, app.lighting_pipeline)
```

```python
from fire_engine.render.bridges.resource_adapter import register_panda_loaders
from fire_engine.resources import default_manager

register_panda_loaders(default_manager)   # call once after ShowBase init
```

## Gotchas

- **BGRA reorder is mandatory** — Panda3D's RAM-image byte order is BGRA even when the Python
  constant says `F_rgba`.  Uploading raw RGBA makes every texture look blue-for-red.
- `to_panda_texture` flips; `to_field_texture` does NOT — using the wrong one for a height field
  produces an upside-down height field that cull-misses rain or grass.
- `to_data_texture_f32` channels are also BGRA — a shader `texelFetch(...).r` returns
  `block[i, col, 0]` (after the swap); shaders must declare channel order accordingly.
- `to_geom_node` with `material_textures` requires `mesh.face_materials` to be set (the faceted
  mesher sets it; the blocky mesher does not) — without it the split is skipped and one un-
  textured Geom is produced.
- `PostProcessPipeline` must be constructed **after** the scene exists (terrain + sky) so
  `FilterManager.renderSceneInto` captures the correct camera.
- `register_panda_loaders` must be called after `ShowBase` is initialised — the global Panda3D
  `Loader` singleton isn't available before that.
- `apply_terrain_shader` calls `pipeline.bind_surface_inputs(terrain_root)` — the lighting
  pipeline must be fully constructed before calling this.
