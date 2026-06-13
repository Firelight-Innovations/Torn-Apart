# world â€” System Doc
keywords: gameobject, game object, transform, component, registry, lifecycle, awake, start, update, late_update, fixed_update, on_enable, on_disable, on_destroy, instantiate, destroy, find_with_tag, find_objects_with_tag, space, camera, app, showbase, panda3d, input, inputstate, unity, clone, hierarchy, parent, children, world matrix, dirty flag, position, rotation, forward, right, up, look_at, transform_point, inverse_transform_point, active_in_hierarchy, set_active, add_component, get_component, get_components, get_component_in_children, remove_component, compare_tag, layer, tag, uuid, batched, bucket, run_frame, fixed_steps, spiral of death, quaternion, z-up, meters, radians, texture_bridge, to_panda_texture, geometry_bridge, to_geom, to_geom_node, resource_adapter, register_panda_loaders, bridge, bgra, f_rgba, ram_image, world meters, terrain_root, chunk_manager, light_sampler, setup_terrain_rendering, stream_and_upload_terrain, nodepath, geom, vertex color, sky, sky_renderer, SkyRendererComponent, sky_shaders, sky dome, skydome, clouds, cloud layer, boxy clouds, raymarch, dda, rain, rain_streak, night_sky, galaxy, stars, twinkle, shooting star, sun disc, moon, moon phase, fog, exponential fog, set_color_scale, terrain_light_scale, background bin, set_shader_input, weather render, day night cycle render, equirect, terrain_shader, apply_terrain_shader, volumetric, GPU lighting, lighting_pipeline, radiance cascade, froxel, ACES, tonemap, normal map, emission map, TBN, make_material_state, texture stage, external_lighting, physical atmosphere, single scattering, sun disc, moon texture, moon_surface, procedural ground, world-space ground, non-repeating ground, ground LUT, u_ground_lut, ground palette, ground_texels_per_m, shimmer fix, fwidth, palette LUT, to_field_texture, build_ground_lut, vertex alpha material id, extra_materials, post_process, PostProcessPipeline, post processing, HDR, hdr buffer, float buffer, rgba16f, FilterManager, renderSceneInto, renderQuadInto, bloom, lens flare, god rays, fxaa, composite, u_hdr_output, u_exposure, gamma, fullscreen quad, post_shaders, textures-power-2, NPOT, offscreen buffer, render target, exposure baked, sun bloom, volumetric clouds, cloud_volumetric, cloud_noise, hue-preserving tonemap, flora, flora_renderer, FloraRendererComponent, flowers, flower, bushes, bush, shrub, scrub, trees, tree, canopy sway, flower_sprite, sprite atlas, atlas variant, crossed quads, vegetation, tree_renderer, TreeRendererComponent, 3D tree, tree mesh, branches, impostor, billboard LOD, tree_shaders, tree.vert, tree.frag, tree_impostor.vert, data texture, to_data_texture_f32, rgba32f, texelFetch, u_inst_tex, variant pool, species, species_mix, sway weight, mesh fade, impostor fade, crossfade, lit_surface, lit surface, lighting any object, u_refine, foliage shadow refine, gfx_foliage_shadow_refine, washed out foliage, foliage lighting, lit-object contract, rain, rain_renderer, RainRendererComponent, volumetric rain, rain streaks, falling rain, rain particles, rain cylinders, gfx_rain_mode, gfx_rain_particles, gfx_rain_occlusion, no rain under roof, rain cover, RainCoverField, u_rain_height_tex, rain cull, precip gate, storm footprint, rain_particles.vert, rain_cylinder.frag, rain_shaders

> One doc per code package; filename matches the package exactly (`docs/systems/world.md` â†” `fire_engine/world/`).

## Role

`world/` is the **World API** â€” Layer 3 in the engine stack.  It provides:

- **Unity-clone object model** (`GameObject`, `Component`, `Transform`): same API shape as Unity (snake_case), Z-up coordinates, quaternion-only rotations.  All authoring uses these types.
- **ComponentRegistry**: the batched frame executor that drives Unity lifecycle order across all components.
- **App** (`world/app.py`): the Panda3D ShowBase wrapper, window, frame loop, input collection, and the **terrain render path** (chunk streaming → Geom upload).  Frame-task step 6 drives the optional `app.lighting_pipeline` (GPU volumetric lighting: `GpuLightingPipeline.update(...)` + `update_surface_inputs(render, sky_state)` each frame — the lit-surface contract lives on `render` so every lit shader inherits it) when main.py wires `config.lighting_backend == "gpu"`.
- **Terrain surface shader** (`world/terrain_shader.py`, GPU backend): `apply_terrain_shader(terrain_root, pipeline, *, seed=0.0, texels_per_m=16.0, extra_materials=None)` replaces the fixed-function texture×vertex-colour pipeline with a GLSL 330 fragment shader that samples the radiance cascades (positions quantised to `light_quant_m` "light pixels"), reads voxel-marched sun/moon visibility for shadows, derives AO from the occupancy volume, applies normal/emission maps (analytic TBN from the dominant face axis — matches the mesher's planar UVs), composites froxel volumetric fog at the fragment's own depth, and ACES-tonemaps with `light_exposure`.  Per-material Geoms carry **(normal, emission)** maps via texture-stage triples built by `geometry_bridge.make_material_state` (stage sorts 0/1/2 → `p3d_Texture0/1/2`).
  - **World-space procedural ground**: the albedo is NOT the tiled `p3d_Texture0` — it is generated in **world space** from a 2-octave integer-hash value noise of the dominant-axis-planar world coords snapped to a `texels_per_m` (≈16 → 0.0625 m) virtual texel grid, posterised through a per-material **palette LUT** (`u_ground_lut`).  The ground therefore **never repeats** across the 1 km map while keeping the crisp pixel-art palette.  A `fwidth`-based shimmer fix collapses sub-pixel texels toward the mid bucket so distant ground stops sparkling (no mipmaps needed).  The LUT (row = material id) is baked by `procedural.textures.ground_lut.build_ground_lut` from the `grass_ground`/`dirt_ground` colour ramps (`GRASS_PALETTE`/`DIRT_PALETTE` + thresholds) — single source of truth shared with the baked previews — and uploaded via `texture_bridge.to_field_texture`.  `seed` is a per-world hash offset (pass one from `core.rng.for_domain`); `extra_materials` adds flat/extra palette rows (e.g. the GI test-room debug materials so they don't clamp to the grass row).  The face material id reaches the shader packed into **vertex-colour alpha** (`surface_nets.py`, `id/255`), so a single NodePath-level shader needs no per-Geom uniforms.
- **CameraComponent** (`world/camera.py`): one-way sync from a `Transform` to the Panda3D camera NodePath.
- **Bridges** â€” the (small) set of files that translate engine data into Panda3D objects, so every layer below stays panda3d-free:
  - `texture_bridge.to_panda_texture` (Phase 2): numpy RGBA â†’ Panda3D `Texture`.
  - `geometry_bridge.to_geom` / `to_geom_node` (Phase 3): `MeshArrays` â†’ Panda3D `Geom` / `GeomNode` (bulk writes).
  - `resource_adapter.register_panda_loaders` (Phase 5): injects Panda3D asset loaders into the `resources.ResourceManager` (inversion of control).
- **DevOverlay** (`world/devtools_overlay.py`): the DirectGUI renderer for the in-game developer overlay (F1). It is the Panda3D half of the `devtools` system â€” see `docs/systems/devtools.md`; the headless brain lives in `fire_engine/devtools/`.
- **SkyRendererComponent** (`world/sky_renderer.py`) + GLSL sources (`world/sky_shaders.py`): the render half of the sky/weather system. The headless simulation lives in `fire_engine/sky/`; this component drives `sky_system.update()` each frame and draws the sky dome — now a **per-pixel physical single-scattering atmosphere** (Rayleigh+Mie, constants mirrored from `sky/atmosphere.py`) with a 2.5×-sized limb-darkened sun disc tinted by its own transmittance, a 2.5×-sized moon disc textured with the seeded procedural `"moon_surface"` texture under the dynamic phase terminator, `"night_sky"` galaxy + twinkle + shooting stars composited post-tonemap — plus volumetric raymarched clouds (rain moved to `RainRendererComponent`, M6).  With `external_lighting=False` (CPU backend) it also applies exponential fog and the global terrain colour-scale; with `external_lighting=True` (GPU backend) it leaves terrain shading entirely to the lighting pipeline and samples the froxel fog's far slice for the dome.

`world/` deliberately does NOT: implement game logic, generate terrain, simulate NPCs, or own any voxel data.  It is the scene-graph boundary: everything above uses the object model; everything below is contacted through direct API calls.

The **object-model modules import ZERO panda3d** â€” `transform.py`, `component.py`, `gameobject.py`, `registry.py` (and `player/fly_controller.py`) are pure Python/numpy and fully headless-testable.  Only the **shell + bridges** import panda3d: `app.py`, `camera.py`, `texture_bridge.py`, `geometry_bridge.py`, `resource_adapter.py`, `sky_renderer.py`, `devtools_overlay.py` (the shader modules `sky_shaders.py`, `grass_shaders.py`, `terrain_shader.py` are importable headless: each now **loads its GLSL from sidecar files in `world/shaders/*.{vert,frag}`** via `core.shader_source.load_glsl` and re-exports them under the same `*_VERTEX`/`*_FRAGMENT` constant names — edit the `.vert`/`.frag` files, not the `.py`).  The package `__init__.py` imports each bridge inside a `try/except ImportError`, so `import fire_engine.world` works headless (the panda3d-backed symbols are then `None`).

## Public API

All symbols below are re-exported from `fire_engine.world` (`__init__.py`).

### Transform (`world/transform.py`)

| Symbol | Description |
|---|---|
| `Transform()` | Per-entity TRS transform, always-present on every GameObject. |
| `Space.SELF` | Enum: operations in the local/object frame. |
| `Space.WORLD` | Enum: operations in world space. |
| `t.local_position: Vec3` | Position relative to parent (meters). Setter marks dirty. |
| `t.local_rotation: Quat` | Rotation relative to parent (unit quaternion). Setter normalises + marks dirty. |
| `t.local_scale: Vec3` | Scale relative to parent. |
| `t.parent: Transform | None` | Parent transform (read). Write via `set_parent`. |
| `t.children: tuple[Transform, ...]` | Read-only children tuple. |
| `t.set_parent(p, keep_world=True)` | Reparent; keep_world=True preserves world position/rotation. |
| `t.position: Vec3` | World-space position (derived, cached). Setter recomputes local_position. |
| `t.rotation: Quat` | World-space rotation (derived through parent chain). Setter recomputes local_rotation. |
| `t.forward: Vec3` | World-space +Y axis (facing direction). |
| `t.right: Vec3` | World-space +X axis. |
| `t.up: Vec3` | World-space +Z axis. |
| `t.translate(v, relative_to=Space.SELF)` | Move by displacement v (meters). |
| `t.rotate(q, relative_to=Space.SELF)` | Apply additional rotation. |
| `t.look_at(target, up=Vec3.UP)` | Orient forward toward target. Degrades gracefully if target == self. |
| `t.transform_point(p)` | Local â†’ world (meters). |
| `t.inverse_transform_point(p)` | World â†’ local (meters). Round-trips with transform_point. |

**Dirty-flag caching**: the 4Ã—4 world matrix is recomputed lazily.  Any write to local TRS or `set_parent` marks the transform and ALL descendants dirty instantly (O(subtree)); the next read of `position`/`rotation` recomputes at most one matrix-multiply per ancestor.

### Component (`world/component.py`)

| Symbol | Description |
|---|---|
| `Component` | Base class for all components. Override any lifecycle method. |
| `c.game_object: GameObject` | Owning GameObject. Set before `awake()`. |
| `c.transform: Transform` | Convenience alias for `c.game_object.transform`. |
| `c.enabled: bool` | When False, `update/late_update/fixed_update` are skipped. `awake/start` still run. |

**Lifecycle order** (guaranteed by `ComponentRegistry.run_frame`):

1. `awake()` â€” immediately after construction; runs even when disabled.
2. `on_enable()` â€” after awake, when component is enabled.
3. `start()` â€” once before the first `update()`; after ALL awakes that frame.
4. `update(dt)` â€” every enabled frame.
5. `late_update(dt)` â€” after ALL updates (camera follow, IK post-processing).
6. `fixed_update(dt)` â€” fixed timestep (0â€“5Ã— per real frame, default 50 Hz).
7. `on_disable()` â€” on disable or hierarchy deactivation.
8. `on_destroy()` â€” end-of-frame teardown.

### GameObject (`world/gameobject.py`)

| Symbol | Description |
|---|---|
| `GameObject(name, tag, layer)` | Create a new entity with an always-present Transform. |
| `go.id: UUID` | Unique identifier (uuid4). |
| `go.name: str` | Human-readable name (not unique). |
| `go.tag: str` | Single primary tag for find_with_tag lookups. |
| `go.layer: int` | Render/physics layer mask index. |
| `go.active_self: bool` | Local active flag. Write via `set_active()`. |
| `go.active_in_hierarchy: bool` | True only when this and all ancestors are active. |
| `go.transform: Transform` | Always present. Cannot be removed. |
| `go.add_component(t, **kw) -> T` | Construct, attach, schedule awake/start. |
| `go.get_component(t) -> T | None` | First component of exact type t. |
| `go.get_components(t) -> list[T]` | All components of type t. |
| `go.get_component_in_children(t) -> T | None` | BFS over transform-child hierarchy. |
| `go.remove_component(c)` | Detach; schedules on_disable + on_destroy end-of-frame. |
| `go.set_active(value)` | Sets active_self; cascades on_enable/on_disable down hierarchy. |
| `go.compare_tag(tag) -> bool` | Exact tag match. |

### ComponentRegistry (`world/registry.py`)

| Symbol | Description |
|---|---|
| `ComponentRegistry` | Module-level singleton. |
| `ComponentRegistry.run_frame(clock)` | Drive one full frame (awake/start/update/late/fixed/destroy). |
| `ComponentRegistry.clear()` | Reset all state (test isolation). |
| `instantiate(template, position, rotation, parent) -> GameObject` | Create + register. |
| `destroy(obj_or_component, delay=0.0)` | Deferred teardown (end of current frame). |
| `find_with_tag(tag) -> GameObject | None` | First registered object with tag. |
| `find_objects_with_tag(tag) -> list[GameObject]` | All registered objects with tag. |

**Execution order** within `run_frame(clock)`:
1. ALL pending `awake()` + `on_enable()` (snapshot before iterating â€” additions go to next frame)
2. ALL pending `start()` (snapshot before iterating)
3. `update(dt)` per type bucket
4. `late_update(dt)` per type bucket
5. `fixed_update(fixed_dt)` Ã— N (driven by `clock.fixed_steps()`)
6. Deferred destroy flush

### App (`world/app.py`)

| Symbol | Description |
|---|---|
| `App(config, clock, event_bus)` | Panda3D ShowBase wrapper. 1280Ã—720, vsync, FPS meter. |
| `app.input_state: InputState` | Read each frame by FlyController. |
| `app.camera_go: GameObject` | The camera entity (spawns at `(0, -20, 10)`). |
| `app.camera_comp: CameraComponent` | Camera sync component. |
| `app.terrain_root: NodePath` | Created in `__init__`; parent of every chunk NodePath. The fallback ground texture + baked-light render state are applied here once. |
| `app.chunk_manager: ChunkManager \| None` | **Injection slot** (set by `main.py` after construction). When present, drained each frame: `stream_frame` â†’ upload `pending_meshes`, remove `unloaded_this_frame`. |
| `app.light_sampler: Callable \| None` | **Injection slot**. Forwarded to `stream_frame` so remeshed chunks bake fresh sunlight. |
| `app.setup_terrain_rendering(ground_texture=None, material_textures=None)` | Call once after injecting `chunk_manager`: stores the materialâ†’texture map (`{MATERIAL_DIRT: dirt_tex, MATERIAL_GRASS: grass_tex}`, forwarded to `to_geom_node` on every chunk upload), applies the optional fallback texture to `terrain_root`, and `set_light_off()` (vertex colours already carry baked sunlight â€” a Panda3D light would double-light). |
| `app.post_process: PostProcessPipeline \| None` | **Injection slot** (set by `main.py` after the scene exists). When present, `_frame_task` calls `post_process.update(lighting_pipeline)` after the lighting step; the HDR composite + effect passes run as render2d cards afterwards. |
| `InputState` | Dataclass: move_forward/backward/left/right/up/down, sprint, mouse_dx/dy, mouse_captured, escape_pressed. |

`App.__init__` sets `render.set_shader_input("u_hdr_output", 0.0)` so surface shaders tonemap internally by default; the post-process pipeline flips it to `1.0`.  When `gfx_post_process` is on it also loads `textures-power-2 none` (NPOT render targets) **before** the GSG is created, so full-window HDR buffers aren't padded to a power-of-two (which would otherwise show the scene in a sub-rectangle).

**Terrain-render injection (not engine-coupled).** `App` does not *require* terrain â€” headless tooling can construct a bare `App`.  `main.py` wires rendering by setting `app.chunk_manager` / `app.light_sampler` after construction, then calling `app.setup_terrain_rendering(fallback_tex, material_textures)`.  The per-frame `App._stream_and_upload_terrain()` (called from `_frame_task` after `run_frame`, before `event_bus.drain()`) streams chunks, converts each `MeshArrays` to a `GeomNode` via `geometry_bridge.to_geom_node`, and parents it under `terrain_root` with **no offset** (mesh positions are absolute world meters).  Lighting needs no per-frame hook: `SunlightComputer` is event-driven (subscribed to the bus), so the lighting step in `_frame_task` is intentionally a `pass`.

### Bridges (panda3d-backed; `None` when panda3d is absent)

| Symbol | File | Description |
|---|---|---|
| `to_panda_texture(rgba) -> Texture` | `texture_bridge.py` | `(H,W,4) uint8` RGBA numpy â†’ Panda3D `Texture` (nearest-neighbour, vertical flip, **RGBAâ†’BGRA reorder**). |
| `to_field_texture(rgba) -> Texture` | `texture_bridge.py` | Data-field upload (e.g. the grass height field): **NO vertical flip** (row 0 â†’ V=0, so `uv = (world_xy - min) / size` samples directly), nearest filter, edge clamp, BGRA reorder. Never use for images. |
| `to_geom(mesh) -> Geom` | `geometry_bridge.py` | `MeshArrays` â†’ Panda3D `Geom` (one bulk memoryview write per array). |
| `to_geom_node(mesh, name, material_textures=None) -> GeomNode` | `geometry_bridge.py` | `to_geom` wrapped in a named `GeomNode`. When `mesh.face_materials` is set (faceted mesher) and `material_textures` (`{material_id: Texture}`) is given, the mesh is split into **one Geom per material**, each added with a `RenderState` carrying its texture (grass vs dirt). Geom states compose OVER node states, so they win over the `terrain_root` fallback texture. |
| `register_panda_loaders(manager)` | `resource_adapter.py` | Register `.egg`/`.bam`/`.gltf`/`.glb`/`.ogg`/`.wav`/`.png`/`.jpg` loaders into a `ResourceManager` (IoC; keeps `resources/` panda3d-free). |
| `DevOverlay(app, manager=None)` | `devtools_overlay.py` | DirectGUI renderer for the in-game dev overlay (F1): stats, click-to-select + outline, live inspector/edit, spawn/actions, day-night/weather. Headless brain in `fire_engine/devtools/` â€” see `docs/systems/devtools.md`. |

### CameraComponent (`world/camera.py`)

| Symbol | Description |
|---|---|
| `CameraComponent(base)` | Syncs owning Transform â†’ `base.camera` NodePath each frame. |
| `cam.sync_to_panda()` | Called by App after all components updated. Math3d â†’ Panda3D conversion happens here only. |

### SkyRendererComponent (`world/sky_renderer.py`, shaders in `world/sky_shaders.py`)

The render half of the sky/weather system (`None` when panda3d is absent).  The headless half is `fire_engine.sky.SkySystem` (Layer 1).

| Symbol | Description |
|---|---|
| `SkyRendererComponent(base, sky_system, terrain_root, clock=None)` | Add to a GameObject via `add_component`. `base` = the `App`; `sky_system` = `fire_engine.sky.SkySystem` (or any duck-typed object with the `SkyState` fields â€” see `tools/screenshot.py --stub-sky`); `terrain_root` receives fog + light scale; `clock` enables star rotation + the deterministic shooting-star schedule. |

**Lifecycle split:** `start()` builds ALL geometry once (bulk numpy â†’ memoryview writes, mirroring `geometry_bridge`); `update(dt)` calls `sky_system.update()` (registry runs every `update` before any `late_update`, so the frame's `SkyState` is fresh without App changes); `late_update(dt)` writes the per-frame render state â€” a fixed handful of `set_shader_input` / `set_pos` / `set_color_scale` calls.

**Sub-systems and render bins** (drawn in bin/sort order; terrain's opaque bin draws after `background`):

| Node | Bin | Depth | Notes |
|---|---|---|---|
| Sky dome (inverted UV-sphere, r = 800 m) | `background` 10 | test+write OFF | Camera-followed by **translation only** (never parented under the camera â€” it must stay world-oriented). Model-space vertex position = view direction in the shader. |
| Volumetric cloud "dome" (inverted sphere, same geometry as the sky dome) | `background` 15 | test+write OFF | Premultiplied-OVER blend (`M_add`, `O_one`/`O_incoming_alpha`; src.a = transmittance). The fragment shader (`cloud_volumetric.frag`) takes the per-pixel world view direction, analytically intersects the horizontal cloud slab `[_VCLOUD_ALT_M, +_VCLOUD_THICK_M]`, and raymarches it sampling the baked tileable 3-D noise (`sky.cloud_noise`); each lit sample marches `gfx_cloud_light_steps` toward the sun for self-shadow (Beer + powder) with an HG forward-scatter phase (silver lining). A bright sun bleeds through thin cloud; thick cloud occludes it; terrain (opaque bin) draws over and occludes clouds behind it. Gated by `gfx_clouds`; `gfx_cloud_steps` controls quality. Noise is disk-cached (`saves/cloud_cache/`, deterministic per seed). **M9 WMO genera:** with `gfx_cloud_genera` on (requires `gfx_weather_map`) the single slab becomes **three altitude bands** — high CIRRUS/CIRROSTRATUS, mid ALTOCUMULUS/ALTOSTRATUS, low CUMULUS/STRATUS/CUMULONIMBUS — all derived in-shader from the same weather-map coverage/density/precip channels (NO new texture data; the marched top rises to the high band, bands are dispatched by the sample's altitude). Off ⇒ the single-slab pre-M9 look, bit-for-bit. The headless companion is `fire_engine.weather.cloud_layers` (see `docs/systems/weather.md`). |
_(Rain moved out of the sky renderer in M6 — see **RainRendererComponent** below.)_

**Dome shader uniforms** (per frame unless noted): `u_sun_dir`, `u_sun_color`, `u_sun_intensity`, `u_moon_dir`, `u_moon_phase`, `u_zenith_color`, `u_horizon_color`, `u_star_visibility`, `u_star_rotation` (radians about +Z; one revolution per game day), `u_time` (real s, twinkle hash), `u_fog_color`, `u_fog_blend`, shooting star `u_ss_active`/`u_ss_start`/`u_ss_travel`/`u_ss_progress`; `p3d_Texture0` = `"night_sky"` equirect.
**Volumetric cloud shader uniforms**: static `u_shape`/`u_detail` (baked 3-D noise), `u_altitude`/`u_thickness`/`u_max_dist`, `u_shape_scale`/`u_detail_scale`/`u_detail_strength`, `u_sigma` (extinction), `u_hg`, `u_steps`/`u_light_steps`/`u_light_step_m`; per frame `u_cam_pos`, `u_sun_dir`/`u_moon_dir`, `u_sun_radiance`/`u_moon_radiance`/`u_sky_ambient` (linear-HDR SkyState contract), `u_coverage` (→ density threshold), `u_cloud_density`, `u_wind` (CPU-integrated drift, meters), `u_time` (jitter). Inherits `u_hdr_output` from `render` (linear HDR under post; ACES+gamma inline in the legacy path). **M4 weather-map** uniforms inherited from `render` (bound by `WeatherMapComponent`): `u_weather_map` (sampler2D, RGBA16F R=coverage G=density B=precip A=fog), `u_wmap_origin` (world XY of the map's min corner), `u_wmap_cell_m`, `u_wmap_cells`, `u_weather_map_enabled` (0 ⇒ flat-ambient pre-M4 look), `u_weather_ambient` (vec2 coverage/density beyond the edge), `u_virga_enabled`. The shader samples the map per march step at the **RAW world XY** (never `+u_wind` — the map already bakes in cell drift); the precip channel lowers/darkens storm bases and adds gray virga shafts below them. A 1×1 dummy `u_weather_map` + disabled state is bound on the cloud node in `_build_clouds` so the shader is valid even without the weather component. **M9 genera** uniforms (static, pushed once in `_build_clouds` from the `cloud_genera_*` config): `u_cloud_genera_enabled` (0 ⇒ single slab), `u_genera_high_alt`/`u_genera_high_thick`, `u_genera_mid_alt`/`u_genera_mid_thick` (the low band reuses `u_altitude`/`u_thickness`), `u_genera_high_floor`/`u_genera_high_cov_w`/`u_genera_high_density`, `u_genera_mid_cov_w`, `u_genera_high_detail`/`u_genera_mid_detail`. No new sampler — the bands read the existing weather-map channels.

**Weather-map render bridge** `world/weather_renderer.py::WeatherMapComponent` (M4): the structural twin of `WindSystemComponent`. Owns a headless `WeatherMap` + an RGBA16F (`T_half_float`/`F_rgba16`) texture (`weather_map_cells`²), re-rasters the four spatial weather channels around the camera (on recenter past half a span, else every `_RERASTER_EVERY_N_FRAMES` frames so drifting cells animate), packs them with `fire_engine.sky.pack_weather_map` (fp16 BGRA, row-major — no transpose), and binds the weather-map contract above on `render` with **committed-origin discipline** (`u_wmap_origin` refreshed only in the same frame as an upload). Reads-only: the `SkyRendererComponent` is the sole driver of `sky_system.update(player_pos)`. Master kill switch `gfx_weather_map`; virga `gfx_cloud_virga`; layered WMO genera `gfx_cloud_genera` (M9; needs `gfx_weather_map`).

### RainRendererComponent (`world/rain_renderer.py`, shaders in `world/rain_shaders.py`, M6)

The volumetric-rain render component (sibling of `DustMoteComponent`), parented under `terrain_root` so it inherits the wind / fog / camera + the M4 weather-map contract. Replaces the old camera-following scrolled cylinders in `sky_renderer.py` (which rained everywhere, even under a roof). Two modes by `config.gfx_rain_mode`:

| Mode | Geometry | Gate path |
|---|---|---|
| `"particles"` (medium+) | `gfx_rain_particles` GPU-instanced falling streaks on a camera-anchored wrapping lattice (the `mote_renderer` pattern: instance XY/Z/fall-phase/sway from `gl_InstanceID` in `rain_particles.vert`, zero CPU per-particle state). View-space billboarded vertical streak; additive, depth-write off. | **per-instance** in the vertex shader (collapses culled quads to zero size) |
| `"cylinders"` (low) | the cheap nested camera-following cylinders, but the fragment shader now applies the gates per fragment at its world XY. | **per-fragment** (`discard`) |
| `"off"` | nothing | — |

Both rendered modes apply two gates at the element world XY:

1. **Rain-cover heightmap cull (the M6 fix).** The component owns a headless `terrain.RainCoverField` — the highest-solid-voxel world Z per 1 m column around the player — and uploads it to `u_rain_height_tex` (single-channel `F_r32`, **nearest**-filtered, edge-clamped) with **committed-origin discipline** (`u_rain_height_origin` refreshed only in the same frame as a texel upload). A streak whose world Z is below the cover height there is under a roof/overhang and is discarded. Toggled by `config.gfx_rain_occlusion`.
2. **Storm-footprint precip gate.** Both shaders sample the inherited `u_weather_map` **precip channel (B)** at the element XY, so rain exists only inside storm cells (fades with precip). When the weather map is off (`u_weather_map_enabled == 0`) they fall back to the scalar `SkyState.rain_intensity` bound as `u_rain_intensity`.

**Cover-heightmap rebuild discipline:** the component subscribes to `ChunkLoadedEvent` / `TerrainEditedEvent`, marks the touched `(cx, cy)` chunk **columns** dirty, recenters the window (full `rebuild_all`) when the player crosses a quarter-span threshold, and otherwise refolds up to `rain_cover_budget_columns` dirty columns per `late_update` (`rebuild_columns`), re-uploading whenever the map changed. GPU lighting backend only (disables itself on the CPU backend or `gfx_rain_mode == "off"`).

**Particle uniforms** (set by the component): `u_hash_seed` (`for_domain("rain","particles")`), `u_rain_box_m`, `u_rain_size_m`, `u_rain_length_m`, `u_rain_fall_mps`, `u_rain_intensity` (scalar fallback), `u_rain_occlusion`, `u_rain_tint`, and `u_time_s` (self-bound + refreshed each frame — NOT inherited, the wind-particles gotcha); cover contract `u_rain_height_tex`/`u_rain_height_origin`/`u_rain_height_cell_m`/`u_rain_height_cells`. `u_cam_pos`, the wind contract (`u_wind_*`), the weather-map contract (`u_weather_map`/`u_wmap_*`), and the froxel fog (`u_fog_integrated`/`u_fog_enabled`/`u_viewport`) are all inherited. **Cylinder uniforms** add `u_rain_tex` (the `rain_streak` texture), `u_rain_alpha`, `u_uv_scroll` (per-layer, set per frame).

**Lighting integration (fog + global light):** each `late_update` the component sets the exponential `panda3d.core.Fog` on `terrain_root` to `SkyState.fog_density` (1/m) and `fog_color`, sets the window clear colour to horizon-blended-with-fog, and calls `terrain_root.set_color_scale(*terrain_light_scale, 1.0)` â€” the **baked vertex sunlight Ã— global day/night scale** product is the whole day-night terrain lighting story (no Panda3D lights involved).

**Shooting stars are deterministic:** game time is split into 30-game-minute slots; `for_domain("sky", "shooting_stars", game_day, slot)` decides spawn (pâ‰ˆ0.5) + start/travel directions; the streak animates over ~1.2 real seconds and only spawns while `star_visibility > 0.5`.

### LightningRendererComponent (`world/lightning_renderer.py`, shaders in `world/lightning_shaders.py` → `shaders/lightning.{vert,frag}`, M7)

The render half of procedural lightning. Subscribes to `LightningStrikeEvent` (published by the headless `WeatherSystem` schedule — see `docs/systems/weather.md`) and, per strike: regrows the bolt with `weather.generate_bolt(event.seed, …)` (deterministic — the event carries only the seed), resolves a **roof-aware ground Z** from a `terrain.RainCoverField` (so a bolt hits a roof, not the floor under it; falls back to `event.ground_pos` Z when no cover is known), uploads the segments to a **pool of two** dynamic-geometry nodes (a quick double-strike doesn't clobber the first bolt), and plays a two-phase envelope: a flickering **leader** revealing the channel top-down (~0.16 s) → a bright HDR **return stroke** (~0.10 s) + afterglow + 1–2 seeded **restrikes**. Each segment is a **camera-facing ribbon** expanded in `lightning.vert` (offset along `segment_dir × view_dir`, hidden below the `u_reveal` front). Parented under `terrain_root` so `u_cam_pos` is inherited.

On a strike it also (1) adds a transient `lighting.PointLight` (`ttl_s ≈ 0.3 s`) at the strike point so the scene lights up, (2) pulses a `u_lightning_flash` uniform **on `base.render`** (inherited by `sky_dome.frag` + `cloud_volumetric.frag`, which add it as an **additive** sky/cloud whitening — NEVER an exposure change, which would fight auto-adaptation), and (3) re-publishes a `ThunderEvent` (camera distance, `delay_s = distance / 343`). Custom vertex format (`vertex` + `a_other` segment-other-endpoint + `a_ribbon = (side, alongT, width, brightness)`). Gated by `config.gfx_lightning_bolts` (off ⇒ disables itself; the headless strike schedule + ThunderEvents still run). **GPU lighting backend only** (needs the live pipeline `.lights` + inherited `u_cam_pos`).

| Symbol | Description |
|---|---|
| `LightningRendererComponent(base, sky_system, chunk_provider, lighting_pipeline, bus)` | Add via `add_component`. `chunk_provider` = anything with a `.chunks` dict (`ChunkManager`) for the roof-aware cover heightmap; `lighting_pipeline` = the active `GpuLightingPipeline` (its `.lights` gets the flash light). |

### GrassRendererComponent (`world/grass_renderer.py`, shaders in `world/grass_shaders.py`)

GPU-only instanced grass for every `tag="grass"` `ZoneVolume` (headless math in `fire_engine/zones/` — see `docs/systems/zones.md`). **GPU lighting backend only**: requires the active `GpuLightingPipeline`; on the legacy CPU backend the component logs and disables itself.

| Symbol | Description |
|---|---|
| `GrassRendererComponent(base, sky_system, zone_store, chunk_provider, lighting_pipeline, bus)` | Add via `add_component`. `chunk_provider` = anything with a `.chunks` dict (`ChunkManager`); `lighting_pipeline` = the active `GpuLightingPipeline`. |

**The CPU stores no blades.** One shared 3-crossed-quad tuft `Geom` (12 verts) is drawn `grass_instance_count(volume, cfg)` times per volume via `set_instance_count`; the vertex shader derives each instance's XY/yaw/scale/sway-phase from `gl_InstanceID` through the lowbias32 hash chain mirrored line-for-line by `zones.instance_attribs`. The only CPU bake is the per-volume **height field** (`zones.bake_grass_height_field`, uploaded with `to_field_texture`): R = terrain surface height inside the volume's Z window, 255 = no ground → the shader collapses that blade (craters cull grass). Re-baked when `TerrainEditedEvent`/`ChunkLoadedEvent` touches the volume (events mark dirty; the re-bake runs in the next `late_update`).

**Lighting/fog by inheritance:** `GpuLightingPipeline.bind_surface_inputs`/`update_surface_inputs` maintain every cascade/fog/celestial uniform on `render`, inherited by the grass root — `grass.frag` includes **`lit_surface.glsl`** (the engine-wide lit-surface contract, see *Lighting any object* below) and gets terrain-identical light: 3-cascade cross-faded GI + voxel-shadowed sun/moon at the blade base (snapped to the `light_quant_m` pixel grid), single-ray celestial shadow refinement behind the `gfx_foliage_shadow_refine` preset knob (`u_refine` bound on `grass_root`), single-tap voxel AO, one-tap froxel fog and the `u_hdr_output`-gated finish.

**Weather sway:** `late_update` maps `SkyState.wind_dir/wind_speed/rain_intensity` to `u_sway_base` (static lean), `u_sway_gust` + `u_gust_freq` (oscillation — storms move grass harder and faster); displacement in the shader is quadratic in blade height so bases stay pinned. Distance fade shrinks blades to zero across `[grass_fade_start_m, grass_fade_end_m]` — no popping or far shimmer. Alpha is a binary cutout (`discard < 0.5`, the pixel-art `"grass_tuft"` texture) so no sorting or transparency bin is needed.

### PostProcessPipeline (`world/post_process.py`, shaders in `world/post_shaders.py`)

The HDR offscreen render target + post-processing chain (`None`/disabled when panda3d is absent or `gfx_post_process` is off).  Built on `direct.filter.FilterManager`: the whole scene (terrain → sky dome → clouds → grass) renders into a linear **RGBA16F float** buffer instead of straight to the window, then fullscreen passes turn that HDR signal into the final image.  This is what lets the sun disc, the grazing-sunrise horizon, and emissive surfaces keep values ≫ 1.0 so they can bloom / flare / tonemap correctly — the old pipeline tonemapped + clamped inside every surface shader, destroying that range first.

| Symbol | Description |
|---|---|
| `PostProcessPipeline(base, config)` | Owns the FilterManager, the HDR scene buffer, and the chained passes. Reads the `gfx_*` graphics-quality knobs. On buffer-allocation failure it disables itself (`enabled=False`) and the surface shaders keep tonemapping internally — never fatal. |
| `pp.enabled: bool` | False when configured off or the GPU couldn't allocate the buffer. |
| `pp.hdr_color_tex / pp.depth_tex` | The linear-HDR scene colour + depth buffers (read by the bloom bright-pass and the lens-flare occlusion test). |
| `pp.final_quad / pp.manager` | The composite card + the FilterManager (used for the bloom/flare/god-ray `renderQuadInto` passes). |
| `pp.update(lighting_pipeline)` | Per-frame refresh — projects the sun to screen space for the god-ray pass (bloom + flare are screen-space self-contained; exposure is applied in the surface shaders so it also scales bloom). |

**HDR output contract (`u_hdr_output`).** A single float shader-input set on `render` (inherited by every surface shader: terrain, sky dome, volumetric clouds, grass, flora).  `0.0` = the shader does its own ACES tonemap + sRGB gamma (legacy path, exact previous look).  `1.0` = the shader outputs **linear HDR with auto-exposure already multiplied in** (`hdr * u_exposure`), and the composite pass does the single tonemap + gamma.  Applying exposure in the surface shaders (not the composite) is deliberate: bloom must operate on the exposed signal.  The sky dome's stars + shooting-star ride along as additive emissive detail; the volumetric clouds emit premultiplied linear HDR.

**Bloom** (`bloom_down.frag` / `bloom_up.frag`): a Call-of-Duty/Jimenez pyramid — a soft-knee bright-pass + Karis-averaged 13-tap **downsample** chain (`gfx_bloom_mips` halvings, all RGBA16F at ≤ half-res so it's iGPU-cheap), then a 3×3-tent **upsample** chain that progressively adds each level back for a smooth, wide, firefly-free glow.  Built in `PostProcessPipeline._build_bloom` via `FilterManager.renderQuadInto`; the result feeds the composite's `u_bloom`.  The sun disc in `sky_dome.frag` is pushed far above 1.0 under HDR (`discGain`/`haloGain`) so bloom bleeds it into a soft, edgeless blob.  Disabled (composite keeps a black dummy, `strength = 0`) when `gfx_bloom` is off.

**Lens flare** (`lens_flare.frag`): image-based, screen-space — reads the HDR scene at quarter-res, isolates the sun (a high HDR threshold), and rebuilds **ghosts** (the source mirrored through the screen centre at several scales, with chromatic fringing) + a **halo** ring.  Because it reads the *rendered* scene, occlusion is automatic: when terrain covers the sun it isn't bright in the buffer, so the flare vanishes — no separate depth test needed.  Built in `_build_flare` (`renderQuadInto` div 4), fed to the composite's `u_flare`.  Geometry constants live on `PostProcessPipeline` (`_FLARE_*`); **strength** (`gfx_lens_flare_strength`, default `0.055`) and the sun-isolation **threshold** (`gfx_lens_flare_threshold`) are config-exposed; gated by `gfx_lens_flare`.

**God rays** (`god_rays.frag`): screen-space crepuscular shafts — half-res radial light-scatter from the sun's screen position (projected per frame in `update`/`_update_godray_sun`, deactivated when the sun is below the horizon or off-screen).  Occlusion is automatic: clouds/terrain that are dark in the scene block the shafts.  Fed to the composite's `u_godray` at `gfx_god_ray_strength` (default `0.4`); gated by `gfx_god_rays` / `gfx_god_ray_samples`.

**FXAA** (`fxaa.frag`): cheap post anti-aliasing, the **last** pass.  When `gfx_fxaa` is on the composite renders into an LDR buffer and the screen quad runs FXAA on it (the HDR offscreen buffer can lose hardware MSAA, so this restores smooth edges); otherwise the composite is itself the screen quad.

**Composite** (`composite.frag`): `out = pow(tonemapHuePreserve(scene + bloom + flare + godray), 1/2.2)` — the single tonemap point.  Built LAST (after the effect passes) so it samples their finished buffers; effect strengths drop to 0 when their pass is disabled.  `tonemapHuePreserve` blends per-channel ACES with a hue-preserving variant (tonemap the peak channel, keep the RGB ratio) so bright saturated sky stays coloured instead of bleaching to white near a high sun (the sun disc, R≈G≈B, still reads white).  The blend amount is the `u_hue_preserve` uniform (config `gfx_tonemap_hue_preserve`, default `0.8` — raise toward `1.0` to fight the low-sun wash-out, lower toward `0.0` for a flatter filmic roll-off).

**Aesthetic vs quality config.** The `[graphics]` `preset` (off/low/medium/high) only drives the heavy *quality* knobs (resolution, step counts, which passes run).  The *look* knobs are deliberately preset-independent so they survive a preset change: post-chain — `gfx_bloom_strength/threshold/knee`, `gfx_lens_flare_strength/threshold`, `gfx_god_ray_strength`, `gfx_tonemap_hue_preserve`; sky/sun (pushed into `sky_dome.frag` once at build in `SkyRendererComponent._build_dome`) — `gfx_sun_disc_intensity`, `gfx_sun_halo_intensity`, `gfx_sun_min_brightness` (floor that keeps a low sunrise/sunset sun bright, hue preserved), `gfx_sky_inscatter_scale` (lower to reduce low-sun wash-out without dimming the sun disc).

### FloraRendererComponent (`world/flora_renderer.py`, shaders in `world/flora_shaders.py`)

GPU-only instanced sprite flora — flowers — for every `tag="flowers"` `ZoneVolume`: the grass idiom generalised over a per-kind spec table (one row today; headless math in `zones/flora_placement.py` — see `docs/systems/zones.md`). Trees and bushes graduated to `TreeRendererComponent` below. **GPU lighting backend only** (logs and disables itself on the CPU backend, like grass).

| Symbol | Description |
|---|---|
| `FloraRendererComponent(base, sky_system, zone_store, chunk_provider, lighting_pipeline, bus)` | Add via `add_component`; identical wiring to `GrassRendererComponent`. |

Per kind it binds a seeded procedural sprite **atlas** (`"flower_sprite"`, 4 cells) on a shared crossed-quad `Geom` and a sway shape (`u_sway_gain` 0.8 vs grass, `u_sway_pivot` 0 = whole-plant blade bend). `flora.vert` is the grass hash chain **plus one link** — `h5` selects the atlas variant — mirrored line-for-line by `zones/flora_placement.py::flora_instance_attribs` (edit both or neither; `tests/test_flora.py` pins the python side and the chain constants). Wind is the same dual path as grass: per-plant `u_wind_tex` sampling when the wind field is live, the scalar SkyState sway fallback otherwise. Height fields (`bake_grass_height_field`), bounds discipline (`BoundingBox` + `set_final`), shader-on-the-instanced-node, per-volume re-bakes on `TerrainEditedEvent`/`ChunkLoadedEvent`, and cascade/fog lighting inherited from `render` are all exactly the grass component's; `u_time_s` is self-bound on `flora_root` (not inherited — the wind-particles gotcha). Config: `flora_flower_density_per_m2` / `_height_m` / `_fade_start_m` / `_fade_end_m` / `_max_instances`; per-volume `params["density"]` overrides the density. The fragment shader (`flora.frag`, also the tree-impostor fragment) is the shared `lit_surface.glsl` contract with a `u_light_offset_m` cascade sample height (0.5 m for flowers) and `u_refine` bound on `flora_root` from `gfx_foliage_shadow_refine`.

### TreeRendererComponent (`world/tree_renderer.py`, shaders in `world/tree_shaders.py`)

Instanced **3-D trees and bushes** for every `tag="trees"` / `tag="bushes"` `ZoneVolume`: per-species variant-mesh pools (`procedural/flora/`, see `docs/content/tree_species_authoring.md`) drawn over CPU-baked placements (`zones/tree_placement.py`), with billboard **impostors past the mesh fade window — the ONLY billboarding trees get**. **GPU lighting backend only.**

| Symbol | Description |
|---|---|
| `TreeRendererComponent(base, sky_system, zone_store, chunk_provider, lighting_pipeline, bus)` | Add via `add_component`; identical wiring to `GrassRendererComponent`/`FloraRendererComponent`. |

**Draw structure** (per volume): one `GeomNode` per `(species, variant)` with ≥ 1 instance — the variant `TreeMesh` uploaded once via `geometry_bridge.to_geom` (its V3N3T2C4 arrays are field-compatible; `colors[:,3]` carries the per-vertex **sway weight**, 0 at the trunk base → ≈1 at leaf tips) and instanced `set_instance_count(n)` times; plus one impostor `GeomNode` per species (crossed quads sized `impostor_width_m × impostor_height_m` — the pool-common raster scale, so the quad overlays every variant exactly at the crossfade). **Per-instance data rides an RGBA32F data texture** (`texture_bridge.to_data_texture_f32`, layout from `zones/tree_placement.py::instances_data_block`): `tree.vert`/`tree_impostor.vert` read it with `texelFetch(u_inst_tex, ivec2(col, gl_InstanceID), 0)` — texel 0 `(x,y,z,yaw)`, texel 1 `(scale,phase,tint,variant)`. No GLSL hash-chain mirror for trees: the CPU owns placement; the (much smaller) contract to keep in sync is this texel layout, pinned by `tests/test_tree_placement.py`.

**LOD crossfade**: meshes shrink to zero over `[<kind>_mesh_fade_start_m, _end_m]` (trees 110–140 m, bushes 60–80 m) while impostors grow in over the same window, then fade out over `[<kind>_impostor_fade_start_m, _end_m]` (trees 300–380 m, bushes 120–150 m). Both stages read the SAME data texture, so they can never desynchronise. `tree.frag` is the full **`lit_surface.glsl`** contract (see *Lighting any object* below) upgraded with **real Lambert against per-face normals** (back-face flip — the tree root is two-sided for leaf quads), cascades sampled at the *fragment's* quantised world position with screen-footprint LOD via `litQuantSize` (trunks darken under their canopy, crowns catch sky, distant crowns don't shimmer), soft-penumbra shadow refinement (`refineVisSoft`) and two-point voxel AO — refinement gated by `gfx_foliage_shadow_refine` (`u_refine` on `tree_root`, covers mesh + impostor draws); the impostor fragment stage is `flora_shaders.FLORA_FRAGMENT` verbatim (same contract, base-anchored sampling at `u_light_offset_m`). Wind: mesh path bends by `color.a²` (trunk pinned, canopy sways), impostor path by the `u_sway_pivot` height ramp; both sample the inherited `u_wind_tex` with the scalar SkyState fallback, `u_time_s` self-bound on `tree_root`. Volumes re-bake placements on `TerrainEditedEvent`/`ChunkLoadedEvent` (trees keep their feet on edited ground). **Trees also CAST into the light grid**: after every (re)bake the component merges all volumes' placements into a `TreeOccluderSet` (heights/canopy radii from the species pool extents × instance scale, bounce colours averaged from the species atlas halves) and pushes it via `GpuLightingPipeline.set_static_occluders` — the ground darkens under canopies, crowns self-shadow, and the foliage shaders' refinement march sees the trunks (opacities: `light_tree_trunk_occ`/`light_tree_canopy_occ`). Config: the `[trees]` table (`tree_*` / `bush_*` density, min spacing, max instances, fade windows, default species); per-volume `params["species_mix"]`/`params["species"]`/`params["density"]` override. A `"trees"` volume is shared infrastructure with the wind system's `LeafLitterComponent` (leaf litter under the same canopy).

**Data-texture gotcha**: Panda3D 4-component RAM images are **BGRA even at float type** — `to_data_texture_f32` reorders channels on upload (and does NOT vertically flip: row 0 = instance 0). Shaders must read it with `texelFetch`, never `texture()`.

### Lighting any object: the lit-surface contract (`world/shaders/lit_surface.glsl`)

Every surface the voxel lighting should touch — terrain, foliage today; buildings, NPCs, props tomorrow — uses ONE shared GLSL library instead of hand-copying lighting code (the pre-library copies drifted until foliage double-tonemapped and washed out). To light a new object:

1. **Include the library** in your fragment shader (expanded by `core.shader_source.load_glsl`):
   ```glsl
   #version 330 core
   #define LIT_REFINE 1            // optional: compile the shadow-refinement march
   //#include "lit_surface.glsl"
   ```
2. **Parent the object anywhere under `render`** — `GpuLightingPipeline.bind_surface_inputs(app.render)` (once, in main.py) and `update_surface_inputs(app.render, sky_state)` (per frame, App step 6) maintain every cascade/fog/celestial uniform on `render` itself, so the whole uniform contract arrives by scene-graph inheritance with zero binding code, wherever the object lives. If you compiled `LIT_REFINE`, bind the one per-object knob: `node.set_shader_input("u_refine", 1.0)` (or the `gfx_foliage_shadow_refine` config value).
3. **Pass a world-space position** from your vertex stage (the fragment's own `v_world` for real meshes — per-fragment light, the "thorough" tier; the instance base for billboards) and, if you have one, a world-space normal.
4. **Sample and compose** (see the library header for the full recipe):
   ```glsl
   vec3 wq = litQuantPos(v_world, litQuantSize(dist * u_px_rad));
   vec3 radiance, vis; float occ;
   sampleCascades(wq, radiance, vis, occ);
   // optional: refineVis/refineVisSoft in the penumbra band, gated by u_refine
   vec3 direct = u_sun_radiance  * (vis.r * max(dot(n, u_sun_dir),  0.0))
               + u_moon_radiance * (vis.g * max(dot(n, u_moon_dir), 0.0));
   vec3 hdr = albedo * (direct + radiance * litAo(occ, occFar));
   frag_color = vec4(litFinish(litFog(hdr, dist)), 1.0);
   ```

**Casting shadows** (the other direction): objects not in the voxel field register a **dynamic-occluder AABB** with the lighting pipeline (`u_box_min`/`u_box_max`, the inject + refinement marches both test them analytically — see `lighting/gpu.py`). The current contract caps at 16 boxes — fine for dev cubes and a few NPCs; crowds/buildings will need either voxelisation into the cascade geometry volumes (the Phase-5 tree-occupancy path) or a bigger occluder structure (future work).

**Budgets**: GL 3.3 guarantees only **16 fragment samplers**; the library uses 10 (9 cascade + 1 fog), terrain sits at 15 total. A building shader with albedo+normal+emission maps lands at 13 — check `tests/test_lit_surface.py::TestSamplerBudget` before adding samplers. The `LIT_REFINE` block also adds 2×16 `vec4` occluder arrays — skip the define on shaders that draw tiny fragments (motes).

## Imports Allowed

`world/transform.py`, `world/component.py`, `world/gameobject.py`, `world/registry.py`:
- `fire_engine.core` (Vec3, Quat, etc.)
- Python standard library, `numpy`
- **No panda3d imports**

`world/app.py`, `world/camera.py`, `world/texture_bridge.py`, `world/geometry_bridge.py`, `world/resource_adapter.py`:
- All of the above PLUS `panda3d.*`, `direct.*`

Per ARCHITECTURE.md Â§4a.2, `world/` may also import: `resources/`, `terrain/`, `lighting/`, `procedural/`, `core/`.  (`app.py` imports `terrain`/`lighting` lazily inside methods so the module stays importable for panda3d-only tooling.)

## Events

### Published
- `ThunderEvent` (deferred) — `LightningRendererComponent` re-publishes one per `LightningStrikeEvent` it renders, carrying the camera distance and `delay_s = distance / 343` for the delayed audio crack (M7).

Otherwise nothing directly from world/.  App calls `event_bus.drain()` each frame; the bus carries events from terrain/lighting/etc.

### Subscribed
- `LightningStrikeEvent` — `LightningRendererComponent` (drives bolts); `ChunkLoadedEvent` / `TerrainEditedEvent` — same component (marks its roof-aware cover heightmap dirty).  Also `ChunkLoadedEvent` / `TerrainEditedEvent` by `RainRendererComponent` / grass / flora / trees for their own dirty-tracking.

Otherwise none directly.  Terrain/lighting integrate via the App integration hooks, not event subscriptions.

## Units & Invariants

### Coordinate System
- **Z-up** (Panda3D native): `forward = +Y`, `right = +X`, `up = +Z`.
- **Not Unity's Y-up** â€” the API shape is Unity's; the axes are ours.
- All distances in **meters**.
- Rotations in **radians** throughout.

### Quaternion Composition (mouse-look)
The FlyController (player/fly_controller.py) composes:
```python
q_yaw   = Quat.from_axis_angle(Vec3.UP,   yaw)    # world Z rotation
q_pitch = Quat.from_axis_angle(Vec3.RIGHT, pitch)  # local X rotation
transform.local_rotation = (q_yaw * q_pitch).normalized()
```
Yaw and pitch are **accumulated floats** (radians), NOT integrated incrementally into the quaternion.  This avoids the roll-drift trap documented in DEVELOPMENT_PLAN.md Known Traps.  Pitch is clamped to Â±89Â°.

### Transform Cache Invariant
`transform.position` and `transform.rotation` are always consistent with the parent chain.  After any write to local TRS or `set_parent`, all descendants are immediately marked dirty.  A stale world matrix is never returned; the cache is recomputed on the next read.

### Lifecycle Ordering Invariant
ALL `awake()` calls within a frame complete before ANY `start()` call begins â€” even when multiple GameObjects are added in the same frame.  Components added during `run_frame` iteration are deferred to the next frame's awake queue.

### No Euler State
Rotations are stored **only as quaternions**.  `Quat.from_euler` and `as_euler` exist solely as a presentation layer; they are never used as stored state.

## Examples

### Creating an entity with a component
```python
from fire_engine.world import instantiate, ComponentRegistry
from fire_engine.world.component import Component
from fire_engine.core.math3d import Vec3, Quat
from fire_engine.core.clock import Clock
from fire_engine.core.event_bus import EventBus

class Rotator(Component):
    def update(self, dt: float) -> None:
        from math import pi
        self.transform.rotate(Quat.from_axis_angle(Vec3.UP, pi * dt))

clock = Clock(fixed_dt=0.02, bus=EventBus())
go = instantiate(position=Vec3(0, 0, 5))
go.add_component(Rotator)

clock.update(0.016)
ComponentRegistry.run_frame(clock)  # awake + start + update
```

### Transform parent/child hierarchy
```python
from fire_engine.world.transform import Transform, Space
from fire_engine.core.math3d import Vec3, Quat
from math import pi

parent = Transform()
child  = Transform()
child.set_parent(parent, keep_world=False)

parent.local_position = Vec3(10, 0, 0)
parent.local_rotation = Quat.from_axis_angle(Vec3.UP, pi / 2)
child.local_position  = Vec3(0, 2, 0)

# Child world position: parent rotated its +Y by 90Â°, so child is at (-2, 0, 0) + (10, 0, 0)
print(child.position)   # Vec3(8, 0, 0)  (approximately)
```

### look_at
```python
cam = Transform()
cam.local_position = Vec3(0, -20, 10)
cam.look_at(Vec3.ZERO)
# cam.forward now points from (0,-20,10) toward the origin
```

### App setup (game entry point)
```python
from fire_engine.core import load_config, Clock, EventBus
from fire_engine.world.app import App

cfg   = load_config()
bus   = EventBus()
clock = Clock(fixed_dt=cfg.fixed_dt, bus=bus)
app   = App(cfg, clock, bus)

# Add player with fly controller
from fire_engine.world import instantiate
from fire_engine.player import FlyController
from fire_engine.core.math3d import Vec3
player_go = instantiate(position=Vec3(0, 0, 20))
player_go.add_component(FlyController, move_speed=10.0)

app.run()
```

## Gotchas

1. **Components added during run_frame go to NEXT frame.** If you call `go.add_component(...)` inside an `update()`, the new component's `awake()` fires in the next `run_frame` call, not the current one.  This is intentional (Unity semantics) and prevents iterator-mutation bugs.

2. **Transform dirty propagation is O(subtree size).** Setting a parent's position marks every descendant dirty.  For deep hierarchies with many descendants, prefer batch-updating leaf positions directly rather than moving the root repeatedly.

3. **`destroy()` is deferred to end-of-frame.** A destroyed component or object continues to exist in memory and will NOT be updated after the destroy call, but `on_destroy()` doesn't fire until the flush at the end of `run_frame`.

4. **Euler angles are not state.** Never try to read back Euler angles from a Transform to modify them incrementally â€” use the quaternion API directly.  The FlyController correctly accumulates yaw/pitch as floats and recomputes the quaternion each frame.

5. **`active_in_hierarchy` walks the parent chain.** It's O(depth), called per-component per-frame for the `_is_active` guard.  Keep hierarchies shallow for performance.

6. **Terrain render is injection, not a hard dependency.** `App.__init__` sets `self.chunk_manager = None` / `self.light_sampler = None`; `main.py` assigns the real objects after construction and calls `setup_terrain_rendering(tex)`.  `App._stream_and_upload_terrain()` no-ops when `chunk_manager is None`, so a bare `App` (headless tooling) runs fine.  There is no `_chunk_manager`/`_light_grid` attribute â€” those were the pre-integration placeholder names; the shipped attributes are `chunk_manager` / `light_sampler`.

7. **Mouse delta is in pixels (normalised by half-window-size).** The FlyController uses `mouse_sensitivity` in radians/pixel.  At 1280Ã—720, a full-screen drag is Â±640px/Â±360px â€” calibrate sensitivity accordingly.

8. **Panda3D `F_rgba` RAM images are BGRA byte order.** `texture_bridge.to_panda_texture` reorders RGBAâ†’BGRA (`flipped[..., [2,1,0,3]]`) before `set_ram_image`, AND flips vertically (OpenGL UV origin is bottom-left).  Both transforms are deliberate â€” "fix" either back and every texture renders blue-for-brown or upside-down.

9. **Chunk mesh vertex positions are ABSOLUTE WORLD METERS.** The mesher emits world-space positions (not chunk-local), so each chunk's NodePath is attached under `terrain_root` at the origin with **no per-chunk offset**.  Adding `chunk_coord * 16 m` here would double the world position.  `terrain_root` itself stays at the origin.

10. **Background-bin sky nodes still need `set_depth_test(False)`.** Putting the dome/cloud quads in the `background` bin only changes draw ORDER; with depth testing on they would still occlude (or be occluded by) terrain incorrectly.  Both depth test AND depth write are off on the dome and cloud quads.

11. **Rain lives in `world/rain_renderer.py`, NOT the sky renderer (M6).** The old camera-following scrolled cylinders rained everywhere — even under a roof — because nothing tested terrain cover. They were replaced by `RainRendererComponent`: GPU-instanced volumetric streaks (or a cheap cylinder fallback) gated by the **rain-cover heightmap** (`terrain.RainCoverField` — highest solid voxel per column; rain below it is under cover and discarded) AND the **weather-map precip channel** (rain only inside storm footprints). The cover heightmap uploads to `u_rain_height_tex` as a single-channel **nearest**-filtered `F_r32` texture with committed-origin discipline. The cylinder mode's V axis is still mirrored to make the streak pattern fall downward (the `to_panda_texture` flip + the `rain_streak` tail-at-high-rows convention), now scrolled via the `u_uv_scroll` uniform in `rain_cylinder.vert` instead of `set_tex_offset`.

12. **`u_coverage` is a quantile threshold, not the coverage value.** The cloud shader's per-cell noise is bell-shaped around ~0.5; thresholding it with raw `cloud_coverage` would render almost no clouds below ~0.3.  `_cloud_value_quantiles` (a numpy float32 port of the same GLSL noise) is sampled once at start; per frame the renderer passes `quantiles[coverage * (n-1)]` so coverage is the ACTUAL fill fraction.  If you change the GLSL `cell_value`, change the numpy port in the same commit.

13. **Adjacent occupied cloud boxes share interior faces.** The DDA shades each box's ENTRY face independently; without the `prev_hit`/`carry_col` continuity in `CLOUD_FRAGMENT`, every cell seam draws a side-coloured grid line across distant cloud ceilings (and float32 precision adds dark gap slivers â€” hence the small Â±0.05 m interval overlap).

14. **The `"night_sky"` equirect needs no flip handling in the dome shader.** The def is authored with array row 0 = zenith specifically so that after `to_panda_texture`'s vertical flip the natural mapping `v = asin(d.z)/Ï€ + 0.5` lands the zenith at v = 1.  Don't add a `1 - v`.

15. **`tools/screenshot.py` releases mouse capture before stepping frames.** The window starts in relative-mouse capture; physical mouse movement during an unattended capture would otherwise feed deltas into `FlyController` and swing the camera mid-warmup.

16. **Mouse capture is reasserted on focus regain.** Alt-tabbing away drops the OS-level hidden-cursor / relative-mouse window properties, and Panda3D does not re-apply them when the window refocuses â€” so the engine would believe the mouse is captured while the desktop shows a free pointer, killing free-look until the next ESC.  `App.windowEvent` overrides ShowBase's handler (calls `super()` first), detects the focus-regain edge via `win.get_properties().get_foreground()`, and re-issues `_set_mouse_capture(input_state.mouse_captured)` (which re-arms `_skip_mouse_delta` so the view doesn't snap).  Complementarily, `main.py`'s `on_click` re-captures when the cursor is free and the overlay is closed, so a click after ESC resumes flying instead of firing the demo explosion.

17. **Instanced nodes are culled by the BASE Geom's bounds.** `set_instance_count` does not grow the cull volume — Panda3D would frustum-cull every grass volume by the single origin-sized tuft Geom and the blades would vanish whenever the origin left the view.  `GrassRendererComponent._build_volumes` therefore sets an explicit `BoundingBox` (volume AABB + blade reach) on each `GeomNode` and calls `set_final(True)`.  The same applies to ANY future `gl_InstanceID`-positioned geometry.

18. **The grass GLSL hash chain has a python twin.** `lowbias32` + the h0–h4 chain in `world/shaders/grass.vert` (loaded by `world/grass_shaders.py`) is mirrored line-for-line by `zones/grass_placement.py::instance_attribs` (which the headless tests pin).  Edit both or neither — a desync makes the tests describe blades that aren't where the GPU draws them.

19. **The procedural ground uses per-octave LOD, NOT a collapse-to-mean.** `terrain.frag`'s `groundNoise` sums three hash octaves (fine 1×, mid 4×, macro 16× larger texels); each octave individually fades toward the hash mean (0.5) via `smoothstep(0.5, 1.4, mpp*texels)` once a screen pixel approaches the size of *that octave's* texels (`mpp` = the ANALYTIC world metres/pixel — gotcha 22).  So the fine detail drops out first, then the mid, leaving the macro colour patches — distant ground stays varied instead of flattening to one green.  The earlier single global `mix(gnoise, 0.5, …)` collapsed *all* scales together and produced the "sea of green after a few metres" — do not reintroduce it.  Each octave has mean 0.5, so the weighted sum's mean stays 0.5 and the posterise buckets stay balanced.  `ground_texels_per_m` sets the fine octave; raise it for crisper near pixels (the LOD prevents the distant shimmer that used to force it down).

20. **Every lit-surface shader samples THREE radiance cascades, finest-first, through ONE shared source.** `sampleCascades` (now in `world/shaders/lit_surface.glsl`, included by `terrain.frag`, `grass.frag`, `flora.frag`, `tree.frag`, `mote_leaf.frag` via `//#include`) cross-fades cascade 0 (0.5 m) → 1 (2 m) → 2 (8 m, the coarse far cascade) → the open-sky fallback.  The cascade uniform sets must stay in sync with `GpuLightingPipeline.bind_surface_inputs`/`update_surface_inputs` — adding a cascade means a matching block in the library and those two methods (one place each now).  Never hand-copy lighting code into a new shader: the pre-library copies drifted until trees and flora double-tonemapped under the HDR pipeline (the "washed-out foliage" bug).  `tests/test_lit_surface.py` pins the contract.

21. **Ground albedo is filtered by ANALYTIC TEXEL COVERAGE: posterise at fixed texel centres, blend the COLOURS by coverage — never sample the hash at camera-dependent positions, and never posterise an averaged noise value.** The ground albedo stacks two hard quantisers (per-texel hash, palette LUT); any scheme that evaluates them at positions that move with the camera pops palette steps under sub-pixel motion ("z-fighting" shimmer).  `terrain.frag` therefore evaluates `groundNoise` ONLY at the 4 nearest fine-texel centres (fixed world points → each corner's posterised colour is constant), runs the LUT per corner (quantise-after-filter re-hardens — keep the order), and blends the 4 colours by the pixel footprint's coverage of each texel (`w = clamp((fract-0.5)/cov + 0.5, 0, 1)`, `cov` = footprint in texels).  The result is a continuous function of surface position: texel edges are crisp ~1 px AA ramps that slide smoothly, interiors saturate to one flat palette colour (pixel art intact).  History (`tools/shimmer_probe.py`, measurements in `tools/out/diag/`): the original average-noise-then-posterise scored far-ground flip fraction 0.0080; posterise-per-sliding-tap fixed flat ground (0.0003) but still boiled on crater walls seen head-on (no octave fade there, taps slid through full-contrast texels); texel-coverage filtering zeroed flat ground and took the crater band to the geometry-edge floor.  The normal map and the light-quant grid were both measured INNOCENT — re-measure with the probe before touching either.

22. **Never use `fwidth()`/`dFdx()` in the terrain shader — compute the screen footprint analytically.** Screen-space derivatives are evaluated on 2×2 pixel quads; wherever a quad straddles two facets of the faceted mesh the helper pixels extrapolate the wrong plane and the derivative is garbage.  The terrain shader computes `mpp = dist * u_px_rad / max(|dot(view_dir, n)|, 0.18)` (exact for planar facets, stable everywhere); `u_px_rad` (radians of view angle per screen pixel) is set per frame in `GpuLightingPipeline.update_surface_inputs` from the lens FOV and window width.  The light-quant LOD also derives from this `mpp`, snapping to power-of-two multiples of `u_quant_m` so coarser light-pixel lattices stay exactly nested and world-anchored (a continuously varying cell size re-seats every cell boundary as the footprint changes — its own shimmer).

23. **Geometry-edge twinkle is MSAA's job — `msaa_samples` in config (default 4, 0 = off).** Facet silhouettes (crater rims, the horizon line) alias no matter how well surfaces are filtered; that residual is rasterisation, not texturing.  `App.__init__` requests `framebuffer-multisample`/`multisamples` via PRC BEFORE the window opens and sets `AntialiasAttrib.M_multisample` (else `M_none`).  MSAA touches polygon edges only — interiors are shaded once per pixel, so the retro texel look is unchanged (verified by pixel-identical crop comparison).  Note for probe users: the FIRST boot after changing the framebuffer config can produce one anomalous sweep (driver shader recompilation mid-run) — re-run before believing a regression.
