# world â€” System Doc
keywords: gameobject, game object, transform, component, registry, lifecycle, awake, start, update, late_update, fixed_update, on_enable, on_disable, on_destroy, instantiate, destroy, find_with_tag, find_objects_with_tag, space, camera, app, showbase, panda3d, input, inputstate, unity, clone, hierarchy, parent, children, world matrix, dirty flag, position, rotation, forward, right, up, look_at, transform_point, inverse_transform_point, active_in_hierarchy, set_active, add_component, get_component, get_components, get_component_in_children, remove_component, compare_tag, layer, tag, uuid, batched, bucket, run_frame, fixed_steps, spiral of death, quaternion, z-up, meters, radians, texture_bridge, to_panda_texture, geometry_bridge, to_geom, to_geom_node, resource_adapter, register_panda_loaders, bridge, bgra, f_rgba, ram_image, world meters, terrain_root, chunk_manager, light_sampler, setup_terrain_rendering, stream_and_upload_terrain, nodepath, geom, vertex color, sky, sky_renderer, SkyRendererComponent, sky_shaders, sky dome, skydome, clouds, cloud layer, boxy clouds, raymarch, dda, rain, rain_streak, night_sky, galaxy, stars, twinkle, shooting star, sun disc, moon, moon phase, fog, exponential fog, set_color_scale, terrain_light_scale, background bin, set_shader_input, weather render, day night cycle render, equirect, terrain_shader, apply_terrain_shader, volumetric, GPU lighting, lighting_pipeline, radiance cascade, froxel, ACES, tonemap, normal map, emission map, TBN, make_material_state, texture stage, external_lighting, physical atmosphere, single scattering, sun disc, moon texture, moon_surface

> One doc per code package; filename matches the package exactly (`docs/systems/world.md` â†” `fire_engine/world/`).

## Role

`world/` is the **World API** â€” Layer 3 in the engine stack.  It provides:

- **Unity-clone object model** (`GameObject`, `Component`, `Transform`): same API shape as Unity (snake_case), Z-up coordinates, quaternion-only rotations.  All authoring uses these types.
- **ComponentRegistry**: the batched frame executor that drives Unity lifecycle order across all components.
- **App** (`world/app.py`): the Panda3D ShowBase wrapper, window, frame loop, input collection, and the **terrain render path** (chunk streaming → Geom upload).  Frame-task step 6 drives the optional `app.lighting_pipeline` (GPU volumetric lighting: `GpuLightingPipeline.update(...)` + `update_surface_inputs(terrain_root, sky_state)` each frame) when main.py wires `config.lighting_backend == "gpu"`.
- **Terrain surface shader** (`world/terrain_shader.py`, GPU backend): `apply_terrain_shader(terrain_root, pipeline)` replaces the fixed-function texture×vertex-colour pipeline with a GLSL 330 fragment shader that samples the radiance cascades (positions quantised to `light_quant_m` "light pixels"), reads voxel-marched sun/moon visibility for shadows, derives AO from the occupancy volume, applies normal/emission maps (analytic TBN from the dominant face axis — matches the mesher's planar UVs), composites froxel volumetric fog at the fragment's own depth, and ACES-tonemaps with `light_exposure`.  Per-material Geoms carry **(albedo, normal, emission)** texture-stage triples built by `geometry_bridge.make_material_state` (stage sorts 0/1/2 → `p3d_Texture0/1/2`); normal maps are derived from albedo luminance by `procedural/maps.py`.
- **CameraComponent** (`world/camera.py`): one-way sync from a `Transform` to the Panda3D camera NodePath.
- **Bridges** â€” the (small) set of files that translate engine data into Panda3D objects, so every layer below stays panda3d-free:
  - `texture_bridge.to_panda_texture` (Phase 2): numpy RGBA â†’ Panda3D `Texture`.
  - `geometry_bridge.to_geom` / `to_geom_node` (Phase 3): `MeshArrays` â†’ Panda3D `Geom` / `GeomNode` (bulk writes).
  - `resource_adapter.register_panda_loaders` (Phase 5): injects Panda3D asset loaders into the `resources.ResourceManager` (inversion of control).
- **DevOverlay** (`world/devtools_overlay.py`): the DirectGUI renderer for the in-game developer overlay (F1). It is the Panda3D half of the `devtools` system â€” see `docs/systems/devtools.md`; the headless brain lives in `fire_engine/devtools/`.
- **SkyRendererComponent** (`world/sky_renderer.py`) + GLSL sources (`world/sky_shaders.py`): the render half of the sky/weather system. The headless simulation lives in `fire_engine/sky/`; this component drives `sky_system.update()` each frame and draws the sky dome — now a **per-pixel physical single-scattering atmosphere** (Rayleigh+Mie, constants mirrored from `sky/atmosphere.py`) with a 2.5×-sized limb-darkened sun disc tinted by its own transmittance, a 2.5×-sized moon disc textured with the seeded procedural `"moon_surface"` texture under the dynamic phase terminator, `"night_sky"` galaxy + twinkle + shooting stars composited post-tonemap — plus boxy raymarched clouds and rain cylinders.  With `external_lighting=False` (CPU backend) it also applies exponential fog and the global terrain colour-scale; with `external_lighting=True` (GPU backend) it leaves terrain shading entirely to the lighting pipeline and samples the froxel fog's far slice for the dome.

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
| `InputState` | Dataclass: move_forward/backward/left/right/up/down, sprint, mouse_dx/dy, mouse_captured, escape_pressed. |

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
| Cloud quads (2 Ã— 2400 m horizontal quads at slab bottom + top) | `background` 20 | test+write OFF | `M_alpha`, two-sided. XY follow snapped to `sky_cloud_cell_m` multiples so the grid never swims. Fragment shader 2D-DDA raymarches the slab `[sky_cloud_altitude_m, +sky_cloud_thickness_m]` (â‰¤48 cell steps, early-out at Î±>0.98); correct from below / inside / above (duplicate plane coverage discarded by camera-height test). |
| Rain (3 nested open cylinders, r = 4/7/11 m, h = 14 m) | transparent (default) | write OFF | `"rain_streak"` texture, additive `ColorBlendAttrib` (`O_incoming_alpha`, `O_one`), per-layer UV scroll rates for parallax, wind tilt; `hide()` when `rain_intensity < 0.05`. |

**Dome shader uniforms** (per frame unless noted): `u_sun_dir`, `u_sun_color`, `u_sun_intensity`, `u_moon_dir`, `u_moon_phase`, `u_zenith_color`, `u_horizon_color`, `u_star_visibility`, `u_star_rotation` (radians about +Z; one revolution per game day), `u_time` (real s, twinkle hash), `u_fog_color`, `u_fog_blend`, shooting star `u_ss_active`/`u_ss_start`/`u_ss_travel`/`u_ss_progress`; `p3d_Texture0` = `"night_sky"` equirect.
**Cloud shader uniforms**: static `u_seed` (from `for_domain("sky", "clouds")`), `u_altitude`, `u_thickness`, `u_cell`, `u_fade_dist`; per frame `u_cam_pos`, `u_coverage` (a noise-threshold QUANTILE, see gotcha 12), `u_opacity`, `u_wind_offset` (CPU-integrated `wind_dir * wind_speed * dt`, meters), `u_top_color`/`u_side_color`/`u_bottom_color` (flat-face lighting computed CPU-side: sunlit tops, medium sides, density-darkened bottoms).

**Lighting integration (fog + global light):** each `late_update` the component sets the exponential `panda3d.core.Fog` on `terrain_root` to `SkyState.fog_density` (1/m) and `fog_color`, sets the window clear colour to horizon-blended-with-fog, and calls `terrain_root.set_color_scale(*terrain_light_scale, 1.0)` â€” the **baked vertex sunlight Ã— global day/night scale** product is the whole day-night terrain lighting story (no Panda3D lights involved).

**Shooting stars are deterministic:** game time is split into 30-game-minute slots; `for_domain("sky", "shooting_stars", game_day, slot)` decides spawn (pâ‰ˆ0.5) + start/travel directions; the streak animates over ~1.2 real seconds and only spawns while `star_visibility > 0.5`.

### GrassRendererComponent (`world/grass_renderer.py`, shaders in `world/grass_shaders.py`)

GPU-only instanced grass for every `tag="grass"` `ZoneVolume` (headless math in `fire_engine/zones/` — see `docs/systems/zones.md`). **GPU lighting backend only**: requires the active `GpuLightingPipeline`; on the legacy CPU backend the component logs and disables itself.

| Symbol | Description |
|---|---|
| `GrassRendererComponent(base, sky_system, zone_store, chunk_provider, lighting_pipeline, bus)` | Add via `add_component`. `chunk_provider` = anything with a `.chunks` dict (`ChunkManager`); `lighting_pipeline` = the active `GpuLightingPipeline`. |

**The CPU stores no blades.** One shared 3-crossed-quad tuft `Geom` (12 verts) is drawn `grass_instance_count(volume, cfg)` times per volume via `set_instance_count`; the vertex shader derives each instance's XY/yaw/scale/sway-phase from `gl_InstanceID` through the lowbias32 hash chain mirrored line-for-line by `zones.instance_attribs`. The only CPU bake is the per-volume **height field** (`zones.bake_grass_height_field`, uploaded with `to_field_texture`): R = terrain surface height inside the volume's Z window, 255 = no ground → the shader collapses that blade (craters cull grass). Re-baked when `TerrainEditedEvent`/`ChunkLoadedEvent` touches the volume (events mark dirty; the re-bake runs in the next `late_update`).

**Lighting/fog by inheritance:** the grass root is parented under `App.terrain_root`, where `GpuLightingPipeline.bind_surface_inputs`/`update_surface_inputs` already maintain every cascade/fog/celestial uniform — the grass fragment shader declares the same names and gets terrain-identical light (radiance-cascade GI + voxel-shadowed sun/moon at the blade base, snapped to the `light_quant_m` pixel grid) plus the one-tap froxel fog, ACES and gamma.

**Weather sway:** `late_update` maps `SkyState.wind_dir/wind_speed/rain_intensity` to `u_sway_base` (static lean), `u_sway_gust` + `u_gust_freq` (oscillation — storms move grass harder and faster); displacement in the shader is quadratic in blade height so bases stay pinned. Distance fade shrinks blades to zero across `[grass_fade_start_m, grass_fade_end_m]` — no popping or far shimmer. Alpha is a binary cutout (`discard < 0.5`, the pixel-art `"grass_tuft"` texture) so no sorting or transparency bin is needed.

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
None directly from world/.  App calls `event_bus.drain()` each frame; the bus carries events from terrain/lighting/etc.

### Subscribed
None directly.  Terrain/lighting integrate via the App integration hooks, not event subscriptions.

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

11. **The rain streak V axis is deliberately mirrored.** `to_panda_texture` vertically flips every upload (gotcha 8), and the `"rain_streak"` def paints each streak's motion-blur tail at higher array rows â€” post-flip the tail sits at LOWER texture v.  `_build_rain_cylinder` therefore maps `v = v_tiles` at the cylinder BOTTOM and `v = 0` at the top, and `_update_rain` scrolls the V offset DOWN (decreasing) â€” together the bright heads lead toward the ground and the pattern falls.  "Fixing" either half alone makes the rain fall upward or flips the streaks.

12. **`u_coverage` is a quantile threshold, not the coverage value.** The cloud shader's per-cell noise is bell-shaped around ~0.5; thresholding it with raw `cloud_coverage` would render almost no clouds below ~0.3.  `_cloud_value_quantiles` (a numpy float32 port of the same GLSL noise) is sampled once at start; per frame the renderer passes `quantiles[coverage * (n-1)]` so coverage is the ACTUAL fill fraction.  If you change the GLSL `cell_value`, change the numpy port in the same commit.

13. **Adjacent occupied cloud boxes share interior faces.** The DDA shades each box's ENTRY face independently; without the `prev_hit`/`carry_col` continuity in `CLOUD_FRAGMENT`, every cell seam draws a side-coloured grid line across distant cloud ceilings (and float32 precision adds dark gap slivers â€” hence the small Â±0.05 m interval overlap).

14. **The `"night_sky"` equirect needs no flip handling in the dome shader.** The def is authored with array row 0 = zenith specifically so that after `to_panda_texture`'s vertical flip the natural mapping `v = asin(d.z)/Ï€ + 0.5` lands the zenith at v = 1.  Don't add a `1 - v`.

15. **`tools/screenshot.py` releases mouse capture before stepping frames.** The window starts in relative-mouse capture; physical mouse movement during an unattended capture would otherwise feed deltas into `FlyController` and swing the camera mid-warmup.

16. **Mouse capture is reasserted on focus regain.** Alt-tabbing away drops the OS-level hidden-cursor / relative-mouse window properties, and Panda3D does not re-apply them when the window refocuses â€” so the engine would believe the mouse is captured while the desktop shows a free pointer, killing free-look until the next ESC.  `App.windowEvent` overrides ShowBase's handler (calls `super()` first), detects the focus-regain edge via `win.get_properties().get_foreground()`, and re-issues `_set_mouse_capture(input_state.mouse_captured)` (which re-arms `_skip_mouse_delta` so the view doesn't snap).  Complementarily, `main.py`'s `on_click` re-captures when the cursor is free and the overlay is closed, so a click after ESC resumes flying instead of firing the demo explosion.

17. **Instanced nodes are culled by the BASE Geom's bounds.** `set_instance_count` does not grow the cull volume — Panda3D would frustum-cull every grass volume by the single origin-sized tuft Geom and the blades would vanish whenever the origin left the view.  `GrassRendererComponent._build_volumes` therefore sets an explicit `BoundingBox` (volume AABB + blade reach) on each `GeomNode` and calls `set_final(True)`.  The same applies to ANY future `gl_InstanceID`-positioned geometry.

18. **The grass GLSL hash chain has a python twin.** `lowbias32` + the h0–h4 chain in `world/shaders/grass.vert` (loaded by `world/grass_shaders.py`) is mirrored line-for-line by `zones/grass_placement.py::instance_attribs` (which the headless tests pin).  Edit both or neither — a desync makes the tests describe blades that aren't where the GPU draws them.
