# lighting — System Doc
keywords: sun, sunlight, moonlight, light grid, occupancy, ambient, GI, global illumination, bounce, flood fill, radiance cascade, volume, volumetric, fog, god rays, froxel, shadow, voxel shadow, AO, ambient occlusion, point light, area light, torch, emission, emissive, albedo, palette, compute shader, GPU, inject, propagate, LightGrid, SunlightComputer, make_light_sampler, occupancy_from_materials, LIGHT_FULL, LIGHT_AMBIENT, VolumeWindow, GeometryVolume, assemble_geometry, EMISSION_SCALE, MaterialPalette, build_default_palette, PointLight, AreaLight, LightSet, GpuLightingPipeline, lighting_backend, light_quant_m, dispatch_compute, ping pong, cascade, cascade 2, third cascade, far cascade, coarse cascade, distant lighting, light_c2_cells, light_c2_cell_m, recenter, hysteresis, thread, threaded, worker, background thread, off-thread, async assembly, CascadeAssemblyWorker, assembly_worker, assemble_packed, AssemblyJob, AssemblyResult, window_chunk_span, pack_volume, needs_recenter, stutter, fps, main thread, shutdown, profile_stream, black flash, crater black, explosion lighting, edit lighting, _apply_edits_sync, sync reassembly, same-frame

> One doc per code package; filename matches the package exactly (`docs/systems/lighting.md` ↔ `fire_engine/lighting/`).

## Role

`lighting/` (Layer 1 — Services) owns all scene lighting.  Two backends, selected by `config.lighting_backend`:

**GPU backend (`"gpu"`, default)** — fully volumetric lighting computed on the GPU each frame:

- **Radiance cascades** — three camera-centered 3-D texture windows, finest-first: cascade 0 at `light_c0_cell_m` = 0.5 m cells (`light_c0_cells` = 96 → a 48 m box), cascade 1 at 2.0 m cells (192 m box, covers the streaming view distance), and **cascade 2** — the coarse FAR cascade — at `light_c2_cell_m` = 8.0 m cells (`light_c2_cells` = 64 → a 512 m box).  Cascade 2 keeps distant terrain (and the GI test room) lit with low-resolution shadows + GI once a surface leaves cascade 1 — chiefly the leading-edge band of newly-streamed chunks the trailing (hysteresis-lagged) cascade-1 window hasn't caught up to, which previously popped to flat sky ambient.  All three ride the identical off-thread assembly + inject + propagate machinery (the per-frame loops iterate `self.cascades`), so "bake far chunks on a separate thread at a lower resolution" needs no new subsystem.  `VolumeWindow` (headless, `volume.py`) owns the grid-snapped window math with recenter hysteresis; `assemble_geometry` slices loaded chunk material arrays into contiguous `uint8` occupancy/albedo/emission blocks (one numpy pass per intersecting chunk, no per-voxel loops).
- **Material palette** — `MaterialPalette` (`palette.py`): 256-row albedo/emission lookup derived from the procedural ground textures' mean linear RGB (`build_default_palette`).  Emissive materials (`with_emission`) inject light into the world from the volume itself.
- **GPU passes** (GLSL 430 compute sources in `lighting/shaders/*.comp` — `inject`, `propagate`, `fog_scatter`, `fog_integrate` — loaded by `glsl.py` via `core.shader_source.load_glsl` and re-exported as `INJECT_COMPUTE` etc.; `glsl.py` still owns the non-GLSL `MAX_LIGHTS`. Dispatched by `GpuLightingPipeline` in `gpu.py` via `GraphicsEngine.dispatch_compute`):
  1. **INJECT** (on change only) — per cell: occupancy raymarch toward the sun/moon (→ the `u_vis` visibility volume = all voxel shadows), straight-up march (→ skylight from `SkyState.sky_ambient`), first-bounce sunlight off solid neighbours tinted by their albedo, emissive-neighbour leak, and dynamic point/area lights with windowed inverse-square falloff + shadow march.
  2. **PROPAGATE** (every frame, `light_prop_iters` ping-pong Jacobi iterations) — exponential diffusion `next = direct·(1−decay) + decay·avg₆`; solid neighbours reflect a cell's own light back tinted by albedo (multi-bounce).  This is the flood-fill **GI**: light visibly flows around corners over a few frames and converges to a stable bounded field.
  3. **FOG_SCATTER / FOG_INTEGRATE** (every frame) — a camera-frustum **froxel** volume (`fog_froxels_x/y/z`, exponential depth slices to `fog_far_m`): weather-driven height fog density, Henyey-Greenstein sun/moon in-scatter shadowed through the cascade occupancy (→ **god rays**), sky/GI ambient scatter; integrated front-to-back into per-slice (accumulated light, transmittance).
- **Dynamic lights** — `LightSet` (`lights.py`): `PointLight` / `AreaLight` registry with TTL transients (explosion flashes fade and expire), packed to one `float32 (64, 12)` uniform array per change.
- **Surface contract** — `GpuLightingPipeline.bind_surface_inputs(node)` / `update_surface_inputs(node, sky_state)` feed `world/terrain_shader.py`, which samples the cascades (quantised to `light_quant_m` = 0.0625 m → 8×8×8 visible "light pixels" per voxel — the snap grid only; the GI *data* resolution is the cascade-0 cell), reads `u_vis` for celestial shadows, derives AO from occupancy, adds emission, and composites the froxel fog at its own depth.

**CPU backend (`"cpu"`, legacy)** — the Phase-4 v0 baked pipeline, fully retained: `LightGrid` (per-chunk `uint8 (16,16,16)` @ 1 m cells), `SunlightComputer` column pass + 3×3×3 box blur, `make_light_sampler` baking into mesh vertex colours.

`lighting/` is the one non-`world/` package allowed to touch the GPU (ARCHITECTURE §4 rule 4); only `gpu.py` imports panda3d — every other module is headless-testable.

## Public API

Headless symbols re-exported from `fire_engine.lighting`; the GPU pipeline imports explicitly from `fire_engine.lighting.gpu`.

| Symbol | Description |
|---|---|
| `VolumeWindow(cells, cell_m, snap_cells=8, margin_cells=8)` | Camera-centered grid-snapped cascade window; `recenter(camera_pos) -> bool` (mutates origin), `needs_recenter(camera_pos) -> bool` (non-mutating drift test, used by the async path), `world_origin_m`, `size_m`. |
| `assemble_geometry(window, chunks, palette, chunk_size, voxel_size) -> GeometryVolume` | Slice chunk materials into `albedo_occ`/`emission` `uint8 (N,N,N,4)` blocks (max-material downsample for coarse cascades). `chunks` values may be chunk objects (`.materials`) or bare ndarray snapshots (async path). |
| `window_chunk_span(origin_cell, cells, cell_m, chunk_size, voxel_size) -> list[coord]` | Chunk coords a window at `origin_cell` intersects — lets the async path snapshot exactly the chunks a reassembly reads. |
| `pack_volume(arr) -> bytes` | Numpy transpose + RGBA→BGRA + contiguous pack of a `(N,N,N,4)` block to Panda3D 3-D-texture RAM bytes (the off-thread half of an upload). |
| `AssemblyJob` / `AssemblyResult` / `CascadeAssemblyWorker()` / `assemble_packed(job)` (`assembly_worker.py`, headless) | Background cascade-volume assembly: `worker.start()/submit(job)/drain_results()/pending()/stop()`. Worker output is byte-identical to a synchronous `assemble_geometry` + `pack_volume`. |
| `GeometryVolume` | Packed block + `origin_cell` + `cell_m`. |
| `EMISSION_SCALE = 8.0` | HDR emission ÷ scale on uint8 pack; × scale in shaders. |
| `MaterialPalette` | `albedo`/`emission` `float32 (256, 3)` lookups; `with_emission(material, rgb)` copy-with-glow. |
| `build_default_palette() -> MaterialPalette` | Albedo from `dirt_ground`/`grass_ground` mean linear RGB. |
| `PointLight(position, color, intensity, radius, ttl_s=None)` | Omni light; HDR intensity (torch ≈ 8, explosion flash ≈ 40). |
| `AreaLight(center, half_extents, color, intensity, radius, ttl_s=None)` | Axis-aligned emissive box (falloff from the box surface). |
| `LightSet()` | Registry: `add/remove/clear/update(dt)/pack(max_lights)`; `version` bumps on packed-data change. |
| `LightGrid`, `SunlightComputer`, `make_light_sampler`, `occupancy_from_materials`, `LIGHT_FULL`, `LIGHT_AMBIENT` | CPU backend — unchanged from Phase 4 v0 (see git history of this doc for the full reference). |

`fire_engine.lighting.gpu` (panda3d — excluded from headless tests):

| Symbol | Description |
|---|---|
| `GpuLightingPipeline(config, base, chunk_provider, bus, palette=None, *, threaded=True)` | Owns cascade textures + compute dispatch; subscribes to `TerrainEditedEvent`/`ChunkLoadedEvent`. `threaded=False` assembles volumes inline (deterministic tooling/tests). |
| `.update(camera_pos, sky_state, dt)` | Per-frame driver (App frame task step 6): schedule async reassembly → drain+upload finished volumes → inject (dirty cascades only) → propagate → fog. |
| `.shutdown()` | Stop the background `CascadeAssemblyWorker`. Call once on app exit (`main()` does, in a `finally`). Idempotent; no-op when unthreaded. |
| `.bind_surface_inputs(node)` / `.update_surface_inputs(node, sky_state)` | Static / per-frame shader-input contract for `world/terrain_shader.py`. |
| `.lights: LightSet` | Public dynamic-light registry (demo: explosion flash, L-key torches). |

`fire_engine.lighting.glsl` — `INJECT_COMPUTE`, `PROPAGATE_COMPUTE`, `FOG_SCATTER_COMPUTE`, `FOG_INTEGRATE_COMPUTE`, `MAX_LIGHTS = 64` (plain strings, headless-importable).

## Imports Allowed

- `numpy`, stdlib, `fire_engine.core` everywhere; `fire_engine.procedural` (palette derivation — foundation layer, callable from anywhere); `fire_engine.terrain` (downward).
- `panda3d` **only in `gpu.py`** — keeps the rest of the package in the headless suite.
- No imports from `world/` or higher.  `world/terrain_shader.py` imports lighting, never the reverse.

## Events

### Subscribed
| Event | Subscriber | Action |
|---|---|---|
| `TerrainEditedEvent` | `SunlightComputer` (CPU) | Recompute affected columns; mark chunks dirty for remesh. |
| `TerrainEditedEvent` | `GpuLightingPipeline` (GPU) | Queue the edited chunk coords; the intersecting cascades reassemble + re-inject **synchronously the same frame** (`_apply_edits_sync`) so the new crater lights immediately — an async (1–2 frame) reassembly leaves the stale occupancy marking the crater solid/shadowed, so it renders black then pops to lit. Edits are discrete events, so the synchronous gather is affordable; cascades with a job already in flight fall back to the batched `_pending_coords` path. |
| `ChunkLoadedEvent` | `SunlightComputer` (CPU) | Recompute the chunk's column. |
| `ChunkLoadedEvent` | `GpuLightingPipeline` (GPU) | Queue coord; batched reassembly (≥ 0.25 s apart) for cascades the chunk actually intersects — streaming-frontier loads never touch cascade 0. |

### Published
None.

## Units & Invariants

- World meters, Z-up.  Cascade texel `(i,j,k)` covers `[(origin_cell + (i,j,k))·cell_m, +cell_m)`; `origin_cell` snaps to `snap_cells` so consecutive windows align on the cell grid (no sub-cell crawl).
- Volume arrays are indexed `[x, y, z]`; GPU upload transposes to Panda3D's page-major `(z, y, x)` and reorders RGBA→BGRA (`_upload_volume`).
- `albedo_occ`: RGB = linear albedo ×255, A = 255 solid / 0 air.  Coarse cascades downsample by **max material id** per block (any solid ⇒ solid; grass skin wins over dirt for bounce colour).
- `emission`: RGB = linear HDR emission ÷ `EMISSION_SCALE` ×255.
- Radiance/visibility volumes are `rgba16f`, GPU-resident only (never read back).
- Propagation is contractive (`decay < 1`, albedo < 1 ⇒ spectral radius < 1): the radiance field is always bounded by the injected sources; reach ≈ `1/(1−decay)` cells (decay = `exp(-cell_m / 4 m)` C0, `exp(-cell_m / 10 m)` C1).
- Light values are **linear HDR RGB** in SkyState units: noon sun ≈ 3.2, skylight ≈ 0.2–0.7, torch intensity ≈ 8.  The terrain/sky shaders ACES-tonemap with `light_exposure`.
- Lighting is **not** `Saveable` — fully derived from terrain + clock + lights each run.
- Determinism (headless half): `assemble_geometry` and `build_default_palette` are pure functions of chunk data/seed — byte-identical across runs (`tests/test_lighting_volume.py`).
- Config (`[lighting]`/`[fog]` tables): `lighting_backend`, `light_c0_cells/_cell_m`, `light_c1_cells/_cell_m`, `light_c2_cells/_cell_m` (coarse far cascade), `light_quant_m` (0.0625 → 8×8×8 light pixels per voxel), `light_prop_iters`, `light_bounce_strength`, `light_ao_strength`, `light_max_point_lights`, `light_exposure`, `fog_enabled`, `fog_froxels_x/y/z`, `fog_far_m`, `fog_anisotropy`.

## Examples

### Boot wiring (GPU backend — what main.py does)
```python
from fire_engine.lighting.gpu import GpuLightingPipeline
from fire_engine.world.terrain_shader import apply_terrain_shader

pipeline = GpuLightingPipeline(cfg, app, chunk_manager, bus)
app.lighting_pipeline = pipeline            # App frame task drives update()
apply_terrain_shader(app.terrain_root, pipeline)
# mesher gets light_sampler=None → vertex colours carry only the facet accent
```

### Dynamic lights
```python
from fire_engine.lighting import PointLight, AreaLight

torch_id = pipeline.lights.add(PointLight(
    position=(8.0, 8.0, 10.5), color=(1.0, 0.62, 0.28),
    intensity=8.0, radius=16.0))
pipeline.lights.add(PointLight(                       # explosion flash
    position=hit, color=(1.0, 0.55, 0.2), intensity=40.0,
    radius=18.0, ttl_s=0.5))                          # fades + auto-removes
pipeline.lights.add(AreaLight(                        # glowing doorway
    center=(0, 4, 9), half_extents=(0.5, 0.1, 1.0),
    color=(1.0, 0.8, 0.5), intensity=2.0, radius=10.0))
pipeline.lights.remove(torch_id)
```

### Emissive material
```python
from fire_engine.lighting import build_default_palette
palette = build_default_palette().with_emission(7, (2.0, 1.2, 0.4))  # lava-ish
pipeline = GpuLightingPipeline(cfg, app, chunk_manager, bus, palette=palette)
```

### Headless volume assembly (tests / tools)
```python
from fire_engine.lighting import VolumeWindow, assemble_geometry, build_default_palette
win = VolumeWindow(cells=96, cell_m=0.5)
win.recenter(camera_pos)
vol = assemble_geometry(win, chunk_manager.chunks, build_default_palette(),
                        chunk_size=cfg.chunk_size, voxel_size=cfg.voxel_size)
```

## Gotchas

1. **Never import `fire_engine.lighting.gpu` from headless code/tests** — it imports panda3d.  The package `__init__` deliberately does not re-export it.
2. **The GPU backend bakes NO light into vertex colours** — main.py passes `light_sampler=None`, so vertex colours hold only the facet accent.  Pointing the old CPU sampler at meshes while the GPU shader is active would double-light.
3. **`SkyRendererComponent` must be constructed with `external_lighting=True`** on the GPU backend, or its `terrain_root.set_color_scale` + Panda3D `Fog` fight the shader.
4. **Light flows over a few frames** (propagation is iterative).  An explosion flash (`ttl_s=0.5`) is visible because injection re-runs on every lights change; don't expect single-frame convergence of large skylight changes — that's the look, not a bug.
5. **Sun/moon shadows live in the `u_vis` volume**, recomputed only when sun direction/radiance/volume/lights change (`_changed` eps ≈ 0.004).  At default `game_time_scale=60` that's a few injects per real minute — near-free.  Time-lapse (`game_time_scale=1800`) re-injects often: expect GPU load.
6. **Hybrid laptops**: Panda3D windows often default to the integrated GPU.  Set python.exe to "High performance" in Windows Graphics Settings to run on the discrete GPU — the difference is ~10×.
7. **Beyond cascade 2** (> ~256 m from the camera) surfaces fall back to sky-ambient and full sun visibility — no voxel shadows.  Cascade 2 (8 m cells, 512 m box) now covers the band between cascade 1's edge and the streaming frontier, so the leading-edge "lighting pops to flat" seam while flying — and the GI room going flat as you back away from it — is gone.  At the current 96 m streaming radius cascade 2 is reached only by geometry the trailing cascade-1 window hasn't caught up to; raising `view_distance_chunks` toward the 1 km goal makes it the primary far-lighting tier.
8. **`assemble_geometry` requires `cell_m` to be an integer multiple of `voxel_size`** and `chunk_size` divisible by the per-cell voxel count — it raises `ValueError` otherwise.
9. **Area lights shadow-march from the closest box point**, so a box buried in terrain lights nothing — keep emissive boxes in open air, like point lights (a light at exactly ground height is half-buried and mostly shadowed).
10. **`LightSet.update(dt)` bumps `version` every frame while any TTL light lives** — transient lights deliberately re-inject per frame to animate the fade.  Many simultaneous transients = many injects; cap effects accordingly.
11. **Cascade volume assembly runs OFF the main thread** (`CascadeAssemblyWorker`).  The CPU gather+pack (~90 ms p99 on a fly-around) was *the* fly-around stutter (`tools/profile_stream.py`); it now overlaps the render thread (numpy releases the GIL).  **Invariant:** a cascade's committed `window.origin_cell` — which drives the `u_c*_origin_m` shader/inject/fog uniforms — only advances when the matching volume is uploaded (`_commit_assembly_result`), so the GPU geom texture and the origin uniform never disagree.  Cost: the lighting volume lags the camera by ≤1–2 frames in position, well inside the existing recenter hysteresis (no visible artifact).  The boot/first frame assembles **synchronously** so the world is lit on frame 1.  One job per cascade is in flight at a time; the snapshot captures *references* to chunk material arrays, so a concurrent in-place brush edit of a captured array is the only race — transient, self-correcting on the next reassembly.
