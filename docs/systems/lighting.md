# lighting — System Doc
keywords: sun, sunlight, moonlight, light grid, occupancy, ambient, GI, global illumination, bounce, flood fill, radiance cascade, volume, volumetric, fog, god rays, froxel, shadow, voxel shadow, AO, ambient occlusion, point light, area light, torch, emission, emissive, albedo, palette, compute shader, GPU, inject, propagate, LightGrid, SunlightComputer, make_light_sampler, occupancy_from_materials, LIGHT_FULL, LIGHT_AMBIENT, VolumeWindow, GeometryVolume, assemble_geometry, EMISSION_SCALE, MaterialPalette, build_default_palette, PointLight, AreaLight, LightSet, GpuLightingPipeline, lighting_backend, light_quant_m, dispatch_compute, ping pong, cascade, cascade 2, third cascade, far cascade, coarse cascade, distant lighting, light_c2_cells, light_c2_cell_m, recenter, hysteresis, thread, threaded, worker, background thread, off-thread, async assembly, CascadeAssemblyWorker, assembly_worker, assemble_packed, AssemblyJob, AssemblyResult, window_chunk_span, pack_volume, needs_recenter, stutter, fps, main thread, shutdown, profile_stream, black flash, crater black, explosion lighting, edit lighting, _apply_edits_sync, sync reassembly, same-frame, fractional occupancy, solid fraction, partial occupancy, hollow room, Cornell room, black hole, ChunkBlockCache, block cache, downsample cache, invalidate, LRU, coarse downsample, solid sub-voxel fraction, radiance shift, shift pass, recenter pop, boost_frames, boot warmup, warmup, dark load-in, margin_cells, recenter hysteresis, near-cascade latency, c0 immediate, chunk-load pop, fog_far_m, SHIFT_COMPUTE, shift.comp

> One doc per code package; filename matches the package exactly (`docs/systems/lighting.md` ↔ `fire_engine/lighting/`).

## Role

`lighting/` (Layer 1 — Services) owns all scene lighting.  Two backends, selected by `config.lighting_backend`:

**GPU backend (`"gpu"`, default)** — fully volumetric lighting computed on the GPU each frame:

- **Radiance cascades** — three camera-centered 3-D texture windows, finest-first: cascade 0 at `light_c0_cell_m` = 0.5 m cells (`light_c0_cells` = 96 → a 48 m box), cascade 1 at 2.0 m cells (192 m box, covers the streaming view distance), and **cascade 2** — the coarse FAR cascade — at `light_c2_cell_m` = 8.0 m cells (`light_c2_cells` = 64 → a 512 m box).  Cascade 2 keeps distant terrain (and the GI test room) lit with low-resolution shadows + GI once a surface leaves cascade 1 — chiefly the leading-edge band of newly-streamed chunks the trailing (hysteresis-lagged) cascade-1 window hasn't caught up to, which previously popped to flat sky ambient.  All three ride the identical off-thread assembly + inject + propagate machinery (the per-frame loops iterate `self.cascades`), so "bake far chunks on a separate thread at a lower resolution" needs no new subsystem.  `VolumeWindow` (headless, `volume.py`) owns the grid-snapped window math with recenter hysteresis; `assemble_geometry` slices loaded chunk material arrays into contiguous `uint8` occupancy/albedo/emission blocks (one numpy pass per intersecting chunk, no per-voxel loops).
- **Material palette** — `MaterialPalette` (`palette.py`): 256-row albedo/emission lookup derived from the procedural ground textures' mean linear RGB (`build_default_palette`).  Emissive materials (`with_emission`) inject light into the world from the volume itself.
- **GPU passes** (GLSL 430 compute sources in `lighting/shaders/*.comp` — `inject`, `propagate`, `shift`, `fog_scatter`, `fog_integrate` — loaded by `glsl.py` via `core.shader_source.load_glsl` and re-exported as `INJECT_COMPUTE`, `SHIFT_COMPUTE` etc.; `glsl.py` still owns the non-GLSL `MAX_LIGHTS`. Dispatched by `GpuLightingPipeline` in `gpu.py` via `GraphicsEngine.dispatch_compute`):
  1. **INJECT** (on change only) — per cell: occupancy raymarch toward the sun/moon (→ the `u_vis` visibility volume = all voxel shadows), straight-up march (→ skylight from `SkyState.sky_ambient`), first-bounce sunlight off solid neighbours tinted by their albedo, emissive-neighbour leak, and dynamic point/area lights with windowed inverse-square falloff + shadow march.
  2. **PROPAGATE** (every frame, `light_prop_iters` ping-pong Jacobi iterations) — exponential diffusion `next = direct·(1−decay) + decay·avg₆`; solid neighbours reflect a cell's own light back tinted by albedo (multi-bounce).  This is the flood-fill **GI**: light visibly flows around corners over a few frames and converges to a stable bounded field.  A cascade that just recentered runs **+6 extra iterations for 4 frames** (a per-cascade `boost_frames` counter) so the border band the shift pass (below) zeroed out re-converges fast.  On the **boot frame only**, after the first inject + normal propagate, every cascade runs a one-shot **48-iteration warmup burst** so the world starts at converged GI instead of brightening over ~1 s.
  3. **SHIFT** (`shift.comp`, on a cascade recenter only) — when a window's committed origin jumps by an integer cell delta, the two radiance ping-pong textures still hold the *previous* window's field at the *old* origin → stale, spatially misaligned GI for the many frames propagate needs to re-converge (the worst recenter pop).  This pass copies the current radiance (read side = `ping`) into the other ping-pong texture **shifted by `u_shift = new_origin_cell − old_origin_cell`** (per-cell `dst[c] = src[c + u_shift]`, `vec4(0)` for source cells outside the previous window), then `ping` is swapped so the next propagate reads the spatially-aligned field.  rgba16f image bindings (`u_src` read, `u_dst` write), dispatched from `_commit_assembly_result` exactly like inject/propagate.  Cascade 0 still uses 1 cell = 1 voxel binary occupancy; the shift is origin-delta only and never touches geometry.
  4. **FOG_SCATTER / FOG_INTEGRATE** (every frame) — a camera-frustum **froxel** volume (`fog_froxels_x/y/z`, exponential depth slices to `fog_far_m` = 192 m, covering the cascade-1 192 m box so fog never cuts off mid-cascade): weather-driven height fog density, Henyey-Greenstein sun/moon in-scatter shadowed through the cascade occupancy (→ **god rays**), sky/GI ambient scatter; integrated front-to-back into per-slice (accumulated light, transmittance).

  **Fractional occupancy** (geom alpha A ∈ [0,1] = solid sub-cell fraction, not binary): cascade 0 stays effectively binary (1 cell = 1 voxel); coarse cascades may carry partial cells so hollow rooms keep lit interiors. INJECT/PROPAGATE and `terrain.frag` treat the fraction as per-cell **opacity** and as a **convex blend weight**, so binary 0/1 data (all that exists until the volume side ships fractions) is bit-for-bit identical to the old behaviour:
     - *Shadow/visibility marches* (`inject.comp::marchVis`) accumulate transmittance `vis *= (1 − A)` per cell instead of a hard `A>0.5 → stop`. Steps are one cell long (no path-length rescale); early-out at `vis < 0.003`. At A=1 the cell zeroes vis (old hard hit), at A=0 it passes (air) → soft low-LOD shadows instead of black blocks.
     - *First-bounce / emissive leak* (`inject.comp`) scale a neighbour's reflected sun/moon light and emission by its solid fraction A (full wall A=1 reflects as before, air A=0 skipped, half-solid reflects half). Only fully-solid cells (A>0.5) still short-circuit to the emission-only "solid emitter" path.
     - *Propagation* (`propagate.comp`) per neighbour mixes the two pre-existing exact terms by A: `mix(conduct, reflect, A)` where `conduct` = neighbour radiance through its air fraction (old air term) and `reflect` = our own light off its solid, albedo-tinted (old wall term). **Stays contractive**: each per-neighbour contribution is a convex combination of own/neighbour radiance (coefficients `(1−A)`, `A·albedo·bounce`, all in [0,1] with albedo<1, bounce≤1); the `u_decay/6` 6-neighbour average is unchanged, so the spectral radius is still < 1 and the field stays bounded. Fully-solid cells (A>0.5) short-circuit to emission-only; partial cells diffuse, keeping coarse-cascade interiors lit.
     - `fog_scatter.comp` needs no change — it shadows via the already-computed `u_c1_vis` texture, which carries the fractional transmittance from INJECT.
- **Dynamic lights** — `LightSet` (`lights.py`): `PointLight` / `AreaLight` registry with TTL transients (explosion flashes fade and expire), packed to one `float32 (64, 12)` uniform array per change.
- **Surface contract** — `GpuLightingPipeline.bind_surface_inputs(node)` / `update_surface_inputs(node, sky_state)` feed `world/terrain_shader.py`, which samples the cascades (quantised to `light_quant_m` = 0.0625 m → 8×8×8 visible "light pixels" per voxel — the snap grid only; the GI *data* resolution is the cascade-0 cell), reads `u_vis` for celestial shadows, derives AO from occupancy, adds emission, and composites the froxel fog at its own depth.
  - **Smooth cascade handoff** (`terrain.frag::sampleCascades`): the finest-first selection is a soft cross-fade, not a hard `inBox` cliff. Each cascade has a containment weight `boxWeight(uv, fade)` = `smoothstep(0, fade, dist-to-nearest-face)` (component-wise min over the [0,1] cube); the result is built coarsest→finest as `mix(coarser, finer, w_finer)`, starting from the sky-ambient fallback (`u_sky_ambient*0.6`, vis=1, occ=0) so nothing ever pops or goes black at a boundary. Fade bands (fraction of half-extent): **c0 0.14, c1 0.12, c2 0.10** → ~3.4 m / ~11.5 m / ~25.6 m wide at the 24/96/256 m half-extents, each wider than the next-coarser cell so the blend never exposes that tier's texel grid. Well inside c0 (`w0==1`) only c0 is sampled (the cheap common case); bands cost ≤2 cascade taps. The AO probe calls it a second time, so the function is self-contained.
  - **Quantisation LOD**: the light-pixel snap cell is clamped to the screen footprint — `eff = max(u_quant_m, mppW*1.2)` with `mppW = max(fwidth(v_world.{x,y,z}))` — so the snap grid can never go sub-pixel (sub-pixel snapping → distance sparkle as fragments hop light-pixels frame to frame). Up close `u_quant_m` dominates and the chunky pixelated lighting is unchanged.

**CPU backend (`"cpu"`, legacy)** — the Phase-4 v0 baked pipeline, fully retained: `LightGrid` (per-chunk `uint8 (16,16,16)` @ 1 m cells), `SunlightComputer` column pass + 3×3×3 box blur, `make_light_sampler` baking into mesh vertex colours.

`lighting/` is the one non-`world/` package allowed to touch the GPU (ARCHITECTURE §4 rule 4); only `gpu.py` imports panda3d — every other module is headless-testable.

## Public API

Headless symbols re-exported from `fire_engine.lighting`; the GPU pipeline imports explicitly from `fire_engine.lighting.gpu`.

| Symbol | Description |
|---|---|
| `VolumeWindow(cells, cell_m, snap_cells=8, margin_cells=8)` | Camera-centered grid-snapped cascade window; `recenter(camera_pos) -> bool` (mutates origin), `needs_recenter(camera_pos) -> bool` (non-mutating drift test, used by the async path), `world_origin_m`, `size_m`.  Per-window `margin_cells` is the recenter hysteresis: cascades 0/1 use the default 8 cells; the coarse **cascade 2 uses 16 cells (= 128 m)** so its ~33k-chunk gather recenters half as often and stops permanently lagging at flight speed (`gpu.py` passes it at construction). |
| `assemble_geometry(window, chunks, palette, chunk_size, voxel_size, cache=None) -> GeometryVolume` | Slice chunk materials into `albedo_occ`/`emission` `uint8 (N,N,N,4)` blocks. Coarse cascades downsample albedo by **max material id** and occupancy A by **solid sub-voxel fraction** (`round(255 × solid / k³)`) per block — fine path (`cell_m == voxel_size`) is byte-identical binary occupancy. `chunks` values may be chunk objects (`.materials`) or bare ndarray snapshots (async path). Optional `cache` (`ChunkBlockCache`) reuses per-chunk downsampled mini-blocks across reassemblies — output is byte-identical with/without it. |
| `ChunkBlockCache(max_entries=4096)` | Thread-safe LRU cache of per-chunk downsampled `(material_id, solid_count)` mini-blocks keyed by `(chunk coord, cell_m)`, so a coarse-cascade recenter copies blocks instead of re-downsampling tens of thousands of chunk arrays. `get/put/invalidate(coord)/clear()`, `len()`. Blocks are palette-**independent** (palette applied after the cache); assumes one terrain geometry per run; LRU-bounded (default 4096 entries ≈ single-digit MB). Guarded by an internal `threading.Lock` so the assembly worker reads/populates while the main thread invalidates. |
| `window_chunk_span(origin_cell, cells, cell_m, chunk_size, voxel_size) -> list[coord]` | Chunk coords a window at `origin_cell` intersects — lets the async path snapshot exactly the chunks a reassembly reads. |
| `pack_volume(arr) -> bytes` | Numpy transpose + RGBA→BGRA + contiguous pack of a `(N,N,N,4)` block to Panda3D 3-D-texture RAM bytes (the off-thread half of an upload). |
| `AssemblyJob` / `AssemblyResult` / `CascadeAssemblyWorker(cache_max_entries=4096)` / `assemble_packed(job, cache=None)` (`assembly_worker.py`, headless) | Background cascade-volume assembly: `worker.start()/submit(job)/drain_results()/pending()/stop()`. Worker output is byte-identical to a synchronous `assemble_geometry` + `pack_volume`. The worker owns a `block_cache: ChunkBlockCache` (reused across reassemblies); the main thread calls `worker.invalidate_chunk(coord)` on terrain edits and `worker.clear_cache()` on reload. |
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
| `.update(camera_pos, sky_state, dt)` | Per-frame driver (App frame task step 6): schedule async reassembly → drain+upload finished volumes (**shift radiance on any recenter so converged GI follows the window**) → inject (dirty cascades only) → propagate (**+6 iters/4 frames per just-recentered cascade; 48-iter one-shot warmup on the boot frame**) → fog.  Newly-streamed chunks that intersect **cascade 0** trigger an *immediate* c0 reassembly (its ~27-chunk gather is cheap); cascades 1/2 keep the 0.25 s `_LOAD_REASSEMBLE_INTERVAL_S` batch.  Terrain edits also call `worker.invalidate_chunk(coord)` so the block cache re-downsamples the edited chunk. |
| `.shutdown()` | Stop the background `CascadeAssemblyWorker`. Call once on app exit (`main()` does, in a `finally`). Idempotent; no-op when unthreaded. |
| `.bind_surface_inputs(node)` / `.update_surface_inputs(node, sky_state)` | Static / per-frame shader-input contract for `world/terrain_shader.py`. |
| `.lights: LightSet` | Public dynamic-light registry (demo: explosion flash, L-key torches). |

`fire_engine.lighting.glsl` — `INJECT_COMPUTE`, `PROPAGATE_COMPUTE`, `SHIFT_COMPUTE` (radiance recenter shift), `FOG_SCATTER_COMPUTE`, `FOG_INTEGRATE_COMPUTE`, `MAX_LIGHTS = 64` (plain strings, headless-importable).

## Imports Allowed

- `numpy`, stdlib, `fire_engine.core` everywhere; `fire_engine.procedural` (palette derivation — foundation layer, callable from anywhere); `fire_engine.terrain` (downward).
- `panda3d` **only in `gpu.py`** — keeps the rest of the package in the headless suite.
- No imports from `world/` or higher.  `world/terrain_shader.py` imports lighting, never the reverse.

## Events

### Subscribed
| Event | Subscriber | Action |
|---|---|---|
| `TerrainEditedEvent` | `SunlightComputer` (CPU) | Recompute affected columns; mark chunks dirty for remesh. |
| `TerrainEditedEvent` | `GpuLightingPipeline` (GPU) | `worker.invalidate_chunk(coord)` for each edited chunk (so the block cache re-downsamples it), then queue the coords; the intersecting cascades reassemble + re-inject **synchronously the same frame** (`_apply_edits_sync`) so the new crater lights immediately — an async (1–2 frame) reassembly leaves the stale occupancy marking the crater solid/shadowed, so it renders black then pops to lit. Edits are discrete events, so the synchronous gather is affordable; cascades with a job already in flight fall back to the batched `_pending_coords` path. |
| `ChunkLoadedEvent` | `SunlightComputer` (CPU) | Recompute the chunk's column. |
| `ChunkLoadedEvent` | `GpuLightingPipeline` (GPU) | Queue coord.  A chunk that intersects **cascade 0** reassembles it *immediately* (cheap ~27-chunk gather — kills the 0.25 s "newly-loaded near terrain renders unshadowed then pops" latency); cascades 1/2 use the batched reassembly (≥ 0.25 s apart) for cascades the chunk actually intersects — streaming-frontier loads still rarely touch cascade 0's 48 m box. |

### Published
None.

## Units & Invariants

- World meters, Z-up.  Cascade texel `(i,j,k)` covers `[(origin_cell + (i,j,k))·cell_m, +cell_m)`; `origin_cell` snaps to `snap_cells` so consecutive windows align on the cell grid (no sub-cell crawl).
- Volume arrays are indexed `[x, y, z]`; GPU upload transposes to Panda3D's page-major `(z, y, x)` and reorders RGBA→BGRA (`_upload_volume`).
- `albedo_occ`: RGB = linear albedo ×255 (selected by **max material id** per block — grass skin wins over dirt for bounce colour). **A = solid sub-voxel fraction ×255** (`round(255 × solid_voxels / k³)`, where `k = cell_m / voxel_size`). At cascade 0 (`cell_m == voxel_size`, `k == 1`) this is exactly 255 (solid) / 0 (air) — byte-identical to the old binary flag. At the coarse cascades it is partial, so a hollow room reads A == 0 in its air interior and only its walls read partly-solid (fixes the Cornell-room "black hole" where any-solid downsample collapsed a hollow box to fully solid).
- `emission`: RGB = linear HDR emission ÷ `EMISSION_SCALE` ×255.
- Radiance/visibility volumes are `rgba16f`, GPU-resident only (never read back).
- Propagation is contractive (`decay < 1`, albedo < 1 ⇒ spectral radius < 1): the radiance field is always bounded by the injected sources; reach ≈ `1/(1−decay)` cells (decay = `exp(-cell_m / 4 m)` C0, `exp(-cell_m / 10 m)` C1).
- Light values are **linear HDR RGB** in SkyState units: noon sun ≈ 3.2, skylight ≈ 0.2–0.7, torch intensity ≈ 8.  The terrain/sky shaders apply `light_exposure` (auto-exposure adapted).  With HDR post-processing on (`[graphics] gfx_post_process`, see `docs/systems/world.md`), the surface shaders still multiply in `u_exposure` but skip the tonemap — they emit linear HDR and the single ACES tonemap + gamma happens in the post-process composite (so bloom sees the exposed HDR signal).  With post off they tonemap internally as before.
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

12. **The worker OWNS the block cache** (`worker.block_cache: ChunkBlockCache`) and threads it into every async job.  The synchronous boot/edit paths (`_assemble_and_upload_sync`) pass that same cache so the boot downsample warms it (the cache's internal lock makes the cross-thread sharing safe).  **A terrain edit MUST call `worker.invalidate_chunk(coord)` before reassembly** — `_on_terrain_edited` does this for every edited coord; skip it and a coarse cascade copies the stale pre-edit mini-block out of the cache and the crater never relights.

13. **Radiance shift on recenter** (`_commit_assembly_result` → `_shift_radiance`): when a commit moves a cascade's origin, the SHIFT pass realigns the converged radiance to the new window (see GPU pass 3) and sets `boost_frames = 4`.  Without it a recenter pops to stale, misaligned GI for ~`1/light_prop_iters` worth of frames.  The shift reads `casc.radiance[ping]` and writes the other texture, then flips `ping`; it runs on the main thread inside the drain step, so `_base.win`/`graphicsEngine` are valid (the boot frame and unthreaded tooling path also reach it).
