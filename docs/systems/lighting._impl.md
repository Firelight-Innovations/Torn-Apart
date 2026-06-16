# lighting._impl — Private Implementation Sub-package
keywords: lighting impl, _impl, cascade, _Cascade, make_volume_texture, ChunkBlockCache, OccluderSet, MAX_OCCLUDERS, GeometryOccupancyProvider, AssemblyJob, AssemblyResult, GeometryVolume, PointLight, AreaLight, SpotLight, inject_and_gather, dispatch_fog, setup_fog, upload_volume, assemble_and_upload_sync, apply_edits_sync, schedule_assembly, submit_assembly, drain_assembly_results, commit_assembly_result, shift_radiance, bind_surface_inputs, update_surface_inputs, sky_inputs, protocols, cache, occluder_set, types, gpu_cascade, gpu_inject_gather, gpu_fog, gpu_cascade_assembly, gpu_surface

> Private internals sub-package created during standards remediation to satisfy the
> ≤10-module structural limit while keeping all public import paths in the parent
> `fire_engine.lighting` package unchanged.  Do NOT import directly from here —
> all public symbols are re-exported from their originating parent modules.

## Role

`fire_engine.lighting._impl` holds the overflow implementation modules extracted
from the parent `fire_engine.lighting` package to satisfy the one-public-class-per-
module and ≤10-modules-per-folder structural rules.  Every module here is a
private extraction from one parent module that had grown too large; the re-exports
in the parent modules preserve all historical import paths.

The sub-package spans five concern areas:

- **Protocols** (`protocols.py`) — `GeometryOccupancyProvider`, the structural hook
  that lets non-terrain geometry (buildings, future props) splat solids into the
  lighting cascades without creating a lighting↔buildings import.
- **Cache** (`cache.py`) — `ChunkBlockCache`, the thread-safe LRU cache of per-chunk
  downsampled geometry mini-blocks used by the async cascade-assembly worker.
- **Occluder set** (`occluder_set.py`) — `OccluderSet` + `MAX_OCCLUDERS`, the
  registry of dynamic shadow-caster AABBs uploaded to the GPU each frame.
- **Types** (`types.py`) — frozen/mutable support dataclasses
  (`AssemblyJob`, `AssemblyResult`, `GeometryVolume`, `PointLight`, `AreaLight`,
  `SpotLight`) grouped here to satisfy the one-public-class rule.
- **GPU dispatch helpers** (`gpu_cascade.py`, `gpu_inject_gather.py`, `gpu_fog.py`,
  `gpu_cascade_assembly.py`, `gpu_surface.py`) — panda3d-dependent functions and the
  `_Cascade` private class extracted from `gpu.py` to keep it under 500 lines.

This sub-package does NOT define any new public API.  It is not headless-clean (the
`gpu_*` modules import panda3d), and it deliberately avoids being listed in the
package's `__init__` re-exports — treat it as a build artifact, not a stable API.

## Public API

All exports are re-exported from their originating parent modules:

From `fire_engine.lighting.volume` (extracted into `cache.py`):
- `ChunkBlockCache` — thread-safe LRU cache, keyed `(chunk_coord, cell_m)`.

From `fire_engine.lighting.lights` (extracted into `occluder_set.py`):
- `OccluderSet` — dynamic AABB shadow-caster registry; `set_boxes`, `pack`, `count`.
- `MAX_OCCLUDERS = 16` — GPU uniform array length cap.

From `fire_engine.lighting` (extracted into `protocols.py`):
- `GeometryOccupancyProvider` — `@runtime_checkable` Protocol;
  `rasterize_occupancy(origin_cell, cells, cell_m, albedo_occ, emission)`.

From `fire_engine.lighting` (extracted into `types.py`):
- `AssemblyJob` — frozen dataclass; one cascade-volume reassembly request.
- `AssemblyResult` — frozen dataclass; finished packed cascade volume.
- `GeometryVolume` — dataclass; assembled `albedo_occ`/`emission` arrays.
- `PointLight` — dataclass; omni punctual light (position, color, intensity, radius, ttl_s).
- `AreaLight` — dataclass; axis-aligned emissive box light.
- `SpotLight` — dataclass; cone-restricted punctual light.

From `fire_engine.lighting.gpu` (extracted into `gpu_cascade.py`):
- `make_volume_texture(name, cells, *, hdr, linear) -> Texture` — allocate a cascade
  3-D texture (rgba16f for GPU-written radiance, rgba8 for CPU-uploaded geometry).
- `_Cascade` — private class; one radiance cascade's window, textures, and compute
  node paths.  Not a public API; accessed only by `gpu.py`.

From `fire_engine.lighting.gpu` (extracted into `gpu_inject_gather.py`):
- `inject_and_gather(pipeline, sun, packed, count, box_min, box_max, n_boxes,
  engine, gsg)` — run injection + GI gather for every dirty cascade.

From `fire_engine.lighting.gpu` (extracted into `gpu_fog.py`):
- `dispatch_fog(pipeline, camera_pos, sun, sky_state, engine, gsg)` — fill/integrate
  the froxel scatter volume for this frame.
- `setup_fog(pipeline)` — allocate froxel textures and configure compute node paths;
  called once from `GpuLightingPipeline.__init__`.

From `fire_engine.lighting.gpu` (extracted into `gpu_cascade_assembly.py`):
- `upload_volume(tex, arr)` — upload a `uint8 (N,N,N,4)` block to a 3-D texture.
- `assemble_and_upload_sync(pipeline, casc)` — synchronous boot-frame gather + upload.
- `apply_edits_sync(pipeline)` — same-frame synchronous reassembly for brush edits.
- `schedule_assembly(pipeline, camera_pos)` — submit async jobs for moved/stale cascades.
- `submit_assembly(pipeline, casc, origin_cell)` — snapshot chunks + enqueue a job.
- `drain_assembly_results(pipeline)` — upload finished volumes (main thread).
- `commit_assembly_result(pipeline, res)` — upload one finished volume + advance origin.
- `shift_radiance(pipeline, casc, old_origin, new_origin)` — SHIFT pass on recenter.

From `fire_engine.lighting.gpu` (extracted into `gpu_surface.py`):
- `bind_surface_inputs(pipeline, node)` — bind static lighting samplers onto a render
  `NodePath` at boot.
- `update_surface_inputs(pipeline, node, sky_state)` — refresh per-frame uniforms.
- `sky_inputs(sky_state) -> tuple` — extract `(sun_dir, sun_rad, moon_dir, moon_rad,
  sky_amb)` from a `SkyState`, with graceful fallbacks for `None`.

## Imports Allowed

Same rules as the parent `fire_engine.lighting` package:

- `numpy`, stdlib.
- `fire_engine.core.*`, `fire_engine.procedural` (read-only).
- `fire_engine.lighting.*` (sibling imports within the package are allowed).
- `panda3d.*` — **only in `gpu_*.py` modules** (same restriction as `gpu.py`; these
  are panda3d dispatch helpers, intentionally non-headless).
- No imports from `fire_engine.world`, `fire_engine.simulation`, or higher.

## Events

None — this sub-package contains no event publishers or subscribers.  Event wiring
lives in the parent modules (`gpu.py` subscribes to `TerrainEditedEvent` /
`ChunkLoadedEvent`; the helpers here are called from those handlers).

## Units & Invariants

Same as parent package `fire_engine.lighting`:

- World coordinates in **meters**, Z-up.
- Cascade texels indexed `[x, y, z]`; GPU upload transposes to Panda3D page-major
  `(z, y, x)` with RGBA→BGRA reorder (`upload_volume` / `pack_volume`).
- `albedo_occ`: `uint8 (N, N, N, 4)` — RGB linear albedo ×255, A = solid sub-voxel
  fraction ×255 (binary 0/255 at cascade 0 where `cell_m == voxel_size`).
- `emission`: `uint8 (N, N, N, 4)` — RGB linear HDR emission ÷ `EMISSION_SCALE` ×255.
- `OccluderSet` boxes are in world meters; GPU arrays are `float32 (16, 3)`, zero-padded
  past `count`.
- `ChunkBlockCache` entries are `(material_id, solid_count)` mini-blocks keyed
  `(chunk_coord, cell_m)`; blocks are palette-independent (palette applied after).
- All `gpu_cascade_assembly` functions treat `casc.window.origin_cell` as the
  **committed** origin; it only advances in `commit_assembly_result` — never before
  the matching volume lands on the GPU.

## Examples

    # Do not import directly from _impl — use the parent module's public surface:
    from fire_engine.lighting import ChunkBlockCache, GeometryOccupancyProvider
    from fire_engine.lighting.lights import OccluderSet, MAX_OCCLUDERS
    from fire_engine.lighting.volume import ChunkBlockCache

    # The GPU helpers are called only from fire_engine.lighting.gpu:
    # (shown for documentation only — do not call from outside lighting/)
    from fire_engine.lighting._impl.gpu_cascade_assembly import (
        schedule_assembly, drain_assembly_results,
    )

## Gotchas

- **Do not import from `_impl` outside `fire_engine.lighting`.**  All symbols are
  re-exported through the parent modules; internal paths may be renamed at any time
  to satisfy structural constraints.
- **`gpu_*.py` import panda3d** — they are NOT headless-safe and are excluded from
  the headless test suite by the same import rule that governs `gpu.py`.
- **`_Cascade` is private by design** — it is not in any `__all__` under the public
  lighting surface and carries no stability guarantee; callers outside `gpu.py`
  access cascades only through `GpuLightingPipeline.cascades`.
- **`ChunkBlockCache` entries are read-only views** (`WRITEABLE` cleared by `put`);
  callers must not mutate returned arrays.
- **Thread safety**: `ChunkBlockCache` is guarded by an internal `threading.Lock`
  (safe for worker-thread reads + main-thread invalidations);  `OccluderSet` is
  main-thread-only (called from `GpuLightingPipeline.update`).
