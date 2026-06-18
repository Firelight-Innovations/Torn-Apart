# terrain.lod — System Doc
keywords: lod, terrain lod, threaded meshing, off-main-thread, worker pool, TerrainLodPool, LodJob, LodResult, build_lod_mesh, mesh job, mesh result, snapshot, immutable snapshot, seq, staleness, stale, monotonic sequence, mesh_style, faceted, blocky, neighbors, neighbor_materials, neighbor_solids, WorkerPool, daemon thread, drain_results, submit, _on_error, sentinel, empty mesh, determinism, Hard Rule 12, main thread orchestration, MeshArrays, chunk meshing, parallel mesh

> One doc per code package; filename matches the package exactly (`docs/systems/world.terrain.lod.md` ↔ `fire_engine/world/terrain/lod/`).

## Role

`terrain/lod/` is the **pure, headless core of the threaded terrain mesher** (Layer 2 — Structure). Hard Rule 12 forbids the main thread from doing heavy work: chunk meshing (surface-nets / culled-face array builds) must run on a worker pool, not the main thread. This sub-package owns the independently-testable pieces of that pipeline:

- **The hand-off types** — `LodJob` (an immutable snapshot: a *copy* of one chunk's `materials` plus its neighbour materials/solidity, tagged with a monotonic `seq`) and `LodResult` (the produced `MeshArrays` + originating `coord` + `seq`).
- **The pure transform** — `build_lod_mesh(job)`: reconstructs a `Chunk` from the snapshot and runs the configured mesher (`build_mesh_faceted` or `build_mesh`). Byte-identical to `ChunkManager.mesh_chunk` for the same chunk + neighbour state. Runs on the worker thread, and is synchronously callable for tests.
- **The pool** — `TerrainLodPool`, a `WorkerPool[LodJob, LodResult]` subclass fanning jobs across N daemon threads.

`terrain.lod/` deliberately does NOT: import panda3d, gather the neighbour arrays (that snapshot logic stays in `ChunkManager._neighbor_materials` / `_neighbor_solids`), assign `seq` (the caller does), bake per-face light (the threaded path always passes `light_sampler=None`), or touch the scene graph. The `ChunkManager`/`app_terrain` integration that copies the arrays, submits jobs, drains results, and uploads geometry lives outside this package.

## Public API

All symbols below are re-exported from `fire_engine.world.terrain.lod` (`__init__.py`). The two frozen hand-off dataclasses live in `terrain/lod/types.py`; `build_lod_mesh` in `terrain/lod/job.py`; `TerrainLodPool` in `terrain/lod/pool.py`.

| Symbol | Description |
|---|---|
| `LodJob(coord, materials, neighbors, chunk_size, voxel_size, shade_strength, mesh_style, seq)` | Frozen immutable hand-off snapshot for one chunk's threaded mesh build. |
| `LodJob.materials` | `uint8 (32,32,32)` snapshot COPY of the chunk's materials, `[x, y, z]`. |
| `LodJob.neighbors` | `dict[offset, uint8 array \| bool array \| WORLD_FLOOR_SOLID]` — 26 offsets (faceted) or 6 face dirs (blocky), each a copy; forwarded verbatim to the mesher. |
| `LodJob.mesh_style` | `"faceted"` (default mesher) or `"blocky"` — selects which mesher `build_lod_mesh` calls. |
| `LodJob.seq` | Monotonic submit sequence for staleness discipline; carried unchanged into `LodResult`. |
| `LodResult(coord, mesh, seq)` | Frozen result: the produced `MeshArrays`, the originating `coord`, and `seq`. |
| `build_lod_mesh(job) -> LodResult` | PURE transform: rebuild `Chunk`, run the configured mesher (`light_sampler=None`), wrap as `LodResult`. |
| `TerrainLodPool(n_workers)` | `WorkerPool[LodJob, LodResult]` subclass; `start`/`submit`/`drain_results`/`pending`/`stop`. `_process` runs `build_lod_mesh`; `_on_error` posts an empty-mesh sentinel. |
| `LodStreamer(chunk_manager, pool, config)` | Async streaming driver (`streamer.py`). `stream_frame(camera_pos)` drains finished meshes into `chunk_manager.pending_meshes` and submits a bounded batch of fresh jobs (dirty-first, then nearest missing). Owns the per-coord `seq` staleness map. Off-thread counterpart of `ChunkManager.stream_frame`; no panda3d. |

## Streaming (`LodStreamer`)

`LodStreamer` (in `terrain/lod/streamer.py`) is the integration that turns the pure pool into an async replacement for `ChunkManager.stream_frame`. It is the ONE place that owns the snapshot/`seq` plumbing the rest of the package deliberately omits. Each `stream_frame(camera_pos)`:

1. **Drains** `pool.drain_results()`: a `LodResult` is kept only if `coord in chunk_manager.chunks` **and** `result.seq == self._node_seq[coord]` (the latest submitted seq for that coord). Kept meshes go into `chunk_manager.pending_meshes`; stale/unloaded results are dropped. Newest-wins, order-independent.
2. Resets `unloaded_this_frame`, computes `desired_set` + `camera_chunk`.
3. **Submits** up to `config.lod_submit_per_frame` jobs: **dirty loaded chunks first** (nearest-first), then **nearest missing desired** chunks (`get_or_create` + publish `ChunkLoadedEvent`). For each, it builds a job via `_make_job` (which bumps `seq`, records `_node_seq[coord]`, copies `materials` and copies each non-`str` neighbour array from `ChunkManager._neighbor_materials`/`_neighbor_solids`), `pool.submit(job)`, and clears `chunk.dirty` (so it isn't resubmitted next frame — a later brush re-dirties → fresh `seq` → newest wins).
4. Calls `ChunkManager._unload_far(camera_chunk)` (shared with the sync path), then prunes `_node_seq` of any coord no longer loaded.

The synchronous `ChunkManager.stream_frame` path stays intact (backward-compat for the headless suite + the baked-light/editor case). `app_terrain.stream_and_upload_terrain` branches to `LodStreamer` when `app.lod_streamer is not None`, and budgets the upload drain to `config.lod_max_uploads_per_frame` nearest-first.

## Coarse ranks (P2 core)

The **coarse LOD** layer downsamples distant terrain into bigger, cheaper tiles. This is the pure, headless core (P2 step 1); the off-thread streamer + render wiring land in P2 step 2.

A **coarse node** at rank `L` covers a cube of `k³` native `L0` chunks (`k = 1 << L`: 2/4/8 for L1/L2/L3) and is meshed as a single 32³ block of `base_voxel_size · k` metre voxels. Because the meshers read one cube dimension (`n = chunk.materials.shape[0]`), the node is snapped on **all three axes** (`node = chunk >> L`) — a deliberate cube interpretation (Z is snapped like X/Y), which is what lets the unchanged meshers run on the shim.

| Symbol (module) | Description |
|---|---|
| `LodNode(rank, nx, ny, nz)` (`node.py`) | Frozen node key + addressing. `LodNode.for_chunk(coord, rank)` snaps `chunk >> L`; `voxel_size(base)`, `world_origin(base_chunk_m)`, `chunk_origin()`, `covered_chunks()`. |
| `rank_factor(rank) -> int` (`node.py`) | `1 << rank` (voxels merged per axis). |
| `downsample_block(tile, rank, mode="any") -> uint8 (32,32,Z)` (`downsample.py`) | Whole-array reduce of a tiled `(32·k, 32·k, Z·k)` block: solidity via ANY (preserves thin walls / horizon silhouette), material via max-id (`"any"`, grass 2 beats dirt 1) or majority (`"majority"`). 6-D reshape, no per-voxel loop. |
| `_CoarseChunk(node, materials, base_voxel_size=, chunk_size=)` (`coarse_chunk.py`) | Read-only `Chunk` shim exposing exactly `materials` / `_voxel_size` / `world_origin` / `is_solid_mask()` so `build_mesh_faceted` / `build_mesh` run unchanged. Wraps a real `Chunk` at the node coord with `voxel_size = base · 2**L`. Not `Saveable`, never in `ChunkManager.chunks`. |
| `desired_node_set(camera_chunk, config, z_band, max_rank, *, near_radius_chunks=, far_radius_chunks=, chunk_meters=) -> NodePlan` (`desired.py`) | Vectorised replacement for the `O(r³)` `desired_set` loop: one `np.meshgrid` over the XY window × Z band, Chebyshev distance, per-cell rank via `np.searchsorted` against the band radii, `cell >> L` snap + `np.unique`. |
| `NodePlan(near_chunks, coarse_nodes)` (`desired.py`) | Frozen partition: `near_chunks: set[(cx,cy,cz)]` (rank 0) + `coarse_nodes: dict[L, set[(L,nx,ny,nz)]]`. Near and every coarse set are disjoint (hard band cuts — no double-draw). |

**Reduction policy.** Solidity is always ANY (thin 1-voxel walls survive any factor). Material id is max-id (`"any"` mode, grass beats dirt) or the most-common nonzero id (`"majority"`). Both are whole-array (Hard Rule 4); `"majority"` loops only over the small distinct-id alphabet, never over voxels.

**Regression invariant.** With `max_rank=0`, `near_radius_chunks = view_distance_chunks = 6`, `z_band = (-2, 4)`, `desired_node_set(...).near_chunks` equals the legacy `ChunkManager.desired_set` output **exactly** — the square (Chebyshev) XY radius-6 × Z[-2..4] set of `13·13·7 = 1183` chunks. This pins the meshgrid/searchsorted refactor to the existing loop so the P1 near path is unchanged when coarse ranks are off.

**Band cuts (P2).** Each cell is in exactly one rank (near OR one coarse node), so a coarse node and its `L0` chunks are mutually exclusive — no double-draw. Cross-rank seam cracks and crossfade are P3.

**Editable radius is authoritative (P2).** When coarse ranks are active, `near_chunks` still contains the **full** Chebyshev radius-`near_r` square × Z band — the editable/lit/saved (rank-0) footprint never shrinks when coarse turns on. `desired_node_set` pre-claims that square before the coarsest→finest sweep, so a boundary coarse block (whose nearest-corner Chebyshev distance is exactly `near_r`) is rejected rather than swallowing the `±near_r` edge ring. Without this, `enter[1] == near_r` with `keep = nearest >= enter[L]` silently cut the brush/save/light radius one chunk ring (~16 m) per side once coarse was on. Coarse fills strictly beyond `near_r`; the partition stays complete and disjoint.

**Config keys (P2):** `lod_max_rank` (3), `lod_near_radius_chunks` (6), `lod_far_radius_chunks` (32), `lod_band_l1_m` (32.0), `lod_band_l2_m` (96.0), `lod_band_l3_m` (192.0), `lod_downsample_mode` (`"any"`), `lod_coarse_submit_per_frame` (4), `lod_coarse_uploads_per_frame` (4). See `docs/systems/core.md`.

## Coarse ranks (P2 render)

The render/streaming half (P2 step 2): the coarse nodes the planner describes are now **meshed off-thread, uploaded, and drawn at distance** — the distant horizon — through the same `TerrainLodPool` + geometry bridge as the near (`L0`) path. Hard band cuts (a coarse node and the `L0` chunks it covers never both draw; pop at the boundary — crossfade is P3).

| Symbol (module) | Description |
|---|---|
| `LodJob.rank` / `LodResult.rank` (`types.py`) | Rank `L` tag (default `0` = native `L0`, byte-identical to P1). `rank > 0` = a coarse node: `coord` is the node coord `(nx,ny,nz)`, `materials` is the already-downsampled 32³ block, `voxel_size = base · 2**L`, `neighbors` empty (open coarse borders in P2). |
| `build_lod_mesh(job)` (`job.py`) | Rank-aware: `rank == 0` rebuilds a real `Chunk` (unchanged P1 path); `rank > 0` wraps the downsampled block in a `_CoarseChunk` and runs the **same** mesher. `TerrainLodPool._process` is unchanged (it just calls `build_lod_mesh`). |
| `assemble_coarse_materials(node, materials_for, *, chunk_size=, mode=) -> uint8 (32,32,32)` (`coarse_assembly.py`) | Gather a node's `k³` `L0` chunk-materials (via a provider) into a `(32·k, 32·k, 32·k)` tile and `downsample_block` it. The only iteration is the `≤8³` chunk gather (not per-voxel; Hard Rule 4). |
| `CoarseLodStreamer(chunk_manager, pool, config)` (`coarse_streamer.py`) | Async coarse driver. `stream_frame(camera_pos)` drains finished coarse meshes into `chunk_manager.pending_coarse_meshes`, plans ranks `1..lod_max_rank` via `desired_node_set`, records nodes that left the desired set in `chunk_manager.unloaded_coarse_this_frame` (the hard band cut), and submits up to `lod_coarse_submit_per_frame` fresh jobs **coarsest-far-first, nearest-within-rank-first**. Uses a SEPARATE `TerrainLodPool` so heavy coarse jobs never steal near results. No panda3d. |

**Channels on `ChunkManager` (P2).** `pending_coarse_meshes: dict[(rank,nx,ny,nz), MeshArrays]` and `unloaded_coarse_this_frame: list[(rank,nx,ny,nz)]` — written by `CoarseLodStreamer`, drained/detached by the render layer. Parallel to the near `pending_meshes` / `unloaded_this_frame`. Coarse-node keys are 4-tuples (the rank is part of the key); near coords are 3-tuples.

**Staleness (per node).** `CoarseLodStreamer._node_seq[(rank,nx,ny,nz)]` is the latest submitted `seq` for that node; a drained coarse result is kept only if its node is still tracked AND `result.seq == _node_seq[key]`, then the entry is cleared (the mesh now lives in `pending_coarse_meshes`). A node that leaves the desired set is retired the same frame (newest-wins, order-independent — same discipline as `LodStreamer`).

**Determinism.** A coarse node is byte-identical regardless of `L0` load order: the materials provider returns a loaded chunk's live (possibly edited) materials when present, else the deterministic `generate_chunk` baseline (Hard Rule 2 — no RNG in the coarse path itself).

**Render wiring (`app_terrain.py`).** `stream_and_upload_terrain` calls `coarse_streamer.stream_frame(pos)` after the near streamer when wired, then `_upload_coarse_terrain` drains `pending_coarse_meshes` nearest-first up to `lod_coarse_uploads_per_frame`, uploading each via the same `to_geom_node` (coarse meshes carry `face_materials`, so the grass/dirt texture split works unchanged) under `terrain_root`, detaching stale/retired coarse `NodePath`s from `App._coarse_nodes`. `main.py` builds the coarse pool + `CoarseLodStreamer` only when `lod_max_rank > 0`, and stops the pool at shutdown.

## Imports Allowed

- `fire_engine.core` (logger, `ChunkLoadedEvent`), `fire_engine.core._impl.worker_pool` (`WorkerPool` base) and `fire_engine.core.math3d` (`Vec3`, used by `node.py`/`coarse_chunk.py` for world-origin math; `Config` under `TYPE_CHECKING` only in `desired.py`).
- `fire_engine.world.terrain.chunk` (`Chunk`), `fire_engine.world.terrain.meshing` (`MeshArrays`, `build_mesh`), `fire_engine.world.terrain.surface_nets` (`build_mesh_faceted`), `fire_engine.world.terrain.generation` (`generate_chunk` — the coarse path's loaded-or-generated materials provider).
- `LodStreamer` / `CoarseLodStreamer` additionally reference `ChunkManager` and `Config` (under `TYPE_CHECKING` only, to avoid an import cycle) and read the `lod_*` config keys.
- **Never** panda3d (Hard Rule 1). No imports from `render/`, `lighting/`, or `simulation/`.

## Events

Published: none. Subscribed: none. The pool is a pure producer/consumer across `queue.Queue`s; state-change notifications stay with the integrating `ChunkManager` (`ChunkLoadedEvent` / `ChunkUnloadedEvent`).

## Units & Invariants

- `materials`: `uint8 (chunk_size,)*3` indexed `[x, y, z]`; 0 = air, ≥1 = solid material id. World space in **meters**; voxel = `voxel_size` (0.5 m); chunk edge = `chunk_size` (32 voxels = 16 m).
- **Immutable-snapshot invariant (Hard Rule 12):** every array a `LodJob` carries is a private copy the caller made before submitting. The worker thread only *reads* them, so the live chunk can be edited on the main thread concurrently with no data race. Never mutate a job's arrays.
- **Determinism:** `build_lod_mesh` is a pure function of the job — same job → byte-identical `MeshArrays` (`positions/normals/uvs/colors/indices/face_materials`). No RNG. It reproduces `ChunkManager.mesh_chunk` exactly for the same chunk + neighbour state.
- **seq staleness:** `seq` is monotonic per the caller's submit order and round-trips unchanged through the result. Because drain order is NOT submit order and jobs run concurrently, the consumer matches results by `coord` and discards a result whose `seq` is older than the latest submitted for that coord.
- **No baked light:** the threaded path always passes `light_sampler=None` (the live GPU game applies sun light on the GPU). Baked-light callers stay on the synchronous `mesh_chunk` path in this phase.
- **Failure sentinel:** a raised `_process` posts `LodResult(coord, <empty MeshArrays>, seq)` so a failed job never wedges the consumer's per-coord seq tracking (mirrors `VenturiWorker` / `CascadeAssemblyWorker`).

## Examples

```python
import numpy as np
from fire_engine.core import EventBus, load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.world.terrain.chunk_manager import ChunkManager
from fire_engine.world.terrain.lod import LodJob, TerrainLodPool, build_lod_mesh

set_world_seed(1337)
config = load_config()
cm = ChunkManager(config, EventBus())
coord = (0, 0, 0)
chunk = cm.get_or_create(coord)

# Caller snapshots the chunk + neighbours (copies) and tags a seq.
neighbors = {k: (v if isinstance(v, str) else v.copy())
             for k, v in cm._neighbor_materials(coord).items()}
job = LodJob(
    coord=coord,
    materials=chunk.materials.copy(),
    neighbors=neighbors,
    chunk_size=int(config.chunk_size),
    voxel_size=float(config.voxel_size),
    shade_strength=float(config.facet_shade_strength),
    mesh_style="faceted",
    seq=1,
)

# Synchronous (e.g. boot / first frame / tests):
result = build_lod_mesh(job)             # == cm.mesh_chunk(coord) byte-for-byte

# Threaded (steady state):
pool = TerrainLodPool(n_workers=2)
pool.start()
pool.submit(job)
# ... later frames ...
for res in pool.drain_results():         # drain order != submit order
    ...                                  # upload res.mesh for res.coord if res.seq is current
pool.stop()
```

## Gotchas

- **Copy before you submit.** `LodJob.materials` and the `neighbors` arrays must be copies; if you pass the live `chunk.materials` and the main thread edits it while a worker reads it, you get a torn read and non-deterministic output. The caller (`ChunkManager`) owns the copy.
- **Drain order is not submit order.** With `n_workers > 1`, results arrive in completion order. Always key results by `coord`; never assume FIFO.
- **`seq` lives here for the consumer's benefit, not the worker's.** The worker echoes it untouched. Staleness logic (drop old-`seq` results) belongs in the integrating consumer.
- **`mesh_style` is read from the job, not from `Config`.** The caller snapshots `config.mesh_style` into the job at submit time, so a mid-stream config flip can't desync an in-flight job.
- **Neighbour-gathering is intentionally absent.** Don't duplicate `_neighbor_materials` / `_neighbor_solids` here — that logic stays in `chunk_manager.py`; this package only consumes the dict the caller hands it.
- **A bad job still returns a result.** `_on_error` posts an empty-mesh sentinel. The consumer should treat an empty mesh as "remove/skip geometry for this coord", not as a missing result.
