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

## Imports Allowed

- `fire_engine.core` (logger) and `fire_engine.core._impl.worker_pool` (`WorkerPool` base).
- `fire_engine.world.terrain.chunk` (`Chunk`), `fire_engine.world.terrain.meshing` (`MeshArrays`, `build_mesh`), `fire_engine.world.terrain.surface_nets` (`build_mesh_faceted`).
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
