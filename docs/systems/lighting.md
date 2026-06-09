# lighting — System Doc
keywords: sun, sunlight, light grid, occupancy, column, ambient, penumbra, blur, vertex color, shadow, LightGrid, SunlightComputer, make_light_sampler, occupancy_from_materials, LIGHT_FULL, LIGHT_AMBIENT, column pass, box blur, diffusion, recompute, dirty, voxel light, light cell, light array, bake, face centers, mesher hook, build_mesh, uint8, float32

> One doc per code package; filename matches the package exactly (`docs/systems/lighting.md` ↔ `torn_apart/lighting/`).

## Role

`lighting/` is the **voxel light grid** package (Layer 1 — Services).  It owns:

- **Occupancy downsampling** — converting a 32³ terrain material array to a 16³ bool occupancy grid (one light cell = 2×2×2 terrain voxels = 1 m³).
- **Per-chunk light storage** — `LightGrid`: a dict of `uint8 (16, 16, 16)` arrays (one per chunk coord), with valid/dirty bookkeeping.
- **CPU sunlight column pass** — `SunlightComputer`: for each (cx, cy) column of loaded chunks, stacks occupancy into a tall grid and sweeps cumulative-OR downward from +Z (sunlight source) to classify every cell as full-sun (255) or ambient (40).
- **Box-blur diffusion** — a 3×3×3 uniform filter that softens the hard sun/shadow boundary into a smooth penumbra, applied to the full column stack before splitting back into per-chunk arrays.
- **Event subscriptions** — `SunlightComputer` subscribes to `TerrainEditedEvent` and `ChunkLoadedEvent` on the `EventBus` to keep light current without manual polling.
- **Mesher integration** — `make_light_sampler` returns a callable matching the `build_mesh` `light_sampler` contract: given face-centre world positions `float32 (F, 3)` it returns per-face light `float32 (F,)` in `[0.0, 1.0]`, which the mesher bakes into vertex colours (default Panda3D shader multiplies texture by vertex colour, so darker = shadowed).

**Phase 4 v0 scope:** CPU only, numpy-vectorised, no panda3d imports.  No custom GLSL — vertex colours are the full lighting pipeline this phase.  The light array layout is already GPU-uploadable as a 3-D texture for Phase 5+.

`lighting/` deliberately does NOT: issue render commands, touch the Panda3D scene graph (those are `world/`'s responsibility), or contain per-cell / per-voxel Python loops (Hard Rule 4 — all hot work is numpy array expressions).

## Public API

All symbols are re-exported from `torn_apart.lighting` (`__init__.py`).

| Symbol | Description |
|---|---|
| `LIGHT_FULL: int = 255` | Full sunlight level (no solid above). |
| `LIGHT_AMBIENT: int = 40` | Ambient/shadowed light level (solid present above). |
| `occupancy_from_materials(materials: uint8[32,32,32]) -> bool[16,16,16]` | Downsample 32³ terrain to 16³ occupancy.  Cell True = any of its 8 voxels is solid. |
| `LightGrid()` | Per-chunk `uint8 (16,16,16)` light store. |
| `LightGrid.get(coord) -> uint8[16,16,16] \| None` | Return stored light array or `None`. |
| `LightGrid.set(coord, arr)` | Store and mark valid. |
| `LightGrid.has_valid(coord) -> bool` | True when the stored array is current. |
| `LightGrid.invalidate(coord)` | Mark stale (keeps array in memory for fallback sampling). |
| `LightGrid.remove(coord)` | Remove all data for an evicted chunk. |
| `LightGrid.loaded_coords() -> list[coord]` | All coords with stored light data. |
| `SunlightComputer(config, chunk_provider, light_grid, bus)` | Column-pass + blur engine; subscribes to bus on construction. |
| `SunlightComputer.recompute_column(cx, cy)` | Recompute one (cx, cy) column. |
| `SunlightComputer.recompute_all_loaded()` | Recompute all loaded columns (initial seed at boot or after load). |
| `make_light_sampler(light_grid, config) -> Callable` | Return a `light_sampler` callable for `build_mesh`. |

## Imports Allowed

Per ARCHITECTURE.md §4a.2, `lighting/` may import:
- `numpy`, Python standard library
- `torn_apart.core` (Config, EventBus + events, get_logger)
- `torn_apart.terrain` (Chunk, occupancy helper) — downward call allowed
- `panda3d` — allowed (lighting is the GPU-upload exception), but Phase 4 v0 uses no panda3d

**No imports from** `world/`, `buildings/`, simulation layers, or anything higher in the stack.

## Events

### Subscribed
| Event | Subscriber | Action |
|---|---|---|
| `TerrainEditedEvent(chunk_coords, brush)` | `SunlightComputer._on_terrain_edited` | Recompute light for every (cx,cy) column affected by the edited coords; mark all chunks in those columns `dirty=True` so `stream_frame` remeshes them with fresh light. |
| `ChunkLoadedEvent(coord)` | `SunlightComputer._on_chunk_loaded` | Recompute the (cx,cy) column containing the newly loaded chunk; mark all chunks in that column dirty. |

### Published
None.  The lighting layer notifies of stale state indirectly by setting `chunk.dirty = True`, which triggers remeshing in `ChunkManager.stream_frame`.

## Units & Invariants

### Coordinate System
- **Z-up** (Panda3D native): sunlight travels in the −Z direction (top → bottom).
- Voxel edge = 0.5 m; **light cell edge = 1.0 m** (`voxel_size * light_grid_scale`).
- Chunk = 32³ voxels = **16 m³** cube; light grid per chunk = **16³ light cells**.
- Light-cell index `(lcx, lcy, lcz)` within a chunk: `lcx = voxel_x // 2`, etc. (0..15).
- World position → chunk coord: `floor(world / 16.0)`.  Chunk origin = `chunk_coord * 16.0 m`.
- World position → light cell in chunk: `floor((world - chunk_origin) / 1.0)`, clamped 0..15.

### Light Value Constants
| Constant | Value | Meaning |
|---|---|---|
| `LIGHT_FULL` | 255 | No solid voxel at or above this cell in its (cx,cy) column. |
| `LIGHT_AMBIENT` | 40 | At least one solid voxel at or above this cell. |

After the box blur, intermediate values between 40 and 255 appear in the penumbra zone adjacent to shadow edges.

### Column Pass Algorithm
1. Collect all loaded chunks for a given (cx, cy), sorted by `cz` ascending (bottom to top).
2. Compute 16³ bool occupancy for each via `occupancy_from_materials`.
3. Concatenate along Z axis: shape `(16, 16, T)` where `T = num_chunks * 16`.
   - Axis 0 = X (light cells across), axis 1 = Y, axis 2 = Z (bottom=0, top=T-1).
4. Sunlight travels downward (−Z).  A cell is **shadowed** if any cell at the **same or higher Z** in its (X,Y) column is occupied.
5. Implementation: reverse Z (`[:, :, ::-1]`), apply `np.maximum.accumulate(axis=2)` (cumulative-OR top→bottom), reverse back.  Result is `1` where shadowed, `0` where lit.
6. Map: `shadowed → LIGHT_AMBIENT`, `not_shadowed → LIGHT_FULL`.

### Box Blur (Penumbra)
- A 3×3×3 uniform box filter is applied to the full column's float light values **before** converting to `uint8`.
- Implementation: pad the `(16, 16, T)` float array by 1 on each face (edge-replicate) → `(18, 18, T+2)`, then sum 27 slices (offsets `i,j,k ∈ {0,1,2}`) and divide by 27.  The outer loop runs exactly 27 iterations regardless of scene size (not a per-cell loop).
- The float result is clamped to `[LIGHT_AMBIENT, LIGHT_FULL]` then rounded to `uint8`.  **Range is conserved:** ambient floor is preserved (shadowed regions never go darker than 40); full ceiling is preserved (fully-lit regions stay at 255).
- The blur produces a gradient (penumbra) of 1–2 cells wide around each shadow edge.

### make_light_sampler Contract (exact mesher hook)
```
sampler(face_centers: float32 (F, 3)) -> float32 (F,)
```
- **Input**: face-centre world positions in **meters**, one row per exposed mesh face.
- **Output**: per-face light in **[0.0, 1.0]** (value is multiplied into greyscale vertex colour by the mesher; alpha = 1.0).
- **Cell mapping** (vectorised, no per-face loop):
  1. `chunk_coord = floor(world / 16.0)` per face.
  2. `chunk_origin = chunk_coord * 16.0`.
  3. `light_cell = floor((world - chunk_origin) / 1.0)`, clamped to `[0, 15]`.
  4. Fancy-index `light_grid.get(chunk_coord)[cx, cy, cz]` → `uint8`.
  5. Divide by 255 → `float32`.
- **Fallback**: chunks with no computed light array return **1.0 (full bright)** to prevent black flashes on freshly streamed geometry.

### Saveable
`lighting/` does NOT implement `Saveable` — light is fully recomputed from the terrain data on load.  No save delta needed.

### Determinism
`occupancy_from_materials` is a pure function of the materials array; the column pass is a pure function of the occupancy stack; the box blur is a pure function of the light column.  Same terrain data → byte-identical light arrays.

## Examples

### Boot sequence (wiring into the orchestrator)
```python
from torn_apart.core import load_config, EventBus
from torn_apart.core.rng import set_world_seed
from torn_apart.terrain import ChunkManager
from torn_apart.lighting import LightGrid, SunlightComputer, make_light_sampler

set_world_seed(1337)
cfg = load_config()
bus = EventBus()

cm = ChunkManager(cfg, bus)
lg = LightGrid()
sc = SunlightComputer(cfg, cm, lg, bus)   # subscribes to bus immediately

# After initial streaming:
sc.recompute_all_loaded()
sampler = make_light_sampler(lg, cfg)

# Each frame (in the main loop, step Q4→Q5 per ARCHITECTURE.md §4a.1):
from torn_apart.core.math3d import Vec3
cm.stream_frame(Vec3(0, 0, 20), light_sampler=sampler)   # light baked into new meshes
# SunlightComputer auto-handles dirty recomputes via event subscriptions.
```

### Manual column recompute
```python
sc.recompute_column(cx=3, cy=-2)   # recompute one column without events
```

### Sampling the light grid directly
```python
import numpy as np
positions = np.array([[8.0, 8.0, 7.5]], dtype=np.float32)   # world meters
light = sampler(positions)   # float32 (1,) in [0.0, 1.0]
```

### occupancy_from_materials
```python
import numpy as np
from torn_apart.lighting import occupancy_from_materials

mat = np.zeros((32, 32, 32), dtype=np.uint8)
mat[0, 0, 0] = 1            # solid voxel in first 2×2×2 block
occ = occupancy_from_materials(mat)   # bool (16, 16, 16)
assert occ[0, 0, 0] == True           # light cell (0,0,0) is occupied
```

## Gotchas

1. **`SunlightComputer` subscribes synchronously on construction.** Construct it *after* the EventBus but *before* publishing `ChunkLoadedEvent` / `TerrainEditedEvent`, or those early events will miss the subscription.  (The startup sequence in ARCHITECTURE.md §4a.1 shows the correct order: lighting is allocated before terrain streaming begins.)

2. **`stream_frame` needs the sampler passed in explicitly.** The `ChunkManager` does not hold a reference to the sampler by default.  The orchestrator calls `cm.stream_frame(camera_pos, light_sampler=sampler)`.  Forgetting the argument produces full-bright terrain (the `None` default), not an error.

3. **Column pass only uses *loaded* chunks.** Chunks above the target chunk that are not yet streamed are treated as air (no occupancy).  This means very tall structures whose top is beyond `view_distance_chunks` will appear lit below in their shadow band — acceptable for v0, fixable in v1 by extending the column buffer above the loaded set.

4. **The 27-iteration blur loop is not a "per-cell loop" (Hard Rule 4).** It iterates 27 times over *constant* neighbourhood offsets, not over terrain cells.  The actual cell work is done by numpy slice operations across the full array.  Do not mistake it for a performance violation.

5. **`make_light_sampler` closes over `light_grid` by reference.** The sampler always reads the current contents of the grid.  Do not replace the `LightGrid` object after creating the sampler — update it in-place via `set()`.

6. **Light arrays are stored by reference in `LightGrid`.** `SunlightComputer` calls `lg.set(coord, arr)` with newly allocated arrays.  Old arrays are garbage-collected.  Do not hold long-lived views into a light array that may be replaced by the next recompute.

7. **`TerrainEditedEvent.chunk_coords` may be a single tuple or a frozenset.** `_on_terrain_edited` handles both cases.  See `core/event_bus.py` for the event definition; `apply_brush` currently emits one event per touched chunk with `chunk_coords` as a single 3-tuple.

8. **Light is not saved.** It is recomputed from terrain on boot and after `apply_delta`.  After `SaveManager.load()`, call `sc.recompute_all_loaded()` (or rely on event-driven recompute as chunks are re-streamed).
