# terrain — System Doc
keywords: voxel, chunk, mesh, mesher, build_mesh, brush, apply_brush, SphereBrush, BoxBrush, CylinderBrush, BrushMode, crater, heightmap, surface_height, carve, cave, overhang, raycast, raycast_voxel, DDA, Hit, streaming, ChunkManager, desired_set, stream_frame, get_or_create, chunk_provider, Saveable, delta, get_delta, apply_delta, generate_chunk, MeshArrays, light_sampler, neighbor_solids, WORLD_FLOOR_SOLID, padding, world_origin, materials, dirty, edited, determinism, geometry_bridge, to_geom

> One doc per code package; filename matches the package exactly (`docs/systems/terrain.md` ↔ `torn_apart/terrain/`).

## Role

`terrain/` is the **voxel terrain** package (Layer 2 — Structure).  It owns:

- **Chunk storage** — `Chunk`, a 32³ `uint8` material array (0 = air, ≥1 = solid material id) with `dirty`/`edited` flags and a world-space origin.
- **Generation** — `generate_chunk`, a *pure function of (world_seed, chunk coord)*: a continuous world-coordinate value-noise heightmap plus a 3-D carve pass for overhangs / shallow caves.
- **Meshing** — `build_mesh`, a fully-vectorised culled-face mesher emitting flat-shaded quads only where a voxel is exposed (the retro hard-edge look).
- **Brush editing** — `apply_brush` with `SphereBrush`/`BoxBrush`/`CylinderBrush`, the *only* terrain mutation path (ARCHITECTURE.md §5.5).
- **Raycasting** — `raycast_voxel`, a voxel DDA that turns a click ray into a `Hit` (used to place a brush centre).
- **Streaming + persistence** — `ChunkManager`: camera-proximity load/unload, the `chunk_provider` for brush/raycast, and the `"terrain"` `Saveable` (delta = edited chunks only).

`terrain/` deliberately does NOT: import panda3d, touch the scene graph, issue render commands, or compute lighting.  It produces pure-numpy `MeshArrays` that the World layer uploads via `world/geometry_bridge.py`.  It is fully headless-testable.

## Public API

All symbols below are re-exported from `torn_apart.terrain` (`__init__.py`).

| Symbol | Description |
|---|---|
| `Chunk(coord, materials=None, *, chunk_size=32, voxel_size=0.5)` | 32³ `uint8` material chunk. |
| `Chunk.materials` | `uint8 (32,32,32)` indexed `[x, y, z]`; 0 = air, ≥1 = solid. |
| `Chunk.dirty` | Needs remesh. |
| `Chunk.edited` | Deviates from generated baseline → goes in the save delta. |
| `Chunk.world_origin -> Vec3` | Min-corner world position = `coord * 16 m`. |
| `Chunk.is_solid_mask() -> bool[32,32,32]` | `materials > 0`. |
| `generate_chunk(coord, config) -> uint8[32,32,32]` | Pure-function chunk generation. |
| `surface_height(world_x, world_y) -> float32` | Continuous terrain height (world Z, meters) at world XY. |
| `build_mesh(chunk, neighbor_solids=None, light_sampler=None) -> MeshArrays` | Culled-face vectorised mesher. |
| `MeshArrays` | Dataclass: `positions`, `normals`, `uvs`, `colors`, `indices` (+ `face_count`/`tri_count`/`vertex_count`/`is_empty`). |
| `WORLD_FLOOR_SOLID` | Sentinel for `neighbor_solids` meaning "pad this face SOLID". |
| `SphereBrush(radius_m)` | Sphere shape. |
| `BoxBrush(half_extents_m: Vec3)` | Axis-aligned box shape. |
| `CylinderBrush(radius_m, height_m)` | Vertical (Z-axis) cylinder shape. |
| `BrushMode.ADD` / `BrushMode.REMOVE` | Brush mode enum. |
| `apply_brush(brush, center, mode, material=1, *, chunk_provider, bus=None) -> set[coord]` | The single mutation path. |
| `raycast_voxel(origin, direction, chunk_provider, max_distance_m=100.0) -> Hit \| None` | Voxel DDA raycast. |
| `Hit` | Dataclass: `point`, `voxel`, `chunk_coord`, `normal`, `distance`. |
| `ChunkManager(config, event_bus)` | Streaming store, `chunk_provider`, `Saveable("terrain")`. |

The render-side bridge `world/geometry_bridge.to_geom(mesh) -> panda3d.core.Geom` (and `to_geom_node`) lives in `world/` (the only file allowed to import panda3d for this handoff).

## Imports Allowed

Per ARCHITECTURE.md §4a.2, `terrain/` may import:
- `numpy`, Python standard library
- `torn_apart.core` (Config, EventBus + events, `for_domain`, Vec3, get_logger)
- `torn_apart.procedural` (`value_noise` and registry — foundation, callable from anywhere)

**No panda3d imports.** Never import from `world/`, `lighting/`, `buildings/`, or any higher layer.  The `terrain → world` mesh handoff is data-only (numpy `MeshArrays` returned to the World layer's geometry bridge), not an import of `world`.

## Events

### Published
| Event | When | Publisher |
|---|---|---|
| `ChunkLoadedEvent(coord)` | A chunk has been generated + meshed | `ChunkManager.stream_frame` |
| `ChunkUnloadedEvent(coord)` | A chunk is evicted (beyond radius+1) | `ChunkManager.stream_frame` |
| `TerrainEditedEvent(chunk_coords, brush)` | One per chunk a brush actually changed | `apply_brush` (when `bus` given) |

### Subscribed
None.  Terrain is mutated by direct `apply_brush` calls, not events.  (Lighting, in Phase 4, *subscribes* to `TerrainEditedEvent` to recompute light.)

## Units & Invariants

### Coordinate System & Index Convention
- **Z-up** (Panda3D native): +Z is world up.  All distances in meters.
- Voxel edge = **0.5 m**; chunk edge = **32 voxels = 16 m**; light cell = 1 m.
- Chunk coord is integer `(cx, cy, cz)`; **world origin (min corner) = `coord * 16 m`**.
- `materials[x, y, z]` — `x, y, z` are local voxel indices `0..31`:
  - `x` → world +X (east), spans `[origin_x, origin_x + 16)`
  - `y` → world +Y (north), spans `[origin_y, origin_y + 16)`
  - `z` → world +Z (up), spans `[origin_z, origin_z + 16)`
  - Voxel `(x,y,z)` centre world position = `world_origin + (x+0.5, y+0.5, z+0.5) * 0.5 m`.

### Determinism Guarantee
`generate_chunk(coord, config)` is a **pure function of (world_seed, coord)** and is **byte-identical** across processes/machines (verified: same SHA-256 in two separate interpreters).  All noise is drawn from `core.rng.for_domain("terrain", ...)` (blake2b key digest, never `hash()`).  No `random.*`, no unseeded `np.random.*`.

### Seamlessness
Both the heightmap and the carve field are sampled in **continuous WORLD coordinates** from global coarse-noise grids (not per-chunk noise), wrapped to a fixed global period.  Neighbouring chunks index the *same* field, so terrain meets seamlessly at chunk borders (no cliffs/gaps; a column straddling two vertically-adjacent chunks is continuous).

### Mesher: edge padding rule (critical)
`build_mesh` builds a `(34, 34, 34)` padded solidity array and computes face masks by **padded-array slicing**, never `np.roll` (roll wraps → face leaks).  Face exposed ⇔ `solid & ~neighbor_solid`.  The 6 one-voxel boundary slabs are filled as follows:

- **Neighbour present** → fill from that neighbour chunk's facing slab (correct cross-chunk culling).
- **Neighbour absent** → pad **AIR** (open/visible world edge) …
- **…except the −Z (bottom) world boundary**, which pads **SOLID** so the map has no see-through floor.  "Bottom of the world" is signalled by the caller passing the `WORLD_FLOOR_SOLID` sentinel for the `(0,0,-1)` direction.  `ChunkManager` emits that sentinel when the chunk below is absent AND this chunk sits at/below the lowest streamed Z band (`cz <= -2`).

Counts: 1 solid voxel in air → **6 faces / 12 tris / 24 verts**; two adjacent → **10 faces** (shared face culled); fully buried → **0 faces**.

### `build_mesh` signature & the `light_sampler` contract (Phase 4 plugs in here)
```python
build_mesh(
    chunk,
    neighbor_solids: dict | None = None,
    light_sampler: Callable[[np.ndarray], np.ndarray] | None = None,
) -> MeshArrays
```
- **`neighbor_solids`**: a dict mapping each of the 6 unit face directions `(dx,dy,dz)` ∈ `{(±1,0,0),(0,±1,0),(0,0,±1)}` to one of:
  - a `bool (32,32,32)` solidity array of that neighbour chunk, OR
  - the `WORLD_FLOOR_SOLID` sentinel (pad that face SOLID — the −Z floor), OR
  - key absent / `None` (pad that face AIR — open edge).
  Passing `None` for the whole dict pads every face air (isolated single-chunk fixtures).
- **`light_sampler`** — the exact Phase-4 contract:
  - **input**: face-centre world positions, `float32` shape `(F, 3)` in **meters** (one row per exposed face, `F == mesh.face_count`, in the mesher's face order).
  - **output**: per-face light, `float32` shape `(F,)` in range **[0.0, 1.0]** (0 = black, 1 = full sun).
  - The value is multiplied into the greyscale base colour and written to all 4 vertices of that face (RGBA, alpha = 1.0).
  - When `None` (default), all faces are full-bright (1.0), so the mesher is testable without lighting.

`MeshArrays` fields: `positions float32[N,3]` (world meters), `normals float32[N,3]` (flat per-face), `uvs float32[N,2]` (planar from world coords ÷ 1 m tile), `colors float32[N,4]` (RGBA [0,1]), `indices uint32[M]` (2 CCW tris/quad).  `N = 4*F`, `M = 6*F`.

### `chunk_provider` contract (brush + raycast)
`chunk_provider(coord) -> Chunk` returns the chunk for an integer coord, **creating/generating it on demand** if not loaded.  `ChunkManager.get_or_create` satisfies it (and `ChunkManager` is itself callable as a provider via `__call__`).  Headless tests use a dict-backed provider that constructs a `Chunk` on miss.

### Saveable delta format
`ChunkManager` implements `Saveable` with `save_key = "terrain"`:
- `get_delta() -> {coord_tuple: materials_uint8_array}` — **edited chunks only** (`chunk.edited == True`).  Values are copies of the `uint8 (32,32,32)` arrays — plain numpy, **no live object refs, no pickle** (Hard Rule 3).
- `apply_delta(delta)` — after baseline regen from seed, for each `coord -> materials`: ensure the chunk exists (generate baseline if needed), overwrite its materials with the saved array, and mark it `edited=True` (re-saves) + `dirty=True` (remeshes next `stream_frame`).

### Streaming invariants
- `desired_set(camera_pos)` is a **pure function**: chunks within `view_distance_chunks` (square XY radius) of the camera chunk, Z in `[-2, +4]` relative to it.  Count = `(2r+1)² * 7`.
- `stream_frame` loads/meshes **≤ 2 chunks per frame** (nearest-first), remeshes dirty chunks within the same budget, and unloads chunks beyond `radius + 1` (hysteresis → no boundary thrash).
- `pending_meshes` holds produced `MeshArrays` for the World layer to upload; `unloaded_this_frame` lists coords whose Geoms the World layer must remove.

### Handoff to `world/`
Terrain produces `MeshArrays` (pure numpy) and records them; the World layer drains `pending_meshes`, calls `world/geometry_bridge.to_geom` (one bulk memoryview write per array — no per-vertex `GeomVertexWriter` loops), and uploads the Geom.

## Examples

### Generate + mesh a chunk
```python
from torn_apart.core import load_config
from torn_apart.core.rng import set_world_seed
from torn_apart.terrain import Chunk, generate_chunk, build_mesh

set_world_seed(1337)
cfg = load_config()
chunk = Chunk((0, 0, 0), generate_chunk((0, 0, 0), cfg))
mesh = build_mesh(chunk, neighbor_solids=None)   # isolated → open edges
print(mesh.face_count, mesh.tri_count)            # exposed quads / triangles
```

### Click → crater (the demo loop)
```python
from torn_apart.core import EventBus, load_config
from torn_apart.core.rng import set_world_seed
from torn_apart.core.math3d import Vec3
from torn_apart.terrain import ChunkManager, raycast_voxel, apply_brush, SphereBrush, BrushMode

set_world_seed(1337)
cm = ChunkManager(load_config(), EventBus())
hit = raycast_voxel(Vec3(8, -4, 30), Vec3(0, 0, -1), cm)   # cm is callable as provider
if hit:
    apply_brush(SphereBrush(2.5), hit.point, BrushMode.REMOVE,
                chunk_provider=cm, bus=cm.bus)              # carves + flags dirty/edited
```

### Streaming each frame
```python
from torn_apart.core.math3d import Vec3
cm.stream_frame(Vec3(0, 0, 20))            # loads up to 2 chunks
for coord, mesh in list(cm.pending_meshes.items()):
    ...                                     # world/ uploads via geometry_bridge.to_geom(mesh)
    del cm.pending_meshes[coord]
```

### Save round-trip
```python
delta = cm.get_delta()                      # {coord: uint8[32,32,32]} for edited chunks only
# ... later, fresh world same seed ...
cm2 = ChunkManager(load_config(), EventBus())
cm2.apply_delta(delta)                       # craters restored, chunks marked dirty+edited
```

### Phase-4 lighting hook (illustrative)
```python
def sun_sampler(face_centers):              # float32 (F,3) world meters
    # ... sample the light grid at each face centre ...
    return light_0_to_1                      # float32 (F,) in [0,1]

mesh = build_mesh(chunk, cm._neighbor_solids(coord), light_sampler=sun_sampler)
```

## Gotchas

1. **Padding, not `np.roll`.** Face masks use padded-array slicing; `np.roll` wraps and leaks faces across the chunk. Never substitute it.
2. **World-floor sentinel.** Absent neighbours pad AIR except −Z at the world floor (`WORLD_FLOOR_SOLID`). Forget it and the map's bottom is see-through. `ChunkManager` supplies it only for `cz <= -2` when the below-chunk is unloaded.
3. **Generation samples WORLD coordinates, never per-chunk noise.** Per-chunk `value_noise` would NOT tile — seams would crack. The global-grid samplers in `generation.py` are what make borders seamless; keep new noise terms world-coordinate.
4. **`materials[x,y,z]` index order is load-bearing.** The mesher, brush, raycast, lighting, and saves all assume `[x,y,z]` (x=east, y=north, z=up). Don't transpose.
5. **`apply_brush` is the ONLY mutation path** (ARCHITECTURE.md §5.5). Don't write `chunk.materials` directly outside it/generation/`apply_delta` — you'd skip the `dirty`/`edited` flags and the `TerrainEditedEvent`, breaking remesh and saves.
6. **Brush only flags/publishes for chunks it actually changed.** A brush whose mask misses a chunk (or changes no voxel, e.g. REMOVE over air) leaves that chunk untouched — no flags, no event, not in the returned set.
7. **The DDA loop is the one allowed Python loop** (≤200 steps, once per click). Everything else — generation, meshing, brush rasterisation, index/vertex assembly — is vectorised numpy (Hard Rule 4).
8. **`get_delta` copies arrays.** It returns copies, so mutating a chunk after `get_delta` doesn't corrupt an in-flight save. `apply_delta` overwrites in place into a generated baseline.
9. **Generation cost is the 3-D carve grid.** ~21 ms/chunk (gen) + ~3.5 ms (mesh) at seed 1337 — under the 30 ms budget. If you add octaves to the carve field, re-check the budget; the 3-D coarse grid dominates.
