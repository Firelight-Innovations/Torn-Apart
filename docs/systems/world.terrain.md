# terrain — System Doc
keywords: voxel, chunk, mesh, mesher, build_mesh, build_mesh_faceted, surface nets, faceted, facet, smooth terrain, semi-smooth, Daggerfall, mesh_style, blocky, facet_shade_strength, grass, dirt, MATERIAL_DIRT, MATERIAL_GRASS, face_materials, verts_per_face, neighbor_materials, NEIGHBOR_OFFSETS_26, brush, apply_brush, SphereBrush, BoxBrush, CylinderBrush, BrushMode, crater, flat, flat terrain, world_size_m, ground_height_m, world footprint, world bounds, world size, seed-independent, authored terrain, heightmap, surface_height, carve, cave, overhang, raycast, raycast_voxel, DDA, Hit, streaming, ChunkManager, desired_set, stream_frame, get_or_create, chunk_provider, Saveable, delta, get_delta, apply_delta, generate_chunk, MeshArrays, light_sampler, neighbor_solids, WORLD_FLOOR_SOLID, padding, world_origin, materials, dirty, edited, determinism, geometry_bridge, to_geom, vertex alpha, material packing, material id alpha, procedural ground, world-space ground, ground_texels_per_m, ground shader palette, rain cover, RainCoverField, rain_cover, cover heightmap, top-solid, highest solid voxel, rain occlusion, no rain under roof, OPEN_SKY_Z, rain_cover_cells, rain_cover_cell_m, rain_cover_budget_columns, recenter, rebuild_all, rebuild_columns

> One doc per code package; filename matches the package exactly (`docs/systems/world.terrain.md` ↔ `fire_engine/world/terrain/`).

## Role

`terrain/` is the **voxel terrain** package (Layer 2 — Structure).  It owns:

- **Chunk storage** — `Chunk`, a 32³ `uint8` material array (0 = air, ≥1 = solid material id; `MATERIAL_DIRT`=1, `MATERIAL_GRASS`=2) with `dirty`/`edited` flags and a world-space origin.
- **Generation** — `generate_chunk`, a *pure function of (coord, config)* producing **flat, seed-independent baseline terrain**: solid below `config.ground_height_m`, clamped to the square `config.world_size_m` footprint (centred on origin), no hills/noise/caves. The topmost solid voxel layer is grass (`MATERIAL_GRASS`); everything deeper is dirt (`MATERIAL_DIRT`) — digging exposes dirt. The terrain is authored semi-procedurally on top of this blank canvas; `world_seed` drives other systems, not terrain.
- **Meshing** — two fully-vectorised meshers selected by `config.mesh_style`:
  - `build_mesh_faceted` (**default**, `mesh_style = "faceted"`) — flat-shaded **naive surface nets**: vertices are pulled onto the solid/air surface so crater walls become chamfered sloped facets instead of stair-steps, but every triangle keeps its own flat normal + a subtle normal-based "facet accent" shade, so individual polygons stay visible (the Daggerfall-ish semi-smooth look). Emits per-face material ids (`MeshArrays.face_materials`) for grass/dirt texturing.
  - `build_mesh` (`mesh_style = "blocky"`) — the classic culled-face cube mesher (hard Minecraft edges), kept for fixtures/comparison.
- **Brush editing** — `apply_brush` with `SphereBrush`/`BoxBrush`/`CylinderBrush`, the *only* terrain mutation path (ARCHITECTURE.md §5.5).
- **Raycasting** — `raycast_voxel`, a voxel DDA that turns a click ray into a `Hit` (used to place a brush centre).
- **Streaming + persistence** — `ChunkManager`: camera-proximity load/unload, the `chunk_provider` for brush/raycast, and the `"terrain"` `Saveable` (delta = edited chunks only).

`terrain/` deliberately does NOT: import panda3d, touch the scene graph, issue render commands, or compute lighting.  It produces pure-numpy `MeshArrays` that the World layer uploads via `world/geometry_bridge.py`.  It is fully headless-testable.

## Public API

All symbols below are re-exported from `fire_engine.world.terrain` (`__init__.py`).

| Symbol | Description |
|---|---|
| `Chunk(coord, materials=None, *, chunk_size=32, voxel_size=0.5)` | 32³ `uint8` material chunk. |
| `Chunk.materials` | `uint8 (32,32,32)` indexed `[x, y, z]`; 0 = air, ≥1 = solid. |
| `Chunk.dirty` | Needs remesh. |
| `Chunk.edited` | Deviates from generated baseline → goes in the save delta. |
| `Chunk.world_origin -> Vec3` | Min-corner world position = `coord * 16 m`. |
| `Chunk.is_solid_mask() -> bool[32,32,32]` | `materials > 0`. |
| `generate_chunk(coord, config) -> uint8[32,32,32]` | Flat baseline chunk: solid below `ground_height_m` within the `world_size_m` footprint (centred on origin), else air. Top solid layer = `MATERIAL_GRASS`, deeper = `MATERIAL_DIRT`. Seed-independent. |
| `MATERIAL_DIRT` / `MATERIAL_GRASS` | Material id constants (1 / 2). The renderer maps them to the `"dirt_ground"` / `"grass_ground"` procedural textures. |
| `surface_height(world_x, world_y, config=None) -> float32` | Flat ground height (world Z, meters); constant `config.ground_height_m` (or 8.0 m default) for all XY. |
| `build_mesh(chunk, neighbor_solids=None, light_sampler=None) -> MeshArrays` | Culled-face cube mesher (`mesh_style = "blocky"`). |
| `build_mesh_faceted(chunk, neighbor_materials=None, light_sampler=None, *, shade_strength=0.25) -> MeshArrays` | Flat-shaded surface-nets mesher (`mesh_style = "faceted"`, default). `neighbor_materials` maps each of the 26 `NEIGHBOR_OFFSETS_26` to a neighbour `uint8` materials array / `WORLD_FLOOR_SOLID` / absent (air). |
| `NEIGHBOR_OFFSETS_26` | All 26 face+edge+corner neighbour offsets the faceted mesher needs. |
| `MeshArrays` | Dataclass: `positions`, `normals`, `uvs`, `colors`, `indices`, `face_materials` (`uint8 (F,)` or `None`), `verts_per_face` (4 blocky / 6 faceted) (+ `face_count`/`tri_count`/`vertex_count`/`is_empty`). |
| `WORLD_FLOOR_SOLID` | Sentinel for `neighbor_solids`/`neighbor_materials` meaning "pad this region SOLID". |
| `SphereBrush(radius_m)` | Sphere shape. |
| `BoxBrush(half_extents_m: Vec3)` | Axis-aligned box shape. |
| `CylinderBrush(radius_m, height_m)` | Vertical (Z-axis) cylinder shape. |
| `BrushMode.ADD` / `BrushMode.REMOVE` | Brush mode enum. |
| `apply_brush(brush, center, mode, material=1, *, chunk_provider, bus=None) -> set[coord]` | The single mutation path. |
| `raycast_voxel(origin, direction, chunk_provider, max_distance_m=100.0) -> Hit \| None` | Voxel DDA raycast. |
| `Hit` | Dataclass: `point`, `voxel`, `chunk_coord`, `normal`, `distance`. |
| `ChunkManager(config, event_bus)` | Streaming store, `chunk_provider`, `Saveable("terrain")`. |
| `ChunkManager.remesh_edited(coords, light_sampler=None) -> int` | Same-frame remesh of brush-edited chunks + their dirty border neighbours, bypassing the `stream_frame` budget. Call right after `apply_brush` with its returned set. |
| `RainCoverField(config)` | Top-down **cover heightmap**: highest-solid-voxel world Z per 1 m column in a `rain_cover_cells²` window centred on the player. Pure numpy (no panda3d); the M6 rain renderer (`world/rain_renderer.py`) uploads it and discards rain below the cover height (no rain under a roof). |
| `RainCoverField.height` | `(cells, cells) float32` world Z (m); `[row=+Y, col=+X]`, `OPEN_SKY_Z` where unknown. |
| `RainCoverField.origin_m -> (float, float)` | Committed min-corner world XY (m) of texel `(0,0)`; world XY → texel = `(xy - origin_m) / cell_m`. |
| `RainCoverField.recenter(center_xy) -> origin_m` | Snap the window's min-corner under `center_xy` (cell-grid snapped — committed-origin discipline). |
| `RainCoverField.rebuild_all(chunks)` | Cold rebuild: clear to `OPEN_SKY_Z`, fold every chunk overlapping the window. |
| `RainCoverField.rebuild_columns(chunks, chunk_columns)` | Incremental refold of the window texels under the given `(cx, cy)` chunk columns (clears then re-folds all loaded Z layers — removing a roof lowers the height). |
| `OPEN_SKY_Z` | Sentinel world Z (−1e9 m) for a column with no known solid voxel (open sky never clips rain). |

The render-side bridge `world/geometry_bridge.to_geom(mesh) -> panda3d.core.Geom` (and `to_geom_node`) lives in `world/` (the only file allowed to import panda3d for this handoff).

## Imports Allowed

Per ARCHITECTURE.md §4a.2, `terrain/` may import:
- `numpy`, Python standard library
- `fire_engine.core` (Config, EventBus + events, `for_domain`, Vec3, get_logger)
- `fire_engine.procedural` (`value_noise` and registry — foundation, callable from anywhere)

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
`generate_chunk(coord, config)` is a **pure function of (coord, config)** and is **byte-identical** across processes/machines.  It is **seed-independent** — baseline terrain is flat/authored, so `world_seed` does not affect it (the seed drives textures, ambient noise, NPC behaviour, etc.).  No `random.*`, no `np.random.*`.

### Flat baseline + world footprint
A voxel is solid iff its centre world Z is below `config.ground_height_m` **and** its centre world X and Y both lie within the square footprint `[-world_size_m/2, +world_size_m/2)` (centred on the origin).  No hills, no caves/overhangs.  Chunks entirely below the ground are fully solid; chunks entirely above it (or entirely outside the footprint) are empty air.  The topmost solid layer (the voxel whose +Z neighbour centre would be at/above the ground height) is `MATERIAL_GRASS`; all deeper solid voxels are `MATERIAL_DIRT` — both pure functions of world Z.  Seamlessness is trivial — solidity *and* material are pure functions of world position, so chunk borders always agree.

### Mesher: edge padding rule (critical)
`build_mesh` builds a `(34, 34, 34)` padded solidity array and computes face masks by **padded-array slicing**, never `np.roll` (roll wraps → face leaks).  Face exposed ⇔ `solid & ~neighbor_solid`.  The 6 one-voxel boundary slabs are filled as follows:

- **Neighbour present** → fill from that neighbour chunk's facing slab (correct cross-chunk culling).
- **Neighbour absent** → pad **AIR** (open/visible world edge) …
- **…except the −Z (bottom) world boundary**, which pads **SOLID** so the map has no see-through floor.  "Bottom of the world" is signalled by the caller passing the `WORLD_FLOOR_SOLID` sentinel for the `(0,0,-1)` direction.  `ChunkManager` emits that sentinel when the chunk below is absent AND this chunk sits at/below the lowest streamed Z band (`cz <= -2`).

Counts: 1 solid voxel in air → **6 faces / 12 tris / 24 verts**; two adjacent → **10 faces** (shared face culled); fully buried → **0 faces**.

### Faceted mesher (`build_mesh_faceted`, the default `mesh_style`)
Naive surface nets over the binary solid/air grid, flat-shaded:

- **Dual cells**: every 2×2×2 block of voxel centres is a cell (33³ per chunk, covering the border). A cell straddling solid/air gets one vertex at the **centroid of its sign-changing edge midpoints**; flat ground therefore stays *exactly* planar at `ground_height_m`, while crater walls become chamfered sloped facets.
- **Faces**: one quad per exposed voxel face — the **same exposure mask as `build_mesh`**, so `faceted.face_count == blocky.face_count` (a tested invariant) and the `light_sampler` contract is unchanged (one sample per face; centres are the deformed quad centroids).
- **Emission**: 2 *independent* flat triangles per face (`verts_per_face = 6`), each with its own normal and a normal-based **facet accent** shade `(1-s) + s*clamp(n·accent, 0, 1)` (s = `config.facet_shade_strength`) multiplied into the baked light — facets stay readable in the lighting-off texture×vertex-colour pipeline. UVs are planar by dominant quad-normal axis (world ÷ 1 m tile).
- **Materials**: `face_materials[f]` = the solid voxel's material id; `world/geometry_bridge.to_geom_node` splits the chunk into one Geom per material and assigns the grass/dirt textures.
- **26-neighbour padding**: border dual cells straddle voxels from up to 4 chunks, so the padded 34³ array is filled from all `NEIGHBOR_OFFSETS_26` (face + edge + corner), not just the 6 face neighbours.
- **Seam guarantee**: both chunks compute shared border cells from the same world voxels → byte-identical vertex positions, no cracks, independent of meshing order. `ChunkManager._neighbor_materials` uses loaded chunks (live edits) or the pure `generate_chunk` baseline for unloaded ones **without inserting them into the store** (no side effects); the two are identical unless the neighbour was edited, and `apply_brush` dirty-flags border neighbours for remesh.
- **World floor**: no sentinel needed in the manager path — below-band neighbour chunks generate fully solid, which culls the bottom naturally (the mesher still honours `WORLD_FLOOR_SOLID` for isolated fixtures).

Counts: 1 solid voxel in air → **6 faces / 12 tris / 36 verts** (vertices pulled inside the voxel cube — an octahedron-ish nugget).

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
  - The value is multiplied into the greyscale base colour and written to all 6 vertices of that face (RGB). **Vertex-colour alpha is NOT 1.0** — the faceted mesher packs the face material id into alpha as `face_material / 255` so the GPU terrain shader can select a per-material world-space procedural palette (see `world/shaders/terrain.frag`). Terrain is opaque (no transparency attrib), so the fixed-function fallback ignores this alpha.
  - When `None` (default), all faces are full-bright (1.0), so the mesher is testable without lighting.

`MeshArrays` fields: `positions float32[N,3]` (world meters), `normals float32[N,3]` (flat per-face), `uvs float32[N,2]` (planar from world coords ÷ 1 m tile), `colors float32[N,4]` (RGB = baked-light×facet grey; **A = face material id / 255** for the GPU ground-palette shader), `indices uint32[M]` (2 CCW tris/quad).  `N = 4*F`, `M = 6*F`.

### `chunk_provider` contract (brush + raycast)
`chunk_provider(coord) -> Chunk` returns the chunk for an integer coord, **creating/generating it on demand** if not loaded.  `ChunkManager.get_or_create` satisfies it (and `ChunkManager` is itself callable as a provider via `__call__`).  Headless tests use a dict-backed provider that constructs a `Chunk` on miss.

### Saveable delta format
`ChunkManager` implements `Saveable` with `save_key = "terrain"`:
- `get_delta() -> {coord_tuple: materials_uint8_array}` — **edited chunks only** (`chunk.edited == True`).  Values are copies of the `uint8 (32,32,32)` arrays — plain numpy, **no live object refs, no pickle** (Hard Rule 3).
- `apply_delta(delta)` — after baseline regen from seed, for each `coord -> materials`: ensure the chunk exists (generate baseline if needed), overwrite its materials with the saved array, and mark it `edited=True` (re-saves) + `dirty=True` (remeshes next `stream_frame`).
- `reset_to_baseline()` — revert **every loaded edited chunk** to its procedural baseline: regenerate `materials` from `generate_chunk(coord, config)`, clear `edited`, set `dirty=True`, and drop the chunk's `pending_meshes` entry.  This is the **F9 revert prelude**: `apply_delta` only touches chunks in the saved delta, so to truly revert to a save (undoing craters dug *after* the save) you must wipe all edits first, then load.  Canonical F9 flow:
  ```python
  cm.reset_to_baseline()        # all edits → baseline, chunks marked dirty
  sm.load("saves/quick.ta")     # apply_delta re-adds ONLY the saved craters
  # subsequent stream_frame()s remesh the dirty chunks; world/ re-uploads Geoms
  ```
  Only *loaded* chunks are reset; unloaded chunks already hold no edits in RAM and regenerate from seed on their next `get_or_create`.

### Streaming invariants
- `desired_set(camera_pos)` is a **pure function**: chunks within `view_distance_chunks` (square XY radius) of the camera chunk, Z in `[-2, +4]` relative to it.  Count = `(2r+1)² * 7`.
- `stream_frame` has a **2-chunk-per-frame budget**: it remeshes **dirty chunks FIRST** (brush edits / relights the player is looking at must never be starved — with ~1.2k chunks in the desired set, loading takes hundreds of frames), then loads/meshes missing desired chunks nearest-first, and unloads chunks beyond `radius + 1` (hysteresis → no boundary thrash).
- **Interactive brush edits must NOT wait on that budget** — until a border neighbour remeshes, the faces the edit newly exposed in it don't exist, so the player sees a black hole through the world.  Call `remesh_edited(touched)` right after `apply_brush`: it immediately remeshes every still-dirty chunk in the touched set's 26-neighbourhood (typical crater: 1–4 chunks ≈ 10–30 ms, one-frame hitch) and leaves unrelated dirty chunks (e.g. an F9 load) on the budgeted path.
- `pending_meshes` holds produced `MeshArrays` for the World layer to upload; `unloaded_this_frame` lists coords whose Geoms the World layer must remove.

### Handoff to `world/`
Terrain produces `MeshArrays` (pure numpy) and records them; the World layer drains `pending_meshes`, calls `world/geometry_bridge.to_geom` (one bulk memoryview write per array — no per-vertex `GeomVertexWriter` loops), and uploads the Geom.

### Rain-cover heightmap (`RainCoverField`, M6)
A top-down cache of the **world Z of the highest solid voxel per 1 m column** around the player, consumed by the GPU rain renderer to cull rain under roofs/overhangs.

- **Window**: a `rain_cover_cells × rain_cover_cells` grid of `rain_cover_cell_m`-edge columns (256 × 1 m by default → a 256 m square), centred on the player and snapped to the cell grid (committed-origin discipline, mirroring wind/weather). `height[row, col]` indexes `row → world +Y`, `col → world +X` (the wind/weather convention); `origin_m` is texel `(0,0)`'s min-corner world XY.
- **Per-chunk reduction (vectorised — Hard Rule 4)**: `argmax` over the **reversed Z axis** of a chunk's `(32,32,32)` solidity mask finds the highest solid voxel per `(x,y)` column in one pass (no Python voxel loop); the top-face world Z = `cz·16 + (z_idx+1)·0.5`. Columns of multiple chunk-Z layers fold with `np.maximum`, so a roof above a floor wins. Open columns stay at `OPEN_SKY_Z` (−1e9 m) so open sky never clips rain.
- **Dirty rebuilds**: the world component marks `(cx, cy)` chunk **columns** dirty on `ChunkLoadedEvent` / `TerrainEditedEvent` and refolds them via `rebuild_columns` (clear the column's window footprint, re-fold all loaded Z layers) so **removing a roof lowers** the height. It recenters (full `rebuild_all`) when the player crosses a threshold and amortises dirty refolds over `rain_cover_budget_columns` columns per refresh.

## Examples

### Generate + mesh a chunk
```python
from fire_engine.core import load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.world.terrain import Chunk, generate_chunk, build_mesh

set_world_seed(1337)
cfg = load_config()
chunk = Chunk((0, 0, 0), generate_chunk((0, 0, 0), cfg))
mesh = build_mesh(chunk, neighbor_solids=None)   # isolated → open edges
print(mesh.face_count, mesh.tri_count)            # exposed quads / triangles
```

### Click → crater (the demo loop)
```python
from fire_engine.core import EventBus, load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.core.math3d import Vec3
from fire_engine.world.terrain import ChunkManager, raycast_voxel, apply_brush, SphereBrush, BrushMode

set_world_seed(1337)
cm = ChunkManager(load_config(), EventBus())
hit = raycast_voxel(Vec3(8, -4, 30), Vec3(0, 0, -1), cm)   # cm is callable as provider
if hit:
    touched = apply_brush(SphereBrush(2.5), hit.point, BrushMode.REMOVE,
                          chunk_provider=cm, bus=cm.bus)    # carves + flags dirty/edited
    cm.remesh_edited(touched)        # same-frame remesh — no see-through hole
```

### Streaming each frame
```python
from fire_engine.core.math3d import Vec3
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
3. **Baseline terrain is flat + seed-independent** (DECISIONS.md 2026-06-09). `generate_chunk` ignores `world_seed`; it only reads `ground_height_m` and `world_size_m` from config. Detail/structure is authored on top (brushes, future content agents) — don't reintroduce procedural heightmap noise into the baseline. If you ever sample world-position noise for terrain, sample in **continuous WORLD coordinates** (per-chunk noise won't tile → cracked seams).
4. **`materials[x,y,z]` index order is load-bearing.** The mesher, brush, raycast, lighting, and saves all assume `[x,y,z]` (x=east, y=north, z=up). Don't transpose.
5. **`apply_brush` is the ONLY mutation path** (ARCHITECTURE.md §5.5). Don't write `chunk.materials` directly outside it/generation/`apply_delta` — you'd skip the `dirty`/`edited` flags and the `TerrainEditedEvent`, breaking remesh and saves.
6. **Brush only flags `edited`/publishes for chunks it actually changed.** A brush whose mask misses a chunk (or changes no voxel, e.g. REMOVE over air) leaves that chunk untouched — no flags, no event, not in the returned set. HOWEVER, when changed voxels touch a chunk **border**, the adjacent chunks are flagged plain `dirty` (remesh only — cross-chunk culling and faceted border vertices depend on them) without `edited` or an event.
7. **The DDA loop is the one allowed Python loop** (≤200 steps, once per click). Everything else — generation, meshing, brush rasterisation, index/vertex assembly — is vectorised numpy (Hard Rule 4).
8. **`get_delta` copies arrays.** It returns copies, so mutating a chunk after `get_delta` doesn't corrupt an in-flight save. `apply_delta` overwrites in place into a generated baseline.
9. **Generation is cheap now.** Flat generation is a couple of vectorised comparisons (sub-millisecond/chunk); meshing (~3.5 ms) dominates. The old 3-D carve-noise cost is gone. If you add authored detail back into the baseline, re-check the per-chunk budget (30 ms).
