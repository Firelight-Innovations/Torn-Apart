# buildings — System Doc
keywords: building, buildings, floorplan, floor plan, storey, story, floor, wall, walls, curved wall, arc, bulge, segment, thickness, opening, window, door, sill, head, lintel, room, rooms, room detection, interior, foundation, slab, roof, flat roof, stairs, stair stub, BuildingDefaults, Building, Storey, Wall, Opening, Room, StairsStub, Foundation, RoofSlab, WallKind, OpeningKind, add_wall, add_opening, add_room, add_storey, add_stairs, set_foundation, set_roof, detect_rooms, storey_base_z, world_aabb, to_dict, from_dict, allocate_eid, element id, plan space, building-local, free-form, non-voxel, Sims, Paralives, BuildingDef, procedural building, house, settlement, seam, gap, corner filler, filler post, junction, butt joint, miter, watertight, pad_hull, pitched roof, gable, hip, shed, roof kind, surface material, per-surface material

## Role
Free-form floorplan buildings (ARCHITECTURE.md §5.7) — the data model and imperative authoring API for structures that are deliberately **not** voxel-aligned. A `Building` has one world transform (position `Vec3` + rotation `Quat`, arbitrary axis) and stacks per-storey 2-D floorplans: walls are straight segments or circular arcs (single `bulge` scalar) with real thickness and parametric window/door openings; rooms are first-class objects (explicitly authored, or auto-detected from wall topology) so future systems can procedurally furnish each room; foundations, floor/ceiling slabs and flat roofs complete the envelope; stairs exist as data stubs only (no geometry yet). This package is the base layer the future tag→building procedural generator (`BuildingDef`) will drive — every generator action is one of these API calls. It deliberately does NOT render (that is `world/building_renderer.py`), does NOT import panda3d (fully headless), and does NOT yet mesh stairs or pitched roofs.

## Public API
- `BuildingDefaults` — frozen per-building fallback dimensions; build via `BuildingDefaults.from_config(cfg)` (single number source = the `Config` `building_*` fields). Fields: `storey_height_m`, `wall_thickness_m`, `slab_thickness_m`, `foundation_depth_m`.
- `Building(name, position, rotation, defaults, tags=None)` — the root object. `add_storey(height_m=None, slab_m=None)`, `set_foundation(polygon=None, depth_m=None)` (None polygon → automatic convex-hull footprint of storey-0 walls padded by half the thickest wall), `set_roof(polygon=None, thickness_m=None, kind=RoofKind.FLAT, pitch_deg=30.0, ridge_dir_rad=0.0, overhang_m=0.0)` (FLAT keeps the flat slab; SHED/GABLE/HIP are pitched over the outline's ridge-aligned bounding rectangle), `storey_base_z(index)`, `total_height_m`, `world_aabb()` (conservative world AABB for lighting invalidation), `allocate_eid()`, `to_dict()`/`from_dict()` (exact round-trip; the Saveable payload).
- `Storey` — one floor; create only via `Building.add_storey`. `add_wall(a, b, *, bulge=0.0, thickness_m=None, height_m=None) -> Wall`, `add_opening(wall_id, kind, *, offset_m, width_m, head_m, sill_m=0.0) -> Opening` (validates length/band/sill<head), `add_room(polygon, tag="", meta=None) -> Room` (winding normalized to CCW), `add_stairs(storey_to, anchor, direction_rad, width_m) -> StairsStub`, `detect_rooms(*, snap_eps_m, arc_segments_per_quarter)` (auto-detection; see fire_engine/buildings/rooms.py).
- `Wall` — one span. `kind` (`WallKind.SEGMENT`/`ARC`, derived from `bulge`), `chord_m()`, `length_m()`, `arc_params() -> (center, radius, start_angle, sweep)`, `tessellate(arc_segments_per_quarter) -> float64 (P, 2)` centerline polyline with exact endpoints.
- `Opening` — rectangular cutout in the wall's local `(s, z)` frame (`s` = arclength from endpoint `a`; `z` above the floor-slab top). Fields `kind` (`OpeningKind.WINDOW`/`DOOR`), `offset_m`, `width_m`, `sill_m`, `head_m`.
- `Room` — enclosed plan region: `polygon` (CCW `float64 (N, 2)`), `tag`, `meta`, `auto`; `area_m2()`, `centroid()`.
- `StairsStub`, `Foundation`, `RoofSlab` — data carriers (see docstrings). `RoofSlab` now also carries `kind` (`RoofKind`), `pitch_deg`, `ridge_dir_rad`, `overhang_m` (all default to a flat roof, so Iteration-1 saves load unchanged).
- `WallKind`, `OpeningKind`, `RoofKind` — enums (str values in dicts). `RoofKind` = `FLAT` / `SHED` / `GABLE` / `HIP`.
- `SurfaceMaterial` — `IntEnum` (`WALL=0` / `FLOOR=1` / `ROOF=2` / `FOUNDATION=3`) the mesher writes into `MeshArrays.face_materials` (one uint8 per triangle); the renderer splits the building geom per id and binds a distinct procedural albedo (`plaster_wall` / `wood_floor` / `roof_shingle` / `stone_foundation`). `WALL=0` so untagged faces default to the wall texture; `face_materials` is `None` when only one material is present.
- All model types have `to_dict()`/`from_dict()` over plain primitives (no numpy in the dicts, no pickle).
- `rooms.detect_room_polygons(walls, *, snap_eps_m, arc_segments_per_quarter) -> list[float64 (N,2)]` — the room auto-detection engine `Storey.detect_rooms` delegates to. Tessellates each wall, snaps endpoints onto a `snap_eps_m` grid, builds a planar half-edge graph and traces its minimal cycles (next dart = the one clockwise of the reverse dart around the shared node); returns every CCW positive-area bounded face. Pure numpy + model; the graph trace is a bounded Python loop over wall segments (dozens, not thousands — flagged per Hard Rule 4).
- `meshing.mesh_building(building, cfg) -> terrain.meshing.MeshArrays` — the whole building as one **building-LOCAL** triangle soup (positions in local meters; the renderer applies `position`/`rotation` as a node transform — move/rotate is never a remesh). `colors` flat white `(1,1,1,1)` (the building shader ignores vertex colour, unlike terrain's material-id-in-alpha), `face_materials` a `uint8` per-triangle `SurfaceMaterial` id (or `None` if a single material), `verts_per_face=3` (the soup mixes wall panels / reveals / caps / slab tris, so `face_count` == triangle count — the per-face counts are not the terrain quad semantics). Also `meshing.mesh_wall(wall, z_bottom, z_top, arc_segments_per_quarter)` and `meshing.mesh_slab(polygon, z0, z1)` for the per-element pieces. The triangle-soup accumulator and its slab/sloped-prism primitives live in `buildings/_impl/soup.py` (`Soup`); pitched-roof geometry (gable/hip/shed planes + gable/shed infill) lives in `buildings/_impl/roofs.py` (`add_roof`); both are package-internal helpers the mesher composes.
- `triangulate.triangulate_polygon(polygon) -> uint32 (T,3)` — dependency-free ear clipping for slab faces (bounded loop over polygon vertices; vectorized ear test).
- `occlusion.BuildingOccupancyRasterizer(manager)` — a structural `lighting.volume.GeometryOccupancyProvider` (duck-typed; imports nothing from `lighting/`) that will splat building solids into the light cascades so buildings shadow the sun and bounce GI with zero shader changes. **v1: documented no-op** — `rasterize_occupancy(...)` returns without touching the arrays (buildings are *lit* but do not yet *occlude*), because the async cascade worker needs an immutable snapshot of the geometry first (future scope; the docstring spells out the intended wall/slab voxelization algorithm). Register via `GpuLightingPipeline.register_geometry_provider`.
- `BuildingManager(config, bus)` — the runtime registry; Saveable `save_key="buildings"`. `add(spec) -> Building` (clones `spec` via `from_dict(to_dict())`, assigns a fresh world id, publishes `"added"` — mutate the *returned* clone, never the argument), `remove(id) -> bool`, `notify_changed(id)` (publish `"modified"` after editing a building in place), `get(id)`, `buildings()` (id-ordered), `version` (monotonic change counter the renderer rebuilds against), `mark_baseline()`/`get_delta()`/`apply_delta()` (full-list delta vs the boot baseline, ZoneStore pattern; `{}` when unchanged).

## Imports Allowed
procedural, terrain, core (ARCHITECTURE.md §4a.2). **Never panda3d** — rendering goes through `world/building_renderer.py`; light occlusion through the structural `GeometryOccupancyProvider` protocol in `lighting/volume.py` (no import in either direction).

## Events
Published: `BuildingChangedEvent(building_id, change, bounds_min, bounds_max)` (defined in `core/event_bus.py`) by `BuildingManager` on every `add` (`"added"`), `notify_changed` (`"modified"`) and `remove` (`"removed"`), and re-emitted as `"added"` for each building during `apply_delta` (so the renderer rebuilds on load). `bounds_*` are the conservative world AABB (`Building.world_aabb()`). The model layer itself publishes nothing.
Subscribed: none.

## Units & Invariants
- Plan space = building-local x/y **meters**; `world = position + rotation.rotate(local)`; Z-up; rotations are quaternions only.
- Building-local `z = 0` = top of the foundation slab; foundation occupies `[-depth_m, 0]`. Storey `i` spans `[base_z, base_z + height_m]` with `base_z = Σ` heights below; its floor slab `[base_z, base_z + slab_m]`; walls `[base_z + slab_m, base_z + height_m]` (a wall's own `height_m` measures from the slab top). The flat roof slab sits on `total_height_m`.
- Bulge: `|bulge| = tan(included_angle / 4)`; **positive bows the wall to the LEFT of a→b**, negative right; `|bulge| = 1` is a semicircle; apex offset = `bulge × chord/2` off the chord midpoint. `arc_params()` returns *signed* sweep `= -4·atan(bulge)` (left-bow arcs walk clockwise about their center).
- Element ids are per-building monotonic ints, stable across save/load (`next_eid` is serialized); building ids are assigned by the manager.
- Model floats are Python float64; meshing converts to float32 at the GPU boundary.
- Determinism: the model layer has no RNG at all — identical authoring calls produce identical `to_dict()` output.

## Examples
```python
from fire_engine.core.config import Config
from fire_engine.core.math3d import Vec3, Quat
from fire_engine.buildings import Building, BuildingDefaults, OpeningKind
import math

defaults = BuildingDefaults.from_config(Config())
b = Building(name="hut", position=Vec3(-24.0, 10.0, 8.0),
             rotation=Quat.from_axis_angle(Vec3.UP, math.radians(15)),
             defaults=defaults, tags=["rural"])
s0 = b.add_storey()                          # 3.0 m floor-to-floor, 0.2 m slab
south = s0.add_wall((0, 0), (8, 0))          # straight, 0.3 m thick
bay   = s0.add_wall((8, 0), (8, 6), bulge=-0.4)  # bows right (outward, east)
s0.add_wall((8, 6), (0, 6))
s0.add_wall((0, 6), (0, 0))
s0.add_opening(south.id, OpeningKind.DOOR, offset_m=3.5, width_m=0.9, head_m=2.0)
s0.add_opening(south.id, OpeningKind.WINDOW, offset_m=1.0, width_m=1.2,
               sill_m=1.0, head_m=2.2)
s0.add_room([(0, 0), (8, 0), (8, 6), (0, 6)], tag="living")
b.set_foundation()                           # auto hull footprint, 0.5 m deep
b.set_roof()                                 # flat roof slab
spec = b.to_dict()                           # save payload; from_dict restores
```

The registered `DemoHouseDef` (`procedural.get("building_demo_house")`, in `buildings/defs.py`) is the in-game **feature-showcase** build: a two-storey ~12×8 m house exercising every v1 capability at once — non-orthogonal chamfer wall, curved east bay (incl. a curved-wall window), variable wall thickness (0.4 m exterior / 0.15 m interior) and a 1.1 m open-plan half-wall, exterior + interior doors, three auto-detected & tagged rooms on storey 0 plus one explicit `add_room("loft")` on storey 1, variable storey heights, auto foundation + a **gable roof** (32° pitch, 0.4 m eave overhang — Iteration-2 showcase), a stair stub, and an 18° yaw. It spawns on open ground one chunk **west** of the demo grass patch (clear of the grass/flower/bush/tree zones) on every launch via `[debug] debug_demo_building` (now **on by default** — set `false` to hide). main.py adds it before `mark_baseline()` so it is part of the save baseline (regenerates on load, ~0-byte delta when untouched).

## Gotchas
- Room auto-detection requires walls to MEET AT ENDPOINTS (within `building_snap_eps_m`, 1 cm): v1 does not split mid-span T-junctions — author the long wall as two spans sharing the junction vertex.
- The automatic `set_foundation()`/`set_roof()` footprint is a **convex hull**: L-shaped/concave footprints must pass an explicit polygon or the slab will bridge the notch.
- `Opening.offset_m` is measured along the wall **arclength** (matters for arcs), from endpoint `a` to the opening's *near edge* — not its center.
- A positive-bulge wall on a CCW perimeter bows **into** the room (left of travel); use negative bulge for outward bay windows on CCW outlines.
- `Building.world_aabb()` is conservative (hull + thickness padding, tessellation at 4 chords/quarter), not tight — fine for lighting invalidation, wrong tool for collision.
- Don't construct `Storey` directly — `Building.add_storey` wires the back-reference that element-id allocation needs (`Storey.from_dict` is the only other sanctioned path).
- Meshing joins walls **butt-to-butt** (each wall is offset along its *own* centerline only — no cross-wall miter at shared corners; the general miter solver is Iteration 3). Iteration 2 closes the resulting corner gap with a **filler post** per shared node: `meshing._add_corner_fillers` extrudes the convex hull of the incident walls' offset corner points over the shared wall band, so corners read as solid without mitering. Lone (free) wall ends keep their butt cap.
- Per-storey floor slabs use the **convex hull** of that storey's wall centerlines, **padded outward to the outer wall faces** (half the thickest wall, via `model._pad_hull_outward`) so the floor edge meets the wall instead of stopping `thickness/2` short (Iteration 2 floor↔wall seam fix). The hull is still convex, so a *concave* plan's floor bridges the notch (true-outline concave slabs are Iteration 3 — same caveat as `set_foundation`).
- A wall mesh assumes its openings' `head_m` stays below the wall top — the top cap spans the full length; an opening reaching the very top would leave the cap floating. Keep `head_m < height_m`.
- Pitched roofs (`kind != FLAT`) are generated over the footprint outline's **ridge-aligned bounding rectangle**, not its true outline — an L-shaped/concave plan gets a roof over its bounding box (same convex caveat as the slabs; true-outline roofs are Iteration 3). Eaves are anchored at the footprint perimeter at the wall-top height and the ridge rises by `halfspan·tan(pitch)`; rotating `ridge_dir_rad` over a non-square footprint changes the span and therefore the ridge height. `overhang_m` projects eaves outward-and-down along each plane's own slope (so eave undersides sit below the wall top).
- `mesh_building` emits **building-local** positions (unlike terrain chunk meshes, which are world-space at the chunk origin). The renderer MUST apply `building.position`/`rotation` as a node transform, and `building.vert` MUST compute `v_world` via `p3d_ModelMatrix` or lighting samples the wrong world cell.

## Known Limitations (v1)
This is the **first iteration**. The list below is the canonical handoff for the agent that takes the next pass — every item is a deliberate v1 cut, not an accidental bug. (Owner/other agents: append your own findings under "Owner-observed", below.)

1. **No light occlusion — buildings don't shadow or bounce GI.** `occlusion.BuildingOccupancyRasterizer.rasterize_occupancy` is a documented no-op, so buildings are *lit* but cast no shadows and contribute no GI; interiors only darken through openings. Enabling it needs a thread-safe immutable geometry snapshot handed to the async cascade-assembly worker (the intended wall/slab voxelization is spelled out in `occlusion.py`'s docstring). **This is the single biggest visual gap.**
2. **Butt-to-butt wall joins — no cross-wall corner miter.** Each wall is offset along *its own* centerline only; at shared corners thick walls overlap. **Iteration 2 update:** the corner *gap* is now closed by a filler post per shared node (`_add_corner_fillers`), so corners are watertight — but this is a fill, not a true miter (faces are not trimmed/extended to meet cleanly). The general miter solver remains Iteration 3.
3. **Convex-hull slabs & footprints.** `set_foundation`/`set_roof` (auto) and the per-storey floor slabs use the **convex hull** of the wall centerlines. Concave / L-shaped / U-shaped / courtyard plans get a slab that bridges the notch (the demo's chamfer notch is bridged this way). Workaround today: pass explicit polygons; proper fix is to triangulate the true room/footprint outlines.
4. **~~Flat roofs only.~~ Iteration 2: SHED / GABLE / HIP added.** `set_roof(kind=...)` now generates pitched roofs (`buildings/_impl/roofs.py`) with `pitch_deg`, `ridge_dir_rad`, and `overhang_m`. Remaining cuts: pitched roofs are built over the footprint's **ridge-aligned bounding rectangle** (concave/true-outline pitched roofs follow concave slabs in Iteration 3); no dormers, no fascia/gutter, no per-roof material yet beyond the per-surface work below.
5. **Stairs are data-only stubs.** `StairsStub` reserves which storeys connect and where, but meshes nothing and feeds no traversal/navmesh — there is no way to actually move between storeys.
6. **Room auto-detection is endpoint-only, single-storey, planar.** Walls must meet at shared endpoints (no mid-span T-junction splitting — author long walls as two spans). Detection runs per storey on its 2-D wall graph; there is no multi-storey volume/room-stack concept, and overlapping/self-touching plans can confuse the half-edge trace.
7. **~~One material.~~ Iteration 2: per-surface materials added.** The mesher tags every face with a `SurfaceMaterial` (`WALL`/`FLOOR`/`ROOF`/`FOUNDATION`) in `MeshArrays.face_materials`; the renderer splits the geom per id (`geometry_bridge.to_geom_node`) and binds a distinct procedural albedo each (`plaster_wall` / `wood_floor` / `roof_shingle` / `stone_foundation`). Remaining cuts: granularity is per **surface-class**, not per-wall / per-room / per-face; no UV scale/offset control; no normal/roughness maps; no trim/fascia material yet.
8. **Openings are bare rectangles.** Windows/doors are rectangular holes with flat reveals — no frames, glazing, mullions, sills-as-geometry, or arched/round tops; doors are just holes (not functional/openable). An opening's `head_m` must stay below the wall top (the top cap spans the full wall length, so a full-height opening would leave the cap floating).
9. **No interior detail.** The next storey's floor slab *is* the ceiling (no distinct ceiling surface), and rooms are polygons + tags only — no baseboards, fixtures, or furniture (furnishing is the whole point of tracking rooms, but it's future work).
10. **No collision / physics.** `world_aabb()` is a conservative broad-phase box for lighting invalidation only; there is no per-wall collision geometry, so the player passes straight through buildings.
11. **No procedural generation yet.** The `BuildingDef.generate(rng, **params)` seam exists, but only `DemoHouseDef` (a fixed hand-authored layout) is implemented — there is no tag/description→building generator and no settlement/town placement.
12. **Variable wall height leaves open tops by design.** A wall shorter than the storey band (e.g. the demo's 1.1 m half-wall) is open above — there's no auto-cap to the ceiling/roof; that's intended for open-plan partitions but means you can't yet make a "wall with a gap" cleanly.

### Owner-observed (fill in)
_Reserved for the owner's and other agents' additional findings — add bullets here._

## Roadmap

### Iteration 2 — IN PROGRESS (branch `feature/procedural-buildings`)
The owner-selected scope for this pass (everything else below is deferred to Iteration 3+):
- **Gap/seam fixing** (limitations 2 & 12, partial): close the visible gaps and overlaps where building *sections* meet — wall→wall at shared corners, wall→floor-slab, and wall-top→roof. This is the "make the existing demo house look solid" pass, not the full general miter solver (that is Iteration 3).
- **Pitched roofs** (limitation 4): gable / hip / shed roof generators over the footprint, replacing flat-only.
- **Per-surface materials** (limitation 7): distinct material ids per surface class (wall / floor / roof / trim / foundation) instead of the single `plaster_wall` albedo, so roofs and floors read differently from walls.

### Iteration 3 — PLANNED (owner-specified 2026-06-16, record so it isn't forgotten)
The headline theme is **authoring + generation tooling** — making buildings something both a human and a coding/AI harness can create and edit:
- **Editor UI for buildings**: in-engine editor tooling to place, edit, and refine buildings *by hand* (walls, openings, storeys, roofs, materials) — see the editor system docs (`devtools` / editor harness) for the host.
- **Harness editing path**: a stable, scriptable API surface so a coding harness (e.g. Claude Code) can author/edit buildings programmatically — the imperative authoring API exists today; this iteration hardens it into the canonical harness-facing contract.
- **Procedural building generation**: tag/description → building generator (limitation 11), runnable BOTH from a coding harness AND from inside the editor UI itself (a "generate a building like X" action). This is the `BuildingDef.generate(rng, **params)` seam finally implemented beyond `DemoHouseDef`.

### Iteration 3+ backlog (unscheduled, ordered roughly by impact)
- **True-outline (concave) slabs** (limitation 3): triangulate actual room/footprint polygons; support L-shapes, courtyards, atria. (Deferred from Iteration 2 by owner.)
- **Cross-wall corner miter** (limitation 2): the *general* mitered-join solver at shared nodes (Iteration 2 only closes the seams cosmetically; this does it properly for arbitrary plans).
- **Wire up light occlusion** (limitation 1): snapshot-safe geometry → cascade worker → buildings shadow + bounce GI. Highest visual payoff; needs the thread-safe immutable geometry snapshot.
- **Procedural room/furniture placer** (limitation 9): the system that consumes `Room` tags to furnish interiors — the reason rooms are first-class objects. Pairs with the Iteration 3 generator.
- **Real stairs** (limitation 5): stair-run geometry + storey traversal.
- **Richer openings** (limitation 8): frames, glazing, arched tops, functional doors, room-connectivity/portal graph.
- **Collision meshes** (limitation 10): per-wall collision for the player/physics.
- **T-junction room detection** (limitation 6): auto-split mid-span wall intersections.
