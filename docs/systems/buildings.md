# buildings — System Doc
keywords: building, buildings, floorplan, floor plan, storey, story, floor, wall, walls, curved wall, arc, bulge, segment, thickness, opening, window, door, sill, head, lintel, room, rooms, room detection, interior, foundation, slab, roof, flat roof, stairs, stair stub, BuildingDefaults, Building, Storey, Wall, Opening, Room, StairsStub, Foundation, RoofSlab, WallKind, OpeningKind, add_wall, add_opening, add_room, add_storey, add_stairs, set_foundation, set_roof, detect_rooms, storey_base_z, world_aabb, to_dict, from_dict, allocate_eid, element id, plan space, building-local, free-form, non-voxel, Sims, Paralives, BuildingDef, procedural building, house, settlement

## Role
Free-form floorplan buildings (ARCHITECTURE.md §5.7) — the data model and imperative authoring API for structures that are deliberately **not** voxel-aligned. A `Building` has one world transform (position `Vec3` + rotation `Quat`, arbitrary axis) and stacks per-storey 2-D floorplans: walls are straight segments or circular arcs (single `bulge` scalar) with real thickness and parametric window/door openings; rooms are first-class objects (explicitly authored, or auto-detected from wall topology) so future systems can procedurally furnish each room; foundations, floor/ceiling slabs and flat roofs complete the envelope; stairs exist as data stubs only (no geometry yet). This package is the base layer the future tag→building procedural generator (`BuildingDef`) will drive — every generator action is one of these API calls. It deliberately does NOT render (that is `world/building_renderer.py`), does NOT import panda3d (fully headless), and does NOT yet mesh stairs or pitched roofs.

## Public API
- `BuildingDefaults` — frozen per-building fallback dimensions; build via `BuildingDefaults.from_config(cfg)` (single number source = the `Config` `building_*` fields). Fields: `storey_height_m`, `wall_thickness_m`, `slab_thickness_m`, `foundation_depth_m`.
- `Building(name, position, rotation, defaults, tags=None)` — the root object. `add_storey(height_m=None, slab_m=None)`, `set_foundation(polygon=None, depth_m=None)` (None polygon → automatic convex-hull footprint of storey-0 walls padded by half the thickest wall), `set_roof(polygon=None, thickness_m=None)`, `storey_base_z(index)`, `total_height_m`, `world_aabb()` (conservative world AABB for lighting invalidation), `allocate_eid()`, `to_dict()`/`from_dict()` (exact round-trip; the Saveable payload).
- `Storey` — one floor; create only via `Building.add_storey`. `add_wall(a, b, *, bulge=0.0, thickness_m=None, height_m=None) -> Wall`, `add_opening(wall_id, kind, *, offset_m, width_m, head_m, sill_m=0.0) -> Opening` (validates length/band/sill<head), `add_room(polygon, tag="", meta=None) -> Room` (winding normalized to CCW), `add_stairs(storey_to, anchor, direction_rad, width_m) -> StairsStub`, `detect_rooms(*, snap_eps_m, arc_segments_per_quarter)` (auto-detection; see fire_engine/buildings/rooms.py).
- `Wall` — one span. `kind` (`WallKind.SEGMENT`/`ARC`, derived from `bulge`), `chord_m()`, `length_m()`, `arc_params() -> (center, radius, start_angle, sweep)`, `tessellate(arc_segments_per_quarter) -> float64 (P, 2)` centerline polyline with exact endpoints.
- `Opening` — rectangular cutout in the wall's local `(s, z)` frame (`s` = arclength from endpoint `a`; `z` above the floor-slab top). Fields `kind` (`OpeningKind.WINDOW`/`DOOR`), `offset_m`, `width_m`, `sill_m`, `head_m`.
- `Room` — enclosed plan region: `polygon` (CCW `float64 (N, 2)`), `tag`, `meta`, `auto`; `area_m2()`, `centroid()`.
- `StairsStub`, `Foundation`, `RoofSlab` — data carriers (see docstrings).
- `WallKind`, `OpeningKind` — enums (str values in dicts).
- All model types have `to_dict()`/`from_dict()` over plain primitives (no numpy in the dicts, no pickle).

## Imports Allowed
procedural, terrain, core (ARCHITECTURE.md §4a.2). **Never panda3d** — rendering goes through `world/building_renderer.py`; light occlusion through the structural `GeometryOccupancyProvider` protocol in `lighting/volume.py` (no import in either direction).

## Events
Published: none yet from the model layer (the manager will publish `BuildingChangedEvent` on add/modify/remove — see `fire_engine/buildings/manager.py`).
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

## Gotchas
- Room auto-detection requires walls to MEET AT ENDPOINTS (within `building_snap_eps_m`, 1 cm): v1 does not split mid-span T-junctions — author the long wall as two spans sharing the junction vertex.
- The automatic `set_foundation()`/`set_roof()` footprint is a **convex hull**: L-shaped/concave footprints must pass an explicit polygon or the slab will bridge the notch.
- `Opening.offset_m` is measured along the wall **arclength** (matters for arcs), from endpoint `a` to the opening's *near edge* — not its center.
- A positive-bulge wall on a CCW perimeter bows **into** the room (left of travel); use negative bulge for outward bay windows on CCW outlines.
- `Building.world_aabb()` is conservative (hull + thickness padding, tessellation at 4 chords/quarter), not tight — fine for lighting invalidation, wrong tool for collision.
- Don't construct `Storey` directly — `Building.add_storey` wires the back-reference that element-id allocation needs (`Storey.from_dict` is the only other sanctioned path).
