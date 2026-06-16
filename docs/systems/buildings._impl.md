# buildings._impl — System Doc
keywords: buildings _impl, private implementation, storey, demo house, DemoHouseDef, Storey, one-public-class-per-module, split, implementation detail, seams, corner filler, filler post, corner_filler_polys, wall junction, gap, seam, Foundation, RoofSlab, soup, Soup, triangle soup, add_prism, prism, slab, add_roof, roofs, pitched roof, gable, hip, shed, ridge, eave, overhang, RoofKind

> Private sub-package of `fire_engine/buildings/`; all public symbols are re-exported from
> the parent package.  Nothing here is part of the public API.

## Role
Private implementation helpers for the `buildings` package — not a public API.  Modules in
this sub-package were split out of the parent package's modules to satisfy the
one-public-class-per-module (Hard Rule 9) and ≤500-lines-per-module (Hard Rule 8) limits.
Every public symbol defined here is re-exported from the corresponding parent module so
callers never import from `_impl` directly.  This package deliberately does NOT export
anything from its `__init__.py`.

## Public API
This sub-package has no public API surface.  All symbols are re-exported from the parent
(or are package-internal helpers the parent modules import directly, exactly as
`model.py` imports `Storey` from here):
- `Storey` — defined in `buildings/_impl/storey.py`, re-exported from `buildings/model.py`.
- `DemoHouseDef` — defined in `buildings/_impl/demo_house.py`, re-exported from `buildings/defs.py`.
- `Foundation`, `RoofSlab` — defined in `buildings/_impl/types.py`, re-exported from `buildings/types.py`.
- `corner_filler_polys` — defined in `buildings/_impl/seams.py`; **internal helper** (not public,
  not re-exported), imported by `buildings/meshing.py` to close the gap at shared wall corners.
  Returns one filler-post polygon + band height per junction node (the convex hull of the
  incident walls' ±thickness/2 offset corner points); the mesher extrudes each over the wall
  band so butt-jointed corners read as solid without the full cross-wall miter (Iteration 3).
- `Soup` — defined in `buildings/_impl/soup.py`; **internal** triangle-soup accumulator shared by
  the wall mesher and the roof generator. Methods: `add_quads` / `add_tris` (low level, auto-flip
  winding to a supplied outward normal), `add_slab` (flat horizontal solid), `add_prism` (a planar
  top polygon dropped vertically by a constant thickness — one pitched-roof panel with real depth),
  `build` → `MeshArrays`. `buildings/meshing.py` keeps a `_Soup = Soup` back-compat alias.
- `add_roof` — defined in `buildings/_impl/roofs.py`; **internal** pitched-roof generator. Dispatches
  on `RoofKind`: `FLAT` → flat slab; `SHED` (one mono-pitch plane + side/end infill); `GABLE` (two
  planes + central ridge + gable-end infill triangles); `HIP` (two trapezoids + two end triangles on
  a shortened ridge). Built over the footprint's ridge-aligned bounding rectangle; eaves anchored at
  the wall-top height, ridge raised by `halfspan·tan(pitch)`, `overhang_m` projecting eaves down-slope.

## Imports Allowed
Same as the parent `buildings` package: `procedural`, `terrain`, `core`
(ARCHITECTURE.md §4a.2).  Never panda3d.

## Events
Published: none.
Subscribed: none.

## Units & Invariants
All units match the parent `buildings` package (plan space in building-local meters, Z-up,
quaternions only, element ids are per-building monotonic ints).

## Examples
Do not import from this sub-package directly.  Use the parent package:

```python
# Correct — via the parent package:
from fire_engine.buildings import Storey
from fire_engine.buildings.defs import DemoHouseDef

# Wrong — never import from _impl directly:
# from fire_engine.buildings._impl.storey import Storey  # forbidden
```

## Gotchas
- Import paths through `_impl` are implementation details and may change without notice;
  always use the parent package's exports.
- The `Storey` class requires a `Building` back-reference (wired by `Building.add_storey`);
  constructing it directly bypasses element-id allocation — use `Building.add_storey` or
  `Storey.from_dict(building, d)` only.
