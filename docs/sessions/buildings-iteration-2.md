# Session — Buildings Iteration 2 (seams, pitched roofs, per-surface materials)

date: 2026-06-16
branch: `feature/procedural-buildings` (off master `11fa2fe`) — PR open, NOT yet merged
keywords: buildings, iteration 2, seam, gap, corner filler, pitched roof, gable, hip, shed,
RoofKind, per-surface material, SurfaceMaterial, face_materials, roof_shingle, wood_floor,
stone_foundation, Soup, add_prism, posterise, building_renderer, material_textures

## What shipped
Three owner-scoped workstreams refining the v1 free-form building system
(`docs/systems/buildings.md`). All headless-tested; full suite green (4727 passed,
1 skipped); standards gate 9/9; ruff + mypy clean.

### 1. Gap/seam fixing — commit `441c48c`
- **Floor↔wall seam:** per-storey floor slabs now pad the wall-centerline convex
  hull out to the **outer wall faces** (`model._pad_hull_outward`, extracted from
  `_auto_footprint`), so the floor edge meets the wall instead of stopping
  `thickness/2` short.
- **Corner gaps:** shared wall corners get a filler post per junction node
  (`buildings/_impl/seams.corner_filler_polys`): the convex hull of the incident
  walls' ±thickness/2 offset corner points, extruded over the shortest incident
  wall band. Watertight, never overhangs a wall's own outer face. This is a
  *gap-closer*, NOT the general cross-wall miter (that's iteration 3).

### 2. Pitched roofs — commit `061d8c7`
- `RoofKind` enum: `FLAT` / `SHED` / `GABLE` / `HIP`. `RoofSlab` gained
  `kind`, `pitch_deg`, `ridge_dir_rad`, `overhang_m` (back-compatible dicts —
  Iteration-1 saves load as `FLAT`).
- `set_roof(kind=, pitch_deg=, ridge_dir_rad=, overhang_m=)` authors them.
- `buildings/_impl/roofs.add_roof` builds planes over the footprint outline's
  **ridge-aligned bounding rectangle**: eaves anchored at the wall-top height,
  ridge raised by `halfspan·tan(pitch)`, overhang projected down each plane's
  own slope; gable/shed get vertical infill so the envelope closes at the walls.
- **Refactor:** the triangle-soup accumulator moved out of `meshing.py` to
  `buildings/_impl/soup.py` (class `Soup`) and gained a sloped `add_prism`
  primitive shared by walls + roofs. `meshing.py` keeps a `_Soup = Soup` alias
  and stays under the 500-line cap.
- Demo/showcase house now wears a 32° gable roof.

### 3. Per-surface materials — commit `88cf10c`
- `SurfaceMaterial` IntEnum (`WALL=0` / `FLOOR=1` / `ROOF=2` / `FOUNDATION=3`).
  The mesher tags every face into `MeshArrays.face_materials` (uint8, or `None`
  when uniform). `Soup.add_*` gained a `material` arg; floors→FLOOR,
  foundation→FOUNDATION, roof planes→ROOF, walls/corner-fillers/gable-infill→WALL.
- `BuildingRendererComponent` splits the geom per material via
  `to_geom_node(mesh, material_textures=...)` and binds a distinct procedural
  albedo each — **no shader change** (this is the terrain material-split idiom).
  Wall albedo is the node-level fallback so a missing content def degrades
  gracefully (`_load_material_textures`).
- New procedural textures in `procedural/textures/ground/` (each with a
  determinism test mirror): `roof_shingle` (slate courses), `wood_floor` (timber
  planks), `stone_foundation` (running-bond rubble). Shared `posterise()` lifted
  into `procedural/textures/base.py`.
  Preview: `python tools/preview_texture.py roof_shingle` (etc.).

## Verified
- Full headless suite: **4727 passed, 1 skipped**.
- Standards gate (`tests/standards/`): **9 passed, 1 deselected** — lint, type,
  structure (≤500 lines / module), docs links, git-hygiene all green.
- Texture previews eyeballed — read clearly as slate / wood / stone.

## NOT verified — do this first in iteration 3
- **In-game GPU appearance is UNVERIFIED.** The render path (per-material binding,
  pitched-roof geometry) can't be headless-tested (Hard Rule 1 — render/ imports
  panda3d). Launch `python main.py` and look at the demo house: it spawns one
  chunk **west** of the demo grass patch (`[debug] debug_demo_building`, on by
  default). Expect plaster walls, a slate gable roof, wood floors visible through
  openings, and a stone foundation course. Watch for: material geom split working
  (4 distinct textures, not all plaster), roof winding/back-faces, and the corner
  fillers not poking out.

## Gotchas / lessons for the next agent
- **Worktree testing:** this branch lives in the git worktree
  `C:/Users/bjsea/Documents/Projects/torn-apart-buildings`, which has **no
  `.venv`**. Run the main checkout's interpreter with `PYTHONPATH` set to the
  worktree, or `import fire_engine` resolves to the editable-installed *main* tree:
  `PYTHONPATH=<worktree> "<main>/.venv/Scripts/python.exe" -m pytest ...`.
- **git-hygiene gate** can falsely flag a sibling worktree branch that sits at
  master HEAD with no commits as "stale merged" — it self-resolves once that
  branch has commits. Don't delete the owner's active worktree branches.
- `triangulate_polygon` accepts either winding and emits CCW; `Soup.add_*`
  auto-flip winding to the supplied outward normal, so you pass an approximate
  outward normal and don't worry about vertex order.
- Pitched roofs use the footprint **bounding rectangle**, not the true outline —
  L-shaped/concave plans get a roof over their bbox (same convex caveat as the
  slabs). Rotating `ridge_dir_rad` over a non-square footprint changes the span
  and therefore the ridge height (caught a wrong test assumption this way).

## Iteration 3 backlog (owner-specified — see `docs/systems/buildings.md` Roadmap)
Headline theme: **authoring + generation tooling.**
1. **Editor UI for buildings** — place/edit walls/openings/storeys/roofs/materials
   by hand in the in-engine editor (`devtools`/editor harness is the host).
2. **Harness editing path** — harden the imperative authoring API into the
   canonical scriptable contract a coding harness (Claude Code) drives.
3. **Procedural generation** — implement `BuildingDef.generate` beyond the demo:
   tag/description→building, runnable from BOTH a harness AND inside the editor.

Also queued (deferred from iter 2, ordered by impact): true-outline **concave
slabs**; the general **cross-wall miter** solver; **light-occlusion** wire-up
(`occlusion.BuildingOccupancyRasterizer` is still a no-op); **room/furniture
placer** consuming `Room` tags. Plus a follow-up: the building renderer meshes on
change-events in `late_update` on the main thread — move off-thread to satisfy
the new Hard Rule 12 (low-frequency today, not a per-frame stall).
