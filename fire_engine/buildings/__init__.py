"""
Building Manager API — free-form floorplan buildings (ARCHITECTURE.md §5.7).

Buildings are **not** voxel-aligned: each :class:`Building` has one world
transform (position + quaternion, arbitrary rotation on any axis) and stacks
per-storey 2-D floorplans whose walls are straight segments or circular arcs
(DXF bulge convention) with real thickness and parametric window/door
openings.  Rooms are first-class objects (auto-detected from wall topology or
explicitly authored) so future systems can procedurally furnish them.  See
``docs/systems/buildings.md`` and the DECISIONS.md entry of 2026-06-12 for
the design rationale.

This package is fully headless (numpy only — no panda3d; Hard Rule 1).
Rendering happens in ``world/building_renderer.py``; light occlusion plugs
into ``lighting/volume.py`` through the structural
``GeometryOccupancyProvider`` protocol.

Imports allowed (ARCHITECTURE.md §4a.2): procedural, terrain, core.

Example
-------
    from fire_engine.core.config import Config
    from fire_engine.core.math3d import Vec3, Quat
    from fire_engine.buildings import Building, BuildingDefaults, OpeningKind

    defaults = BuildingDefaults.from_config(Config())
    b = Building(name="hut", position=Vec3(0, 0, 8),
                 rotation=Quat.identity(), defaults=defaults)
    s0 = b.add_storey()
    w = s0.add_wall((0.0, 0.0), (6.0, 0.0))
    s0.add_opening(w.id, OpeningKind.DOOR, offset_m=2.5, width_m=0.9,
                   head_m=2.0)
    b.set_foundation()
"""

from fire_engine.buildings.model import (
    Building,
    BuildingDefaults,
    Foundation,
    Opening,
    OpeningKind,
    RoofSlab,
    Room,
    StairsStub,
    Storey,
    Wall,
    WallKind,
)

__all__ = [
    "Building",
    "BuildingDefaults",
    "Foundation",
    "Opening",
    "OpeningKind",
    "RoofSlab",
    "Room",
    "StairsStub",
    "Storey",
    "Wall",
    "WallKind",
]
