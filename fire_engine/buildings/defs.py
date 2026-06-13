"""
buildings/defs.py — procedural building definitions (the tag→building seam).

A ``BuildingDef`` is a :class:`~fire_engine.procedural.defs.ProceduralDef` whose
``generate`` returns a fully-authored :class:`~fire_engine.buildings.model.Building`
(every generator action is one of the imperative authoring-API calls).  This is
the slot the future tag/description→building generator plugs into: it will read
``params`` (footprint, storey count, room program, style tags) and emit walls /
openings / rooms / slabs accordingly.  v1 ships one concrete proof —
``DemoHouseDef`` ("building_demo_house") — exercising the whole pipeline
(curved bay wall, split perimeter so room detection finds two rooms, a door and
windows with reveals, two storeys, auto foundation + flat roof, a stair stub).

Determinism: ``generate`` uses no RNG for the demo (a fixed layout), so
``get("building_demo_house")`` twice yields byte-identical ``to_dict()`` — the
property the manager's clone-on-add and the save round-trip both rely on.

Imports allowed: procedural, core (Hard Rule 1 — no panda3d).
"""

from __future__ import annotations

import math

import numpy as np

from fire_engine.buildings.model import Building, BuildingDefaults, OpeningKind
from fire_engine.core.config import Config
from fire_engine.core.math3d import Quat, Vec3
from fire_engine.procedural.defs import ProceduralDef, register_def

__all__ = ["BuildingDef", "DemoHouseDef"]


class BuildingDef(ProceduralDef):
    """
    Abstract base for procedural buildings: ``generate(rng, **params) -> Building``.

    Subclasses author a building through the imperative API on
    :class:`Building` / :class:`Storey` and return it.  The registry caches the
    result by ``(name, params digest)``; callers hand it to
    ``BuildingManager.add`` (which clones it and assigns a world id), so a
    ``generate`` MUST be a pure function of ``rng`` + ``params`` and must not
    keep references to its return value.

    Contract for the (future) tag→building generator
    -----------------------------------------------
    - Read placement from ``params`` (``position``, ``ground_z``, ``yaw_rad``)
      and program from style ``params`` (``footprint``, ``storeys``, room
      ``tags``).  Use ``rng`` (seeded by the registry) for every random choice —
      never ``random``/unseeded numpy (Hard Rule 2).
    - Build dimensions from :meth:`BuildingDefaults.from_config` (config is the
      single number source), overridable per call via ``params``.
    - Return the ``Building``; the manager assigns its id, the renderer meshes
      and draws it, and the lighting provider (future) voxelizes it.

    Example
    -------
        from fire_engine.procedural import get
        house = get("building_demo_house", ground_z=8.0)   # a Building
    """

    def generate(self, rng: np.random.Generator, **params) -> Building:  # noqa: D401
        raise NotImplementedError(
            "BuildingDef subclasses must implement generate() -> Building. "
            "See ARCHITECTURE.md §5.7 and docs/systems/buildings.md.")


@register_def
class DemoHouseDef(BuildingDef):
    """
    Two-storey ~8×6 m demo cottage — the building-system vertical slice.

    Storey 0 has a curved (bulged) east bay wall and a central divider with the
    perimeter split at the divider node, so :meth:`Storey.detect_rooms` finds
    two rooms; a 0.9 m door and two 1.2 m windows (with sill/head reveals) cut
    the walls.  Storey 1 is a plain perimeter with two windows.  Auto convex-
    hull foundation (0.5 m) + flat roof slab cap it, and one ``StairsStub``
    reserves the run between storeys.  Yawed ~15° to prove the node transform.

    Registered name
    ---------------
    ``"building_demo_house"``

    Params
    ------
    position : (x, y, z) — world origin (default ``(-24, 10, ground_z)``).
    ground_z : float — local z=0 world height when ``position`` is omitted
        (default 8.0; main.py passes ``Config.ground_height_m``).
    yaw_deg  : float — yaw about world +Z in degrees (default 15).
    """

    name = "building_demo_house"

    def generate(self, rng: np.random.Generator, **params) -> Building:
        ground_z = float(params.get("ground_z", 8.0))
        pos = params.get("position", (-24.0, 10.0, ground_z))
        yaw = math.radians(float(params.get("yaw_deg", 15.0)))
        defaults = BuildingDefaults.from_config(Config())

        b = Building(
            name="demo_house",
            position=Vec3(float(pos[0]), float(pos[1]), float(pos[2])),
            rotation=Quat.from_axis_angle(Vec3.UP, yaw),
            defaults=defaults,
            tags=["demo", "rural", "cottage"])

        # --- storey 0: split perimeter + curved east bay + divider --------
        s0 = b.add_storey()
        sw = s0.add_wall((0, 0), (4, 0))            # south-west (door here)
        se = s0.add_wall((4, 0), (8, 0))            # south-east (window here)
        s0.add_wall((8, 0), (8, 6), bulge=-0.4)     # east — bows out (bay)
        s0.add_wall((8, 6), (4, 6))                 # north-east
        s0.add_wall((4, 6), (0, 6))                 # north-west
        west = s0.add_wall((0, 6), (0, 0))          # west (window here)
        s0.add_wall((4, 0), (4, 6))                 # divider (meets endpoints)
        s0.add_opening(sw.id, OpeningKind.DOOR, offset_m=1.5, width_m=0.9,
                       head_m=2.0)
        s0.add_opening(se.id, OpeningKind.WINDOW, offset_m=1.4, width_m=1.2,
                       sill_m=1.0, head_m=2.2)
        s0.add_opening(west.id, OpeningKind.WINDOW, offset_m=2.4, width_m=1.2,
                       sill_m=1.0, head_m=2.2)
        s0.detect_rooms(snap_eps_m=Config().building_snap_eps_m,
                        arc_segments_per_quarter=Config()
                        .building_arc_segments_per_quarter)
        s0.add_stairs(storey_to=1, anchor=(2.0, 5.0),
                      direction_rad=0.0, width_m=1.0)

        # --- storey 1: plain perimeter + two windows ----------------------
        s1 = b.add_storey()
        s1.add_wall((0, 0), (8, 0))
        e1 = s1.add_wall((8, 0), (8, 6))
        s1.add_wall((8, 6), (0, 6))
        w1 = s1.add_wall((0, 6), (0, 0))
        s1.add_opening(w1.id, OpeningKind.WINDOW, offset_m=2.4, width_m=1.2,
                       sill_m=1.0, head_m=2.2)
        s1.add_opening(e1.id, OpeningKind.WINDOW, offset_m=2.4, width_m=1.2,
                       sill_m=1.0, head_m=2.2)

        b.set_foundation()                          # auto hull, 0.5 m deep
        b.set_roof()                                # flat roof slab
        return b
