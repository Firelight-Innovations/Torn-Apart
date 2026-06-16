"""
buildings/_impl/demo_house.py — DemoHouseDef procedural building definition.

Split from buildings/defs.py to satisfy the one-public-class-per-module limit.
Re-exported from ``fire_engine.buildings.defs`` so all existing import paths
remain valid.

Docs: docs/systems/buildings.md
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from fire_engine.buildings.defs import BuildingDef
from fire_engine.buildings.enums import OpeningKind
from fire_engine.buildings.model import Building, BuildingDefaults
from fire_engine.core.config import Config
from fire_engine.core.math3d import Quat, Vec3
from fire_engine.procedural.defs import register_def

__all__ = ["DemoHouseDef"]


@register_def
class DemoHouseDef(BuildingDef):
    """
    Two-storey ~12×8 m demo house — the building-system feature showcase.

    This is the in-game evaluation build: a single procedural house that
    exercises **every** v1 capability so the whole system can be judged at a
    glance.  Feature checklist (what to look for in-game):

    - **Free-form / non-orthogonal walls** — the south-west corner is a
      45° chamfer wall (with its own small window), not a right angle.
    - **Curved walls (bulge arcs)** — the east elevation bows out into a bay
      on both storeys (``bulge=-0.4``); storey 0's bay carries a curved window.
    - **Variable wall thickness** — exterior walls are 0.4 m, interior
      partitions 0.15 m.
    - **Variable wall height** — the east interior partition is a 1.1 m
      open-plan half-wall (``height_m`` override) splitting the great room
      from the kitchen without reaching the ceiling.
    - **Doors & windows with reveals** — one exterior door (south), two
      interior doors through the spine wall, and windows of differing
      width / sill / head on most elevations.
    - **Rooms as first-class objects** — storey 0 auto-detects **three**
      rooms (living / dining / kitchen) from wall topology; storey 1 declares
      one **explicit** tagged room ("loft").
    - **Multiple storeys, variable storey height** — ground 3.5 m, upper
      2.5 m, stacked over a single floor slab each.
    - **Foundation, floor slabs, flat roof** — auto convex-hull foundation
      (0.5 m) and a flat roof slab capping the upper storey.
    - **Stairs** — one ``StairsStub`` reserving the run up from the living
      room (data only in v1).
    - **Arbitrary-axis rotation** — yawed ~18° about world +Z to prove the
      node transform (the model supports any-axis quaternions; a grounded
      yaw is what reads sensibly in-world).

    Registered name
    ---------------
    ``"building_demo_house"``

    Params
    ------
    position : (x, y, z) — world origin (default ``(-26, 6, ground_z)`` — on
        the open ground one chunk west of the demo grass patch, clear of the
        grass/flower/bush/tree zones so the building is judged on bare terrain
        rather than poking through groundcover).
    ground_z : float — local z=0 world height when ``position`` is omitted
        (default 8.0; main.py passes ``Config.ground_height_m``).
    yaw_deg  : float — yaw about world +Z in degrees (default 18).
    """

    name = "building_demo_house"

    # Plan dimensions (meters, building-local).  Width along +x, depth +y.
    _W = 12.0  # overall width  (x: 0 .. 12)
    _D = 8.0  # overall depth  (y: 0 .. 8)
    _XSPINE = 6.0  # x of the vertical spine wall
    _YSPLIT = 4.0  # y of the east half-wall split (kitchen / dining)
    _CHAMFER = 1.5  # 45° corner cut at the south-west corner
    _EXT_T = 0.4  # exterior wall thickness
    _INT_T = 0.15  # interior partition thickness

    def generate(self, rng: np.random.Generator, **params: Any) -> Building:
        ground_z = float(params.get("ground_z", 8.0))
        pos = params.get("position", (-26.0, 6.0, ground_z))
        yaw = math.radians(float(params.get("yaw_deg", 18.0)))
        cfg = Config()
        defaults = BuildingDefaults.from_config(cfg)

        W, D = self._W, self._D
        xs, ys = self._XSPINE, self._YSPLIT
        ch = self._CHAMFER
        ext, inte = self._EXT_T, self._INT_T

        b = Building(
            name="demo_house",
            position=Vec3(float(pos[0]), float(pos[1]), float(pos[2])),
            rotation=Quat.from_axis_angle(Vec3.UP, yaw),
            defaults=defaults,
            tags=["demo", "rural", "cottage", "showcase"],
        )

        # ---- storey 0: 3 rooms (living + dining + kitchen) ----------------
        # Taller ground storey (3.5 m) to contrast with the 2.5 m upper one.
        s0 = b.add_storey(height_m=3.5)
        # Perimeter (CCW), split at every spine/divider junction node so room
        # detection sees walls meeting only at shared endpoints (no T-joins).
        sw = s0.add_wall((ch, 0), (xs, 0), thickness_m=ext)  # S-W (door)
        se = s0.add_wall((xs, 0), (W, 0), thickness_m=ext)  # S-E (window)
        bay0 = s0.add_wall(
            (W, 0),
            (W, ys),
            bulge=-0.4,  # E bay (curved)
            thickness_m=ext,
        )
        s0.add_wall((W, ys), (W, D), thickness_m=ext)  # E upper (window)
        s0.add_wall((W, D), (xs, D), thickness_m=ext)  # N-E
        nw = s0.add_wall((xs, D), (0, D), thickness_m=ext)  # N-W (window)
        west = s0.add_wall((0, D), (0, ch), thickness_m=ext)  # W (window)
        cham = s0.add_wall((0, ch), (ch, 0), thickness_m=ext)  # SW chamfer
        # Interior spine (thin) — split at the east half-wall node (xs, ys).
        spine_lo = s0.add_wall((xs, 0), (xs, ys), thickness_m=inte)
        spine_hi = s0.add_wall((xs, ys), (xs, D), thickness_m=inte)
        # East half-wall — a LOW open-plan partition (variable height).
        s0.add_wall((xs, ys), (W, ys), thickness_m=inte, height_m=1.1)

        # Openings: 1 exterior door, 2 interior doors, varied windows.
        s0.add_opening(sw.id, OpeningKind.DOOR, offset_m=1.6, width_m=1.0, head_m=2.2)
        s0.add_opening(se.id, OpeningKind.WINDOW, offset_m=2.4, width_m=1.6, sill_m=0.9, head_m=2.3)
        s0.add_opening(
            bay0.id, OpeningKind.WINDOW, offset_m=1.5, width_m=1.4, sill_m=0.9, head_m=2.0
        )  # curved-wall window
        s0.add_opening(nw.id, OpeningKind.WINDOW, offset_m=2.0, width_m=1.8, sill_m=1.0, head_m=2.4)
        s0.add_opening(
            west.id, OpeningKind.WINDOW, offset_m=2.6, width_m=1.2, sill_m=1.0, head_m=2.2
        )
        s0.add_opening(
            cham.id, OpeningKind.WINDOW, offset_m=0.6, width_m=0.8, sill_m=1.0, head_m=1.9
        )  # window on angled wall
        s0.add_opening(
            spine_lo.id, OpeningKind.DOOR, offset_m=1.4, width_m=0.9, head_m=2.1
        )  # living -> dining
        s0.add_opening(
            spine_hi.id, OpeningKind.DOOR, offset_m=1.4, width_m=0.9, head_m=2.1
        )  # living -> kitchen

        rooms = s0.detect_rooms(
            snap_eps_m=cfg.building_snap_eps_m,
            arc_segments_per_quarter=cfg.building_arc_segments_per_quarter,
        )
        # Tag the detected rooms by centroid so future furnishing has labels.
        for room in rooms:
            cx, cy = room.centroid()
            if cx < xs:
                room.tag = "living"
            elif cy < ys:
                room.tag = "dining"
            else:
                room.tag = "kitchen"
        s0.add_stairs(storey_to=1, anchor=(2.0, D - 2.0), direction_rad=0.0, width_m=1.1)

        # ---- storey 1: same shell (matching bay + chamfer) + explicit room
        s1 = b.add_storey(height_m=2.5)
        s1.add_wall((ch, 0), (W, 0), thickness_m=ext)  # south
        sbay = s1.add_wall((W, 0), (W, ys), bulge=-0.4, thickness_m=ext)
        s1.add_wall((W, ys), (W, D), thickness_m=ext)  # east upper
        n1 = s1.add_wall((W, D), (0, D), thickness_m=ext)  # north (window)
        w1 = s1.add_wall((0, D), (0, ch), thickness_m=ext)  # west (window)
        s1.add_wall((0, ch), (ch, 0), thickness_m=ext)  # chamfer
        s1.add_opening(
            sbay.id, OpeningKind.WINDOW, offset_m=1.5, width_m=1.2, sill_m=0.8, head_m=2.0
        )
        s1.add_opening(n1.id, OpeningKind.WINDOW, offset_m=4.0, width_m=2.0, sill_m=0.8, head_m=2.1)
        s1.add_opening(w1.id, OpeningKind.WINDOW, offset_m=2.6, width_m=1.2, sill_m=0.8, head_m=2.0)
        # Explicit (authored) room — proves add_room alongside auto-detection.
        s1.add_room(
            [(0.5, 0.5), (W - 0.5, 0.5), (W - 0.5, D - 0.5), (0.5, D - 0.5)],
            tag="loft",
            meta={"furnish": "open"},
        )

        b.set_foundation()  # auto hull, 0.5 m deep
        b.set_roof()  # flat roof slab caps top
        return b
