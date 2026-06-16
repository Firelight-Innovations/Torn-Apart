"""
buildings/model.py — Free-form floorplan building data model.

Buildings are **not** voxels.  A :class:`Building` is an authored floorplan:
one world transform (position + quaternion — arbitrary rotation on any axis),
and per-storey 2-D plans whose walls are straight segments or circular arcs
with real thickness and parametric openings (windows/doors).  Rooms are
first-class objects so future systems can procedurally furnish them.

Coordinate conventions
----------------------
- **Plan space** is building-local x/y in meters (Z-up world).  A local point
  ``(x, y, z)`` maps to world space as
  ``world = building.position + building.rotation.rotate(Vec3(x, y, z))``.
- Building-local ``z = 0`` is the **top of the foundation slab**; the
  foundation occupies ``[-foundation.depth_m, 0]``.
- Storey ``i`` spans ``[base_z, base_z + height_m]`` where ``base_z`` is the
  sum of the heights of all storeys below (see :meth:`Building.storey_base_z`).
  Its floor slab occupies ``[base_z, base_z + slab_m]``; walls span
  ``[base_z + slab_m, base_z + height_m]`` (the next storey's floor slab is
  this storey's ceiling).  The optional flat roof slab caps the top storey.

Arc walls — the bulge convention
--------------------------------
A wall is one class for both straight and curved spans, parameterised by a
single *bulge* scalar: ``|bulge| = tan(included_angle / 4)``.  The rule is
deliberately simple: **positive bulge bows the wall to the left of the a→b
direction, negative to the right**; ``bulge == 0`` is a straight segment and
``|bulge| == 1`` a semicircle (the arc's apex sits ``bulge × chord/2`` meters
off the chord midpoint).  Topology (room detection) only ever needs the
endpoints, which stay first-class; arc geometry (center/radius/sweep) is
derived on demand.

Serialisation
-------------
Every type round-trips through plain dicts of primitives via
``to_dict()``/``from_dict()`` (lists of floats, str enum values — never
pickle, never live references) so :class:`~fire_engine.buildings.manager.\
BuildingManager` can persist buildings through the ``Saveable`` protocol.
Model-layer floats are Python floats (float64); meshing converts to float32
at the GPU boundary.

Example
-------
    from fire_engine.core.config import Config
    from fire_engine.core.math3d import Vec3, Quat
    from fire_engine.buildings import Building, BuildingDefaults, OpeningKind

    defaults = BuildingDefaults.from_config(Config())
    b = Building(name="hut", position=Vec3(0, 0, 8),
                 rotation=Quat.identity(), defaults=defaults)
    s0 = b.add_storey()                      # 3.0 m storey, 0.2 m slab
    w = s0.add_wall((0, 0), (6, 0))          # straight, default 0.3 m thick
    s0.add_wall((6, 0), (6, 4), bulge=0.4)   # curved (CCW arc)
    s0.add_opening(w.id, OpeningKind.DOOR, offset_m=2.5, width_m=0.9,
                   head_m=2.0)
    b.set_foundation()                       # auto outline, default depth

Docs: docs/systems/buildings.md
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fire_engine.buildings._impl.storey import Storey
from fire_engine.buildings.enums import OpeningKind, WallKind
from fire_engine.buildings.types import (
    BuildingDefaults,
    Foundation,
    Opening,
    PlanPoint,
    RoofSlab,
    Room,
    StairsStub,
    Wall,
)
from fire_engine.core.math3d import Quat, Vec3

# Re-exports for backward-compatible imports from this module are included in __all__.
__all__ = [
    "Building",
    "BuildingDefaults",
    "Foundation",
    "Opening",
    "OpeningKind",
    "PlanPoint",
    "RoofSlab",
    "Room",
    "StairsStub",
    "Storey",
    "Wall",
    "WallKind",
]


class Building:
    """
    A free-form building: one world transform + stacked storey floorplans.

    The single authoring entry point for the building system — the future
    tag→building procedural generator emits exactly these calls.  See the
    module docstring for the coordinate/elevation conventions.

    Attributes
    ----------
    id        : int — assigned by ``BuildingManager.add`` (0 = unmanaged).
    name      : str — human/agent-readable label.
    position  : Vec3 — world position of the building origin (meters); local
                z=0 (foundation top) lands exactly here.
    rotation  : Quat — world rotation (arbitrary axis; quaternions only).
    tags      : list[str] — semantic tags for procedural generation
                ("rural", "tavern", "ruined", ...).
    defaults  : BuildingDefaults — fallback dimensions from config.
    storeys   : list[Storey] — index order, ground first.
    foundation: Foundation | None
    roof      : RoofSlab | None

    Example
    -------
    See the module docstring.

    Docs: docs/systems/buildings.md
    """

    def __init__(
        self,
        name: str,
        position: Vec3,
        rotation: Quat,
        defaults: BuildingDefaults,
        tags: list[str] | None = None,
    ) -> None:
        self.id: int = 0
        self.name = name
        self.position = position
        self.rotation = rotation
        self.tags: list[str] = list(tags or [])
        self.defaults = defaults
        self.storeys: list[Storey] = []
        self.foundation: Foundation | None = None
        self.roof: RoofSlab | None = None
        self._next_eid: int = 1

    # ------------------------------------------------------------------
    # Element ids
    # ------------------------------------------------------------------

    def allocate_eid(self) -> int:
        """
        Allocate the next per-building element id (monotonic int).

        Ids are unique within one building and stable across save/load (the
        counter itself is serialized).

        Docs: docs/systems/buildings.md
        """
        eid = self._next_eid
        self._next_eid += 1
        return eid

    # ------------------------------------------------------------------
    # Authoring API
    # ------------------------------------------------------------------

    def add_storey(self, height_m: float | None = None, slab_m: float | None = None) -> Storey:
        """
        Append a storey on top of the existing stack and return it.

        Parameters
        ----------
        height_m : float | None
            Floor-to-floor height; ``None`` → ``defaults.storey_height_m``.
        slab_m : float | None
            Floor slab thickness; ``None`` → ``defaults.slab_thickness_m``.

        Docs: docs/systems/buildings.md
        """
        storey = Storey(
            self,
            eid=self.allocate_eid(),
            index=len(self.storeys),
            height_m=(self.defaults.storey_height_m if height_m is None else float(height_m)),
            slab_m=(self.defaults.slab_thickness_m if slab_m is None else float(slab_m)),
        )
        self.storeys.append(storey)
        return storey

    def set_foundation(
        self, polygon: Any | None = None, depth_m: float | None = None
    ) -> Foundation:
        """
        Define the foundation slab (local z ``[-depth_m, 0]``).

        Parameters
        ----------
        polygon : array-like (N, 2) | None
            Simple CCW footprint polygon; ``None`` → the convex hull of
            storey 0's tessellated wall centerlines padded outward by half
            the thickest wall (an automatic footprint — concave footprints
            should pass an explicit polygon).
        depth_m : float | None
            Slab depth; ``None`` → ``defaults.foundation_depth_m``.

        Docs: docs/systems/buildings.md
        """
        poly = self._auto_footprint() if polygon is None else np.array(polygon, dtype=np.float64)
        self.foundation = Foundation(
            polygon=poly,
            depth_m=(self.defaults.foundation_depth_m if depth_m is None else float(depth_m)),
        )
        return self.foundation

    def set_roof(self, polygon: Any | None = None, thickness_m: float | None = None) -> RoofSlab:
        """
        Define the flat roof slab capping the top storey.

        Parameters
        ----------
        polygon : array-like (N, 2) | None
            Simple CCW roof outline; ``None`` → same automatic footprint as
            :meth:`set_foundation`.
        thickness_m : float | None
            Slab thickness; ``None`` → ``defaults.slab_thickness_m``.

        Docs: docs/systems/buildings.md
        """
        poly = self._auto_footprint() if polygon is None else np.array(polygon, dtype=np.float64)
        self.roof = RoofSlab(
            polygon=poly,
            thickness_m=(
                self.defaults.slab_thickness_m if thickness_m is None else float(thickness_m)
            ),
        )
        return self.roof

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def storey_base_z(self, index: int) -> float:
        """
        Local z of storey ``index``'s floor-slab bottom in meters
        (sum of the heights of all storeys below; storey 0 → 0.0).

        Docs: docs/systems/buildings.md
        """
        if not 0 <= index < len(self.storeys):
            raise IndexError(f"storey index {index} out of range")
        return float(sum(s.height_m for s in self.storeys[:index]))

    @property
    def total_height_m(self) -> float:
        """Local z of the top of the highest storey (roof slab excluded).

        Docs: docs/systems/buildings.md
        """
        return float(sum(s.height_m for s in self.storeys))

    def world_aabb(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """
        Conservative world-space AABB of the whole building in meters.

        Computed from the local extent box (all tessellated wall centerlines
        padded by half the thickest wall, plus foundation/roof polygons and
        the full z range) with its 8 corners transformed by the building's
        rotation + position.  Used for lighting invalidation
        (``BuildingChangedEvent``) and broad-phase queries.

        Docs: docs/systems/buildings.md
        """
        # Wall centerlines get padded by half the thickest wall; slab
        # polygons are already true outlines and are merged unpadded.
        wall_pts: list[np.ndarray] = []
        slab_pts: list[np.ndarray] = []
        max_thick = 0.0
        for storey in self.storeys:
            for wall in storey.walls:
                wall_pts.append(wall.tessellate(arc_segments_per_quarter=4))
                max_thick = max(max_thick, wall.thickness_m)
        if self.foundation is not None:
            slab_pts.append(self.foundation.polygon)
        if self.roof is not None:
            slab_pts.append(self.roof.polygon)
        lows: list[np.ndarray] = []
        highs: list[np.ndarray] = []
        if wall_pts:
            xy = np.concatenate(wall_pts, axis=0)
            pad = max_thick / 2.0
            lows.append(xy.min(axis=0) - pad)
            highs.append(xy.max(axis=0) + pad)
        if slab_pts:
            xy = np.concatenate(slab_pts, axis=0)
            lows.append(xy.min(axis=0))
            highs.append(xy.max(axis=0))
        if lows:
            lo_x, lo_y = np.minimum.reduce(lows)
            hi_x, hi_y = np.maximum.reduce(highs)
        else:
            lo_x = lo_y = hi_x = hi_y = 0.0
        lo_z = -(self.foundation.depth_m if self.foundation else 0.0)
        hi_z = self.total_height_m + (self.roof.thickness_m if self.roof else 0.0)
        # Transform the 8 local box corners (a fixed-size loop, not per-vertex).
        corners = [(x, y, z) for x in (lo_x, hi_x) for y in (lo_y, hi_y) for z in (lo_z, hi_z)]
        world = np.array(
            [tuple(self.position + self.rotation.rotate(Vec3(*c))) for c in corners],
            dtype=np.float64,
        )
        mn = world.min(axis=0)
        mx = world.max(axis=0)
        return (
            (float(mn[0]), float(mn[1]), float(mn[2])),
            (float(mx[0]), float(mx[1]), float(mx[2])),
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Full building spec as a dict of primitives (delta-save payload).

        Round-trips exactly: ``Building.from_dict(b.to_dict()).to_dict()
        == b.to_dict()``.

        Docs: docs/systems/buildings.md
        """
        return {
            "id": int(self.id),
            "name": str(self.name),
            "position": [self.position.x, self.position.y, self.position.z],
            "rotation": [self.rotation.w, self.rotation.x, self.rotation.y, self.rotation.z],
            "tags": list(self.tags),
            "defaults": self.defaults.to_dict(),
            "next_eid": int(self._next_eid),
            "storeys": [s.to_dict() for s in self.storeys],
            "foundation": (None if self.foundation is None else self.foundation.to_dict()),
            "roof": None if self.roof is None else self.roof.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Building:
        """Inverse of :meth:`to_dict`.

        Docs: docs/systems/buildings.md
        """
        pos = d["position"]
        rot = d["rotation"]
        building = cls(
            name=str(d["name"]),
            position=Vec3(float(pos[0]), float(pos[1]), float(pos[2])),
            rotation=Quat(float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])),
            defaults=BuildingDefaults.from_dict(d["defaults"]),
            tags=list(d.get("tags", ())),
        )
        building.id = int(d.get("id", 0))
        building.storeys = [Storey.from_dict(building, s) for s in d.get("storeys", ())]
        f = d.get("foundation")
        building.foundation = None if f is None else Foundation.from_dict(f)
        r = d.get("roof")
        building.roof = None if r is None else RoofSlab.from_dict(r)
        building._next_eid = int(d["next_eid"])
        return building

    # ------------------------------------------------------------------

    def _auto_footprint(self) -> np.ndarray:
        """
        Automatic footprint: convex hull of storey 0's tessellated wall
        centerlines, padded outward by half the thickest wall.

        Monotone-chain hull — a bounded loop over a few dozen plan points
        (allowed small loop; the bulk tessellation itself is numpy).
        Concave (L-shaped) footprints should pass explicit polygons to
        :meth:`set_foundation` / :meth:`set_roof` instead.
        """
        if not self.storeys or not self.storeys[0].walls:
            raise ValueError("auto footprint needs at least one wall on storey 0")
        walls = self.storeys[0].walls
        xy = np.concatenate([w.tessellate(arc_segments_per_quarter=8) for w in walls], axis=0)
        pad = max(w.thickness_m for w in walls) / 2.0
        return _pad_hull_outward(_convex_hull(xy), pad)


def _pad_hull_outward(hull: np.ndarray, pad: float) -> np.ndarray:
    """
    Offset a CCW convex hull outward by ``pad`` meters along each vertex's
    angle bisector (vectorized; ``float64 (H, 2)``).

    Shared by :meth:`Building._auto_footprint` (auto foundation/roof outlines)
    and the mesher's per-storey floor slab, which pads the wall-centerline hull
    out to the **outer wall faces** so the floor edge meets the wall instead of
    stopping ``thickness/2`` short (no floor↔wall seam). ``pad <= 0`` or a
    degenerate hull (<3 vertices) returns the hull unchanged.

    Docs: docs/systems/buildings.md
    """
    if hull.shape[0] < 3 or pad <= 0.0:
        return hull
    prev = np.roll(hull, 1, axis=0)
    nxt = np.roll(hull, -1, axis=0)
    e_in = hull - prev
    e_out = nxt - hull
    # Outward normals of the two adjacent edges (hull is CCW → right normal
    # points outward).
    n_in = np.stack([e_in[:, 1], -e_in[:, 0]], axis=1)
    n_out = np.stack([e_out[:, 1], -e_out[:, 0]], axis=1)
    n_in /= np.linalg.norm(n_in, axis=1, keepdims=True)
    n_out /= np.linalg.norm(n_out, axis=1, keepdims=True)
    bis = n_in + n_out
    norm = np.linalg.norm(bis, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    bis /= norm
    # Scale so the *edge* offset is pad (1/cos(half-angle) factor).
    cos_half = np.clip(np.sum(bis * n_out, axis=1, keepdims=True), 0.2, 1.0)
    result: np.ndarray = hull + bis * (pad / cos_half)
    return result


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """
    Convex hull (CCW, ``float64 (H, 2)``) via Andrew's monotone chain.

    Python loop over a few dozen sorted plan points — an allowed bounded
    small loop (this is plan-authoring code, never per-voxel/per-vertex).
    """
    pts = np.unique(points.round(decimals=9), axis=0)  # sorted lexicographic
    if pts.shape[0] <= 2:
        return pts.copy()

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=np.float64)
