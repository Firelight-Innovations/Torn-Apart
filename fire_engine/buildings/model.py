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
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from fire_engine.core.config import Config
from fire_engine.core.math3d import Quat, Vec3

__all__ = [
    "WallKind",
    "OpeningKind",
    "BuildingDefaults",
    "Opening",
    "Wall",
    "Room",
    "StairsStub",
    "Foundation",
    "RoofSlab",
    "Storey",
    "Building",
]

# Plan-space point: (x, y) meters in building-local space.
PlanPoint = tuple[float, float]


class WallKind(Enum):
    """Derived wall geometry kind — see :attr:`Wall.kind` (never stored)."""

    SEGMENT = "segment"
    ARC = "arc"


class OpeningKind(Enum):
    """What an :class:`Opening` cuts out of a wall."""

    WINDOW = "window"
    DOOR = "door"


@dataclass(frozen=True)
class BuildingDefaults:
    """
    Per-building fallback dimensions, sourced from ``core.config``.

    There are deliberately **no field defaults** here — the single source of
    the canonical numbers is the ``Config`` dataclass (``building_*`` fields);
    construct via :meth:`from_config` (tests pass ``Config()`` for the stock
    values).

    Fields (all meters)
    -------------------
    storey_height_m    : storey floor-to-floor height when ``add_storey``
                         gets no explicit ``height_m``.
    wall_thickness_m   : wall thickness when ``add_wall`` gets none.
    slab_thickness_m   : floor/ceiling/roof slab thickness.
    foundation_depth_m : foundation slab depth below local z=0.
    """

    storey_height_m: float
    wall_thickness_m: float
    slab_thickness_m: float
    foundation_depth_m: float

    @classmethod
    def from_config(cls, cfg: Config) -> "BuildingDefaults":
        """
        Build defaults from the engine :class:`~fire_engine.core.config.Config`.

        Example
        -------
        >>> BuildingDefaults.from_config(Config()).wall_thickness_m
        0.3
        """
        return cls(
            storey_height_m=cfg.building_default_storey_height_m,
            wall_thickness_m=cfg.building_default_wall_thickness_m,
            slab_thickness_m=cfg.building_slab_thickness_m,
            foundation_depth_m=cfg.building_foundation_depth_m,
        )

    def to_dict(self) -> dict:
        """Plain-primitive dict (delta-save payload)."""
        return {
            "storey_height_m": float(self.storey_height_m),
            "wall_thickness_m": float(self.wall_thickness_m),
            "slab_thickness_m": float(self.slab_thickness_m),
            "foundation_depth_m": float(self.foundation_depth_m),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BuildingDefaults":
        """Inverse of :meth:`to_dict`."""
        return cls(
            storey_height_m=float(d["storey_height_m"]),
            wall_thickness_m=float(d["wall_thickness_m"]),
            slab_thickness_m=float(d["slab_thickness_m"]),
            foundation_depth_m=float(d["foundation_depth_m"]),
        )


@dataclass
class Opening:
    """
    A rectangular cutout (window or door) in a wall.

    Openings live in the wall's local ``(s, z)`` frame: ``s`` is arclength in
    meters along the wall centerline from endpoint ``a``; ``z`` is height in
    meters above the **top of the floor slab** the wall stands on.  The mesher
    partitions the wall face around the rectangle — no CSG.

    Fields
    ------
    id       : per-building element id (stable across save/load).
    kind     : ``OpeningKind.WINDOW`` or ``OpeningKind.DOOR``.
    offset_m : arclength from wall start ``a`` to the opening's near edge (m).
    width_m  : opening width along the wall (m).
    sill_m   : bottom of the opening above the floor-slab top (m); doors use
               ``0.0`` (they reach the wall base).
    head_m   : top of the opening above the floor-slab top (m); must exceed
               ``sill_m``.
    """

    id: int
    kind: OpeningKind
    offset_m: float
    width_m: float
    sill_m: float
    head_m: float

    def to_dict(self) -> dict:
        """Plain-primitive dict (delta-save payload)."""
        return {
            "id": int(self.id),
            "kind": self.kind.value,
            "offset_m": float(self.offset_m),
            "width_m": float(self.width_m),
            "sill_m": float(self.sill_m),
            "head_m": float(self.head_m),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Opening":
        """Inverse of :meth:`to_dict`."""
        return cls(
            id=int(d["id"]),
            kind=OpeningKind(d["kind"]),
            offset_m=float(d["offset_m"]),
            width_m=float(d["width_m"]),
            sill_m=float(d["sill_m"]),
            head_m=float(d["head_m"]),
        )


@dataclass
class Wall:
    """
    One wall span: a straight segment or circular arc with thickness.

    The centerline runs from plan point ``a`` to ``b``; ``bulge`` selects the
    arc (see module docstring).  Thickness extends ``thickness_m / 2`` to each
    side of the centerline.  Vertically the wall fills the storey's wall band
    unless ``height_m`` overrides it (measured from the floor-slab top).

    Fields
    ------
    id          : per-building element id.
    a, b        : centerline endpoints, plan-space meters.
    bulge       : ``|bulge| = tan(included_angle/4)``; 0 = straight, >0 bows
                  left of a→b, <0 bows right, ±1 = semicircle.
    thickness_m : wall thickness in meters.
    height_m    : wall height above the floor-slab top in meters, or ``None``
                  to fill the storey band (``storey.height_m - storey.slab_m``).
    openings    : cutouts along this wall (windows/doors).

    Example
    -------
    >>> w = Wall(id=1, a=(0.0, 0.0), b=(4.0, 0.0), bulge=0.0,
    ...          thickness_m=0.3)
    >>> w.kind is WallKind.SEGMENT and abs(w.length_m() - 4.0) < 1e-9
    True
    """

    id: int
    a: PlanPoint
    b: PlanPoint
    bulge: float = 0.0
    thickness_m: float = 0.3
    height_m: float | None = None
    openings: list[Opening] = field(default_factory=list)

    # Straight/arc discrimination threshold: below this the sagitta is
    # sub-millimeter for any realistic wall, so treat as straight.
    _BULGE_EPS = 1e-9

    @property
    def kind(self) -> WallKind:
        """Derived geometry kind: ``ARC`` when ``|bulge|`` is significant."""
        return WallKind.ARC if abs(self.bulge) > self._BULGE_EPS else WallKind.SEGMENT

    # ------------------------------------------------------------------
    # Geometry queries
    # ------------------------------------------------------------------

    def chord_m(self) -> float:
        """Straight-line distance |b - a| in meters."""
        return math.hypot(self.b[0] - self.a[0], self.b[1] - self.a[1])

    def arc_params(self) -> tuple[PlanPoint, float, float, float]:
        """
        Derived circle parameters for an arc wall.

        Returns
        -------
        (center, radius, start_angle, sweep)
            ``center`` plan point; ``radius`` meters (positive);
            ``start_angle`` radians of endpoint ``a`` about the center;
            ``sweep`` signed angular traversal in radians from ``a`` to ``b``
            (``sweep = -4·atan(bulge)`` — a left-bowing arc walks clockwise
            about its center, so positive bulge yields negative sweep).

        Raises
        ------
        ValueError
            For straight walls (``kind == SEGMENT``) or degenerate chords.
        """
        if self.kind is not WallKind.ARC:
            raise ValueError("arc_params() on a straight wall")
        c = self.chord_m()
        if c <= self._BULGE_EPS:
            raise ValueError("degenerate arc wall: coincident endpoints")
        # Sagitta (apex offset along the left normal) and signed radius.
        s = self.bulge * c / 2.0
        r = c * (1.0 + self.bulge * self.bulge) / (4.0 * self.bulge)  # signed
        mx = (self.a[0] + self.b[0]) / 2.0
        my = (self.a[1] + self.b[1]) / 2.0
        # Left-hand unit normal of a→b.  The apex sits at mid + perp·s; the
        # center sits opposite it on the bisector at mid + perp·(s - r)
        # (bulge = 1 → s == r → center on the chord, a true semicircle).
        px = -(self.b[1] - self.a[1]) / c
        py = (self.b[0] - self.a[0]) / c
        cx = mx + px * (s - r)
        cy = my + py * (s - r)
        sweep = -4.0 * math.atan(self.bulge)
        start = math.atan2(self.a[1] - cy, self.a[0] - cx)
        return (cx, cy), abs(r), start, sweep

    def length_m(self) -> float:
        """Centerline arclength in meters (chord length for straight walls)."""
        if self.kind is WallKind.SEGMENT:
            return self.chord_m()
        _, radius, _, sweep = self.arc_params()
        return radius * abs(sweep)

    def tessellate(self, arc_segments_per_quarter: int) -> np.ndarray:
        """
        Centerline polyline including both endpoints, ``float64 (P, 2)``.

        Straight walls return exactly ``[a, b]``.  Arcs are split into
        ``ceil(|sweep| / 90°) × arc_segments_per_quarter`` chords sampled by
        one vectorized ``np.linspace`` over angle; the first/last points are
        snapped to the exact endpoints so room detection sees identical
        coordinates regardless of float drift.

        Parameters
        ----------
        arc_segments_per_quarter : int
            Chords per quarter circle (``Config.building_arc_segments_per_quarter``).
        """
        if self.kind is WallKind.SEGMENT:
            return np.array([self.a, self.b], dtype=np.float64)
        (cx, cy), radius, start, sweep = self.arc_params()
        n = max(1, math.ceil(abs(sweep) / (math.pi / 2.0)) * int(arc_segments_per_quarter))
        theta = np.linspace(start, start + sweep, n + 1)
        pts = np.empty((n + 1, 2), dtype=np.float64)
        pts[:, 0] = cx + radius * np.cos(theta)
        pts[:, 1] = cy + radius * np.sin(theta)
        pts[0] = self.a
        pts[-1] = self.b
        return pts

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Plain-primitive dict (delta-save payload)."""
        return {
            "id": int(self.id),
            "a": [float(self.a[0]), float(self.a[1])],
            "b": [float(self.b[0]), float(self.b[1])],
            "bulge": float(self.bulge),
            "thickness_m": float(self.thickness_m),
            "height_m": None if self.height_m is None else float(self.height_m),
            "openings": [o.to_dict() for o in self.openings],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Wall":
        """Inverse of :meth:`to_dict`."""
        h = d.get("height_m")
        return cls(
            id=int(d["id"]),
            a=(float(d["a"][0]), float(d["a"][1])),
            b=(float(d["b"][0]), float(d["b"][1])),
            bulge=float(d["bulge"]),
            thickness_m=float(d["thickness_m"]),
            height_m=None if h is None else float(h),
            openings=[Opening.from_dict(o) for o in d.get("openings", ())],
        )


@dataclass
class Room:
    """
    One enclosed plan region on a storey — the unit of future furnishing.

    A room is **data, not geometry**: a simple CCW polygon in plan space plus
    a tag and free-form metadata.  Auto-detected rooms (``auto=True``) come
    from :func:`fire_engine.buildings.rooms.detect_rooms`; explicit rooms are
    authored via :meth:`Storey.add_room`.

    Fields
    ------
    id      : per-building element id.
    polygon : ``float64 (N, 2)`` CCW plan-space vertices (not closed — the
              last vertex does not repeat the first).
    tag     : semantic label ("kitchen", "living", ...) for procedural
              furnishing; free-form.
    meta    : free-form dict of primitives for future systems.
    auto    : True when produced by room auto-detection (re-derivable),
              False when explicitly authored.
    """

    id: int
    polygon: np.ndarray
    tag: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    auto: bool = False

    def area_m2(self) -> float:
        """Polygon area in m² (shoelace, vectorized; positive for CCW)."""
        x = self.polygon[:, 0]
        y = self.polygon[:, 1]
        return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    def centroid(self) -> PlanPoint:
        """Polygon centroid in plan-space meters (vectorized shoelace form)."""
        x = self.polygon[:, 0]
        y = self.polygon[:, 1]
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        a = float(np.sum(cross)) / 2.0
        if abs(a) < 1e-12:
            return float(np.mean(x)), float(np.mean(y))
        cx = float(np.sum((x + np.roll(x, -1)) * cross)) / (6.0 * a)
        cy = float(np.sum((y + np.roll(y, -1)) * cross)) / (6.0 * a)
        return cx, cy

    def to_dict(self) -> dict:
        """Plain-primitive dict (delta-save payload)."""
        return {
            "id": int(self.id),
            "polygon": [[float(p[0]), float(p[1])] for p in self.polygon],
            "tag": str(self.tag),
            "meta": dict(self.meta),
            "auto": bool(self.auto),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Room":
        """Inverse of :meth:`to_dict`."""
        return cls(
            id=int(d["id"]),
            polygon=np.array(d["polygon"], dtype=np.float64),
            tag=str(d.get("tag", "")),
            meta=dict(d.get("meta", {})),
            auto=bool(d.get("auto", False)),
        )


@dataclass
class StairsStub:
    """
    Data-model placeholder for a stair run — **no geometry in v1**.

    Reserves the concept (which storeys it connects, where it starts, which
    way it runs) so floorplan generators can already place stairs; meshing
    and traversal are future scope.

    Fields
    ------
    id            : per-building element id.
    storey_from   : index of the lower storey.
    storey_to     : index of the upper storey (normally ``storey_from + 1``).
    anchor        : plan point of the bottom step's center.
    direction_rad : plan-space heading of ascent in radians (0 = +x, CCW).
    width_m       : stair width in meters.
    """

    id: int
    storey_from: int
    storey_to: int
    anchor: PlanPoint
    direction_rad: float
    width_m: float

    def to_dict(self) -> dict:
        """Plain-primitive dict (delta-save payload)."""
        return {
            "id": int(self.id),
            "storey_from": int(self.storey_from),
            "storey_to": int(self.storey_to),
            "anchor": [float(self.anchor[0]), float(self.anchor[1])],
            "direction_rad": float(self.direction_rad),
            "width_m": float(self.width_m),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StairsStub":
        """Inverse of :meth:`to_dict`."""
        return cls(
            id=int(d["id"]),
            storey_from=int(d["storey_from"]),
            storey_to=int(d["storey_to"]),
            anchor=(float(d["anchor"][0]), float(d["anchor"][1])),
            direction_rad=float(d["direction_rad"]),
            width_m=float(d["width_m"]),
        )


@dataclass
class Foundation:
    """
    The foundation slab under the building.

    Occupies local z ``[-depth_m, 0]`` across ``polygon`` (simple CCW
    plan-space polygon, ``float64 (N, 2)``, not closed).
    """

    polygon: np.ndarray
    depth_m: float

    def to_dict(self) -> dict:
        """Plain-primitive dict (delta-save payload)."""
        return {
            "polygon": [[float(p[0]), float(p[1])] for p in self.polygon],
            "depth_m": float(self.depth_m),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Foundation":
        """Inverse of :meth:`to_dict`."""
        return cls(polygon=np.array(d["polygon"], dtype=np.float64), depth_m=float(d["depth_m"]))


@dataclass
class RoofSlab:
    """
    Flat roof slab capping the top storey (pitched roofs are future scope).

    Occupies local z ``[top, top + thickness_m]`` where ``top`` is the top of
    the highest storey, across ``polygon`` (simple CCW plan-space polygon).
    """

    polygon: np.ndarray
    thickness_m: float

    def to_dict(self) -> dict:
        """Plain-primitive dict (delta-save payload)."""
        return {
            "polygon": [[float(p[0]), float(p[1])] for p in self.polygon],
            "thickness_m": float(self.thickness_m),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RoofSlab":
        """Inverse of :meth:`to_dict`."""
        return cls(
            polygon=np.array(d["polygon"], dtype=np.float64), thickness_m=float(d["thickness_m"])
        )


class Storey:
    """
    One floor of a building: a 2-D plan of walls plus its rooms and stairs.

    Created via :meth:`Building.add_storey` (which wires the back-reference
    used for element-id allocation) — do not construct directly.

    Attributes
    ----------
    id       : per-building element id.
    index    : 0-based vertical position (0 = ground storey).
    height_m : floor-to-floor height in meters (includes the floor slab).
    slab_m   : floor slab thickness in meters; the wall band starts above it.
    walls    : list[Wall] — this storey's plan.
    rooms    : list[Room] — explicit + auto-detected rooms.
    stairs   : list[StairsStub].

    Example
    -------
    >>> s = building.add_storey()
    >>> w = s.add_wall((0, 0), (4, 0))
    >>> s.add_opening(w.id, OpeningKind.WINDOW, offset_m=1.0, width_m=1.2,
    ...               sill_m=1.0, head_m=2.2)  # doctest: +ELLIPSIS
    Opening(...)
    """

    def __init__(
        self, building: "Building", eid: int, index: int, height_m: float, slab_m: float
    ) -> None:
        self._building = building
        self.id = eid
        self.index = index
        self.height_m = height_m
        self.slab_m = slab_m
        self.walls: list[Wall] = []
        self.rooms: list[Room] = []
        self.stairs: list[StairsStub] = []

    # ------------------------------------------------------------------
    # Authoring API
    # ------------------------------------------------------------------

    def add_wall(
        self,
        a: PlanPoint,
        b: PlanPoint,
        *,
        bulge: float = 0.0,
        thickness_m: float | None = None,
        height_m: float | None = None,
    ) -> Wall:
        """
        Add a wall span to this storey's plan and return it.

        Parameters
        ----------
        a, b : (float, float)
            Centerline endpoints in plan-space meters.  For room
            auto-detection, adjoining walls must share endpoint coordinates
            (within ``Config.building_snap_eps_m``) — v1 does not split
            mid-span T-junctions.
        bulge : float
            ``tan(sweep/4)`` arc parameter; 0 = straight (see module doc).
        thickness_m : float | None
            Wall thickness; ``None`` → ``building.defaults.wall_thickness_m``.
        height_m : float | None
            Wall height above the floor-slab top; ``None`` → fill the storey
            band (``height_m - slab_m``).
        """
        if math.hypot(b[0] - a[0], b[1] - a[1]) <= 1e-9:
            raise ValueError("wall endpoints are coincident")
        wall = Wall(
            id=self._building.allocate_eid(),
            a=(float(a[0]), float(a[1])),
            b=(float(b[0]), float(b[1])),
            bulge=float(bulge),
            thickness_m=(
                self._building.defaults.wall_thickness_m
                if thickness_m is None
                else float(thickness_m)
            ),
            height_m=None if height_m is None else float(height_m),
        )
        self.walls.append(wall)
        return wall

    def add_opening(
        self,
        wall_id: int,
        kind: OpeningKind,
        *,
        offset_m: float,
        width_m: float,
        head_m: float,
        sill_m: float = 0.0,
    ) -> Opening:
        """
        Cut a window/door into one of this storey's walls and return it.

        Parameters
        ----------
        wall_id : int
            Id of a wall previously returned by :meth:`add_wall`.
        kind : OpeningKind
            ``WINDOW`` or ``DOOR``.
        offset_m : float
            Arclength from the wall's ``a`` endpoint to the opening's near
            edge (meters).
        width_m : float
            Opening width along the wall (meters).
        head_m : float
            Opening top above the floor-slab top (meters).
        sill_m : float
            Opening bottom above the floor-slab top (meters); leave at 0.0
            for doors (they reach the wall base).

        Raises
        ------
        KeyError   — unknown ``wall_id`` on this storey.
        ValueError — opening exceeds the wall's length or height band, or
                     ``sill_m >= head_m``.
        """
        wall = next((w for w in self.walls if w.id == wall_id), None)
        if wall is None:
            raise KeyError(f"no wall id={wall_id} on storey index={self.index}")
        if sill_m >= head_m:
            raise ValueError(f"sill_m={sill_m} must be below head_m={head_m}")
        length = wall.length_m()
        if offset_m < 0.0 or offset_m + width_m > length + 1e-9:
            raise ValueError(
                f"opening [{offset_m}, {offset_m + width_m}] m exceeds wall length {length:.3f} m"
            )
        band = wall.height_m if wall.height_m is not None else self.height_m - self.slab_m
        if head_m > band + 1e-9:
            raise ValueError(f"head_m={head_m} exceeds wall band height {band:.3f} m")
        opening = Opening(
            id=self._building.allocate_eid(),
            kind=kind,
            offset_m=float(offset_m),
            width_m=float(width_m),
            sill_m=float(sill_m),
            head_m=float(head_m),
        )
        wall.openings.append(opening)
        return opening

    def add_room(self, polygon: Any, tag: str = "", meta: dict[str, Any] | None = None) -> Room:
        """
        Explicitly declare a room region (independent of wall topology).

        Parameters
        ----------
        polygon : array-like (N, 2)
            Simple CCW plan-space polygon in meters (≥ 3 vertices, not
            closed).  CW input is reversed to CCW.
        tag : str
            Semantic label for future furnishing ("kitchen", ...).
        meta : dict | None
            Free-form primitives for future systems.
        """
        poly = np.array(polygon, dtype=np.float64)
        if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2:
            raise ValueError("room polygon must be (N>=3, 2)")
        # Normalize winding to CCW (positive shoelace area).
        x, y = poly[:, 0], poly[:, 1]
        if float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) < 0.0:
            poly = poly[::-1].copy()
        room = Room(
            id=self._building.allocate_eid(),
            polygon=poly,
            tag=tag,
            meta=dict(meta or {}),
            auto=False,
        )
        self.rooms.append(room)
        return room

    def add_stairs(
        self, storey_to: int, anchor: PlanPoint, direction_rad: float, width_m: float
    ) -> StairsStub:
        """
        Reserve a stair run from this storey upward — data only in v1
        (no geometry; see :class:`StairsStub`).
        """
        stub = StairsStub(
            id=self._building.allocate_eid(),
            storey_from=self.index,
            storey_to=int(storey_to),
            anchor=(float(anchor[0]), float(anchor[1])),
            direction_rad=float(direction_rad),
            width_m=float(width_m),
        )
        self.stairs.append(stub)
        return stub

    def detect_rooms(self, *, snap_eps_m: float, arc_segments_per_quarter: int) -> list[Room]:
        """
        Auto-detect enclosed rooms from this storey's wall topology.

        Replaces previously auto-detected rooms (``auto=True``) with the
        fresh result; explicit rooms are kept.  Walls must meet at endpoints
        (within ``snap_eps_m``) — v1 does not split mid-span T-junctions.

        Parameters
        ----------
        snap_eps_m : float
            Endpoint-snap tolerance (``Config.building_snap_eps_m``).
        arc_segments_per_quarter : int
            Arc tessellation density (must match meshing for consistent
            polygons).

        Returns
        -------
        list[Room]
            The newly detected rooms (also appended to :attr:`rooms`).
        """
        from fire_engine.buildings.rooms import detect_room_polygons

        polygons = detect_room_polygons(
            self.walls, snap_eps_m=snap_eps_m, arc_segments_per_quarter=arc_segments_per_quarter
        )
        self.rooms = [r for r in self.rooms if not r.auto]
        detected: list[Room] = []
        for poly in polygons:
            room = Room(id=self._building.allocate_eid(), polygon=poly, auto=True)
            self.rooms.append(room)
            detected.append(room)
        return detected

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Plain-primitive dict (delta-save payload)."""
        return {
            "id": int(self.id),
            "index": int(self.index),
            "height_m": float(self.height_m),
            "slab_m": float(self.slab_m),
            "walls": [w.to_dict() for w in self.walls],
            "rooms": [r.to_dict() for r in self.rooms],
            "stairs": [s.to_dict() for s in self.stairs],
        }

    @classmethod
    def from_dict(cls, building: "Building", d: dict) -> "Storey":
        """Inverse of :meth:`to_dict` (re-wires the building back-ref)."""
        storey = cls(
            building,
            eid=int(d["id"]),
            index=int(d["index"]),
            height_m=float(d["height_m"]),
            slab_m=float(d["slab_m"]),
        )
        storey.walls = [Wall.from_dict(w) for w in d.get("walls", ())]
        storey.rooms = [Room.from_dict(r) for r in d.get("rooms", ())]
        storey.stairs = [StairsStub.from_dict(s) for s in d.get("stairs", ())]
        return storey


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
        """
        if not 0 <= index < len(self.storeys):
            raise IndexError(f"storey index {index} out of range")
        return float(sum(s.height_m for s in self.storeys[:index]))

    @property
    def total_height_m(self) -> float:
        """Local z of the top of the highest storey (roof slab excluded)."""
        return float(sum(s.height_m for s in self.storeys))

    def world_aabb(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """
        Conservative world-space AABB of the whole building in meters.

        Computed from the local extent box (all tessellated wall centerlines
        padded by half the thickest wall, plus foundation/roof polygons and
        the full z range) with its 8 corners transformed by the building's
        rotation + position.  Used for lighting invalidation
        (``BuildingChangedEvent``) and broad-phase queries.
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

    def to_dict(self) -> dict:
        """
        Full building spec as a dict of primitives (delta-save payload).

        Round-trips exactly: ``Building.from_dict(b.to_dict()).to_dict()
        == b.to_dict()``.
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
    def from_dict(cls, d: dict) -> "Building":
        """Inverse of :meth:`to_dict`."""
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
        hull = _convex_hull(xy)
        # Pad outward along each hull vertex's angle bisector (vectorized).
        prev = np.roll(hull, 1, axis=0)
        nxt = np.roll(hull, -1, axis=0)
        e_in = hull - prev
        e_out = nxt - hull
        # Outward normals of the two adjacent edges (hull is CCW → right
        # normal points outward).
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
        return hull + bis * (pad / cos_half)


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
