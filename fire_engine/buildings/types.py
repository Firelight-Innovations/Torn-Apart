"""
buildings/types.py — dataclass support types for the building model.

Groups all ``@dataclass`` support types used across the buildings package:
primitive value objects, geometry containers, and per-element plan entities.
All types round-trip through plain dicts via ``to_dict()``/``from_dict()``
(never pickle, never live references) so the ``Saveable`` protocol can persist
them through delta saves.
Docs: docs/systems/buildings.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

# Foundation + RoofSlab (the horizontal-slab value objects) live in
# buildings/_impl/types.py to keep this module under the line cap; re-exported
# here so `from fire_engine.buildings.types import Foundation` still resolves.
from fire_engine.buildings._impl.types import Foundation as Foundation
from fire_engine.buildings._impl.types import RoofSlab as RoofSlab
from fire_engine.buildings.enums import OpeningKind, WallKind
from fire_engine.core.config import Config

if TYPE_CHECKING:
    pass

__all__ = [
    "BuildingDefaults",
    "Foundation",
    "Opening",
    "PlanPoint",
    "RoofSlab",
    "Room",
    "StairsStub",
    "Wall",
]

# Plan-space point: (x, y) meters in building-local space.
PlanPoint = tuple[float, float]


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
    Docs: docs/systems/buildings.md
    """

    storey_height_m: float
    wall_thickness_m: float
    slab_thickness_m: float
    foundation_depth_m: float

    @classmethod
    def from_config(cls, cfg: Config) -> BuildingDefaults:
        """
        Build defaults from the engine :class:`~fire_engine.core.config.Config`.

        Example
        -------
        >>> BuildingDefaults.from_config(Config()).wall_thickness_m
        0.3
        Docs: docs/systems/buildings.md
        """
        return cls(
            storey_height_m=cfg.building_default_storey_height_m,
            wall_thickness_m=cfg.building_default_wall_thickness_m,
            slab_thickness_m=cfg.building_slab_thickness_m,
            foundation_depth_m=cfg.building_foundation_depth_m,
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain-primitive dict (delta-save payload).
        Docs: docs/systems/buildings.md
        """
        return {
            "storey_height_m": float(self.storey_height_m),
            "wall_thickness_m": float(self.wall_thickness_m),
            "slab_thickness_m": float(self.slab_thickness_m),
            "foundation_depth_m": float(self.foundation_depth_m),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BuildingDefaults:
        """Inverse of :meth:`to_dict`.
        Docs: docs/systems/buildings.md
        """
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
    Docs: docs/systems/buildings.md
    """

    id: int
    kind: OpeningKind
    offset_m: float
    width_m: float
    sill_m: float
    head_m: float

    def to_dict(self) -> dict[str, Any]:
        """Plain-primitive dict (delta-save payload).
        Docs: docs/systems/buildings.md
        """
        return {
            "id": int(self.id),
            "kind": self.kind.value,
            "offset_m": float(self.offset_m),
            "width_m": float(self.width_m),
            "sill_m": float(self.sill_m),
            "head_m": float(self.head_m),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Opening:
        """Inverse of :meth:`to_dict`.
        Docs: docs/systems/buildings.md
        """
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
    arc (see ``buildings/model.py`` module docstring).  Thickness extends
    ``thickness_m / 2`` to each side of the centerline.  Vertically the wall
    fills the storey's wall band unless ``height_m`` overrides it (measured
    from the floor-slab top).

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
    Docs: docs/systems/buildings.md
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
        """Derived geometry kind: ``ARC`` when ``|bulge|`` is significant.
        Docs: docs/systems/buildings.md
        """
        return WallKind.ARC if abs(self.bulge) > self._BULGE_EPS else WallKind.SEGMENT

    # Geometry queries

    def chord_m(self) -> float:
        """Straight-line distance |b - a| in meters.
        Docs: docs/systems/buildings.md
        """
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
        Docs: docs/systems/buildings.md
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
        """Centerline arclength in meters (chord length for straight walls).
        Docs: docs/systems/buildings.md
        """
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
        Docs: docs/systems/buildings.md
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

    # Serialisation

    def to_dict(self) -> dict[str, Any]:
        """Plain-primitive dict (delta-save payload).
        Docs: docs/systems/buildings.md
        """
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
    def from_dict(cls, d: dict[str, Any]) -> Wall:
        """Inverse of :meth:`to_dict`.
        Docs: docs/systems/buildings.md
        """
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
    Docs: docs/systems/buildings.md
    """

    id: int
    polygon: np.ndarray
    tag: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    auto: bool = False

    def area_m2(self) -> float:
        """Polygon area in m² (shoelace, vectorized; positive for CCW).
        Docs: docs/systems/buildings.md
        """
        x = self.polygon[:, 0]
        y = self.polygon[:, 1]
        return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    def centroid(self) -> PlanPoint:
        """Polygon centroid in plan-space meters (vectorized shoelace form).
        Docs: docs/systems/buildings.md
        """
        x = self.polygon[:, 0]
        y = self.polygon[:, 1]
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        a = float(np.sum(cross)) / 2.0
        if abs(a) < 1e-12:
            return float(np.mean(x)), float(np.mean(y))
        cx = float(np.sum((x + np.roll(x, -1)) * cross)) / (6.0 * a)
        cy = float(np.sum((y + np.roll(y, -1)) * cross)) / (6.0 * a)
        return cx, cy

    def to_dict(self) -> dict[str, Any]:
        """Plain-primitive dict (delta-save payload).
        Docs: docs/systems/buildings.md
        """
        return {
            "id": int(self.id),
            "polygon": [[float(p[0]), float(p[1])] for p in self.polygon],
            "tag": str(self.tag),
            "meta": dict(self.meta),
            "auto": bool(self.auto),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Room:
        """Inverse of :meth:`to_dict`.
        Docs: docs/systems/buildings.md
        """
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
    Docs: docs/systems/buildings.md
    """

    id: int
    storey_from: int
    storey_to: int
    anchor: PlanPoint
    direction_rad: float
    width_m: float

    def to_dict(self) -> dict[str, Any]:
        """Plain-primitive dict (delta-save payload).
        Docs: docs/systems/buildings.md
        """
        return {
            "id": int(self.id),
            "storey_from": int(self.storey_from),
            "storey_to": int(self.storey_to),
            "anchor": [float(self.anchor[0]), float(self.anchor[1])],
            "direction_rad": float(self.direction_rad),
            "width_m": float(self.width_m),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StairsStub:
        """Inverse of :meth:`to_dict`.
        Docs: docs/systems/buildings.md
        """
        return cls(
            id=int(d["id"]),
            storey_from=int(d["storey_from"]),
            storey_to=int(d["storey_to"]),
            anchor=(float(d["anchor"][0]), float(d["anchor"][1])),
            direction_rad=float(d["direction_rad"]),
            width_m=float(d["width_m"]),
        )
