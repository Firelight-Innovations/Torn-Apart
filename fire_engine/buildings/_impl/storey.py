"""
buildings/_impl/storey.py — Storey class (one floor of a Building).

Split from buildings/model.py to satisfy the one-public-class-per-module
limit.  Re-exported from ``fire_engine.buildings.model`` so all existing
import paths remain valid.

Docs: docs/systems/buildings.md
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from fire_engine.buildings.enums import OpeningKind
from fire_engine.buildings.types import Opening, Room, StairsStub, Wall

if TYPE_CHECKING:
    from fire_engine.buildings.model import Building

__all__ = ["Storey"]


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

    Docs: docs/systems/buildings.md
    """

    def __init__(
        self, building: Building, eid: int, index: int, height_m: float, slab_m: float
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
        a: tuple[float, float],
        b: tuple[float, float],
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

        Docs: docs/systems/buildings.md
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

        Docs: docs/systems/buildings.md
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

        Docs: docs/systems/buildings.md
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
        self,
        storey_to: int,
        anchor: tuple[float, float],
        direction_rad: float,
        width_m: float,
    ) -> StairsStub:
        """
        Reserve a stair run from this storey upward — data only in v1
        (no geometry; see :class:`StairsStub`).

        Docs: docs/systems/buildings.md
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

        Docs: docs/systems/buildings.md
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

    def to_dict(self) -> dict[str, Any]:
        """Plain-primitive dict (delta-save payload).

        Docs: docs/systems/buildings.md
        """
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
    def from_dict(cls, building: Building, d: dict[str, Any]) -> Storey:
        """Inverse of :meth:`to_dict` (re-wires the building back-ref).

        Docs: docs/systems/buildings.md
        """
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
