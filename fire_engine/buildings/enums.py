"""
buildings/enums.py — shared enumerations for the building model.

Groups all public Enum types used across the buildings package.

Docs: docs/systems/buildings.md
"""

from __future__ import annotations

from enum import Enum, IntEnum

__all__ = [
    "OpeningKind",
    "RoofKind",
    "SurfaceMaterial",
    "WallKind",
]


class WallKind(Enum):
    """Derived wall geometry kind — see :attr:`Wall.kind` (never stored).

    Docs: docs/systems/buildings.md
    """

    SEGMENT = "segment"
    ARC = "arc"


class OpeningKind(Enum):
    """What an :class:`Opening` cuts out of a wall.

    Docs: docs/systems/buildings.md
    """

    WINDOW = "window"
    DOOR = "door"


class RoofKind(Enum):
    """
    Roof shape over a building's top storey (see :class:`RoofSlab`).

    - ``FLAT``  — a single horizontal slab (Iteration-1 behaviour; the default
      so existing saves and ``set_roof()`` calls are unchanged).
    - ``SHED``  — one mono-pitch plane sloping from a low eave to a high eave.
    - ``GABLE`` — two planes meeting at a central ridge, with vertical gable
      infill triangles closing the two ends.
    - ``HIP``   — four planes (two trapezoids + two end triangles) meeting at a
      shortened ridge; self-closing (no gable infill).

    Pitched kinds are generated over the footprint's ridge-aligned bounding
    rectangle (true-outline / concave pitched roofs are Iteration 3).

    Docs: docs/systems/buildings.md
    """

    FLAT = "flat"
    SHED = "shed"
    GABLE = "gable"
    HIP = "hip"


class SurfaceMaterial(IntEnum):
    """
    Which material/texture a building surface is drawn with.

    The mesher tags every face with one of these ids (stored in
    ``MeshArrays.face_materials``); the renderer splits the building geom by id
    and binds a distinct procedural albedo per material (``WALL`` →
    ``plaster_wall``, ``FLOOR`` → ``wood_floor``, ``ROOF`` → ``roof_shingle``,
    ``FOUNDATION`` → ``stone_foundation``).  ``WALL`` is ``0`` so an untagged
    face defaults to the wall material.  ``IntEnum`` so the value drops straight
    into the ``uint8`` ``face_materials`` array.

    Docs: docs/systems/buildings.md
    """

    WALL = 0
    FLOOR = 1
    ROOF = 2
    FOUNDATION = 3
