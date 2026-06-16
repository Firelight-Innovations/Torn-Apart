"""
buildings/enums.py — shared enumerations for the building model.

Groups all public Enum types used across the buildings package.

Docs: docs/systems/buildings.md
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "OpeningKind",
    "RoofKind",
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
