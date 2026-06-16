"""
buildings/enums.py — shared enumerations for the building model.

Groups all public Enum types used across the buildings package.

Docs: docs/systems/buildings.md
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "OpeningKind",
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
