"""
Shared enumerations for fire_engine.devtools.

Groups FieldKind, GizmoMode, and HandleType into one exempt module so every
other devtools module can import from here without circular-import risk.

Docs: docs/systems/devtools.md
"""

from __future__ import annotations

from enum import Enum, auto


class FieldKind(Enum):
    """
    The widget family a :class:`~fire_engine.devtools.types.Field` maps to.

    The renderer chooses a concrete control per kind:

    LABEL   — read-only text (a formatted string from ``get``).
    FLOAT   — single numeric entry; ``get``→float, ``set``(float).
    INT     — single integer entry; ``get``→int, ``set``(int).
    BOOL    — toggle; ``get``→bool, ``set``(bool).
    STRING  — text entry; ``get``→str, ``set``(str).
    VEC3    — three numeric entries; ``get``→(x, y, z) tuple of float,
              ``set``((x, y, z)).  Used for positions, scales, and
              euler-angle views of rotations.
    ENUM    — choice from ``choices``; ``get``→str, ``set``(str).

    Docs: docs/systems/devtools.md
    """

    LABEL = auto()
    FLOAT = auto()
    INT = auto()
    BOOL = auto()
    STRING = auto()
    VEC3 = auto()
    ENUM = auto()


class GizmoMode(Enum):
    """
    Which manipulator is active (mirrors Unity's W/E/R tools).

    Docs: docs/systems/devtools.md
    """

    TRANSLATE = "translate"
    ROTATE = "rotate"
    SCALE = "scale"


class HandleType(Enum):
    """
    The kind of handle a ray can grab.

    AXIS    — a single-axis arrow (translate) or stalk (scale).
    PLANE   — a two-axis square (translate on the plane whose *normal* is ``axis``).
    RING    — a rotation ring in the plane whose *normal* is ``axis``.
    UNIFORM — the centre cube (uniform scale on all axes; ``axis`` ignored).

    Docs: docs/systems/devtools.md
    """

    AXIS = "axis"
    PLANE = "plane"
    RING = "ring"
    UNIFORM = "uniform"


__all__ = ["FieldKind", "GizmoMode", "HandleType"]
