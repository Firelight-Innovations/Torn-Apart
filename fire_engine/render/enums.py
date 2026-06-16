"""
render/enums.py — Shared enumeration types for the render package.

Grouping module (exempt from one-class rule per pyproject.toml [tool.firelight]).

Docs: docs/systems/render.md
"""

from __future__ import annotations

from enum import Enum, auto


class Space(Enum):
    """
    Reference-frame selector used by Transform.translate and Transform.rotate.

    SELF  — operations are expressed in the transform's own local frame.
    WORLD — operations are expressed in world space.

    Example
    -------
        t.translate(Vec3(0, 1, 0), relative_to=Space.SELF)   # move 1 m forward
        t.translate(Vec3(0, 1, 0), relative_to=Space.WORLD)  # move 1 m along world +Y
    """

    SELF = auto()
    WORLD = auto()
