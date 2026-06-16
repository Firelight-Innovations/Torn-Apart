"""
terrain/enums.py — Shared enumerations for the terrain package.

Docs: docs/systems/world.terrain.md
"""

from __future__ import annotations

from enum import Enum


class BrushMode(Enum):
    """Whether a brush adds solid material or removes it."""

    ADD = "add"
    REMOVE = "remove"
