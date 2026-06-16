"""
buildings/_impl/types.py — slab dataclass support types.

Holds the two horizontal-slab value objects (:class:`Foundation`, :class:`RoofSlab`)
split out of ``buildings/types.py`` to keep that module under the line cap. They
are re-exported from ``buildings.types`` (and therefore ``fire_engine.buildings``),
so every historical import path still resolves. Grouping module (one-public-class
rule exempt). Both round-trip through plain dicts (never pickle).

Docs: docs/systems/buildings._impl.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Foundation:
    """
    The foundation slab under the building.

    Occupies local z ``[-depth_m, 0]`` across ``polygon`` (simple CCW
    plan-space polygon, ``float64 (N, 2)``, not closed).
    Docs: docs/systems/buildings._impl.md
    """

    polygon: np.ndarray
    depth_m: float

    def to_dict(self) -> dict[str, Any]:
        """Plain-primitive dict (delta-save payload).
        Docs: docs/systems/buildings._impl.md
        """
        return {
            "polygon": [[float(p[0]), float(p[1])] for p in self.polygon],
            "depth_m": float(self.depth_m),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Foundation:
        """Inverse of :meth:`to_dict`.
        Docs: docs/systems/buildings._impl.md
        """
        return cls(polygon=np.array(d["polygon"], dtype=np.float64), depth_m=float(d["depth_m"]))


@dataclass
class RoofSlab:
    """
    Flat roof slab capping the top storey (pitched roofs are future scope).

    Occupies local z ``[top, top + thickness_m]`` where ``top`` is the top of
    the highest storey, across ``polygon`` (simple CCW plan-space polygon).
    Docs: docs/systems/buildings._impl.md
    """

    polygon: np.ndarray
    thickness_m: float

    def to_dict(self) -> dict[str, Any]:
        """Plain-primitive dict (delta-save payload).
        Docs: docs/systems/buildings._impl.md
        """
        return {
            "polygon": [[float(p[0]), float(p[1])] for p in self.polygon],
            "thickness_m": float(self.thickness_m),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RoofSlab:
        """Inverse of :meth:`to_dict`.
        Docs: docs/systems/buildings._impl.md
        """
        return cls(
            polygon=np.array(d["polygon"], dtype=np.float64), thickness_m=float(d["thickness_m"])
        )
