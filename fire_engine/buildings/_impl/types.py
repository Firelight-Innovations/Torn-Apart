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

from fire_engine.buildings.enums import RoofKind


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
    Roof capping the top storey — flat slab or a pitched shape.

    The base sits on local z ``top`` (top of the highest storey) across
    ``polygon`` (simple CCW plan-space outline).  ``kind`` selects the shape
    (:class:`~fire_engine.buildings.enums.RoofKind`); for ``FLAT`` the roof is
    a horizontal slab in ``[top, top + thickness_m]`` (Iteration-1 behaviour),
    and the pitched fields are ignored.

    Pitched fields (used when ``kind != FLAT``)
    -------------------------------------------
    pitch_deg     : slope of the roof planes from horizontal, degrees.
    ridge_dir_rad : plan-space heading of the ridge line, radians (0 = +x);
                    the roof is generated over the footprint's bounding
                    rectangle aligned to this direction.
    overhang_m    : how far the eaves extend beyond the footprint, meters.
    thickness_m   : roof-plane depth (and flat-slab thickness), meters.

    Docs: docs/systems/buildings._impl.md
    """

    polygon: np.ndarray
    thickness_m: float
    kind: RoofKind = RoofKind.FLAT
    pitch_deg: float = 30.0
    ridge_dir_rad: float = 0.0
    overhang_m: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Plain-primitive dict (delta-save payload).
        Docs: docs/systems/buildings._impl.md
        """
        return {
            "polygon": [[float(p[0]), float(p[1])] for p in self.polygon],
            "thickness_m": float(self.thickness_m),
            "kind": self.kind.value,
            "pitch_deg": float(self.pitch_deg),
            "ridge_dir_rad": float(self.ridge_dir_rad),
            "overhang_m": float(self.overhang_m),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RoofSlab:
        """Inverse of :meth:`to_dict` (pitched keys default to a flat roof).
        Docs: docs/systems/buildings._impl.md
        """
        return cls(
            polygon=np.array(d["polygon"], dtype=np.float64),
            thickness_m=float(d["thickness_m"]),
            kind=RoofKind(d.get("kind", RoofKind.FLAT.value)),
            pitch_deg=float(d.get("pitch_deg", 30.0)),
            ridge_dir_rad=float(d.get("ridge_dir_rad", 0.0)),
            overhang_m=float(d.get("overhang_m", 0.0)),
        )
