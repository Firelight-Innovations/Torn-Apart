"""
terrain/types.py — Shared support types (frozen dataclasses) for the terrain package.

Contains the brush-shape dataclasses used by ``apply_brush``.  The behavioural
logic (rasterisation, chunk iteration) lives in ``brush.py``.

Docs: docs/systems/world.terrain.md
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fire_engine.core.math3d import Vec3


@dataclass(frozen=True)
class SphereBrush:
    """
    Solid sphere brush.

    Parameters
    ----------
    radius_m : float
        Sphere radius in meters.

    Example
    -------
    >>> SphereBrush(radius_m=2.5)        # a 2.5 m explosion radius
    SphereBrush(radius_m=2.5)
    """

    radius_m: float

    def aabb(self, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World-space axis-aligned bounds (min, max) in meters."""
        r = np.array([self.radius_m] * 3, dtype=np.float64)
        return center - r, center + r

    def mask(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, center: np.ndarray) -> np.ndarray:
        """Boolean mask: voxel centres within ``radius_m`` of ``center``."""
        cx, cy, cz = center
        result: np.ndarray = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 <= self.radius_m**2
        return result


@dataclass(frozen=True)
class BoxBrush:
    """
    Axis-aligned box brush.

    Parameters
    ----------
    half_extents_m : Vec3
        Half-size of the box along each world axis in meters.

    Example
    -------
    >>> from fire_engine.core.math3d import Vec3
    >>> BoxBrush(half_extents_m=Vec3(1.0, 1.0, 2.0))   # 2×2×4 m block
    BoxBrush(half_extents_m=Vec3(1.0, 1.0, 2.0))
    """

    half_extents_m: Vec3

    def aabb(self, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World-space axis-aligned bounds (min, max) in meters."""
        he = self.half_extents_m.to_numpy().astype(np.float64)
        return center - he, center + he

    def mask(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, center: np.ndarray) -> np.ndarray:
        """Boolean mask: voxel centres inside the axis-aligned box."""
        cx, cy, cz = center
        hx, hy, hz = self.half_extents_m.to_numpy()
        result: np.ndarray = (
            (np.abs(X - cx) <= hx) & (np.abs(Y - cy) <= hy) & (np.abs(Z - cz) <= hz)
        )
        return result


@dataclass(frozen=True)
class CylinderBrush:
    """
    Vertical cylinder brush (axis along world +Z).

    Parameters
    ----------
    radius_m : float
        Cylinder radius in meters (in the XY plane).
    height_m : float
        Full cylinder height in meters (centred on ``center`` along Z).

    Example
    -------
    >>> CylinderBrush(radius_m=1.5, height_m=4.0)
    CylinderBrush(radius_m=1.5, height_m=4.0)
    """

    radius_m: float
    height_m: float

    def aabb(self, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World-space axis-aligned bounds (min, max) in meters."""
        half = np.array([self.radius_m, self.radius_m, self.height_m / 2.0], dtype=np.float64)
        return center - half, center + half

    def mask(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, center: np.ndarray) -> np.ndarray:
        """Boolean mask: voxel centres inside the vertical cylinder."""
        cx, cy, cz = center
        half_h = self.height_m / 2.0
        radial = (X - cx) ** 2 + (Y - cy) ** 2 <= self.radius_m**2
        vertical = np.abs(Z - cz) <= half_h
        result: np.ndarray = radial & vertical
        return result


# Union type alias for all brush shapes.
Brush = SphereBrush | BoxBrush | CylinderBrush
