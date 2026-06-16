"""
core/math3d.py — Pure-numpy 3D math primitives for the Torn Apart engine.

Coordinate system (Z-up, Panda3D native):
  forward = +Y
  right   = +X
  up      = +Z

All types are float32 numpy-backed for memory efficiency and SIMD readiness.
No panda3d imports — this module is fully headless-testable.

The ``Quat`` class lives in :mod:`fire_engine.core.quat` and is re-exported
here so the conventional ``from fire_engine.core.math3d import Quat`` path
stays valid.

Usage example:
    from fire_engine.core.math3d import Vec3, Quat
    from math import pi

    pos   = Vec3(1.0, 2.0, 3.0)                        # meters
    rot   = Quat.from_axis_angle(Vec3.UP, pi / 2)       # 90° yaw about world Z
    fwd   = rot.rotate(Vec3.FORWARD)                    # → approx Vec3(-1, 0, 0)

Docs: docs/systems/core.md
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar

import numpy as np

# ---------------------------------------------------------------------------
# Vec3
# ---------------------------------------------------------------------------


class Vec3:
    """
    Immutable (by convention) 3-component float32 vector backed by a numpy array.

    Units: world distances are in **meters** unless otherwise stated by the caller.
    Coordinate convention: Z-up (forward=+Y, right=+X, up=+Z).

    Class constants
    ---------------
    Vec3.ZERO    → (0, 0, 0)
    Vec3.ONE     → (1, 1, 1)
    Vec3.UP      → (0, 0, 1)  — world up (+Z)
    Vec3.FORWARD → (0, 1, 0)  — world forward (+Y)
    Vec3.RIGHT   → (1, 0, 0)  — world right (+X)

    Example
    -------
    >>> v = Vec3(1, 2, 3)
    >>> v.normalized().dot(Vec3.UP)
    np.float32(0.8017837)

    Docs: docs/systems/core.md
    """

    __slots__ = ("_data",)

    # World-space constants. Declared here for type-checkers; the values are
    # assigned just after the class body (a Vec3 can't be constructed inside
    # its own definition). Keep the two in sync.
    ZERO: ClassVar[Vec3]
    ONE: ClassVar[Vec3]
    UP: ClassVar[Vec3]
    FORWARD: ClassVar[Vec3]
    RIGHT: ClassVar[Vec3]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        """
        Create a Vec3 from three scalar components.

        Parameters
        ----------
        x, y, z : float
            Components in meters (or any consistent unit).
        """
        self._data: np.ndarray = np.array([x, y, z], dtype=np.float32)

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> Vec3:
        """
        Create a Vec3 from a length-3 numpy array (copied to float32).

        Parameters
        ----------
        arr : np.ndarray
            Shape (3,) array.

        Example
        -------
        >>> Vec3.from_numpy(np.array([0, 0, 1], dtype=np.float32))
        Vec3(0.0, 0.0, 1.0)

        Docs: docs/systems/core.md
        """
        v = cls.__new__(cls)
        v._data = np.asarray(arr, dtype=np.float32).copy()
        return v

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def x(self) -> float:
        """X component (meters).

        Docs: docs/systems/core.md
        """
        return float(self._data[0])

    @property
    def y(self) -> float:
        """Y component (meters).

        Docs: docs/systems/core.md
        """
        return float(self._data[1])

    @property
    def z(self) -> float:
        """Z component (meters).

        Docs: docs/systems/core.md
        """
        return float(self._data[2])

    @property
    def length(self) -> float:
        """Euclidean length in meters.

        Docs: docs/systems/core.md
        """
        return float(np.sqrt(np.dot(self._data, self._data)))

    @property
    def length_squared(self) -> float:
        """Squared Euclidean length (avoids sqrt when only relative comparisons are needed).

        Docs: docs/systems/core.md
        """
        return float(np.dot(self._data, self._data))

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3.from_numpy(self._data + other._data)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3.from_numpy(self._data - other._data)

    def __mul__(self, scalar: float) -> Vec3:
        """Component-wise multiply by scalar."""
        return Vec3.from_numpy(self._data * float(scalar))

    def __rmul__(self, scalar: float) -> Vec3:
        """Scalar * Vec3 — commutative convenience."""
        return Vec3.from_numpy(self._data * float(scalar))

    def __neg__(self) -> Vec3:
        return Vec3.from_numpy(-self._data)

    def __truediv__(self, scalar: float) -> Vec3:
        return Vec3.from_numpy(self._data / float(scalar))

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Exact floating-point equality. Use approx_eq for numerical tolerance."""
        if not isinstance(other, Vec3):
            return NotImplemented
        return bool(np.array_equal(self._data, other._data))

    def approx_eq(self, other: Vec3, eps: float = 1e-6) -> bool:
        """
        Component-wise approximate equality within absolute tolerance eps.

        Parameters
        ----------
        other : Vec3
        eps   : float, default 1e-6

        Example
        -------
        >>> Vec3(1, 0, 0).approx_eq(Vec3(1 + 1e-7, 0, 0))
        True

        Docs: docs/systems/core.md
        """
        return bool(np.all(np.abs(self._data - other._data) <= eps))

    # ------------------------------------------------------------------
    # Vector math
    # ------------------------------------------------------------------

    def dot(self, other: Vec3) -> float:
        """
        Dot product: self · other.

        Example
        -------
        >>> Vec3.UP.dot(Vec3.UP)
        1.0
        >>> Vec3.UP.dot(Vec3.FORWARD)
        0.0

        Docs: docs/systems/core.md
        """
        return float(np.dot(self._data, other._data))

    def cross(self, other: Vec3) -> Vec3:
        """
        Cross product: self × other (right-handed, Z-up).

        Example
        -------
        >>> Vec3.RIGHT.cross(Vec3.FORWARD)  # +X × +Y = +Z
        Vec3(0.0, 0.0, 1.0)

        Docs: docs/systems/core.md
        """
        return Vec3.from_numpy(np.cross(self._data, other._data))

    def normalized(self) -> Vec3:
        """
        Return a unit-length copy. Raises ValueError if the vector is zero-length.

        Example
        -------
        >>> Vec3(3, 0, 0).normalized()
        Vec3(1.0, 0.0, 0.0)

        Docs: docs/systems/core.md
        """
        n = self.length
        if n < 1e-12:
            raise ValueError("Cannot normalize a zero-length Vec3.")
        return Vec3.from_numpy(self._data / n)

    def lerp(self, other: Vec3, t: float) -> Vec3:
        """
        Linear interpolation: self + t * (other - self).

        Parameters
        ----------
        other : Vec3
        t     : float  0 → self, 1 → other (extrapolates outside [0,1])

        Example
        -------
        >>> Vec3(0, 0, 0).lerp(Vec3(2, 0, 0), 0.5)
        Vec3(1.0, 0.0, 0.0)

        Docs: docs/systems/core.md
        """
        return Vec3.from_numpy(self._data + float(t) * (other._data - self._data))

    # ------------------------------------------------------------------
    # Conversion / iteration
    # ------------------------------------------------------------------

    def to_numpy(self) -> np.ndarray:
        """Return a float32 copy of the underlying (3,) array.

        Docs: docs/systems/core.md
        """
        return self._data.copy()

    def __iter__(self) -> Iterator[float]:
        """Iterate as (x, y, z) — enables tuple(vec3) and argument unpacking."""
        yield from (float(self._data[0]), float(self._data[1]), float(self._data[2]))

    def __getitem__(self, idx: int) -> float:
        """Index access: v[0]=x, v[1]=y, v[2]=z."""
        return float(self._data[idx])

    def __repr__(self) -> str:
        return f"Vec3({self._data[0]:.6g}, {self._data[1]:.6g}, {self._data[2]:.6g})"

    def __hash__(self) -> int:
        return hash(tuple(self._data.tolist()))


# Class constants (defined after class body so Vec3 is in scope)
Vec3.ZERO = Vec3(0.0, 0.0, 0.0)
Vec3.ONE = Vec3(1.0, 1.0, 1.0)
Vec3.UP = Vec3(0.0, 0.0, 1.0)  # +Z
Vec3.FORWARD = Vec3(0.0, 1.0, 0.0)  # +Y
Vec3.RIGHT = Vec3(1.0, 0.0, 0.0)  # +X

# Re-export Quat from its own module so the historical import paths
# (fire_engine.core.math3d.Quat and fire_engine.core.Quat) keep resolving.
from fire_engine.core._impl.quat import Quat as Quat  # noqa: E402

__all__ = ["Quat", "Vec3"]
