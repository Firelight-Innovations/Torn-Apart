"""
core/math3d.py — Pure-numpy 3D math primitives for the Torn Apart engine.

Coordinate system (Z-up, Panda3D native):
  forward = +Y
  right   = +X
  up      = +Z

All types are float32 numpy-backed for memory efficiency and SIMD readiness.
No panda3d imports — this module is fully headless-testable.

Usage example:
    from fire_engine.core.math3d import Vec3, Quat
    from math import pi

    pos   = Vec3(1.0, 2.0, 3.0)                        # meters
    rot   = Quat.from_axis_angle(Vec3.UP, pi / 2)       # 90° yaw about world Z
    fwd   = rot.rotate(Vec3.FORWARD)                    # → approx Vec3(-1, 0, 0)
"""

from __future__ import annotations

import math
from collections.abc import Iterator

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
    """

    __slots__ = ("_data",)

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
        """
        v = cls.__new__(cls)
        v._data = np.asarray(arr, dtype=np.float32).copy()
        return v

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def x(self) -> float:
        """X component (meters)."""
        return float(self._data[0])

    @property
    def y(self) -> float:
        """Y component (meters)."""
        return float(self._data[1])

    @property
    def z(self) -> float:
        """Z component (meters)."""
        return float(self._data[2])

    @property
    def length(self) -> float:
        """Euclidean length in meters."""
        return float(np.sqrt(np.dot(self._data, self._data)))

    @property
    def length_squared(self) -> float:
        """Squared Euclidean length (avoids sqrt when only relative comparisons are needed)."""
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
        """
        return float(np.dot(self._data, other._data))

    def cross(self, other: Vec3) -> Vec3:
        """
        Cross product: self × other (right-handed, Z-up).

        Example
        -------
        >>> Vec3.RIGHT.cross(Vec3.FORWARD)  # +X × +Y = +Z
        Vec3(0.0, 0.0, 1.0)
        """
        return Vec3.from_numpy(np.cross(self._data, other._data))

    def normalized(self) -> Vec3:
        """
        Return a unit-length copy. Raises ValueError if the vector is zero-length.

        Example
        -------
        >>> Vec3(3, 0, 0).normalized()
        Vec3(1.0, 0.0, 0.0)
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
        """
        return Vec3.from_numpy(self._data + float(t) * (other._data - self._data))

    # ------------------------------------------------------------------
    # Conversion / iteration
    # ------------------------------------------------------------------

    def to_numpy(self) -> np.ndarray:
        """Return a float32 copy of the underlying (3,) array."""
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


# ---------------------------------------------------------------------------
# Quat
# ---------------------------------------------------------------------------


class Quat:
    """
    Unit quaternion rotation, stored as float32 numpy array in **scalar-first** order:
    ``[w, x, y, z]``.

    Coordinate system
    -----------------
    Z-up (Panda3D native): forward=+Y, right=+X, up=+Z.
    This is NOT Unity's Y-up; the API shape mirrors Unity but the axes are ours.

    Multiplication semantics
    ------------------------
    ``q1 * q2`` applies q2 FIRST, then q1 (standard Hamilton product, matching
    scipy Rotation and Unity's Transform hierarchy).  A rotation R composed as
    ``yaw * pitch`` therefore yaws in world space, then pitches in the yaw-rotated
    frame — exactly the expected camera behaviour.

    Handedness check (verified by tests/test_math3d.py)
    ------
    ``Quat.from_axis_angle(Vec3.UP, pi/2).rotate(Vec3.FORWARD)`` ≈ ``(-1, 0, 0)``.
    Rotating +Y by +90° about +Z (CCW when viewed from above) gives −X.

    HPR Euler convention
    --------------------
    H (heading/yaw)   — rotation about world +Z
    P (pitch)         — rotation about +X
    R (roll)          — rotation about +Y
    Composition order: H then P then R (i.e. qH * qP * qR applied R-first).

    Example
    -------
    >>> from math import pi
    >>> q = Quat.from_axis_angle(Vec3.UP, pi / 2)
    >>> q.rotate(Vec3.FORWARD)      # +Y rotated 90° CCW about +Z → −X
    Vec3(-1.0, ..., ...)
    >>> q.inverse().rotate(q.rotate(Vec3.FORWARD)).approx_eq(Vec3.FORWARD)
    True
    """

    __slots__ = ("_data",)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        """
        Construct from scalar-first components (w, x, y, z).

        For most uses, prefer the factory classmethods (identity, from_axis_angle,
        from_euler) instead of calling this directly.
        """
        self._data: np.ndarray = np.array([w, x, y, z], dtype=np.float32)

    @classmethod
    def identity(cls) -> Quat:
        """
        Return the identity quaternion (no rotation).

        Example
        -------
        >>> Quat.identity().rotate(Vec3.FORWARD)
        Vec3(0.0, 1.0, 0.0)
        """
        q = cls.__new__(cls)
        q._data = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return q

    @classmethod
    def from_axis_angle(cls, axis: Vec3, radians: float) -> Quat:
        """
        Create a quaternion representing a right-handed rotation about *axis*
        by *radians*.

        Parameters
        ----------
        axis    : Vec3   — rotation axis (does not need to be normalised)
        radians : float  — rotation angle in radians (positive = CCW when axis
                           points toward the viewer)

        Returns
        -------
        Quat (normalised)

        Example
        -------
        >>> from math import pi
        >>> Quat.from_axis_angle(Vec3.UP, pi/2).rotate(Vec3.FORWARD)
        Vec3(-1.0, ~0, ~0)   # +Y rotates to −X under 90° about +Z
        """
        n = axis.length
        if n < 1e-12:
            raise ValueError("Rotation axis must not be zero-length.")
        ax = axis._data / n
        half = float(radians) * 0.5
        s = math.sin(half)
        q = cls.__new__(cls)
        q._data = np.array(
            [math.cos(half), s * ax[0], s * ax[1], s * ax[2]],
            dtype=np.float32,
        )
        return q

    @classmethod
    def from_euler(cls, h: float, p: float, r: float) -> Quat:
        """
        Build a quaternion from Panda3D HPR Euler angles (all in **radians**).

        H (heading/yaw)  : rotation about world +Z (+Z is up)
        P (pitch)        : rotation about +X
        R (roll)         : rotation about +Y

        Composition order: heading is applied first (outermost), then pitch,
        then roll — equivalent to ``q_h * q_p * q_r``.

        Parameters
        ----------
        h : float — heading angle in radians
        p : float — pitch angle in radians
        r : float — roll angle in radians

        Returns
        -------
        Quat (normalised)

        Example
        -------
        >>> from math import pi
        >>> q = Quat.from_euler(pi/2, 0, 0)   # 90° yaw
        >>> q.rotate(Vec3.FORWARD).approx_eq(Vec3(-1, 0, 0), eps=1e-5)
        True
        """
        q_h = cls.from_axis_angle(Vec3.UP, h)  # yaw  about +Z
        q_p = cls.from_axis_angle(Vec3.RIGHT, p)  # pitch about +X
        q_r = cls.from_axis_angle(Vec3(0, 1, 0), r)  # roll  about +Y
        return q_h * q_p * q_r

    # ------------------------------------------------------------------
    # Properties / components
    # ------------------------------------------------------------------

    @property
    def w(self) -> float:
        return float(self._data[0])

    @property
    def x(self) -> float:
        return float(self._data[1])

    @property
    def y(self) -> float:
        return float(self._data[2])

    @property
    def z(self) -> float:
        return float(self._data[3])

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def __mul__(self, other: Quat) -> Quat:
        """
        Hamilton product: ``self * other`` applies *other* first, then *self*.

        (Matches scipy.spatial.transform.Rotation and Unity's Quaternion.*)

        Parameters
        ----------
        other : Quat

        Returns
        -------
        Quat (normalised)
        """
        w1, x1, y1, z1 = self._data
        w2, x2, y2, z2 = other._data
        q = Quat.__new__(Quat)
        q._data = np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float32,
        )
        return q

    def rotate(self, v: Vec3) -> Vec3:
        """
        Rotate vector *v* by this quaternion using the sandwich product:
        ``q * [0,v] * q^{-1}``.

        Parameters
        ----------
        v : Vec3 — the vector to rotate (meters)

        Returns
        -------
        Vec3 — rotated vector, same units as input

        Example
        -------
        >>> from math import pi
        >>> Quat.from_axis_angle(Vec3.UP, pi/2).rotate(Vec3.FORWARD)
        Vec3(-1.0, ~0, ~0)
        """
        # Efficient sandwich product using the Rodrigues formula:
        # v' = v + 2w(q × v) + 2(q × (q × v))
        # where q = (x, y, z) part of the quaternion
        w = float(self._data[0])
        qv = self._data[1:]  # (x, y, z) as float32 array
        vv = v._data  # float32 array
        t = 2.0 * np.cross(qv, vv)
        result = vv + w * t + np.cross(qv, t)
        return Vec3.from_numpy(result.astype(np.float32))

    def normalized(self) -> Quat:
        """
        Return a unit-norm copy of this quaternion.

        Raises ValueError if the quaternion is zero.
        """
        n = float(np.linalg.norm(self._data))
        if n < 1e-12:
            raise ValueError("Cannot normalise a zero quaternion.")
        q = Quat.__new__(Quat)
        q._data = (self._data / n).astype(np.float32)
        return q

    def inverse(self) -> Quat:
        """
        Return the inverse (conjugate for unit quaternions) of this quaternion.

        For a unit quaternion q = [w, x, y, z], the inverse = [w, -x, -y, -z].

        Example
        -------
        >>> q = Quat.from_axis_angle(Vec3.UP, 1.0)
        >>> (q * q.inverse()).approx_eq(Quat.identity())
        True
        """
        q = Quat.__new__(Quat)
        q._data = self._data * np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        return q

    def as_euler(self) -> tuple[float, float, float]:
        """
        Decompose this quaternion back into HPR Euler angles **(radians)**.

        H (heading/yaw)  : rotation about world +Z
        P (pitch)        : rotation about +X
        R (roll)         : rotation about +Y

        Returns
        -------
        (h, p, r) : tuple[float, float, float] in radians

        Note
        ----
        Euler decomposition is not unique near gimbal-lock singularities
        (pitch ≈ ±90°).  The round-trip ``from_euler(*as_euler())`` will still
        produce a functionally identical rotation, but the numeric values may
        differ at singularities.

        Example
        -------
        >>> from math import pi
        >>> q = Quat.from_euler(pi/4, pi/6, pi/8)
        >>> h, p, r = q.as_euler()
        >>> Quat.from_euler(h, p, r).approx_eq(q, eps=1e-5)
        True
        """
        w, x, y, z = [float(c) for c in self._data]

        # --- heading (yaw, about +Z) ---
        # Derived from the rotation matrix element R[0,1] and R[1,1]
        # R[0,1] = 2(xy - wz),  R[1,1] = 1 - 2(x²+z²)
        # h = atan2(R[1,0], R[0,0]) where:
        # R[1,0] = 2(xy + wz), R[0,0] = 1 - 2(y²+z²)
        #
        # We use the full HPR = qH * qP * qR decomposition.
        # Converting the 3x3 rotation matrix entries:
        #
        #   R = [[1-2(y²+z²),  2(xy-wz),   2(xz+wy)],
        #        [2(xy+wz),   1-2(x²+z²),  2(yz-wx)],
        #        [2(xz-wy),    2(yz+wx),  1-2(x²+y²)]]
        #
        # For HPR decomposition (H about Z, P about X, R about Y):
        # This is equivalent to an intrinsic ZXY rotation.
        # Combined rotation matrix for ZXY: R = Rz(h) * Rx(p) * Ry(r)
        #
        # From the matrix:
        # sin(p) = R[2,1] = 2(yz + wx)  (pitch from row 2, col 1)

        # sin(p) = 2(yz + wx)
        sin_p = 2.0 * (y * z + w * x)
        sin_p = max(-1.0, min(1.0, sin_p))  # clamp for numerical safety
        p_angle = math.asin(sin_p)

        cos_p = math.cos(p_angle)

        if abs(cos_p) > 1e-6:
            # h = atan2(-R[0,1], R[1,1]) = atan2(-(2(xy-wz)), 1-2(x²+z²))
            # r = atan2(-R[2,0], R[2,2]) = atan2(-(2(xz-wy)), 1-2(x²+y²))
            h_angle = math.atan2(-2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z))
            r_angle = math.atan2(-2.0 * (x * z - w * y), 1.0 - 2.0 * (x * x + y * y))
        else:
            # Gimbal lock: pitch ≈ ±90°, distribute between h and r
            h_angle = math.atan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))
            r_angle = 0.0

        return (h_angle, p_angle, r_angle)

    @staticmethod
    def slerp(a: Quat, b: Quat, t: float) -> Quat:
        """
        Spherical linear interpolation between quaternions *a* and *b*.

        Parameters
        ----------
        a : Quat — start rotation (t=0)
        b : Quat — end rotation  (t=1)
        t : float — interpolation parameter; 0 → a, 1 → b

        Returns
        -------
        Quat (normalised)

        Note
        ----
        If the dot product is negative, *b* is negated to take the short arc.
        Falls back to linear interpolation + normalise near antipodal inputs.

        Example
        -------
        >>> from math import pi
        >>> q0 = Quat.identity()
        >>> q1 = Quat.from_axis_angle(Vec3.UP, pi)
        >>> Quat.slerp(q0, q1, 0.5).approx_eq(Quat.from_axis_angle(Vec3.UP, pi/2))
        True
        """
        da = a._data.copy()
        db = b._data.copy()

        dot = float(np.dot(da, db))

        # Take the short arc
        if dot < 0.0:
            db = -db
            dot = -dot

        dot = min(1.0, dot)  # clamp for acos numerical safety

        if dot > 0.9995:
            # Nearly identical: fall back to lerp + normalise
            result = da + float(t) * (db - da)
            n = float(np.linalg.norm(result))
            q = Quat.__new__(Quat)
            q._data = (result / n).astype(np.float32)
            return q

        theta_0 = math.acos(dot)
        theta = theta_0 * float(t)
        sin_theta = math.sin(theta)
        sin_theta_0 = math.sin(theta_0)

        s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0

        result = s0 * da + s1 * db
        n = float(np.linalg.norm(result))
        q = Quat.__new__(Quat)
        q._data = (result / n).astype(np.float32)
        return q

    # ------------------------------------------------------------------
    # Comparison / display
    # ------------------------------------------------------------------

    def approx_eq(self, other: Quat, eps: float = 1e-6) -> bool:
        """
        Approximate equality, accounting for the q ≡ -q double-cover of SO(3).

        Parameters
        ----------
        other : Quat
        eps   : float, default 1e-6

        Returns
        -------
        bool

        Example
        -------
        >>> Quat.identity().approx_eq(Quat.identity())
        True
        """
        # Two quaternions represent the same rotation if q ≈ other OR q ≈ -other
        diff_pos = np.max(np.abs(self._data - other._data))
        diff_neg = np.max(np.abs(self._data + other._data))
        return bool(min(float(diff_pos), float(diff_neg)) <= eps)

    def __repr__(self) -> str:
        w, x, y, z = self._data
        return f"Quat(w={w:.6g}, x={x:.6g}, y={y:.6g}, z={z:.6g})"
