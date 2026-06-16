"""
Unit quaternion rotation type for the Torn Apart engine.

Moved to :mod:`fire_engine.core._impl.quat` to satisfy the per-directory
module-count limit.  The conventional import paths remain valid:

    from fire_engine.core.math3d import Quat          # re-exported by math3d
    from fire_engine.core import Quat                 # re-exported by __init__

No circular imports: ``Quat`` accesses ``Vec3`` at runtime through duck-typed
attribute access only (``._data`` numpy array); the type annotation uses
``TYPE_CHECKING`` to import ``Vec3`` for type-checker support without a
module-level dependency.

Docs: docs/systems/core.md
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fire_engine.core.math3d import Vec3


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
    >>> from fire_engine.core.math3d import Vec3, Quat
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
        >>> from fire_engine.core.math3d import Vec3, Quat
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
        >>> from fire_engine.core.math3d import Vec3, Quat
        >>> from math import pi
        >>> Quat.from_axis_angle(Vec3.UP, pi/2).rotate(Vec3.FORWARD)
        Vec3(-1.0, ~0, ~0)   # +Y rotates to −X under 90° about +Z
        """
        # axis._data is a float32 numpy (3,) array; axis.length is a float
        n = float(axis.length)
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
        >>> from fire_engine.core.math3d import Vec3, Quat
        >>> from math import pi
        >>> q = Quat.from_euler(pi/2, 0, 0)   # 90° yaw
        >>> q.rotate(Vec3.FORWARD).approx_eq(Vec3(-1, 0, 0), eps=1e-5)
        True
        """
        # Import Vec3 at call time to avoid a circular module-level import.
        # math3d imports quat at the bottom (after Vec3 is defined), so this
        # call-time import is always safe once math3d has been initialised.
        from fire_engine.core.math3d import Vec3

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
        >>> from fire_engine.core.math3d import Vec3, Quat
        >>> from math import pi
        >>> Quat.from_axis_angle(Vec3.UP, pi/2).rotate(Vec3.FORWARD)
        Vec3(-1.0, ~0, ~0)
        """
        # Import Vec3 at call time to avoid a circular module-level import.
        from fire_engine.core.math3d import Vec3

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
        >>> from fire_engine.core.math3d import Vec3, Quat
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
        >>> from fire_engine.core.math3d import Vec3, Quat
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
        >>> from fire_engine.core.math3d import Vec3, Quat
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
