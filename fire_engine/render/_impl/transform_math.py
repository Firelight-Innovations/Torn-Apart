"""
render/_impl/transform_math.py — Pure-numpy math helpers for Transform.

Extracted from render/transform.py to keep that module under 500 lines (C0302).
Not part of the public API.

Docs: docs/systems/render.md
"""

from __future__ import annotations

import numpy as np

from fire_engine.core.math3d import Quat, Vec3


def trs_matrix(pos: Vec3, rot: Quat, scale: Vec3) -> np.ndarray:
    """
    Build a 4×4 TRS (translation × rotation × scale) matrix.

    Uses float64 internally; callers may cast if needed.

    Parameters
    ----------
    pos   : Vec3 — translation in meters
    rot   : Quat — rotation quaternion (unit)
    scale : Vec3 — scale per axis

    Returns
    -------
    np.ndarray shape (4, 4) float64
    """
    w, x, y, z = (float(c) for c in rot._data)
    sx, sy, sz = float(scale.x), float(scale.y), float(scale.z)

    # Rotation matrix from quaternion
    m = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 0.0],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), 0.0],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # Apply scale to rotation columns
    m[0, :3] *= sx
    m[1, :3] *= sy
    m[2, :3] *= sz

    # Translation
    m[0, 3] = float(pos.x)
    m[1, 3] = float(pos.y)
    m[2, 3] = float(pos.z)

    return m


def mat3_to_quat(m: np.ndarray) -> Quat:
    """
    Convert a 3×3 rotation matrix to a unit quaternion (Shepperd's method).

    Parameters
    ----------
    m : np.ndarray shape (3, 3) — orthonormal rotation matrix.
        Rows = output axes in the *input* frame:
            m[0] = new X-axis (right)
            m[1] = new Y-axis (forward)
            m[2] = new Z-axis (up)

    Returns
    -------
    Quat — unit quaternion

    Note
    ----
    The matrix is stored with rows as the destination axes for each local
    basis vector (i.e. the *transpose* of a column-basis matrix).
    """
    trace = m[0, 0] + m[1, 1] + m[2, 2]

    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    q = Quat.__new__(Quat)
    q._data = np.array([w, x, y, z], dtype=np.float32)
    return q.normalized()
