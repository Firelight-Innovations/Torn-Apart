"""
tests/render/_impl/test_transform_math.py — Headless tests for render/_impl/transform_math.py.

Tests trs_matrix and mat3_to_quat with known mathematical inputs.  No panda3d.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fire_engine.core.math3d import Quat, Vec3
from fire_engine.render._impl.transform_math import mat3_to_quat, trs_matrix


class TestTrsMatrix:
    """Known-input / known-output tests for trs_matrix."""

    def test_identity_trs_is_identity_matrix(self) -> None:
        m = trs_matrix(Vec3(0, 0, 0), Quat.identity(), Vec3(1, 1, 1))
        assert m.shape == (4, 4)
        np.testing.assert_allclose(m, np.eye(4), atol=1e-6)

    def test_translation_fills_last_column(self) -> None:
        m = trs_matrix(Vec3(1.0, 2.0, 3.0), Quat.identity(), Vec3(1, 1, 1))
        assert abs(m[0, 3] - 1.0) < 1e-6
        assert abs(m[1, 3] - 2.0) < 1e-6
        assert abs(m[2, 3] - 3.0) < 1e-6
        # Bottom-right is always 1
        assert abs(m[3, 3] - 1.0) < 1e-6

    def test_scale_multiplies_rotation_columns(self) -> None:
        m = trs_matrix(Vec3(0, 0, 0), Quat.identity(), Vec3(2.0, 3.0, 4.0))
        # For identity rotation, diagonal of upper-left 3x3 = scale values.
        assert abs(m[0, 0] - 2.0) < 1e-6
        assert abs(m[1, 1] - 3.0) < 1e-6
        assert abs(m[2, 2] - 4.0) < 1e-6

    def test_return_type_is_float64(self) -> None:
        m = trs_matrix(Vec3(0, 0, 0), Quat.identity(), Vec3(1, 1, 1))
        assert m.dtype == np.float64

    def test_90_deg_z_rotation_matrix(self) -> None:
        """90-degree yaw around Z maps +X → +Y, +Y → -X."""
        q = Quat.from_axis_angle(Vec3.UP, math.pi / 2)
        m = trs_matrix(Vec3(0, 0, 0), q, Vec3(1, 1, 1))
        # Row 0 = [0, -1, 0, 0] (new X axis)
        np.testing.assert_allclose(m[0, :3], [0.0, -1.0, 0.0], atol=1e-5)
        # Row 1 = [1, 0, 0, 0] (new Y axis)
        np.testing.assert_allclose(m[1, :3], [1.0, 0.0, 0.0], atol=1e-5)

    def test_determinism(self) -> None:
        """Same inputs always produce the same matrix."""
        pos = Vec3(5.0, -3.0, 1.0)
        rot = Quat.from_axis_angle(Vec3.RIGHT, math.pi / 4)
        scale = Vec3(2.0, 2.0, 2.0)
        a = trs_matrix(pos, rot, scale)
        b = trs_matrix(pos, rot, scale)
        np.testing.assert_array_equal(a, b)


class TestMat3ToQuat:
    """Known-input / known-output tests for mat3_to_quat."""

    def test_identity_matrix_gives_identity_quaternion(self) -> None:
        q = mat3_to_quat(np.eye(3))
        # Identity quaternion: w=1, x=y=z=0
        data = q._data  # [w, x, y, z] float32
        assert abs(data[0] - 1.0) < 1e-5
        assert abs(data[1]) < 1e-5
        assert abs(data[2]) < 1e-5
        assert abs(data[3]) < 1e-5

    def test_returns_quat_instance(self) -> None:
        q = mat3_to_quat(np.eye(3))
        assert isinstance(q, Quat)

    def test_round_trip_90_deg_z(self) -> None:
        """TRS rotation matrix extracted back to a quaternion matches the original."""
        q_in = Quat.from_axis_angle(Vec3.UP, math.pi / 2)
        m = trs_matrix(Vec3(0, 0, 0), q_in, Vec3(1, 1, 1))
        m3 = m[:3, :3]
        q_out = mat3_to_quat(m3)
        # Compare normalised quaternions (note: q and -q represent same rotation)
        a = q_in.normalized()._data
        b = q_out.normalized()._data
        # Allow sign flip: dot product of abs values should be close to 1
        dot = float(np.abs(np.dot(a.astype(np.float64), b.astype(np.float64))))
        assert dot > 0.9999, f"Quaternion mismatch: dot={dot}"

    def test_round_trip_45_deg_x(self) -> None:
        q_in = Quat.from_axis_angle(Vec3.RIGHT, math.pi / 4)
        m = trs_matrix(Vec3(0, 0, 0), q_in, Vec3(1, 1, 1))
        q_out = mat3_to_quat(m[:3, :3])
        a = q_in.normalized()._data.astype(np.float64)
        b = q_out.normalized()._data.astype(np.float64)
        dot = float(np.abs(np.dot(a, b)))
        assert dot > 0.9999, f"Quaternion mismatch for 45° X: dot={dot}"

    def test_output_is_unit_quaternion(self) -> None:
        """mat3_to_quat always returns a normalised (unit) quaternion."""
        q = Quat.from_axis_angle(Vec3(1, 1, 0), math.pi / 3)
        m = trs_matrix(Vec3(0, 0, 0), q, Vec3(1, 1, 1))
        q_out = mat3_to_quat(m[:3, :3])
        mag = float(np.linalg.norm(q_out._data.astype(np.float64)))
        assert abs(mag - 1.0) < 1e-5, f"Not unit: |q|={mag}"

    def test_determinism(self) -> None:
        m = np.eye(3)
        a = mat3_to_quat(m)._data
        b = mat3_to_quat(m)._data
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "angle",
    [0.0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2, math.pi * 2 / 3],
)
def test_trs_round_trip_various_angles(angle: float) -> None:
    """trs_matrix → mat3_to_quat round-trip holds for many rotation angles."""
    q_in = Quat.from_axis_angle(Vec3(0, 0, 1), angle)
    m = trs_matrix(Vec3(0, 0, 0), q_in, Vec3(1, 1, 1))
    q_out = mat3_to_quat(m[:3, :3])
    a = q_in.normalized()._data.astype(np.float64)
    b = q_out.normalized()._data.astype(np.float64)
    dot = float(np.abs(np.dot(a, b)))
    assert dot > 0.9999, f"angle={angle:.4f}: dot={dot}"
