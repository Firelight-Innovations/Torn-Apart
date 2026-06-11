"""
tests/test_math3d.py — Correctness tests for core/math3d.py.

Covers:
  - Vec3: arithmetic, dot, cross, normalized, lerp, constants
  - Quat: handedness check, slerp boundary, euler round-trip,
    rotate∘inverse identity, axis-angle correctness
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fire_engine.core.math3d import Vec3, Quat


# ===========================================================================
# Vec3 tests
# ===========================================================================

class TestVec3Constants:
    def test_zero(self):
        assert Vec3.ZERO.approx_eq(Vec3(0, 0, 0))

    def test_one(self):
        assert Vec3.ONE.approx_eq(Vec3(1, 1, 1))

    def test_up_is_z(self):
        """Z-up: UP must be (0, 0, 1)."""
        assert Vec3.UP.approx_eq(Vec3(0, 0, 1))

    def test_forward_is_y(self):
        """Z-up, forward=+Y."""
        assert Vec3.FORWARD.approx_eq(Vec3(0, 1, 0))

    def test_right_is_x(self):
        assert Vec3.RIGHT.approx_eq(Vec3(1, 0, 0))


class TestVec3Arithmetic:
    def test_add(self):
        a = Vec3(1, 2, 3)
        b = Vec3(4, 5, 6)
        assert (a + b).approx_eq(Vec3(5, 7, 9))

    def test_sub(self):
        a = Vec3(4, 5, 6)
        b = Vec3(1, 2, 3)
        assert (a - b).approx_eq(Vec3(3, 3, 3))

    def test_scalar_mul(self):
        assert Vec3(1, 2, 3).__mul__(2.0).approx_eq(Vec3(2, 4, 6))

    def test_scalar_rmul(self):
        assert (3.0 * Vec3(1, 0, 0)).approx_eq(Vec3(3, 0, 0))

    def test_neg(self):
        assert (-Vec3(1, -2, 3)).approx_eq(Vec3(-1, 2, -3))

    def test_eq_exact(self):
        v = Vec3(1.0, 2.0, 3.0)
        assert v == Vec3(1.0, 2.0, 3.0)
        assert v != Vec3(1.0, 2.0, 3.1)

    def test_approx_eq(self):
        assert Vec3(1, 0, 0).approx_eq(Vec3(1 + 1e-7, 0, 0))
        assert not Vec3(1, 0, 0).approx_eq(Vec3(2, 0, 0))


class TestVec3VectorMath:
    def test_dot_parallel(self):
        assert abs(Vec3.UP.dot(Vec3.UP) - 1.0) < 1e-6

    def test_dot_orthogonal(self):
        assert abs(Vec3.UP.dot(Vec3.FORWARD)) < 1e-6

    def test_dot_formula(self):
        a = Vec3(1, 2, 3)
        b = Vec3(4, 5, 6)
        assert abs(a.dot(b) - (4 + 10 + 18)) < 1e-5

    def test_cross_right_hand(self):
        """X × Y = Z (Z-up right-handed)."""
        c = Vec3.RIGHT.cross(Vec3.FORWARD)
        assert c.approx_eq(Vec3.UP, eps=1e-5)

    def test_cross_anticommutative(self):
        a = Vec3(1, 0, 0)
        b = Vec3(0, 1, 0)
        assert (a.cross(b)).approx_eq(-(b.cross(a)), eps=1e-5)

    def test_normalized(self):
        v = Vec3(3, 0, 0).normalized()
        assert v.approx_eq(Vec3(1, 0, 0))

    def test_normalized_length_one(self):
        v = Vec3(1, 2, 3).normalized()
        assert abs(v.length - 1.0) < 1e-6

    def test_normalized_zero_raises(self):
        with pytest.raises(ValueError):
            Vec3.ZERO.normalized()

    def test_length_squared(self):
        v = Vec3(3, 4, 0)
        assert abs(v.length_squared - 25.0) < 1e-5
        assert abs(v.length - 5.0) < 1e-5

    def test_lerp_midpoint(self):
        a = Vec3(0, 0, 0)
        b = Vec3(2, 0, 0)
        mid = a.lerp(b, 0.5)
        assert mid.approx_eq(Vec3(1, 0, 0))

    def test_lerp_t0_returns_self(self):
        a = Vec3(1, 2, 3)
        b = Vec3(4, 5, 6)
        assert a.lerp(b, 0.0).approx_eq(a)

    def test_lerp_t1_returns_other(self):
        a = Vec3(1, 2, 3)
        b = Vec3(4, 5, 6)
        assert a.lerp(b, 1.0).approx_eq(b)


class TestVec3Iteration:
    def test_iter(self):
        v = Vec3(1, 2, 3)
        assert list(v) == pytest.approx([1.0, 2.0, 3.0])

    def test_indexing(self):
        v = Vec3(7, 8, 9)
        assert v[0] == pytest.approx(7.0)
        assert v[1] == pytest.approx(8.0)
        assert v[2] == pytest.approx(9.0)

    def test_to_numpy_copy(self):
        v = Vec3(1, 2, 3)
        arr = v.to_numpy()
        assert arr.dtype == np.float32
        arr[0] = 99.0           # mutating copy should not affect v
        assert v.x == pytest.approx(1.0)

    def test_from_numpy(self):
        arr = np.array([5, 6, 7], dtype=np.float64)
        v = Vec3.from_numpy(arr)
        assert v.approx_eq(Vec3(5, 6, 7))


# ===========================================================================
# Quat tests
# ===========================================================================

class TestQuatHandedness:
    """
    THE critical handedness check for the whole engine.

    Rotating +Y (FORWARD) by +90° about +Z (UP) under a right-handed
    Z-up coordinate system must yield −X.

    Intuition: stand at the origin, face +Y (north). Turn left 90°
    (CCW when viewed from above = positive angle about +Z). You now face −X (west).
    """

    def test_90_about_z_rotates_forward_to_neg_x(self):
        q = Quat.from_axis_angle(Vec3.UP, math.pi / 2)
        result = q.rotate(Vec3.FORWARD)
        expected = Vec3(-1, 0, 0)
        assert result.approx_eq(expected, eps=1e-5), (
            f"Expected Vec3(-1,0,0) got {result} — handedness wrong!"
        )

    def test_90_about_z_rotates_right_to_forward(self):
        """Rotate +X by +90° about +Z → +Y."""
        q = Quat.from_axis_angle(Vec3.UP, math.pi / 2)
        result = q.rotate(Vec3.RIGHT)
        expected = Vec3(0, 1, 0)
        assert result.approx_eq(expected, eps=1e-5)

    def test_180_about_z_negates_forward(self):
        q = Quat.from_axis_angle(Vec3.UP, math.pi)
        result = q.rotate(Vec3.FORWARD)
        expected = Vec3(0, -1, 0)
        assert result.approx_eq(expected, eps=1e-5)


class TestQuatIdentity:
    def test_identity_rotates_forward_unchanged(self):
        q = Quat.identity()
        assert q.rotate(Vec3.FORWARD).approx_eq(Vec3.FORWARD)

    def test_identity_rotates_up_unchanged(self):
        assert Quat.identity().rotate(Vec3.UP).approx_eq(Vec3.UP)

    def test_identity_components(self):
        q = Quat.identity()
        assert abs(q.w - 1.0) < 1e-6
        assert abs(q.x) < 1e-6
        assert abs(q.y) < 1e-6
        assert abs(q.z) < 1e-6


class TestQuatInverse:
    def test_inverse_cancels_rotation(self):
        """q.inverse().rotate(q.rotate(v)) ≈ v for any v."""
        q = Quat.from_axis_angle(Vec3.UP, math.pi / 3)
        v = Vec3(1, 2, 3)
        recovered = q.inverse().rotate(q.rotate(v))
        assert recovered.approx_eq(v, eps=1e-5)

    def test_q_times_q_inverse_is_identity(self):
        q = Quat.from_axis_angle(Vec3(1, 1, 0).normalized(), 1.2)
        prod = q * q.inverse()
        assert prod.approx_eq(Quat.identity(), eps=1e-5)


class TestQuatSlerp:
    def test_slerp_t0_returns_a(self):
        a = Quat.from_axis_angle(Vec3.UP, 0.3)
        b = Quat.from_axis_angle(Vec3.UP, 1.2)
        result = Quat.slerp(a, b, 0.0)
        assert result.approx_eq(a, eps=1e-5)

    def test_slerp_t1_returns_b(self):
        a = Quat.from_axis_angle(Vec3.UP, 0.3)
        b = Quat.from_axis_angle(Vec3.UP, 1.2)
        result = Quat.slerp(a, b, 1.0)
        assert result.approx_eq(b, eps=1e-5)

    def test_slerp_midpoint_angle(self):
        """Slerp at t=0.5 should give half the angle."""
        a = Quat.identity()
        b = Quat.from_axis_angle(Vec3.UP, math.pi)
        mid = Quat.slerp(a, b, 0.5)
        expected = Quat.from_axis_angle(Vec3.UP, math.pi / 2)
        assert mid.approx_eq(expected, eps=1e-4)

    def test_slerp_identity_to_self_is_self(self):
        q = Quat.from_axis_angle(Vec3.RIGHT, 0.7)
        assert Quat.slerp(q, q, 0.5).approx_eq(q, eps=1e-5)


class TestQuatEulerRoundTrip:
    """
    from_euler → as_euler → from_euler must produce functionally identical
    rotations (i.e. rotate the same test vectors to the same places).
    """

    @pytest.mark.parametrize("h,p,r", [
        (0.5, 0.3, 0.1),
        (1.0, -0.4, 0.2),
        (-0.8, 0.6, -0.3),
        (0.0, 0.0, 0.0),
        (math.pi / 4, math.pi / 6, math.pi / 8),
    ])
    def test_euler_round_trip(self, h: float, p: float, r: float):
        q1 = Quat.from_euler(h, p, r)
        h2, p2, r2 = q1.as_euler()
        q2 = Quat.from_euler(h2, p2, r2)
        # Check that both quats rotate the canonical vectors identically
        for v in (Vec3.FORWARD, Vec3.RIGHT, Vec3.UP):
            r1 = q1.rotate(v)
            r2 = q2.rotate(v)
            assert r1.approx_eq(r2, eps=1e-4), (
                f"Round-trip failed for H={h},P={p},R={r}: "
                f"v={v} → q1 gave {r1}, q2 gave {r2}"
            )


class TestQuatNormalized:
    def test_unnormalised_normalises(self):
        # Manually build a non-unit quaternion
        q = Quat(2.0, 0.0, 0.0, 0.0)
        n = q.normalized()
        assert abs(n.w - 1.0) < 1e-6

    def test_normalized_preserves_rotation(self):
        q = Quat.from_axis_angle(Vec3.UP, 1.0)
        n = q.normalized()
        assert n.approx_eq(q, eps=1e-5)


class TestQuatMultiplication:
    def test_two_half_turns_is_full_turn(self):
        q_half = Quat.from_axis_angle(Vec3.UP, math.pi / 2)
        q_full = q_half * q_half
        expected = Quat.from_axis_angle(Vec3.UP, math.pi)
        assert q_full.approx_eq(expected, eps=1e-5)

    def test_mul_order_applies_right_first(self):
        """
        q1 * q2 applies q2 first, then q1.
        Rotate +Y by 90° about +Z → −X, then 90° about +Z again → −Y.
        """
        q = Quat.from_axis_angle(Vec3.UP, math.pi / 2)
        combined = q * q  # q applied twice = 180° about +Z
        result = combined.rotate(Vec3.FORWARD)
        expected = Vec3(0, -1, 0)
        assert result.approx_eq(expected, eps=1e-5)


class TestQuatAxisAngleVariants:
    def test_zero_angle_is_identity(self):
        q = Quat.from_axis_angle(Vec3.UP, 0.0)
        assert q.approx_eq(Quat.identity(), eps=1e-5)

    def test_360_is_identity(self):
        q = Quat.from_axis_angle(Vec3.UP, 2 * math.pi)
        # q or -q is identity (double cover)
        assert q.approx_eq(Quat.identity(), eps=1e-5)

    def test_axis_need_not_be_normalized(self):
        """from_axis_angle should normalise the axis internally."""
        q1 = Quat.from_axis_angle(Vec3.UP, 1.0)
        q2 = Quat.from_axis_angle(Vec3(0, 0, 5), 1.0)
        assert q1.approx_eq(q2, eps=1e-5)
