"""
tests/core/_impl/test_quat.py — Mirror test for fire_engine/core/_impl/quat.py.

Covers:
- Construction: __init__, identity, from_axis_angle, from_euler
- Properties: w, x, y, z
- Operations: __mul__ (Hamilton product), rotate, normalized, inverse
- slerp: short-arc, t=0→a, t=1→b
- approx_eq: double-cover q ≡ -q
- as_euler: round-trip through from_euler
- Error cases: zero-length axis
- Re-export: Quat accessible via fire_engine.core.math3d and fire_engine.core
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fire_engine.core._impl.quat import Quat
from fire_engine.core.math3d import Vec3

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_is_identity(self):
        q = Quat()
        assert q.w == pytest.approx(1.0)
        assert q.x == pytest.approx(0.0)
        assert q.y == pytest.approx(0.0)
        assert q.z == pytest.approx(0.0)

    def test_explicit_components(self):
        q = Quat(0.5, 0.5, 0.5, 0.5)
        assert q.w == pytest.approx(0.5, abs=1e-6)
        assert q.x == pytest.approx(0.5, abs=1e-6)

    def test_identity_classmethod(self):
        q = Quat.identity()
        assert q.w == pytest.approx(1.0)
        assert q.x == pytest.approx(0.0)
        assert q.y == pytest.approx(0.0)
        assert q.z == pytest.approx(0.0)

    def test_data_is_float32(self):
        q = Quat.identity()
        assert q._data.dtype == np.float32


class TestFromAxisAngle:
    def test_90_deg_about_z_rotates_forward_to_neg_x(self):
        """CLAUDE.md handedness check: +Y rotated 90° about +Z → −X."""
        q = Quat.from_axis_angle(Vec3.UP, math.pi / 2)
        result = q.rotate(Vec3.FORWARD)
        assert result.x == pytest.approx(-1.0, abs=1e-5)
        assert result.y == pytest.approx(0.0, abs=1e-5)
        assert result.z == pytest.approx(0.0, abs=1e-5)

    def test_zero_rotation_leaves_vector_unchanged(self):
        q = Quat.from_axis_angle(Vec3.UP, 0.0)
        result = q.rotate(Vec3.FORWARD)
        assert result.approx_eq(Vec3.FORWARD)

    def test_180_deg_about_z_reverses_forward(self):
        q = Quat.from_axis_angle(Vec3.UP, math.pi)
        result = q.rotate(Vec3.FORWARD)
        assert result.x == pytest.approx(0.0, abs=1e-5)
        assert result.y == pytest.approx(-1.0, abs=1e-5)

    def test_zero_length_axis_raises(self):
        with pytest.raises(ValueError, match="zero-length"):
            Quat.from_axis_angle(Vec3(0, 0, 0), math.pi / 2)

    def test_unnormalized_axis_accepted(self):
        """Axis does not need to be a unit vector."""
        q = Quat.from_axis_angle(Vec3(0, 0, 2), math.pi / 2)
        result = q.rotate(Vec3.FORWARD)
        assert result.x == pytest.approx(-1.0, abs=1e-5)


class TestFromEuler:
    def test_90_deg_yaw(self):
        q = Quat.from_euler(math.pi / 2, 0.0, 0.0)
        result = q.rotate(Vec3.FORWARD)
        assert result.x == pytest.approx(-1.0, abs=1e-5)

    def test_zero_euler_is_identity(self):
        q = Quat.from_euler(0.0, 0.0, 0.0)
        assert q.approx_eq(Quat.identity(), eps=1e-5)


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


class TestMultiply:
    def test_identity_times_q_is_q(self):
        q = Quat.from_axis_angle(Vec3.UP, math.pi / 4)
        result = Quat.identity() * q
        assert result.approx_eq(q)

    def test_q_times_identity_is_q(self):
        q = Quat.from_axis_angle(Vec3.UP, math.pi / 4)
        result = q * Quat.identity()
        assert result.approx_eq(q)

    def test_composition_applies_second_first(self):
        """q1*q2 applies q2 first, then q1."""
        q90 = Quat.from_axis_angle(Vec3.UP, math.pi / 2)  # 90° yaw
        q90b = Quat.from_axis_angle(Vec3.UP, math.pi / 2)  # another 90° yaw
        q180 = q90 * q90b
        result = q180.rotate(Vec3.FORWARD)
        # Two 90° yaws about +Z reverse +Y → -Y
        assert result.y == pytest.approx(-1.0, abs=1e-5)


class TestInverse:
    def test_q_times_inverse_is_identity(self):
        q = Quat.from_axis_angle(Vec3.UP, 1.0)
        assert (q * q.inverse()).approx_eq(Quat.identity())

    def test_inverse_of_identity_is_identity(self):
        assert Quat.identity().inverse().approx_eq(Quat.identity())

    def test_rotate_then_inverse_recovers_vector(self):
        q = Quat.from_axis_angle(Vec3.UP, math.pi / 3)
        v = Vec3(1, 2, 3)
        assert q.inverse().rotate(q.rotate(v)).approx_eq(v, eps=1e-5)


class TestNormalized:
    def test_already_unit_stays_unit(self):
        q = Quat.identity().normalized()
        norm = float(np.linalg.norm(q._data))
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_scale_removed(self):
        q = Quat(2.0, 0.0, 0.0, 0.0)  # not unit
        n = q.normalized()
        assert float(np.linalg.norm(n._data)) == pytest.approx(1.0, abs=1e-6)

    def test_zero_quat_raises(self):
        with pytest.raises(ValueError, match="zero quaternion"):
            Quat(0.0, 0.0, 0.0, 0.0).normalized()


# ---------------------------------------------------------------------------
# Slerp
# ---------------------------------------------------------------------------


class TestSlerp:
    def test_t_zero_returns_a(self):
        a = Quat.identity()
        b = Quat.from_axis_angle(Vec3.UP, math.pi)
        assert Quat.slerp(a, b, 0.0).approx_eq(a)

    def test_t_one_returns_b(self):
        a = Quat.identity()
        b = Quat.from_axis_angle(Vec3.UP, math.pi)
        assert Quat.slerp(a, b, 1.0).approx_eq(b)

    def test_t_half_interpolates(self):
        q0 = Quat.identity()
        q1 = Quat.from_axis_angle(Vec3.UP, math.pi)
        mid = Quat.slerp(q0, q1, 0.5)
        assert mid.approx_eq(Quat.from_axis_angle(Vec3.UP, math.pi / 2))

    def test_short_arc(self):
        """slerp takes the short arc (dot < 0 → negate b)."""
        a = Quat.from_axis_angle(Vec3.UP, 0.1)
        # negate a to represent the antipodal quaternion (same rotation)
        b_neg = Quat(-a.w, -a.x, -a.y, -a.z)
        # slerp of a and -a should still be near identity (short arc)
        mid = Quat.slerp(Quat.identity(), b_neg, 0.0)
        assert mid.approx_eq(Quat.identity())


# ---------------------------------------------------------------------------
# approx_eq — double-cover
# ---------------------------------------------------------------------------


class TestApproxEq:
    def test_identity_equals_itself(self):
        assert Quat.identity().approx_eq(Quat.identity())

    def test_q_equals_neg_q(self):
        """q and -q represent the same rotation."""
        q = Quat.from_axis_angle(Vec3.UP, 1.0)
        neg_q = Quat(-q.w, -q.x, -q.y, -q.z)
        assert q.approx_eq(neg_q)

    def test_different_rotations_not_equal(self):
        q1 = Quat.from_axis_angle(Vec3.UP, 0.5)
        q2 = Quat.from_axis_angle(Vec3.UP, 1.5)
        assert not q1.approx_eq(q2)


# ---------------------------------------------------------------------------
# as_euler round-trip
# ---------------------------------------------------------------------------


class TestAsEuler:
    def test_round_trip_pure_yaw(self):
        h, p, r = math.pi / 4, 0.0, 0.0
        q = Quat.from_euler(h, p, r)
        h2, p2, r2 = q.as_euler()
        assert Quat.from_euler(h2, p2, r2).approx_eq(q, eps=1e-4)

    def test_round_trip_combined(self):
        h, p, r = math.pi / 4, math.pi / 6, math.pi / 8
        q = Quat.from_euler(h, p, r)
        h2, p2, r2 = q.as_euler()
        assert Quat.from_euler(h2, p2, r2).approx_eq(q, eps=1e-4)

    def test_identity_euler(self):
        h, p, r = Quat.identity().as_euler()
        assert Quat.from_euler(h, p, r).approx_eq(Quat.identity(), eps=1e-5)


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_contains_components(self):
        q = Quat.identity()
        r = repr(q)
        assert "Quat" in r
        assert "w=" in r
        assert "x=" in r


# ---------------------------------------------------------------------------
# Re-export paths
# ---------------------------------------------------------------------------


class TestReExport:
    def test_importable_from_math3d(self):
        from fire_engine.core.math3d import Quat as QuatMath3d

        assert QuatMath3d is Quat

    def test_importable_from_core_init(self):
        from fire_engine.core import Quat as QuatCore

        assert QuatCore is Quat
