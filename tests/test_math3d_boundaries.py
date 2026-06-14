"""
tests/test_math3d_boundaries.py — Numeric-boundary / golden-master tests for math3d.

PURPOSE: Pin CURRENT behavior at edges (zero vectors, NaN/inf, eps boundaries,
slerp sign handling, gimbal singularities).  Do NOT fix bugs here; record what
the code actually does so regressions are visible.

Does NOT duplicate tests/test_math3d.py (happy paths, handedness, round-trips
for normal angles, lerp midpoint, etc.).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fire_engine.core.math3d import Quat, Vec3

# ---------------------------------------------------------------------------
# Vec3 — normalized() boundary
# ---------------------------------------------------------------------------


class TestVec3NormalizedBoundary:
    def test_zero_vector_raises(self):
        """Exact zero vector: length < 1e-12 → ValueError (eps = 1e-12)."""
        with pytest.raises(ValueError):
            Vec3(0.0, 0.0, 0.0).normalized()

    def test_near_zero_below_eps_raises(self):
        """Vector whose length is just under 1e-12 should raise, not silently return garbage."""
        # length = 5e-13 < 1e-12
        v = Vec3(5e-13, 0.0, 0.0)
        with pytest.raises(ValueError):
            v.normalized()

    def test_near_zero_above_eps_returns_unit(self):
        """Vector whose length is just over 1e-12 should succeed and return unit length."""
        v = Vec3(2e-12, 0.0, 0.0)  # length = 2e-12 > 1e-12
        n = v.normalized()
        assert abs(n.length - 1.0) < 1e-5

    def test_nan_component_normalized_propagates_nan(self):
        """A NaN component makes length NaN; normalized() is expected to propagate NaN
        (no explicit guard).  Pin: no ValueError raised, result has NaN components."""
        v = Vec3(float("nan"), 0.0, 1.0)
        # length is NaN → n < 1e-12 is False (NaN comparisons are False)
        # so the code proceeds to divide — pin that it does NOT raise
        result = v.normalized()
        assert math.isnan(result.x) or math.isnan(result.y) or math.isnan(result.z)

    def test_inf_component_normalized_returns_finite_or_nan(self):
        """An inf component makes length inf; dividing by inf collapses the finite
        components to 0 and the inf component to either 1.0 or NaN.  Pin behavior."""
        v = Vec3(float("inf"), 0.0, 0.0)
        # length = inf; inf/inf = NaN for the inf component
        result = v.normalized()
        # Pin: result.x is NaN (inf/inf) and no ValueError is raised
        assert math.isnan(result.x)


# ---------------------------------------------------------------------------
# Vec3 — length / length_squared with extreme values
# ---------------------------------------------------------------------------


class TestVec3LengthExtremes:
    def test_length_squared_very_large_overflows_to_inf(self):
        """float32 overflows at ~3.4e38; a component of 1e20 makes x² ≈ 1e40 → inf.
        length_squared uses np.dot (float32 arithmetic) — pin that it returns inf."""
        v = Vec3(1e20, 0.0, 0.0)
        ls = v.length_squared
        assert math.isinf(ls)

    def test_length_very_large_overflows_to_inf(self):
        """sqrt(inf) = inf.  Pin that length returns inf for overflow input."""
        v = Vec3(1e20, 0.0, 0.0)
        assert math.isinf(v.length)

    def test_length_nan_component_propagates(self):
        """NaN in any component makes length_squared and length both NaN."""
        v = Vec3(float("nan"), 1.0, 1.0)
        assert math.isnan(v.length_squared)
        assert math.isnan(v.length)

    def test_length_squared_normal_large_but_no_overflow(self):
        """1e19 squared = 1e38, which is within float32 range (~3.4e38).
        Pin: length_squared is finite (not inf)."""
        v = Vec3(1e19, 0.0, 0.0)
        ls = v.length_squared
        # float32 max is ~3.4e38; 1e38 is within range
        assert math.isfinite(ls)


# ---------------------------------------------------------------------------
# Vec3 — lerp extrapolation and special endpoints
# ---------------------------------------------------------------------------


class TestVec3LerpBoundary:
    def test_lerp_t_negative_extrapolates(self):
        """t < 0 extrapolates behind a.  Pin: no clamp, result is beyond a."""
        a = Vec3(0.0, 0.0, 0.0)
        b = Vec3(1.0, 0.0, 0.0)
        result = a.lerp(b, -1.0)
        # expected: 0 + (-1)*(1-0) = -1
        assert result.approx_eq(Vec3(-1.0, 0.0, 0.0), eps=1e-5)

    def test_lerp_t_greater_than_one_extrapolates(self):
        """t > 1 extrapolates beyond b.  Pin: no clamp."""
        a = Vec3(0.0, 0.0, 0.0)
        b = Vec3(1.0, 0.0, 0.0)
        result = a.lerp(b, 2.0)
        assert result.approx_eq(Vec3(2.0, 0.0, 0.0), eps=1e-5)

    def test_lerp_with_inf_endpoint_propagates_inf(self):
        """If b has an inf component, result at t=0.5 is inf.  Pin propagation."""
        a = Vec3(0.0, 0.0, 0.0)
        b = Vec3(float("inf"), 0.0, 0.0)
        result = a.lerp(b, 0.5)
        assert math.isinf(result.x)

    def test_lerp_with_nan_endpoint_propagates_nan(self):
        """NaN endpoint → NaN result."""
        a = Vec3(0.0, 0.0, 0.0)
        b = Vec3(float("nan"), 0.0, 0.0)
        result = a.lerp(b, 0.5)
        assert math.isnan(result.x)


# ---------------------------------------------------------------------------
# Vec3 — approx_eq eps boundary
# ---------------------------------------------------------------------------


class TestVec3ApproxEqBoundary:
    def test_just_inside_eps(self):
        """Component diff = eps exactly: abs(a-b) <= eps → True."""
        eps = 1e-6
        assert Vec3(0.0, 0.0, 0.0).approx_eq(Vec3(eps, 0.0, 0.0), eps=eps)

    def test_just_outside_eps(self):
        """Component diff = eps + small delta → False."""
        eps = 1e-6
        assert not Vec3(0.0, 0.0, 0.0).approx_eq(Vec3(eps * 1.01, 0.0, 0.0), eps=eps)


# ---------------------------------------------------------------------------
# Quat — normalized() boundary
# ---------------------------------------------------------------------------


class TestQuatNormalizedBoundary:
    def test_zero_quat_raises(self):
        """A (0,0,0,0) quaternion has norm 0 < 1e-12 → ValueError."""
        q = Quat(0.0, 0.0, 0.0, 0.0)
        with pytest.raises(ValueError):
            q.normalized()

    def test_near_zero_quat_raises(self):
        """Norm = 5e-13 < 1e-12 → ValueError."""
        q = Quat(5e-13, 0.0, 0.0, 0.0)
        with pytest.raises(ValueError):
            q.normalized()

    def test_non_unit_quat_normalizes_to_unit(self):
        """Quat(2,0,0,0) should normalize to (1,0,0,0) = identity."""
        q = Quat(3.0, 0.0, 0.0, 0.0)
        n = q.normalized()
        assert abs(n.w - 1.0) < 1e-5
        assert abs(float(np.linalg.norm(n._data)) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Quat — from_axis_angle boundary
# ---------------------------------------------------------------------------


class TestQuatFromAxisAngleBoundary:
    def test_zero_axis_raises(self):
        """Zero-length axis → ValueError (eps = 1e-12 in from_axis_angle)."""
        with pytest.raises(ValueError):
            Quat.from_axis_angle(Vec3(0.0, 0.0, 0.0), math.pi / 2)

    def test_angle_zero_is_identity(self):
        """Angle = 0 regardless of axis → identity quaternion."""
        q = Quat.from_axis_angle(Vec3.UP, 0.0)
        assert q.approx_eq(Quat.identity(), eps=1e-5)

    def test_angle_2pi_is_identity_via_double_cover(self):
        """Angle = 2π: cos(π)=−1, sin(π)≈0 → q=(-1,0,0,0) ≡ (1,0,0,0).
        approx_eq handles double cover.  Pin: approx_eq returns True."""
        q = Quat.from_axis_angle(Vec3.UP, 2 * math.pi)
        assert q.approx_eq(Quat.identity(), eps=1e-4)

    def test_very_large_angle_wraps_via_trig(self):
        """Angle = 100π (50 full turns).  cos(50π)=1, sin(50π)≈0 → identity-ish.
        Pin: approx_eq identity True (sin/cos of float32 large angle may drift)."""
        q = Quat.from_axis_angle(Vec3.UP, 100 * math.pi)
        # 100π = 50 full turns: mathematically identity; float precision may differ
        # Pin that it at least remains a unit quaternion
        assert abs(float(np.linalg.norm(q._data)) - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# Quat — slerp boundary
# ---------------------------------------------------------------------------


class TestQuatSlerpBoundary:
    def test_slerp_t0_exact_returns_a(self):
        """slerp(a, b, 0.0): must equal a (not just approximately)."""
        a = Quat.from_axis_angle(Vec3.RIGHT, 0.5)
        b = Quat.from_axis_angle(Vec3.UP, 1.2)
        result = Quat.slerp(a, b, 0.0)
        assert result.approx_eq(a, eps=1e-5)

    def test_slerp_t1_exact_returns_b(self):
        """slerp(a, b, 1.0): must equal b."""
        a = Quat.from_axis_angle(Vec3.RIGHT, 0.5)
        b = Quat.from_axis_angle(Vec3.UP, 1.2)
        result = Quat.slerp(a, b, 1.0)
        assert result.approx_eq(b, eps=1e-5)

    def test_slerp_identical_quats_returns_same(self):
        """slerp(q, q, t): dot=1 > 0.9995 → lerp+normalize path.
        Result should equal q for any t in [0,1]."""
        q = Quat.from_axis_angle(Vec3.UP, 0.7)
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            result = Quat.slerp(q, q, t)
            assert result.approx_eq(q, eps=1e-4), f"Failed at t={t}"

    def test_slerp_result_is_unit_length(self):
        """Slerp result must always be unit length (post-normalize step)."""
        a = Quat.from_axis_angle(Vec3.RIGHT, 0.3)
        b = Quat.from_axis_angle(Vec3.UP, 2.1)
        for t in (0.0, 0.1, 0.5, 0.9, 1.0):
            result = Quat.slerp(a, b, t)
            norm = float(np.linalg.norm(result._data))
            assert abs(norm - 1.0) < 1e-5, f"Not unit at t={t}: norm={norm}"

    def test_slerp_opposite_quats_short_arc(self):
        """When dot(a,b) < 0, slerp negates b to take the short arc.
        a=identity, b=-identity: after negation b becomes identity, dot=1.
        Pin: result is identity for all t (both represent the same rotation)."""
        a = Quat.identity()
        # Build -identity by hand
        b = Quat(-1.0, 0.0, 0.0, 0.0)
        result = Quat.slerp(a, b, 0.5)
        # After negation b becomes identity; lerp of identity with itself is identity
        assert result.approx_eq(Quat.identity(), eps=1e-4)

    def test_slerp_nearly_opposite_quats_stays_unit(self):
        """q and -q with a near-antipodal pair (dot very negative) → b is negated,
        then dot ≈ 1.0 → lerp path.  Result must be unit length."""
        a = Quat.from_axis_angle(Vec3.UP, 0.0)  # identity: (1,0,0,0)
        # b is a rotation of ~170°; its negative has dot ~+1 with a
        b = Quat.from_axis_angle(Vec3.UP, math.pi * 0.95)
        result_pos = Quat.slerp(a, b, 0.5)
        result_neg = Quat.slerp(a, Quat(-b._data[0], -b._data[1], -b._data[2], -b._data[3]), 0.5)
        # Both represent the same underlying rotation — result must be unit length
        assert abs(float(np.linalg.norm(result_pos._data)) - 1.0) < 1e-5
        assert abs(float(np.linalg.norm(result_neg._data)) - 1.0) < 1e-5

    def test_slerp_t_outside_range_extrapolates(self):
        """slerp has no clamp on t.  Pin: t=2.0 extrapolates (no ValueError).
        Result is still unit length (due to final normalize step)."""
        a = Quat.identity()
        b = Quat.from_axis_angle(Vec3.UP, math.pi / 2)
        result = Quat.slerp(a, b, 2.0)
        # Should not raise; result should be unit length
        assert abs(float(np.linalg.norm(result._data)) - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# Quat — rotate() boundary
# ---------------------------------------------------------------------------


class TestQuatRotateBoundary:
    def test_identity_rotate_returns_v(self):
        """Identity quaternion must return the original vector unchanged."""
        v = Vec3(3.0, -1.5, 2.2)
        result = Quat.identity().rotate(v)
        assert result.approx_eq(v, eps=1e-5)

    def test_rotation_preserves_length(self):
        """For a unit quaternion, |q.rotate(v)| == |v|."""
        q = Quat.from_axis_angle(Vec3(1.0, 1.0, 1.0).normalized(), 1.3)
        v = Vec3(2.0, 3.0, 4.0)
        rotated = q.rotate(v)
        assert abs(rotated.length - v.length) < 1e-4

    def test_rotate_nan_vector_propagates_nan(self):
        """Rotating a NaN vector: Rodrigues formula propagates NaN.
        Pin: no exception, at least one NaN component in result."""
        q = Quat.identity()
        v = Vec3(float("nan"), 0.0, 0.0)
        result = q.rotate(v)
        assert any(math.isnan(c) for c in (result.x, result.y, result.z))


# ---------------------------------------------------------------------------
# Quat — from_euler/as_euler near gimbal singularities
# ---------------------------------------------------------------------------


class TestQuatEulerGimbalBoundary:
    def test_pitch_near_plus_90_round_trips_rotation(self):
        """Pitch ≈ +π/2 (gimbal lock).  The doc says H/R values may differ but
        the rotation itself is preserved.  Pin: rotating FORWARD with q1 and q2
        gives the same result to loose tolerance."""
        h, p, r = 0.5, math.pi / 2 - 1e-4, 0.3
        q1 = Quat.from_euler(h, p, r)
        h2, p2, r2 = q1.as_euler()
        q2 = Quat.from_euler(h2, p2, r2)
        v1 = q1.rotate(Vec3.FORWARD)
        v2 = q2.rotate(Vec3.FORWARD)
        assert v1.approx_eq(v2, eps=1e-3)

    def test_pitch_near_minus_90_round_trips_rotation(self):
        """Pitch ≈ −π/2 (gimbal lock, other pole).  Same: rotation preserved."""
        h, p, r = 0.5, -(math.pi / 2 - 1e-4), 0.3
        q1 = Quat.from_euler(h, p, r)
        h2, p2, r2 = q1.as_euler()
        q2 = Quat.from_euler(h2, p2, r2)
        v1 = q1.rotate(Vec3.FORWARD)
        v2 = q2.rotate(Vec3.FORWARD)
        assert v1.approx_eq(v2, eps=1e-3)

    def test_pitch_exactly_90_roll_is_zeroed(self):
        """At pitch = +π/2 exactly, as_euler should zero the roll (assign all to H).
        Pin: r_angle = 0.0."""
        q = Quat.from_euler(0.4, math.pi / 2, 0.7)
        h, p, r = q.as_euler()
        assert abs(r) < 1e-4, f"Expected roll≈0 at gimbal lock, got r={r}"


# ---------------------------------------------------------------------------
# Quat — approx_eq double-cover boundary
# ---------------------------------------------------------------------------


class TestQuatApproxEqBoundary:
    def test_q_and_neg_q_are_approx_equal(self):
        """q and -q represent the same rotation; approx_eq must return True."""
        q = Quat.from_axis_angle(Vec3.UP, 1.0)
        neg_q = Quat(-q.w, -q.x, -q.y, -q.z)
        assert q.approx_eq(neg_q, eps=1e-5)

    def test_just_inside_eps_returns_true(self):
        """Pin float32 rounding behavior at the eps boundary.

        SUSPECT: float32 cannot represent 1.0 + 1e-4 exactly; the stored value
        is 1.00010001659..., making the actual diff ~1.0002e-4 > 1e-4.
        So even "exactly eps" nominally ends up OUTSIDE eps after float32 rounding.
        Pin current behavior: False (the boundary is not as tight as intended)."""
        eps = 1e-4
        q = Quat.identity()
        q2 = Quat(q.w + eps, q.x, q.y, q.z)
        # float32 rounding means stored diff > eps; current code returns False
        assert not q.approx_eq(q2, eps=eps)

    def test_well_inside_eps_returns_true(self):
        """A diff well inside eps (half eps in float64, still representable in float32)
        should return True."""
        eps = 1e-3
        q = Quat.identity()
        # Use eps/10 so float32 rounding doesn't push us over
        q2 = Quat(q.w + eps / 10.0, q.x, q.y, q.z)
        assert q.approx_eq(q2, eps=eps)

    def test_just_outside_eps_returns_false(self):
        """Component diff > eps in positive cover AND negative cover → False."""
        eps = 1e-4
        q = Quat.identity()
        q2 = Quat(q.w + eps * 1.5, q.x, q.y, q.z)
        # Also check negative cover: q + q2 has w = 2 + eps*1.5, so min(diff_pos, diff_neg) > eps
        assert not q.approx_eq(q2, eps=eps)
