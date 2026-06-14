"""
tests/test_transform.py — Headless tests for world/transform.py.

Covers:
  - Parent/child world-position composition
  - look_at makes forward point at target
  - Dirty-flag correctness (moving parent updates child world position)
  - transform_point / inverse_transform_point round-trip

NO panda3d imports allowed in this file.
"""

from __future__ import annotations

import math

import pytest

from fire_engine.render.transform import Transform, Space
from fire_engine.core.math3d import Vec3, Quat


EPS = 1e-4  # float32 accumulation tolerance for composed matrices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transform(pos: Vec3 = None, rot: Quat = None) -> Transform:
    t = Transform()
    if pos is not None:
        t.local_position = pos
    if rot is not None:
        t.local_rotation = rot
    return t


# ---------------------------------------------------------------------------
# Basic local TRS
# ---------------------------------------------------------------------------


class TestLocalTRS:
    def test_default_position_zero(self):
        t = Transform()
        assert t.local_position.approx_eq(Vec3.ZERO, eps=EPS)

    def test_default_rotation_identity(self):
        t = Transform()
        assert t.local_rotation.approx_eq(Quat.identity(), eps=EPS)

    def test_default_scale_one(self):
        t = Transform()
        assert t.local_scale.approx_eq(Vec3.ONE, eps=EPS)

    def test_set_local_position(self):
        t = Transform()
        t.local_position = Vec3(1, 2, 3)
        assert t.local_position.approx_eq(Vec3(1, 2, 3), eps=EPS)

    def test_set_local_rotation_normalises(self):
        t = Transform()
        q = Quat(2.0, 0.0, 0.0, 0.0)  # not unit
        t.local_rotation = q
        n = t.local_rotation
        assert abs(n.w**2 + n.x**2 + n.y**2 + n.z**2 - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# World-space position with no parent
# ---------------------------------------------------------------------------


class TestWorldPositionNoParent:
    def test_world_pos_equals_local_when_no_parent(self):
        t = _make_transform(pos=Vec3(5, 3, 1))
        assert t.position.approx_eq(Vec3(5, 3, 1), eps=EPS)

    def test_world_rot_equals_local_when_no_parent(self):
        q = Quat.from_axis_angle(Vec3.UP, math.pi / 4)
        t = _make_transform(rot=q)
        assert t.rotation.approx_eq(q, eps=EPS)


# ---------------------------------------------------------------------------
# Parent/child world-position composition
# ---------------------------------------------------------------------------


class TestParentChildComposition:
    def test_child_offset_under_translated_parent(self):
        """Child at local (0,5,0) under parent at world (10,0,0) → world (10,5,0)."""
        parent = _make_transform(pos=Vec3(10, 0, 0))
        child = _make_transform(pos=Vec3(0, 5, 0))
        child.set_parent(parent, keep_world=False)
        assert child.position.approx_eq(Vec3(10, 5, 0), eps=EPS)

    def test_child_offset_under_rotated_parent(self):
        """
        Parent at origin, rotated 90° about Z (+Y→−X).
        Child local (0,1,0) should map to world (−1,0,0).
        """
        parent = _make_transform(rot=Quat.from_axis_angle(Vec3.UP, math.pi / 2))
        child = _make_transform(pos=Vec3(0, 1, 0))
        child.set_parent(parent, keep_world=False)
        expected = Vec3(-1, 0, 0)
        assert child.position.approx_eq(expected, eps=1e-3)

    def test_child_offset_under_translated_and_rotated_parent(self):
        """
        Parent at (5,0,0), rotated 90° about Z.
        Child local (0,2,0) → parent rotates it to (−2,0,0) → world (3,0,0).
        """
        parent = _make_transform(
            pos=Vec3(5, 0, 0),
            rot=Quat.from_axis_angle(Vec3.UP, math.pi / 2),
        )
        child = _make_transform(pos=Vec3(0, 2, 0))
        child.set_parent(parent, keep_world=False)
        expected = Vec3(5 - 2, 0, 0)  # (3,0,0)
        assert child.position.approx_eq(expected, eps=1e-3)

    def test_deeply_nested_hierarchy(self):
        """Three-level hierarchy: positions accumulate correctly."""
        a = _make_transform(pos=Vec3(1, 0, 0))
        b = _make_transform(pos=Vec3(0, 1, 0))
        c = _make_transform(pos=Vec3(0, 0, 1))
        b.set_parent(a, keep_world=False)
        c.set_parent(b, keep_world=False)
        assert c.position.approx_eq(Vec3(1, 1, 1), eps=EPS)


# ---------------------------------------------------------------------------
# Dirty-flag propagation
# ---------------------------------------------------------------------------


class TestDirtyFlag:
    def test_moving_parent_updates_child_world_position(self):
        """After moving the parent, child world position should update lazily."""
        parent = _make_transform(pos=Vec3(0, 0, 0))
        child = _make_transform(pos=Vec3(0, 0, 5))
        child.set_parent(parent, keep_world=False)

        # Before move
        assert child.position.approx_eq(Vec3(0, 0, 5), eps=EPS)

        # Move parent
        parent.local_position = Vec3(10, 0, 0)

        # Child world position must reflect parent move
        assert child.position.approx_eq(Vec3(10, 0, 5), eps=EPS)

    def test_multiple_children_all_update(self):
        parent = _make_transform(pos=Vec3(0, 0, 0))
        c1 = _make_transform(pos=Vec3(1, 0, 0))
        c2 = _make_transform(pos=Vec3(2, 0, 0))
        c1.set_parent(parent, keep_world=False)
        c2.set_parent(parent, keep_world=False)

        parent.local_position = Vec3(0, 5, 0)
        assert c1.position.approx_eq(Vec3(1, 5, 0), eps=EPS)
        assert c2.position.approx_eq(Vec3(2, 5, 0), eps=EPS)

    def test_rotating_parent_updates_child_direction(self):
        """Rotating the parent should change the child's world position."""
        parent = _make_transform()
        child = _make_transform(pos=Vec3(0, 1, 0))  # 1 m forward
        child.set_parent(parent, keep_world=False)

        # After 90° yaw, child should be at (−1, 0, 0)
        parent.local_rotation = Quat.from_axis_angle(Vec3.UP, math.pi / 2)
        assert child.position.approx_eq(Vec3(-1, 0, 0), eps=1e-3)


# ---------------------------------------------------------------------------
# look_at
# ---------------------------------------------------------------------------


class TestLookAt:
    def test_forward_points_at_target_basic(self):
        """After look_at, transform.forward should point toward target."""
        t = _make_transform(pos=Vec3(0, 0, 0))
        target = Vec3(0, 10, 0)
        t.look_at(target)
        fwd = t.forward
        expected = Vec3(0, 1, 0)  # +Y
        assert fwd.approx_eq(expected, eps=1e-3)

    def test_forward_points_at_target_diagonal(self):
        t = _make_transform(pos=Vec3(0, 0, 0))
        target = Vec3(1, 1, 0)
        t.look_at(target)
        fwd = t.forward
        expected = Vec3(1, 1, 0).normalized()
        assert fwd.approx_eq(expected, eps=1e-3)

    def test_look_at_from_nonzero_position(self):
        t = _make_transform(pos=Vec3(5, 5, 0))
        target = Vec3(5, 10, 0)
        t.look_at(target)
        fwd = t.forward
        expected = Vec3(0, 1, 0)
        assert fwd.approx_eq(expected, eps=1e-3)

    def test_look_at_up_component(self):
        """After look_at toward a target above, forward has positive Z component."""
        t = _make_transform(pos=Vec3(0, 0, 0))
        target = Vec3(0, 1, 1)  # 45° up-forward
        t.look_at(target)
        fwd = t.forward
        expected = Vec3(0, 1, 1).normalized()
        assert fwd.approx_eq(expected, eps=1e-3)

    def test_look_at_no_op_when_target_coincides(self):
        """look_at to own position should not crash or change rotation."""
        q0 = Quat.from_axis_angle(Vec3.UP, 1.0)
        t = _make_transform(pos=Vec3(5, 5, 5), rot=q0)
        t.look_at(Vec3(5, 5, 5))
        # Rotation unchanged
        assert t.rotation.approx_eq(q0, eps=1e-3)


# ---------------------------------------------------------------------------
# transform_point / inverse_transform_point
# ---------------------------------------------------------------------------


class TestTransformPoint:
    def test_transform_point_translation_only(self):
        t = _make_transform(pos=Vec3(3, 4, 5))
        # Local origin → world (3,4,5)
        assert t.transform_point(Vec3.ZERO).approx_eq(Vec3(3, 4, 5), eps=EPS)
        # Local (1,0,0) → world (4,4,5)
        assert t.transform_point(Vec3(1, 0, 0)).approx_eq(Vec3(4, 4, 5), eps=EPS)

    def test_transform_point_rotation_only(self):
        """90° yaw: local +Y → world −X."""
        t = _make_transform(rot=Quat.from_axis_angle(Vec3.UP, math.pi / 2))
        world = t.transform_point(Vec3(0, 1, 0))
        assert world.approx_eq(Vec3(-1, 0, 0), eps=1e-3)

    def test_inverse_transform_point_round_trip(self):
        """transform_point then inverse_transform_point returns the original local point."""
        t = _make_transform(
            pos=Vec3(7, -3, 2),
            rot=Quat.from_axis_angle(Vec3.UP, math.pi / 3),
        )
        local = Vec3(1, 2, 3)
        world = t.transform_point(local)
        back = t.inverse_transform_point(world)
        assert back.approx_eq(local, eps=1e-3)

    def test_inverse_transform_point_translation(self):
        t = _make_transform(pos=Vec3(10, 0, 0))
        world_pt = Vec3(10, 5, 0)
        local = t.inverse_transform_point(world_pt)
        assert local.approx_eq(Vec3(0, 5, 0), eps=EPS)

    def test_transform_point_nested_hierarchy(self):
        """A two-level hierarchy: grandchild local origin → correct world point."""
        parent = _make_transform(pos=Vec3(10, 0, 0))
        child = _make_transform(pos=Vec3(0, 5, 0))
        child.set_parent(parent, keep_world=False)

        # transform_point of child local origin = child world position
        world = child.transform_point(Vec3.ZERO)
        assert world.approx_eq(Vec3(10, 5, 0), eps=EPS)


# ---------------------------------------------------------------------------
# Space enum usage
# ---------------------------------------------------------------------------


class TestTranslate:
    def test_translate_world(self):
        t = _make_transform(pos=Vec3(0, 0, 0))
        t.translate(Vec3(1, 0, 0), relative_to=Space.WORLD)
        assert t.local_position.approx_eq(Vec3(1, 0, 0), eps=EPS)

    def test_translate_self_follows_rotation(self):
        """translate(Space.SELF) moves along local forward (+Y)."""
        t = _make_transform(rot=Quat.from_axis_angle(Vec3.UP, math.pi / 2))
        # Moving 1 unit "forward" in local space → world −X direction
        t.translate(Vec3(0, 1, 0), relative_to=Space.SELF)
        # After 90° yaw, local +Y mapped to world −X
        assert t.local_position.approx_eq(Vec3(-1, 0, 0), eps=1e-3)

    def test_translate_self_no_rotation(self):
        t = _make_transform()
        t.translate(Vec3(0, 3, 0), relative_to=Space.SELF)
        assert t.local_position.approx_eq(Vec3(0, 3, 0), eps=EPS)


class TestDirectionVectors:
    def test_default_forward_is_plus_y(self):
        t = Transform()
        assert t.forward.approx_eq(Vec3.FORWARD, eps=EPS)

    def test_default_right_is_plus_x(self):
        t = Transform()
        assert t.right.approx_eq(Vec3.RIGHT, eps=EPS)

    def test_default_up_is_plus_z(self):
        t = Transform()
        assert t.up.approx_eq(Vec3.UP, eps=EPS)

    def test_90_yaw_rotates_forward_to_minus_x(self):
        t = Transform()
        t.local_rotation = Quat.from_axis_angle(Vec3.UP, math.pi / 2)
        assert t.forward.approx_eq(Vec3(-1, 0, 0), eps=1e-3)
