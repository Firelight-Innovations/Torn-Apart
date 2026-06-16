"""
tests/devtools/test_gizmo.py — tests for fire_engine/devtools/gizmo.py.

Covers the headless gizmo math: ray_plane_intersect, closest_on_axis, Gizmo
pick/begin, update_drag for translate/rotate/scale/uniform, determinism, and
re-export symbols. Fully headless; no panda3d imports.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fire_engine.core.math3d import Quat, Vec3
from fire_engine.devtools.enums import GizmoMode, HandleType
from fire_engine.devtools.gizmo import (
    Gizmo,
    closest_on_axis,
    ray_plane_intersect,
    update_drag,
)
from fire_engine.devtools.types import DragState, Handle

# ---------------------------------------------------------------------------
# ray_plane_intersect
# ---------------------------------------------------------------------------


class TestRayPlaneIntersect:
    def test_hit_straight_down_onto_z0(self):
        o = np.array([1.0, 2.0, 5.0])
        d = np.array([0.0, 0.0, -1.0])
        hit = ray_plane_intersect(o, d, np.zeros(3), np.array([0.0, 0.0, 1.0]))
        assert hit is not None
        assert np.allclose(hit, [1.0, 2.0, 0.0])

    def test_parallel_ray_returns_none(self):
        o = np.array([0.0, 0.0, 1.0])
        d = np.array([1.0, 0.0, 0.0])
        assert ray_plane_intersect(o, d, np.zeros(3), np.array([0.0, 0.0, 1.0])) is None

    def test_ray_behind_plane_returns_none(self):
        o = np.array([0.0, 0.0, -2.0])
        d = np.array([0.0, 0.0, -1.0])
        assert ray_plane_intersect(o, d, np.zeros(3), np.array([0.0, 0.0, 1.0])) is None

    def test_hit_lies_on_plane(self):
        o = np.array([3.0, -1.0, 4.0])
        d = np.array([0.5, 0.5, -1.0])
        n = np.array([0.0, 0.0, 1.0])
        p = np.array([0.0, 0.0, 2.0])
        hit = ray_plane_intersect(o, d, p, n)
        assert hit is not None
        assert abs(float(n.dot(hit - p))) < 1e-9

    def test_origin_on_plane_returns_origin(self):
        o = np.array([1.0, 2.0, 0.0])
        d = np.array([0.0, 0.0, -1.0])
        hit = ray_plane_intersect(o, d, np.zeros(3), np.array([0.0, 0.0, 1.0]))
        assert hit is not None
        assert np.allclose(hit, o)

    def test_deterministic(self):
        o = np.array([1.5, -2.0, 4.0])
        d = np.array([0.1, 0.2, -1.0])
        n = np.array([0.0, 0.0, 1.0])
        p = np.zeros(3)
        r1 = ray_plane_intersect(o, d, p, n)
        r2 = ray_plane_intersect(o, d, p, n)
        assert np.allclose(r1, r2)


# ---------------------------------------------------------------------------
# closest_on_axis
# ---------------------------------------------------------------------------


class TestClosestOnAxis:
    def test_perpendicular_crossing_x_axis(self):
        o = np.array([3.0, 0.0, 5.0])
        d = np.array([0.0, 0.0, -1.0])
        axis_t, _ray_s, dist = closest_on_axis(o, d, np.zeros(3), np.array([1.0, 0.0, 0.0]))
        assert abs(axis_t - 3.0) < 1e-9
        assert dist < 1e-9

    def test_return_types_are_float(self):
        result = closest_on_axis(
            np.array([1.0, 0.0, 5.0]),
            np.array([0.0, 0.0, -1.0]),
            np.zeros(3),
            np.array([1.0, 0.0, 0.0]),
        )
        assert len(result) == 3
        for v in result:
            assert isinstance(v, float)

    def test_deterministic(self):
        o = np.array([3.0, 1.0, 5.0])
        d = np.array([0.0, 0.0, -1.0])
        p = np.zeros(3)
        a = np.array([1.0, 0.0, 0.0])
        assert closest_on_axis(o, d, p, a) == closest_on_axis(o, d, p, a)


# ---------------------------------------------------------------------------
# Gizmo construction
# ---------------------------------------------------------------------------


class TestGizmoConstruction:
    def test_stores_pivot_size_mode(self):
        giz = Gizmo(Vec3(1, 2, 3), 2.5, GizmoMode.TRANSLATE)
        assert giz.pivot.approx_eq(Vec3(1, 2, 3))
        assert giz.size == pytest.approx(2.5)
        assert giz.mode == GizmoMode.TRANSLATE

    def test_size_stored_as_float(self):
        giz = Gizmo(Vec3.ZERO, 1, GizmoMode.SCALE)
        assert isinstance(giz.size, float)


# ---------------------------------------------------------------------------
# Gizmo.pick
# ---------------------------------------------------------------------------


class TestGizmoPick:
    def test_translate_picks_x_axis(self):
        giz = Gizmo(Vec3(0, 0, 0), 1.0, GizmoMode.TRANSLATE)
        h = giz.pick(Vec3(0.5, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert h is not None
        assert h.type == HandleType.AXIS and h.axis == 0

    def test_translate_picks_xy_plane(self):
        giz = Gizmo(Vec3(0, 0, 0), 1.0, GizmoMode.TRANSLATE)
        h = giz.pick(Vec3(0.3, 0.3, 5.0), Vec3(0.0, 0.0, -1.0))
        assert h is not None
        assert h.type == HandleType.PLANE and h.axis == 2

    def test_translate_miss_returns_none(self):
        giz = Gizmo(Vec3(0, 0, 0), 1.0, GizmoMode.TRANSLATE)
        assert giz.pick(Vec3(10.0, 10.0, 5.0), Vec3(0.0, 0.0, -1.0)) is None

    def test_rotate_picks_z_ring(self):
        giz = Gizmo(Vec3(0, 0, 0), 1.0, GizmoMode.ROTATE)
        h = giz.pick(Vec3(1.0, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert h is not None
        assert h.type == HandleType.RING and h.axis == 2

    def test_rotate_miss_returns_none(self):
        giz = Gizmo(Vec3(0, 0, 0), 1.0, GizmoMode.ROTATE)
        assert giz.pick(Vec3(10.0, 10.0, 5.0), Vec3(0.0, 0.0, -1.0)) is None


# ---------------------------------------------------------------------------
# update_drag — translate axis
# ---------------------------------------------------------------------------


class TestUpdateDragTranslateAxis:
    def test_x_axis_drag_delta(self):
        giz = Gizmo(Vec3(0, 0, 0), 1.0, GizmoMode.TRANSLATE)
        h = giz.pick(Vec3(0.5, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert h is not None
        drag = giz.begin(
            h,
            Vec3(2.0, 0.0, 5.0),
            Vec3(0.0, 0.0, -1.0),
            Vec3(0, 0, 0),
            Quat.identity(),
            Vec3(1, 1, 1),
        )
        pos, rot, scl = update_drag(drag, Vec3(5.0, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert pos.approx_eq(Vec3(3.0, 0.0, 0.0), eps=1e-5)
        assert rot.approx_eq(Quat.identity())
        assert scl.approx_eq(Vec3(1, 1, 1))

    def test_x_axis_drag_at_rest_unchanged(self):
        giz = Gizmo(Vec3(0, 0, 0), 1.0, GizmoMode.TRANSLATE)
        h = giz.pick(Vec3(0.5, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert h is not None
        start = Vec3(7.0, -3.0, 2.0)
        drag = giz.begin(
            h,
            Vec3(0.5, 0.0, 5.0),
            Vec3(0.0, 0.0, -1.0),
            start,
            Quat.identity(),
            Vec3(1, 1, 1),
        )
        pos, _, _ = update_drag(drag, Vec3(0.5, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert pos.approx_eq(start, eps=1e-4)


# ---------------------------------------------------------------------------
# update_drag — rotate ring
# ---------------------------------------------------------------------------


class TestUpdateDragRotateRing:
    def test_z_ring_90_degree_spin(self):
        giz = Gizmo(Vec3(0, 0, 0), 1.0, GizmoMode.ROTATE)
        h = giz.pick(Vec3(1.0, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert h is not None and h.type == HandleType.RING and h.axis == 2
        drag = giz.begin(
            h,
            Vec3(1.0, 0.0, 5.0),
            Vec3(0.0, 0.0, -1.0),
            Vec3(0, 0, 0),
            Quat.identity(),
            Vec3(1, 1, 1),
        )
        _, rot, _ = update_drag(drag, Vec3(0.0, 1.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert rot.approx_eq(Quat.from_axis_angle(Vec3.UP, math.pi / 2), eps=1e-4)


# ---------------------------------------------------------------------------
# update_drag — scale axis
# ---------------------------------------------------------------------------


class TestUpdateDragScaleAxis:
    def test_x_axis_doubles_on_size_drag(self):
        giz = Gizmo(Vec3(0, 0, 0), 1.0, GizmoMode.SCALE)
        h = giz.pick(Vec3(0.5, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert h is not None and h.type == HandleType.AXIS and h.axis == 0
        drag = giz.begin(
            h,
            Vec3(0.5, 0.0, 5.0),
            Vec3(0.0, 0.0, -1.0),
            Vec3(0, 0, 0),
            Quat.identity(),
            Vec3(1, 1, 1),
        )
        _, _, scl = update_drag(drag, Vec3(1.5, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert scl.approx_eq(Vec3(2.0, 1.0, 1.0), eps=1e-5)

    def test_scale_clamped_to_minimum(self):
        giz = Gizmo(Vec3(0, 0, 0), 1.0, GizmoMode.SCALE)
        h = giz.pick(Vec3(0.5, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert h is not None
        drag = giz.begin(
            h,
            Vec3(0.5, 0.0, 5.0),
            Vec3(0.0, 0.0, -1.0),
            Vec3(0, 0, 0),
            Quat.identity(),
            Vec3(1, 1, 1),
        )
        _, _, scl = update_drag(drag, Vec3(-1000.0, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert scl.x == pytest.approx(0.01, abs=1e-6)


# ---------------------------------------------------------------------------
# update_drag — uniform scale
# ---------------------------------------------------------------------------


class TestUpdateDragUniformScale:
    def test_doubles_when_dist_doubles(self):
        handle = Handle(HandleType.UNIFORM, 0)
        ref_dist = 0.2
        state = DragState(
            mode=GizmoMode.SCALE,
            handle=handle,
            pivot=Vec3(0, 0, 0),
            size=1.0,
            start_position=Vec3(0, 0, 0),
            start_rotation=Quat.identity(),
            start_scale=Vec3(1, 1, 1),
            ref_dist=ref_dist,
        )
        _, _, scl = update_drag(state, Vec3(float(ref_dist * 2), 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert scl.approx_eq(Vec3(2.0, 2.0, 2.0), eps=1e-4)

    def test_degenerate_ref_dist_returns_start_scale(self):
        state = DragState(
            mode=GizmoMode.SCALE,
            handle=Handle(HandleType.UNIFORM, 0),
            pivot=Vec3(0, 0, 0),
            size=1.0,
            start_position=Vec3(0, 0, 0),
            start_rotation=Quat.identity(),
            start_scale=Vec3(3, 3, 3),
            ref_dist=0.0,
        )
        _, _, scl = update_drag(state, Vec3(0.0, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert scl.approx_eq(Vec3(3, 3, 3))


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


def test_module_reexports_all_symbols():
    import fire_engine.devtools.gizmo as mod

    for name in ("DragState", "Gizmo", "GizmoMode", "Handle", "HandleType", "update_drag"):
        assert hasattr(mod, name), f"{name!r} not re-exported from gizmo"
