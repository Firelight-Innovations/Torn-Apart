"""
tests/test_devtools_gizmo.py — Characterisation (golden-master) tests for
``fire_engine/devtools/gizmo.py``.

Pins current behaviour for:
- ray_plane_intersect: hits, parallel ray (None), ray starting on plane
- closest_on_axis: foot-of-perpendicular math with known configs
- Gizmo construction + handle membership (TRANSLATE mode)
- update_drag for TRANSLATE axis and PLANE handles (numeric deltas)
- update_drag for SCALE axis and UNIFORM handles
- Determinism (same inputs → identical outputs)
- Edge cases: zero-length ray dir, degenerate scale, back-facing plane ray
- Enum members: GizmoMode, HandleType
- Handle dataclass attributes (origin comes from gizmo state, not Handle itself)

No panda3d imports; all headless.
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.core.math3d import Quat, Vec3
from fire_engine.devtools import (
    DragState,
    Gizmo,
    GizmoMode,
    Handle,
    HandleType,
    update_drag,
)
from fire_engine.devtools.gizmo import closest_on_axis, ray_plane_intersect

# ===========================================================================
# Helpers
# ===========================================================================


def _np(v: Vec3) -> np.ndarray:
    """Convert Vec3 to float64 array (matches gizmo.py internal convention)."""
    return v.to_numpy().astype(np.float64)


# ===========================================================================
# 1. Enum membership
# ===========================================================================


class TestEnumMembers:
    def test_gizmo_mode_members_exist(self):
        assert GizmoMode.TRANSLATE is not None
        assert GizmoMode.ROTATE is not None
        assert GizmoMode.SCALE is not None

    def test_handle_type_members_exist(self):
        assert HandleType.AXIS is not None
        assert HandleType.PLANE is not None
        assert HandleType.RING is not None
        assert HandleType.UNIFORM is not None

    def test_gizmo_mode_values(self):
        # Pin current string values — change if the code ever renames them.
        assert GizmoMode.TRANSLATE.value == "translate"
        assert GizmoMode.ROTATE.value == "rotate"
        assert GizmoMode.SCALE.value == "scale"

    def test_handle_type_values(self):
        assert HandleType.AXIS.value == "axis"
        assert HandleType.PLANE.value == "plane"
        assert HandleType.RING.value == "ring"
        assert HandleType.UNIFORM.value == "uniform"


# ===========================================================================
# 2. ray_plane_intersect
# ===========================================================================


class TestRayPlaneIntersect:
    def test_ray_hits_z0_plane_straight_down(self):
        """Ray from above straight down onto z=0; hit point has z=0."""
        o = np.array([1.0, 2.0, 5.0])
        d = np.array([0.0, 0.0, -1.0])
        n = np.array([0.0, 0.0, 1.0])
        p = np.array([0.0, 0.0, 0.0])
        hit = ray_plane_intersect(o, d, p, n)
        assert hit is not None
        assert np.allclose(hit, [1.0, 2.0, 0.0])

    def test_hit_lies_on_plane(self):
        """The returned point satisfies dot(point - plane_origin, normal) ≈ 0."""
        o = np.array([3.0, -1.0, 4.0])
        d = np.array([0.5, 0.5, -1.0])
        n = np.array([0.0, 0.0, 1.0])
        p = np.array([0.0, 0.0, 2.0])
        hit = ray_plane_intersect(o, d, p, n)
        assert hit is not None
        residual = float(n.dot(hit - p))
        assert abs(residual) < 1e-9

    def test_oblique_ray_hits_xy_plane(self):
        """Oblique ray from (0,0,3) with direction (1,0,-1) hits z=0 at (3,0,0)."""
        o = np.array([0.0, 0.0, 3.0])
        d = np.array([1.0, 0.0, -1.0])
        n = np.array([0.0, 0.0, 1.0])
        p = np.array([0.0, 0.0, 0.0])
        hit = ray_plane_intersect(o, d, p, n)
        assert hit is not None
        assert np.allclose(hit, [3.0, 0.0, 0.0])

    def test_parallel_ray_returns_none(self):
        """Ray parallel to the plane → None."""
        o = np.array([0.0, 0.0, 1.0])
        d = np.array([1.0, 0.0, 0.0])  # lies in z=const, normal is Z
        n = np.array([0.0, 0.0, 1.0])
        p = np.array([0.0, 0.0, 0.0])
        assert ray_plane_intersect(o, d, p, n) is None

    def test_ray_behind_plane_returns_none(self):
        """Ray pointing away from the plane (s < 0) → None."""
        o = np.array([0.0, 0.0, -2.0])
        d = np.array([0.0, 0.0, -1.0])  # moving further below z=0
        n = np.array([0.0, 0.0, 1.0])
        p = np.array([0.0, 0.0, 0.0])
        assert ray_plane_intersect(o, d, p, n) is None

    def test_ray_origin_on_plane(self):
        """Ray starting exactly on the plane (s=0) → returns the origin itself."""
        o = np.array([1.0, 2.0, 0.0])
        d = np.array([0.0, 0.0, -1.0])
        n = np.array([0.0, 0.0, 1.0])
        p = np.array([0.0, 0.0, 0.0])
        hit = ray_plane_intersect(o, d, p, n)
        # s = dot(n, p-o)/dot(n,d) = dot([0,0,1],[0,0,0]-[1,2,0])/dot([0,0,1],[0,0,-1])
        # s = 0 / -1 = 0, so hit = o + 0*d = o
        assert hit is not None
        assert np.allclose(hit, o)

    def test_nearly_parallel_ray_returns_none(self):
        """Ray direction nearly parallel (dot < 1e-9) → None (current threshold)."""
        o = np.array([0.0, 0.0, 1.0])
        d = np.array([1.0, 0.0, 1e-10])  # almost parallel to z=0
        n = np.array([0.0, 0.0, 1.0])
        p = np.array([0.0, 0.0, 0.0])
        assert ray_plane_intersect(o, d, p, n) is None


# ===========================================================================
# 3. closest_on_axis
# ===========================================================================


class TestClosestOnAxis:
    def test_perpendicular_crossing_x_axis(self):
        """
        Ray straight down at (3, 0, 5) crossing X-axis perpendicularly.
        Closest axis point must be at t=3 on the X-axis with dist≈0.
        """
        o = np.array([3.0, 0.0, 5.0])
        d = np.array([0.0, 0.0, -1.0])
        p = np.array([0.0, 0.0, 0.0])
        a = np.array([1.0, 0.0, 0.0])
        axis_t, _ray_s, dist = closest_on_axis(o, d, p, a)
        assert abs(axis_t - 3.0) < 1e-9
        assert dist < 1e-9

    def test_ray_above_y_axis_offset(self):
        """
        Ray at (2, 5, 0) pointing in -Z crosses above Y-axis at y=5.
        Closest point on Y-axis at t=5 from origin, distance = 2.
        """
        o = np.array([2.0, 5.0, 0.0])
        d = np.array([0.0, 0.0, -1.0])
        p = np.array([0.0, 0.0, 0.0])
        a = np.array([0.0, 1.0, 0.0])
        axis_t, _ray_s, dist = closest_on_axis(o, d, p, a)
        assert abs(axis_t - 5.0) < 1e-9
        # The closest point on the ray to the Y-axis is at (2,5,0); dist from Y-axis = 2.
        assert abs(dist - 2.0) < 1e-9

    def test_parallel_ray_axis_degenerate(self):
        """
        Ray parallel to axis: the two lines are coplanar → dist depends on offset,
        axis_t uses the E/0 branch. Pin: axis_t = E, ray_s = -D/A.
        """
        # Ray along X at z=1 offset → parallel to X-axis at z=0
        o = np.array([0.0, 0.0, 1.0])
        d = np.array([1.0, 0.0, 0.0])
        p = np.array([0.0, 0.0, 0.0])
        a = np.array([1.0, 0.0, 0.0])
        axis_t, ray_s, dist = closest_on_axis(o, d, p, a)
        # w0 = o - p = (0,0,1); E = a.dot(w0) = 0; A=1; D = d.dot(w0) = 0
        # denom = A - B*B = 1 - 1 = 0 → degenerate branch
        # axis_t = E = 0, ray_s = -D/A = 0
        assert abs(axis_t - 0.0) < 1e-9
        assert abs(ray_s - 0.0) < 1e-9
        # dist between (0,0,1)+0*(1,0,0) and (0,0,0)+0*(1,0,0) = 1
        assert abs(dist - 1.0) < 1e-9

    def test_return_types(self):
        """closest_on_axis returns a 3-tuple of floats."""
        o = np.array([1.0, 0.0, 5.0])
        d = np.array([0.0, 0.0, -1.0])
        p = np.array([0.0, 0.0, 0.0])
        a = np.array([1.0, 0.0, 0.0])
        result = closest_on_axis(o, d, p, a)
        assert len(result) == 3
        for v in result:
            assert isinstance(v, float)


# ===========================================================================
# 4. Gizmo construction and handles
# ===========================================================================


class TestGizmoConstruction:
    def test_gizmo_stores_pivot_size_mode(self):
        giz = Gizmo(Vec3(1.0, 2.0, 3.0), 2.5, GizmoMode.TRANSLATE)
        assert giz.pivot.approx_eq(Vec3(1.0, 2.0, 3.0))
        assert giz.size == pytest.approx(2.5)
        assert giz.mode == GizmoMode.TRANSLATE

    def test_gizmo_size_stored_as_float(self):
        giz = Gizmo(Vec3.ZERO, 1, GizmoMode.SCALE)  # int input
        assert isinstance(giz.size, float)

    def test_translate_gizmo_picks_x_axis(self):
        """Ray along X stalk at y=z=0 → AXIS handle with axis==0."""
        giz = Gizmo(Vec3(0, 0, 0), 1.0, GizmoMode.TRANSLATE)
        h = giz.pick(Vec3(0.5, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert h is not None
        assert h.type == HandleType.AXIS
        assert h.axis == 0

    def test_translate_gizmo_picks_y_axis(self):
        """Ray over the Y stalk at x=z=0 → AXIS handle with axis==1."""
        giz = Gizmo(Vec3(0, 0, 0), 1.0, GizmoMode.TRANSLATE)
        h = giz.pick(Vec3(0.0, 0.5, 5.0), Vec3(0.0, 0.0, -1.0))
        assert h is not None
        assert h.type == HandleType.AXIS
        assert h.axis == 1

    def test_translate_gizmo_picks_z_axis(self):
        """
        Ray over the Z stalk at x=y=0 → AXIS handle with axis==2.

        SUSPECTED BUG: When the ray origin lies exactly on the axis handle
        (ray_s == 0.0 at the closest point), consider() rejects it because it
        guards with ``depth <= 0.0: return``.  A ray AT the handle is not
        detected.  Use a slight offset so ray_s > 0.
        """
        giz = Gizmo(Vec3(0, 0, 0), 1.0, GizmoMode.TRANSLATE)
        # Ray from slightly in front of the Z stalk, looking backward along Y,
        # so that closest_on_axis returns a positive ray_s.
        h = giz.pick(Vec3(0.0, 1.0, 0.5), Vec3(0.0, -1.0, 0.0))
        assert h is not None
        assert h.type == HandleType.AXIS
        assert h.axis == 2

    def test_translate_gizmo_picks_xy_plane_handle(self):
        """Ray into +X+Y quadrant (in z=0 plane) → PLANE handle with axis==2 (Z normal)."""
        giz = Gizmo(Vec3(0, 0, 0), 1.0, GizmoMode.TRANSLATE)
        h = giz.pick(Vec3(0.3, 0.3, 5.0), Vec3(0.0, 0.0, -1.0))
        assert h is not None
        assert h.type == HandleType.PLANE
        assert h.axis == 2

    def test_translate_gizmo_miss_returns_none(self):
        """Ray far from all handles → None."""
        giz = Gizmo(Vec3(0, 0, 0), 1.0, GizmoMode.TRANSLATE)
        h = giz.pick(Vec3(10.0, 10.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert h is None

    def test_scale_gizmo_picks_uniform_center(self):
        """
        SUSPECTED BUG: A ray aimed exactly through the pivot is expected to pick
        the UNIFORM centre-cube handle.  However, the axis loop runs first and
        checks X-axis (axis_t=0, dist=0) and Y-axis (axis_t=0, dist=0) — both
        satisfy ``0 <= axis_t <= R and dist <= axis_r`` because they intersect
        the stalk at the pivot tip (t=0).  The first AXIS handle wins (depth=5,
        same as UNIFORM), preventing UNIFORM from ever being selected via a
        perfectly centred ray.

        Pin CURRENT behaviour: the pick returns an AXIS handle (not UNIFORM)
        when the ray passes directly through the pivot.
        """
        giz = Gizmo(Vec3(0, 0, 0), 1.0, GizmoMode.SCALE)
        h = giz.pick(Vec3(0.0, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert h is not None
        # Current behaviour: AXIS wins over UNIFORM for a perfectly centred ray.
        assert h.type == HandleType.AXIS

    def test_rotate_gizmo_picks_z_ring(self):
        """Ray at ring radius from pivot on Z-ring → RING handle with axis==2."""
        giz = Gizmo(Vec3(0, 0, 0), 1.0, GizmoMode.ROTATE)
        # Ring radius ≈ 1.0 (size); hit at (1, 0, 5) going down → Z ring
        h = giz.pick(Vec3(1.0, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert h is not None
        assert h.type == HandleType.RING
        assert h.axis == 2


# ===========================================================================
# 5. Handle dataclass
# ===========================================================================


class TestHandleDataclass:
    def test_handle_is_frozen(self):
        h = Handle(HandleType.AXIS, 0)
        with pytest.raises((AttributeError, TypeError)):
            h.axis = 99  # type: ignore[misc]

    def test_handle_equality(self):
        assert Handle(HandleType.AXIS, 1) == Handle(HandleType.AXIS, 1)
        assert Handle(HandleType.AXIS, 0) != Handle(HandleType.AXIS, 1)
        assert Handle(HandleType.PLANE, 2) != Handle(HandleType.AXIS, 2)


# ===========================================================================
# 6. update_drag — TRANSLATE axis
# ===========================================================================


class TestUpdateDragTranslateAxis:
    def _setup(self, axis: int) -> tuple[Gizmo, Handle]:
        pivot = Vec3(0, 0, 0)
        giz = Gizmo(pivot, 1.0, GizmoMode.TRANSLATE)
        if axis == 0:
            h = giz.pick(Vec3(0.5, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        elif axis == 1:
            h = giz.pick(Vec3(0.0, 0.5, 5.0), Vec3(0.0, 0.0, -1.0))
        else:
            raise ValueError("Use axis 0 or 1 for this helper")
        assert h is not None and h.axis == axis
        return giz, h

    def test_translate_x_axis_drag_3_units(self):
        """
        Grab X handle at x=2, drag to x=5 → delta=+3 applied to start_pos (0,0,0).
        """
        giz, handle = self._setup(0)
        drag = giz.begin(
            handle,
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

    def test_translate_x_axis_drag_negative(self):
        """Drag from x=2 back to x=0.5 → delta=-1.5."""
        giz, handle = self._setup(0)
        drag = giz.begin(
            handle,
            Vec3(2.0, 0.0, 5.0),
            Vec3(0.0, 0.0, -1.0),
            Vec3(0, 0, 0),
            Quat.identity(),
            Vec3(1, 1, 1),
        )
        pos, _rot, _scl = update_drag(drag, Vec3(0.5, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert pos.approx_eq(Vec3(-1.5, 0.0, 0.0), eps=1e-4)

    def test_translate_y_axis_drag(self):
        """Drag Y handle from y=0.5 to y=3.5 → delta=+3 on Y."""
        giz, handle = self._setup(1)
        drag = giz.begin(
            handle,
            Vec3(0.0, 0.5, 5.0),
            Vec3(0.0, 0.0, -1.0),
            Vec3(0, 0, 0),
            Quat.identity(),
            Vec3(1, 1, 1),
        )
        pos, _rot, _scl = update_drag(drag, Vec3(0.0, 3.5, 5.0), Vec3(0.0, 0.0, -1.0))
        assert pos.approx_eq(Vec3(0.0, 3.0, 0.0), eps=1e-4)

    def test_translate_axis_drag_at_rest_returns_start_pos(self):
        """Cursor doesn't move → position unchanged from start."""
        giz = Gizmo(Vec3(0, 0, 0), 1.0, GizmoMode.TRANSLATE)
        handle = giz.pick(Vec3(0.5, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert handle is not None
        start_pos = Vec3(7.0, -3.0, 2.0)
        drag = giz.begin(
            handle,
            Vec3(0.5, 0.0, 5.0),
            Vec3(0.0, 0.0, -1.0),
            start_pos,
            Quat.identity(),
            Vec3(1, 1, 1),
        )
        pos, _, _ = update_drag(drag, Vec3(0.5, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert pos.approx_eq(start_pos, eps=1e-4)


# ===========================================================================
# 7. update_drag — TRANSLATE plane
# ===========================================================================


class TestUpdateDragTranslatePlane:
    def test_plane_drag_moves_in_xy_plane(self):
        """
        Grab XY-plane handle, move from (0.3, 0.3) to (1.3, 0.8).
        Expected delta: (1.0, 0.5, 0.0) in XY.
        """
        giz = Gizmo(Vec3(0, 0, 0), 1.0, GizmoMode.TRANSLATE)
        # Pick the XY plane handle (normal Z, axis==2)
        handle = giz.pick(Vec3(0.3, 0.3, 5.0), Vec3(0.0, 0.0, -1.0))
        assert handle is not None and handle.type == HandleType.PLANE

        drag = giz.begin(
            handle,
            Vec3(0.3, 0.3, 5.0),
            Vec3(0.0, 0.0, -1.0),
            Vec3(0, 0, 0),
            Quat.identity(),
            Vec3(1, 1, 1),
        )
        pos, _rot, _scl = update_drag(drag, Vec3(1.3, 0.8, 5.0), Vec3(0.0, 0.0, -1.0))
        # ref_point was (0.3, 0.3, 0.0) on z=0 plane; new hit is (1.3, 0.8, 0.0)
        assert pos.approx_eq(Vec3(1.0, 0.5, 0.0), eps=1e-4)


# ===========================================================================
# 8. update_drag — SCALE axis
# ===========================================================================


class TestUpdateDragScaleAxis:
    def test_scale_x_axis_doubles_on_size_drag(self):
        """
        Drag X stalk from 0.5 to 1.5 (delta = +size = 1.0):
        factor = 1 + 1.0 / max(1.0, 1e-6) = 2.0 → X scale doubles.
        """
        giz = Gizmo(Vec3(0, 0, 0), 1.0, GizmoMode.SCALE)
        handle = giz.pick(Vec3(0.5, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert handle is not None and handle.type == HandleType.AXIS and handle.axis == 0
        drag = giz.begin(
            handle,
            Vec3(0.5, 0.0, 5.0),
            Vec3(0.0, 0.0, -1.0),
            Vec3(0, 0, 0),
            Quat.identity(),
            Vec3(1, 1, 1),
        )
        _, _, scl = update_drag(drag, Vec3(1.5, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert scl.approx_eq(Vec3(2.0, 1.0, 1.0), eps=1e-5)

    def test_scale_axis_clamped_at_minimum(self):
        """
        Dragging far negative cannot make factor < 0.01 (clamped).
        """
        giz = Gizmo(Vec3(0, 0, 0), 1.0, GizmoMode.SCALE)
        handle = giz.pick(Vec3(0.5, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert handle is not None
        drag = giz.begin(
            handle,
            Vec3(0.5, 0.0, 5.0),
            Vec3(0.0, 0.0, -1.0),
            Vec3(0, 0, 0),
            Quat.identity(),
            Vec3(1, 1, 1),
        )
        # drag to -1000 → delta = -1000.5 → factor = 1 + (-1000.5)/1.0 ≪ 0.01
        _, _, scl = update_drag(drag, Vec3(-1000.0, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert scl.x == pytest.approx(0.01, abs=1e-6)
        assert scl.y == pytest.approx(1.0, abs=1e-6)
        assert scl.z == pytest.approx(1.0, abs=1e-6)


# ===========================================================================
# 9. update_drag — SCALE uniform
# ===========================================================================


class TestUpdateDragScaleUniform:
    def test_uniform_scale_doubles_when_dist_doubles(self):
        """
        Move ray origin so the closest-point-on-ray to pivot has distance = 2×ref_dist
        → all scale components double.

        NOTE: The UNIFORM handle cannot be picked via Gizmo.pick() with a centred
        ray (see test_scale_gizmo_picks_uniform_center for the suspected bug).
        We construct the DragState manually to test update_drag logic in isolation.
        """
        # Build a UNIFORM DragState manually with a known ref_dist.
        handle = Handle(HandleType.UNIFORM, 0)
        ref_dist = 0.2  # arbitrary non-zero
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
        # update: ray at (2*ref_dist, 0, 5) → closest dist to pivot = 2*ref_dist
        new_x = ref_dist * 2.0
        _, _, scl = update_drag(state, Vec3(float(new_x), 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        expected = Vec3(2.0, 2.0, 2.0)
        assert scl.approx_eq(expected, eps=1e-4)

    def test_uniform_scale_degenerate_ref_dist(self):
        """
        If ref_dist < 1e-6 (grab exactly at pivot), update_drag returns start_scale.
        """
        Gizmo(Vec3(0, 0, 0), 1.0, GizmoMode.SCALE)
        handle = Handle(HandleType.UNIFORM, 0)
        # Build a DragState with ref_dist = 0 manually
        state = DragState(
            mode=GizmoMode.SCALE,
            handle=handle,
            pivot=Vec3(0, 0, 0),
            size=1.0,
            start_position=Vec3(0, 0, 0),
            start_rotation=Quat.identity(),
            start_scale=Vec3(3, 3, 3),
            ref_dist=0.0,
        )
        _, _, scl = update_drag(state, Vec3(0.0, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert scl.approx_eq(Vec3(3, 3, 3))


# ===========================================================================
# 10. Determinism
# ===========================================================================


class TestDeterminism:
    def test_ray_plane_intersect_deterministic(self):
        o = np.array([1.5, -2.0, 4.0])
        d = np.array([0.1, 0.2, -1.0])
        n = np.array([0.0, 0.0, 1.0])
        p = np.array([0.0, 0.0, 0.0])
        r1 = ray_plane_intersect(o, d, p, n)
        r2 = ray_plane_intersect(o, d, p, n)
        assert np.allclose(r1, r2)

    def test_closest_on_axis_deterministic(self):
        o = np.array([3.0, 1.0, 5.0])
        d = np.array([0.0, 0.0, -1.0])
        p = np.array([0.0, 0.0, 0.0])
        a = np.array([1.0, 0.0, 0.0])
        r1 = closest_on_axis(o, d, p, a)
        r2 = closest_on_axis(o, d, p, a)
        assert r1 == r2

    def test_update_drag_deterministic(self):
        """Same drag state + same ray → identical (pos, rot, scl) twice."""
        giz = Gizmo(Vec3(0, 0, 0), 1.0, GizmoMode.TRANSLATE)
        handle = giz.pick(Vec3(0.5, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert handle is not None
        drag = giz.begin(
            handle,
            Vec3(0.5, 0.0, 5.0),
            Vec3(0.0, 0.0, -1.0),
            Vec3(1, 2, 3),
            Quat.identity(),
            Vec3(1, 1, 1),
        )
        r1 = update_drag(drag, Vec3(2.5, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        r2 = update_drag(drag, Vec3(2.5, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        p1, q1, s1 = r1
        p2, q2, s2 = r2
        assert p1.approx_eq(p2)
        assert q1.approx_eq(q2)
        assert s1.approx_eq(s2)


# ===========================================================================
# 11. Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_zero_length_ray_direction_plane(self):
        """
        ray_plane_intersect with zero-direction → denom = 0 → None.
        (The gizmo math never normalises d, so a zero-length direction must not crash.)
        """
        o = np.array([0.0, 0.0, 5.0])
        d = np.array([0.0, 0.0, 0.0])
        n = np.array([0.0, 0.0, 1.0])
        p = np.array([0.0, 0.0, 0.0])
        result = ray_plane_intersect(o, d, p, n)
        assert result is None

    def test_degenerate_scale_translate_still_works(self):
        """
        Even with a scale of Vec3(0,0,0) on the object (degenerate), a TRANSLATE
        drag doesn't crash — scale is not involved in translation.
        """
        giz = Gizmo(Vec3(0, 0, 0), 1.0, GizmoMode.TRANSLATE)
        handle = giz.pick(Vec3(0.5, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert handle is not None
        drag = giz.begin(
            handle,
            Vec3(0.5, 0.0, 5.0),
            Vec3(0.0, 0.0, -1.0),
            Vec3(0, 0, 0),
            Quat.identity(),
            Vec3(0, 0, 0),
        )
        pos, _rot, scl = update_drag(drag, Vec3(1.5, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert pos.approx_eq(Vec3(1.0, 0.0, 0.0), eps=1e-4)
        assert scl.approx_eq(Vec3(0, 0, 0))

    def test_begin_plane_with_parallel_ray_sets_ref_point_none(self):
        """
        If the grab ray is parallel to the plane normal at begin-time,
        ref_point will be None; update_drag should return start pos unchanged.
        """
        handle = Handle(HandleType.PLANE, 2)  # Z-normal plane
        # Ray parallel to z=0: direction has no Z component
        state = DragState(
            mode=GizmoMode.TRANSLATE,
            handle=handle,
            pivot=Vec3(0, 0, 0),
            size=1.0,
            start_position=Vec3(5, 5, 5),
            start_rotation=Quat.identity(),
            start_scale=Vec3(1, 1, 1),
            ref_point=None,
        )
        # update_drag should return start_position unchanged when ref_point is None
        pos, _, _ = update_drag(state, Vec3(1.0, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        # SUSPECT: if hit is found but ref_point is None, code returns (pos, rot, scl)
        # unchanged — pin this behaviour
        assert pos.approx_eq(Vec3(5, 5, 5))

    def test_gizmo_pivot_offset_translate_drag(self):
        """
        Gizmo anchored at (10, 0, 0); axis drag still resolves a delta of +2 on X.
        """
        pivot = Vec3(10, 0, 0)
        giz = Gizmo(pivot, 1.0, GizmoMode.TRANSLATE)
        # X stalk now runs from (10,0,0) to (11,0,0)
        h = giz.pick(Vec3(10.5, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert h is not None and h.axis == 0
        drag = giz.begin(
            h,
            Vec3(10.5, 0.0, 5.0),
            Vec3(0.0, 0.0, -1.0),
            Vec3(10, 0, 0),
            Quat.identity(),
            Vec3(1, 1, 1),
        )
        pos, _, _ = update_drag(drag, Vec3(12.5, 0.0, 5.0), Vec3(0.0, 0.0, -1.0))
        assert pos.approx_eq(Vec3(12.0, 0.0, 0.0), eps=1e-4)
