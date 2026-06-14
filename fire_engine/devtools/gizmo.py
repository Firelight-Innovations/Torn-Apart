"""
devtools/gizmo.py — headless math for the Unity-style transform gizmo.

The dev overlay draws a 3-axis manipulator on the selected object (the bright
arrows / planes / rings you grab to move, rotate, or scale it), exactly like the
Unity Scene-view tools.  *All the geometry math lives here* — handle picking
(which arrow/plane/ring is under the cursor) and drag resolution (how a mouse
ray turns into a new position / rotation / scale).  The Panda3D half
(``world/devtools_overlay.py``) only draws the lines and feeds this module the
cursor ray; swapping the renderer touches nothing here.

Conventions
-----------
- World axes (global orientation, like Unity's "Global" pivot mode): axis index
  ``0=X``, ``1=Y``, ``2=Z`` map to ``Vec3.RIGHT / FORWARD / UP``.
- ``size`` is the gizmo's world-space radius in **meters** (the renderer scales
  it with camera distance so it looks constant on screen); handle thicknesses
  are fractions of it.
- A drag is resolved **absolutely from its start pose** (the reference scalar /
  point / angle captured at ``begin``), never incrementally — so it never
  drifts and releasing + re-grabbing is exact.
- Rotations compose in world space: ``new = axis_delta * start`` (premultiply),
  matching :class:`~fire_engine.core.math3d.Quat` semantics.

No panda3d imports — headless-testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from fire_engine.core.math3d import Vec3, Quat


class GizmoMode(Enum):
    """Which manipulator is active (mirrors Unity's W/E/R tools)."""

    TRANSLATE = "translate"
    ROTATE = "rotate"
    SCALE = "scale"


class HandleType(Enum):
    """
    The kind of handle a ray can grab.

    AXIS    — a single-axis arrow (translate) or stalk (scale).
    PLANE   — a two-axis square (translate on the plane whose *normal* is ``axis``).
    RING    — a rotation ring in the plane whose *normal* is ``axis``.
    UNIFORM — the centre cube (uniform scale on all axes; ``axis`` ignored).
    """

    AXIS = "axis"
    PLANE = "plane"
    RING = "ring"
    UNIFORM = "uniform"


@dataclass(frozen=True)
class Handle:
    """
    One grabbable part of the gizmo.

    Parameters
    ----------
    type : HandleType
    axis : int
        ``0=X / 1=Y / 2=Z``.  For PLANE/RING it is the plane's *normal* axis;
        for UNIFORM it is unused (always 0).
    """

    type: HandleType
    axis: int


@dataclass
class DragState:
    """
    Captured reference pose for an in-progress drag (returned by :meth:`Gizmo.begin`).

    Holds the object's pose at grab time plus the one reference quantity the
    handle needs (axis parameter, plane point, ring angle, or radial distance),
    so :func:`update_drag` can compute an absolute new pose each frame.
    """

    mode: GizmoMode
    handle: Handle
    pivot: Vec3
    size: float
    start_position: Vec3
    start_rotation: Quat
    start_scale: Vec3
    ref_scalar: float = 0.0
    ref_point: Optional[np.ndarray] = None
    ref_angle: float = 0.0
    ref_dist: float = 0.0


# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------

_AXIS_NP = (
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
)
# Right-handed pair (j, k) with j × k == axis i — used for ring angle + plane.
_OTHER = {0: (1, 2), 1: (2, 0), 2: (0, 1)}


def _np(v: Vec3) -> np.ndarray:
    return v.to_numpy().astype(np.float64)


def _axis_vec(i: int) -> Vec3:
    a = _AXIS_NP[i]
    return Vec3(float(a[0]), float(a[1]), float(a[2]))


# ---------------------------------------------------------------------------
# Ray geometry primitives (numpy float64, pure functions)
# ---------------------------------------------------------------------------


def ray_plane_intersect(
    o: np.ndarray, d: np.ndarray, p: np.ndarray, n: np.ndarray
) -> Optional[np.ndarray]:
    """
    Intersect ray ``o + s·d`` (s ≥ 0) with the plane through ``p`` normal ``n``.

    Returns the hit point (length-3 array) or ``None`` if the ray is parallel to
    the plane or only hits it behind the origin.
    """
    denom = float(n.dot(d))
    if abs(denom) < 1e-9:
        return None
    s = float(n.dot(p - o) / denom)
    if s < 0.0:
        return None
    return o + s * d


def closest_on_axis(
    o: np.ndarray, d: np.ndarray, p: np.ndarray, a: np.ndarray
) -> tuple[float, float, float]:
    """
    Closest approach between ray ``o + s·d`` and axis line ``p + t·a`` (``a`` unit).

    Returns ``(axis_t, ray_s, dist)``: the parameter along the axis (meters), the
    parameter along the ray, and the distance between the two closest points.
    """
    w0 = o - p
    A = float(d.dot(d))
    B = float(d.dot(a))
    D = float(d.dot(w0))
    E = float(a.dot(w0))
    denom = A - B * B  # a·a is 1 for a unit axis
    if abs(denom) < 1e-9:
        axis_t = E
        ray_s = -D / A if A > 1e-12 else 0.0
    else:
        ray_s = (B * E - D) / denom
        axis_t = (A * E - B * D) / denom
    cp_ray = o + ray_s * d
    cp_axis = p + axis_t * a
    dist = float(np.linalg.norm(cp_ray - cp_axis))
    return axis_t, ray_s, dist


def _ring_angle(point: np.ndarray, pivot: np.ndarray, axis: int) -> float:
    """Angle (radians) of ``point`` around ``axis`` in that axis's ring plane."""
    j, k = _OTHER[axis]
    v = point - pivot
    return math.atan2(float(v.dot(_AXIS_NP[k])), float(v.dot(_AXIS_NP[j])))


# ---------------------------------------------------------------------------
# Gizmo: picking + drag begin
# ---------------------------------------------------------------------------


class Gizmo:
    """
    A transform manipulator anchored at ``pivot`` with on-screen radius ``size``.

    Construct one per frame from the selected object's world position, the
    camera-scaled size, and the active :class:`GizmoMode`; use :meth:`pick` to
    hit-test the cursor ray (for hover highlight and grab) and :meth:`begin` to
    start a drag.  Stateless apart from the three construction args.

    Example
    -------
        giz = Gizmo(obj.transform.position, size, GizmoMode.TRANSLATE)
        handle = giz.pick(ray_o, ray_d)
        if handle is not None:
            drag = giz.begin(handle, ray_o, ray_d, pos, rot, scale)
    """

    def __init__(self, pivot: Vec3, size: float, mode: GizmoMode) -> None:
        self.pivot = pivot
        self.size = float(size)
        self.mode = mode

    # -- picking --------------------------------------------------------

    def pick(self, ray_o: Vec3, ray_d: Vec3) -> Optional[Handle]:
        """
        Return the handle under the cursor ray (nearest to camera), or ``None``.

        Parameters
        ----------
        ray_o, ray_d : Vec3
            World-space ray origin and direction (need not be normalised).
        """
        o, d, p = _np(ray_o), _np(ray_d), _np(self.pivot)
        R = self.size
        axis_r = R * 0.18
        best: Optional[tuple[float, Handle]] = None

        def consider(depth: float, handle: Handle) -> None:
            nonlocal best
            if depth <= 0.0:
                return
            if best is None or depth < best[0]:
                best = (depth, handle)

        if self.mode in (GizmoMode.TRANSLATE, GizmoMode.SCALE):
            for i in range(3):
                a = _AXIS_NP[i]
                axis_t, ray_s, dist = closest_on_axis(o, d, p, a)
                if 0.0 <= axis_t <= R and dist <= axis_r:
                    consider(ray_s, Handle(HandleType.AXIS, i))

        if self.mode == GizmoMode.TRANSLATE:
            lo, hi = R * 0.15, R * 0.55
            for i in range(3):
                hit = ray_plane_intersect(o, d, p, _AXIS_NP[i])
                if hit is None:
                    continue
                j, k = _OTHER[i]
                cj = float((hit - p).dot(_AXIS_NP[j]))
                ck = float((hit - p).dot(_AXIS_NP[k]))
                if lo <= cj <= hi and lo <= ck <= hi:
                    consider(_ray_s_of(o, d, hit), Handle(HandleType.PLANE, i))

        if self.mode == GizmoMode.SCALE:
            cp_s = _ray_s_of(o, d, p)
            cp = o + cp_s * d
            if cp_s > 0.0 and float(np.linalg.norm(cp - p)) <= R * 0.2:
                consider(cp_s, Handle(HandleType.UNIFORM, 0))

        if self.mode == GizmoMode.ROTATE:
            for i in range(3):
                hit = ray_plane_intersect(o, d, p, _AXIS_NP[i])
                if hit is None:
                    continue
                r = float(np.linalg.norm(hit - p))
                if abs(r - R) <= R * 0.12:
                    consider(_ray_s_of(o, d, hit), Handle(HandleType.RING, i))

        return None if best is None else best[1]

    # -- drag begin -----------------------------------------------------

    def begin(
        self,
        handle: Handle,
        ray_o: Vec3,
        ray_d: Vec3,
        position: Vec3,
        rotation: Quat,
        scale: Vec3,
    ) -> DragState:
        """
        Capture the reference pose for a drag of ``handle``.

        Parameters
        ----------
        handle : Handle — the grabbed handle (from :meth:`pick`).
        ray_o, ray_d : Vec3 — the grabbing ray.
        position, rotation, scale : Vec3 / Quat / Vec3 — the object's pose now.

        Returns
        -------
        DragState — feed to :func:`update_drag` each frame until release.
        """
        o, d, p = _np(ray_o), _np(ray_d), _np(self.pivot)
        st = DragState(
            mode=self.mode,
            handle=handle,
            pivot=self.pivot,
            size=self.size,
            start_position=position,
            start_rotation=rotation,
            start_scale=scale,
        )
        if handle.type == HandleType.AXIS:
            st.ref_scalar = closest_on_axis(o, d, p, _AXIS_NP[handle.axis])[0]
        elif handle.type == HandleType.PLANE:
            st.ref_point = ray_plane_intersect(o, d, p, _AXIS_NP[handle.axis])
        elif handle.type == HandleType.RING:
            hit = ray_plane_intersect(o, d, p, _AXIS_NP[handle.axis])
            st.ref_angle = _ring_angle(hit, p, handle.axis) if hit is not None else 0.0
        elif handle.type == HandleType.UNIFORM:
            cp_s = _ray_s_of(o, d, p)
            st.ref_dist = float(np.linalg.norm((o + cp_s * d) - p))
        return st


def _ray_s_of(o: np.ndarray, d: np.ndarray, point: np.ndarray) -> float:
    """Ray parameter ``s`` (point = o + s·d) of a point's projection onto the ray."""
    dd = float(d.dot(d))
    return float((point - o).dot(d) / dd) if dd > 1e-12 else 0.0


# ---------------------------------------------------------------------------
# Drag resolution (absolute from the captured reference)
# ---------------------------------------------------------------------------


def update_drag(state: DragState, ray_o: Vec3, ray_d: Vec3) -> tuple[Vec3, Quat, Vec3]:
    """
    Resolve the current cursor ray into the object's new ``(position, rotation,
    scale)`` for an active drag.

    The result is computed **absolutely** from ``state``'s captured reference, so
    the object never drifts and a paused cursor leaves it still.  Components not
    affected by the handle are returned unchanged from the start pose.

    Parameters
    ----------
    state : DragState — from :meth:`Gizmo.begin`.
    ray_o, ray_d : Vec3 — the current cursor ray.

    Returns
    -------
    (Vec3, Quat, Vec3) — new local position, rotation, scale to assign.
    """
    o, d, p = _np(ray_o), _np(ray_d), _np(state.pivot)
    pos, rot, scl = state.start_position, state.start_rotation, state.start_scale
    h = state.handle

    if h.type == HandleType.AXIS:
        a = _AXIS_NP[h.axis]
        s_now = closest_on_axis(o, d, p, a)[0]
        delta = s_now - state.ref_scalar
        if state.mode == GizmoMode.TRANSLATE:
            return state.start_position + _axis_vec(h.axis) * delta, rot, scl
        # SCALE along one axis: drag out by `size` ≈ ×2.
        factor = max(1.0 + delta / max(state.size, 1e-6), 0.01)
        arr = scl.to_numpy()
        arr[h.axis] *= factor
        return pos, rot, Vec3.from_numpy(arr)

    if h.type == HandleType.PLANE:
        hit = ray_plane_intersect(o, d, p, _AXIS_NP[h.axis])
        if hit is None or state.ref_point is None:
            return pos, rot, scl
        return state.start_position + Vec3.from_numpy(hit - state.ref_point), rot, scl

    if h.type == HandleType.RING:
        hit = ray_plane_intersect(o, d, p, _AXIS_NP[h.axis])
        if hit is None:
            return pos, rot, scl
        dtheta = _ring_angle(hit, p, h.axis) - state.ref_angle
        new_rot = Quat.from_axis_angle(_axis_vec(h.axis), dtheta) * state.start_rotation
        return pos, new_rot, scl

    if h.type == HandleType.UNIFORM:
        cp_s = _ray_s_of(o, d, p)
        dist = float(np.linalg.norm((o + cp_s * d) - p))
        if state.ref_dist < 1e-6:
            return pos, rot, scl
        factor = max(dist / state.ref_dist, 0.01)
        return pos, rot, Vec3.from_numpy(scl.to_numpy() * factor)

    return pos, rot, scl
