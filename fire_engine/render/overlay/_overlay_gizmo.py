"""
render/overlay/_overlay_gizmo.py — Transform gizmo drawing + drag logic.

Extracted from devtools_overlay.py; called as free functions taking the overlay
instance as first argument (C0302 fat-class split pattern).

Docs: docs/systems/render.overlay.md
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from panda3d.core import LineSegs

from fire_engine.core.math3d import Vec3
from fire_engine.devtools import (
    Button,
    Field,
    FieldKind,
    Gizmo,
    GizmoMode,
    Handle,
    Section,
    is_chunk,
    update_drag,
)

if TYPE_CHECKING:
    from fire_engine.render.overlay.devtools_overlay import DevOverlay


# ---------------------------------------------------------------------------
# Class-level constant tables (kept here so the draw functions can use them)
# ---------------------------------------------------------------------------

_AXIS_DIR: tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

_AXIS_COL: tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
] = ((1.0, 0.35, 0.35, 1.0), (0.4, 1.0, 0.4, 1.0), (0.45, 0.6, 1.0, 1.0))

_HL_COL: tuple[float, float, float, float] = (1.0, 1.0, 0.3, 1.0)

_OTHER_AXES: dict[int, tuple[int, int]] = {0: (1, 2), 1: (2, 0), 2: (0, 1)}


# ---------------------------------------------------------------------------
# Panel builder
# ---------------------------------------------------------------------------


def build_gizmo_panel(self_obj: DevOverlay) -> tuple[list[Section], list[Button]]:
    """Build the Gizmo panel: a current-mode read-out + tool buttons.

    Docs: docs/systems/render.overlay.md
    """

    def mode_label() -> str:
        return self_obj._gizmo_mode.value if self_obj._gizmo_mode is not None else "off"

    sections = [Section("", [Field("tool", FieldKind.LABEL, mode_label)])]
    buttons = [
        Button("Move", lambda: set_gizmo_mode(self_obj, GizmoMode.TRANSLATE)),
        Button("Rotate", lambda: set_gizmo_mode(self_obj, GizmoMode.ROTATE)),
        Button("Scale", lambda: set_gizmo_mode(self_obj, GizmoMode.SCALE)),
        Button("Off", lambda: set_gizmo_mode(self_obj, None)),
    ]
    return sections, buttons


def set_gizmo_mode(self_obj: DevOverlay, mode: GizmoMode | None) -> None:
    """Switch the active gizmo tool (``None`` hides the gizmo).

    Docs: docs/systems/render.overlay.md
    """
    self_obj._gizmo_mode = mode
    self_obj._gizmo_drag = None  # cancel any in-flight drag on a mode switch


# ---------------------------------------------------------------------------
# Gizmo target / pivot helpers
# ---------------------------------------------------------------------------


def gizmo_target(self_obj: DevOverlay) -> Any:
    """
    The object the gizmo currently manipulates, or ``None``.

    Only a registered, pickable GameObject qualifies — that excludes the
    camera (no AABB; ``FlyController`` overwrites its rotation anyway) and
    picked terrain chunks (not GameObjects), and requires an active mode.

    Docs: docs/systems/render.overlay.md
    """
    if self_obj._gizmo_mode is None:
        return None
    go = self_obj.manager.selection.current
    if go is None or is_chunk(go):
        return None
    if self_obj.manager.find_selectable(go) is None:
        return None
    return go


def gizmo_pivot_size(self_obj: DevOverlay, go: Any) -> tuple[Vec3, float]:
    """Gizmo pivot (object origin) + a camera-distance-scaled world size.

    Docs: docs/systems/render.overlay.md
    """
    pivot = go.transform.local_position
    cam = self_obj._app.camera_go.transform.position
    dist = (pivot - cam).length
    return pivot, max(dist * 0.14, 0.3)


# ---------------------------------------------------------------------------
# Drag begin / update / end
# ---------------------------------------------------------------------------


def begin_gizmo(self_obj: DevOverlay, origin: Vec3, direction: Vec3) -> bool:
    """
    If a gizmo handle is under the cursor, start dragging it.

    Returns ``True`` (click consumed) when a drag began, so the click does
    not also re-select or deselect.

    Docs: docs/systems/render.overlay.md
    """
    go = gizmo_target(self_obj)
    if go is None or self_obj._gizmo_mode is None:
        return False
    pivot, size = gizmo_pivot_size(self_obj, go)
    giz = Gizmo(pivot, size, self_obj._gizmo_mode)
    handle = giz.pick(origin, direction)
    if handle is None:
        return False
    tf = go.transform
    self_obj._gizmo_drag = giz.begin(
        handle,
        origin,
        direction,
        tf.local_position,
        tf.local_rotation,
        tf.local_scale,
    )
    self_obj._gizmo_go = go
    return True


def update_gizmo(self_obj: DevOverlay) -> None:
    """Per-frame: apply an active drag and redraw the gizmo (or clear it).

    Docs: docs/systems/render.overlay.md
    """
    if self_obj._gizmo_np is not None:
        self_obj._gizmo_np.remove_node()
        self_obj._gizmo_np = None

    go = gizmo_target(self_obj) if self_obj.manager.enabled else None
    if go is None:
        self_obj._gizmo_drag = None
        return

    ray = self_obj._cursor_ray()
    hovered = None
    if self_obj._gizmo_drag is not None:
        # Resolve the live drag and write the new pose back to the object.
        if ray is not None:
            pos, rot, scl = update_drag(self_obj._gizmo_drag, ray[0], ray[1])
            tf = go.transform
            tf.local_position = pos
            tf.local_rotation = rot
            tf.local_scale = scl
        hovered = self_obj._gizmo_drag.handle
    elif ray is not None and self_obj._base.mouseWatcherNode.get_over_region() is None:
        # Hover highlight when not dragging and not over a panel.
        pivot, size = gizmo_pivot_size(self_obj, go)
        if self_obj._gizmo_mode is not None:
            hovered = Gizmo(pivot, size, self_obj._gizmo_mode).pick(ray[0], ray[1])

    pivot, size = gizmo_pivot_size(self_obj, go)
    draw_gizmo(self_obj, pivot, size, self_obj._gizmo_mode, hovered)


# ---------------------------------------------------------------------------
# Axis colour helper
# ---------------------------------------------------------------------------


def gizmo_axis_col(i: int, htype: Any, hovered: Handle | None) -> tuple[float, float, float, float]:
    """Return the per-axis colour (highlighted when the handle is hovered).

    Docs: docs/systems/render.overlay.md
    """
    from fire_engine.devtools import HandleType

    hot = (
        hovered is not None
        and hovered.type == htype
        and (htype == HandleType.UNIFORM or hovered.axis == i)
    )
    return _HL_COL if hot else _AXIS_COL[i]


# ---------------------------------------------------------------------------
# Geometry drawing helpers
# ---------------------------------------------------------------------------


def draw_gizmo_axes(
    ls: Any,
    px: float,
    py: float,
    pz: float,
    size: float,
    hovered: Handle | None,
) -> None:
    """Draw the three axis arrows with cross-tip markers (translate + scale).

    Docs: docs/systems/render.overlay.md
    """
    from fire_engine.devtools import HandleType

    for i, a in enumerate(_AXIS_DIR):
        ls.set_color(*gizmo_axis_col(i, HandleType.AXIS, hovered))
        ex, ey, ez = px + a[0] * size, py + a[1] * size, pz + a[2] * size
        ls.move_to(px, py, pz)
        ls.draw_to(ex, ey, ez)
        j, k = _OTHER_AXES[i]
        t = size * 0.12
        jd, kd = _AXIS_DIR[j], _AXIS_DIR[k]
        for sgn in (t, -t):
            ls.move_to(ex, ey, ez)
            ls.draw_to(ex + jd[0] * sgn, ey + jd[1] * sgn, ez + jd[2] * sgn)
            ls.move_to(ex, ey, ez)
            ls.draw_to(ex + kd[0] * sgn, ey + kd[1] * sgn, ez + kd[2] * sgn)


def draw_gizmo_rings(
    ls: Any,
    px: float,
    py: float,
    pz: float,
    size: float,
    hovered: Handle | None,
) -> None:
    """Draw three rotation rings (one per axis).

    Docs: docs/systems/render.overlay.md
    """
    from fire_engine.devtools import HandleType

    seg = 48
    for i in range(3):
        ls.set_color(*gizmo_axis_col(i, HandleType.RING, hovered))
        j, k = _OTHER_AXES[i]
        jd, kd = _AXIS_DIR[j], _AXIS_DIR[k]
        for n in range(seg + 1):
            ang = 2.0 * math.pi * n / seg
            cj, ck = math.cos(ang) * size, math.sin(ang) * size
            x = px + jd[0] * cj + kd[0] * ck
            y = py + jd[1] * cj + kd[1] * ck
            z = pz + jd[2] * cj + kd[2] * ck
            (ls.move_to if n == 0 else ls.draw_to)(x, y, z)


def draw_gizmo(
    self_obj: DevOverlay,
    pivot: Vec3,
    size: float,
    mode: GizmoMode | None,
    hovered: Handle | None,
) -> None:
    """Assemble and attach the gizmo LineSegs node to the scene.

    Docs: docs/systems/render.overlay.md
    """
    from fire_engine.devtools import HandleType

    ls = LineSegs("gizmo")
    ls.set_thickness(2.5)
    px, py, pz = pivot.x, pivot.y, pivot.z

    if mode in (GizmoMode.TRANSLATE, GizmoMode.SCALE):
        draw_gizmo_axes(ls, px, py, pz, size, hovered)

    if mode == GizmoMode.TRANSLATE:
        lo, hi = size * 0.2, size * 0.45
        for i in range(3):
            ls.set_color(*gizmo_axis_col(i, HandleType.PLANE, hovered))
            j, k = _OTHER_AXES[i]
            jd, kd = _AXIS_DIR[j], _AXIS_DIR[k]
            corners = [(lo, lo), (hi, lo), (hi, hi), (lo, hi), (lo, lo)]
            for n, (cj, ck) in enumerate(corners):
                x = px + jd[0] * cj + kd[0] * ck
                y = py + jd[1] * cj + kd[1] * ck
                z = pz + jd[2] * cj + kd[2] * ck
                (ls.move_to if n == 0 else ls.draw_to)(x, y, z)

    if mode == GizmoMode.SCALE:
        uni_hot = hovered is not None and hovered.type == HandleType.UNIFORM
        ls.set_color(*(_HL_COL if uni_hot else (0.9, 0.9, 0.9, 1.0)))
        c = size * 0.1
        box = [(-c, -c), (c, -c), (c, c), (-c, c), (-c, -c)]
        for n, (dx, dz) in enumerate(box):
            (ls.move_to if n == 0 else ls.draw_to)(px + dx, py, pz + dz)

    if mode == GizmoMode.ROTATE:
        draw_gizmo_rings(ls, px, py, pz, size, hovered)

    node = self_obj._base.render.attach_new_node(ls.create())
    node.set_light_off()
    node.set_depth_test(False)  # always visible through geometry
    node.set_depth_write(False)
    node.set_bin("fixed", 110)  # above the selection outline (bin 100)
    self_obj._gizmo_np = node
