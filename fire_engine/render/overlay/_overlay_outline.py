"""
render/overlay/_overlay_outline.py — Selection wireframe box drawing.

Extracted from devtools_overlay.py; called as free functions taking the overlay
instance as first argument (C0302 fat-class split pattern).

Docs: docs/systems/render.overlay.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from panda3d.core import LineSegs

if TYPE_CHECKING:
    from fire_engine.render.overlay.devtools_overlay import DevOverlay

_OUTLINE_COLOR = (0.40, 1.0, 0.35, 1.0)


def update_outline(self_obj: DevOverlay) -> None:
    """Remove the previous outline and draw a fresh one around the selection.

    Docs: docs/systems/render.overlay.md
    """
    if self_obj._outline_np is not None:
        self_obj._outline_np.remove_node()
        self_obj._outline_np = None

    go = self_obj.manager.selection.current
    if not self_obj.manager.enabled or go is None:
        return
    box = selection_aabb(self_obj, go)
    if box is None:
        return  # selection has no drawable box (e.g. the camera)
    draw_box(self_obj, *box)


def selection_aabb(self_obj: DevOverlay, go: Any) -> tuple[Any, Any] | None:
    """
    World-space AABB ``(min, max)`` to outline for the current selection.

    A picked terrain chunk outlines its full 16 m cube (origin → origin +
    size); a registered object uses its :class:`Selectable` box; anything
    else (e.g. the camera, which has no box) returns ``None``.

    Docs: docs/systems/render.overlay.md
    """
    from fire_engine.devtools import is_chunk

    if is_chunk(go):
        o = go.world_origin
        m = go.chunk_meters
        return (o.x, o.y, o.z), (o.x + m, o.y + m, o.z + m)
    sel = self_obj.manager.find_selectable(go)
    if sel is None:
        return None
    return sel.world_aabb()


def draw_box(self_obj: DevOverlay, bmin: Any, bmax: Any) -> None:
    """Attach a LineSegs wireframe box to the scene graph.

    Docs: docs/systems/render.overlay.md
    """
    x0, y0, z0 = float(bmin[0]), float(bmin[1]), float(bmin[2])
    x1, y1, z1 = float(bmax[0]), float(bmax[1]), float(bmax[2])
    corners = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),  # bottom
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),  # top
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # verticals
    ]
    ls = LineSegs("selection_outline")
    ls.set_color(*_OUTLINE_COLOR)
    ls.set_thickness(2.5)
    for a, b in edges:
        ls.move_to(*corners[a])
        ls.draw_to(*corners[b])
    node = self_obj._base.render.attach_new_node(ls.create())
    # Draw on top so the box is always visible through geometry.
    node.set_light_off()
    node.set_depth_test(False)
    node.set_depth_write(False)
    node.set_bin("fixed", 100)
    self_obj._outline_np = node
