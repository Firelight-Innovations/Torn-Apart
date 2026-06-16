"""
render/overlay/_overlay_spawn.py — Dev-prop spawning and emissive-light toggling.

Extracted from devtools_overlay.py; called as free functions taking the overlay
instance as first argument (C0302 fat-class split pattern).

Docs: docs/systems/render.overlay.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from panda3d.core import LQuaternionf

from fire_engine.core.math3d import Vec3
from fire_engine.lighting.lights import AreaLight
from fire_engine.render.registry import instantiate

if TYPE_CHECKING:
    from fire_engine.render.overlay.devtools_overlay import DevOverlay


def spawn_cube(self_obj: DevOverlay) -> Any:
    """
    Spawn a 1 m cube 5 m in front of the camera, select it, and make it
    pickable.  The cube has no components — it's a transform you can move and
    edit live in the Inspector (proof the edit round-trip works end to end).

    Returns
    -------
    GameObject — the spawned object.
    """
    cam_tf = self_obj._app.camera_go.transform
    pos = cam_tf.position + cam_tf.forward * 5.0
    go = instantiate(position=pos)
    self_obj._spawn_count += 1
    go.name = f"Cube{self_obj._spawn_count}"
    go.tag = "devspawn"

    model = self_obj._base.loader.load_model("models/misc/rgbCube")
    if model is None or model.is_empty():
        # Fallback: a plain box model name some Panda3D builds ship instead.
        model = self_obj._base.loader.load_model("box")
    if model is not None and not model.is_empty():
        model.set_scale(0.5)  # rgbCube spans -1..1 → a 1 m cube (half-extent 0.5)
        model.reparent_to(self_obj._base.render)
        model.set_light_off()
        self_obj._spawned[go] = model

    self_obj.manager.add_selectable(go, Vec3(0.5, 0.5, 0.5))
    self_obj.manager.selection.set(go)
    return go


def toggle_emissive(self_obj: DevOverlay) -> None:
    """
    Toggle the SELECTED spawned prop between inert and emissive.

    Emissive props register an :class:`~fire_engine.lighting.lights.AreaLight`
    matching their world bounds on the GPU lighting pipeline — the cube
    becomes a glowing box light feeding the GI gather and the froxel
    fog (the emission-map path for dynamic objects).  The visual gets a
    bright warm colour-scale so the prop itself reads as glowing.
    No-op without the GPU lighting backend or with nothing selected.
    """
    go = self_obj.manager.selection.current
    pipeline = getattr(self_obj._app, "lighting_pipeline", None)
    if go is None or go not in self_obj._spawned or pipeline is None:
        return
    np_ = self_obj._spawned[go]
    if go in self_obj._emissive:
        light_id, _ = self_obj._emissive.pop(go)
        pipeline.lights.remove(light_id)
        np_.clear_color_scale()
        return
    bounds = np_.get_tight_bounds()
    p = go.transform.position
    center = (p.x, p.y, p.z)
    half = (0.5, 0.5, 0.5)
    if bounds is not None:
        mn, mx = bounds
        center = ((mn.x + mx.x) * 0.5, (mn.y + mx.y) * 0.5, (mn.z + mx.z) * 0.5)
        half = (
            max((mx.x - mn.x) * 0.5, 0.05),
            max((mx.y - mn.y) * 0.5, 0.05),
            max((mx.z - mn.z) * 0.5, 0.05),
        )
    light = AreaLight(
        center=center, half_extents=half, color=(1.0, 0.78, 0.45), intensity=10.0, radius=14.0
    )
    self_obj._emissive[go] = (pipeline.lights.add(light), light)
    np_.set_color_scale(2.2, 1.8, 1.1, 1.0)  # the prop visibly glows


def sync_spawned(self_obj: DevOverlay) -> None:
    """
    Mirror each spawned GameObject's transform onto its NodePath, then
    push the props' world AABBs to the lighting pipeline as dynamic
    occluders (shadow casting / god-ray cutting) and keep any emissive
    prop's AreaLight glued to its box.  ``OccluderSet.set_boxes`` is
    change-detected internally, so static props cost nothing.
    """
    pipeline = getattr(self_obj._app, "lighting_pipeline", None)
    boxes: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    lights_dirty = False
    for go, np_ in self_obj._spawned.items():
        p = go.transform.position
        q = go.transform.rotation
        s = go.transform.local_scale
        np_.set_pos(p.x, p.y, p.z)
        np_.set_quat(LQuaternionf(q.w, q.x, q.y, q.z))
        np_.set_scale(0.5 * s.x, 0.5 * s.y, 0.5 * s.z)
        if pipeline is None:
            continue
        bounds = np_.get_tight_bounds()  # world AABB (includes rotation)
        if bounds is None:
            continue
        mn, mx = bounds
        boxes.append(((mn.x, mn.y, mn.z), (mx.x, mx.y, mx.z)))
        em = self_obj._emissive.get(go)
        if em is not None:
            light = em[1]
            center = ((mn.x + mx.x) * 0.5, (mn.y + mx.y) * 0.5, (mn.z + mx.z) * 0.5)
            if any(abs(a - b) > 0.01 for a, b in zip(center, light.center, strict=True)):
                light.center = center
                light.half_extents = (
                    max((mx.x - mn.x) * 0.5, 0.05),
                    max((mx.y - mn.y) * 0.5, 0.05),
                    max((mx.z - mn.z) * 0.5, 0.05),
                )
                lights_dirty = True
    if pipeline is not None:
        pipeline.occluders.set_boxes(boxes)
        if lights_dirty:
            pipeline.lights.notify_changed()
