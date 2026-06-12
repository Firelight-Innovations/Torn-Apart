"""
world/scene_visuals.py — panda3d visuals for editor-authored scene objects.

The render half of :class:`fire_engine.scene.runtime.SceneRuntime` (which is
headless). The runtime instantiates GameObjects from a loaded scene and calls
:meth:`SceneVisualFactory.attach` per object; this factory gives each kind its
in-game representation:

    cube    1 m stock-cube model (scales with the object's local_scale)
    sphere  1 m procedural UV-sphere
    light   a real ``PointLight`` on the GPU lighting pipeline (warm torch
            defaults; skipped with a log line on the CPU backend)
    empty   nothing (a grouping transform)
    spawn   nothing (the runtime exposes spawn_position; main.py applies it)

Every visual object is also registered click-pickable with the dev overlay, so
F1 → click inspects/moves authored objects exactly like dev-spawned cubes.
A per-frame task mirrors GameObject transforms onto the NodePaths/lights AND
writes gizmo edits back into the runtime's store, so F5 after moving an object
in-game saves the moved transform (documented in docs/systems/devtools.md).

Example (wired by main.py)::

    factory = SceneVisualFactory(app, lighting_pipeline, overlay)
    runtime = SceneRuntime(visual_factory=factory)
    factory.runtime = runtime          # enables store write-back
    save_manager.register(runtime)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from panda3d.core import GeomNode, LQuaternionf, NodePath  # type: ignore[import]

from fire_engine.core.math3d import Vec3
from fire_engine.world.primitives import (
    CUBE_MODEL_SCALE,
    build_sphere_geom,
    load_cube_model,
)

if TYPE_CHECKING:
    from fire_engine.scene.runtime import SceneRuntime
    from fire_engine.world.gameobject import GameObject

log = logging.getLogger(__name__)

# Warm torch defaults for authored "light" objects — matches main.py's
# on_drop_torch so an editor-placed light reads like the familiar dev torch.
# (DECISIONS.md 2026-06-12: SceneObject has no params field yet; when one is
# added these become per-object overrides.)
LIGHT_COLOR: tuple[float, float, float] = (1.0, 0.62, 0.28)
LIGHT_INTENSITY: float = 8.0
LIGHT_RADIUS_M: float = 16.0

_MOVE_EPS = 1e-4  # transform-change threshold for light re-injection/write-back


class SceneVisualFactory:
    """Creates and synchronises panda3d visuals for authored scene objects.

    Args:
        app: The App/ShowBase (owns ``render``, ``loader``, ``taskMgr``).
        lighting_pipeline: ``GpuLightingPipeline`` or ``None`` (CPU backend —
            authored lights are skipped with one log line).
        dev_overlay: ``DevOverlay`` or ``None``; when present, authored objects
            register as click-pickable in the F1 overlay.
    """

    def __init__(self, app, lighting_pipeline=None, dev_overlay=None) -> None:
        self._app = app
        self._pipeline = lighting_pipeline
        self._overlay = dev_overlay
        self.runtime: "SceneRuntime | None" = None  # set by main.py for write-back
        self._nodes: dict["GameObject", NodePath] = {}
        self._node_scale: dict["GameObject", float] = {}  # model-unit fixup
        self._light_ids: dict["GameObject", int] = {}
        self._ids: dict["GameObject", int] = {}           # go -> scene object id
        self._last_synced: dict["GameObject", tuple] = {}
        self._warned_no_pipeline = False
        app.taskMgr.add(self._sync_task, "scene-visuals-sync")

    # ------------------------------------------------------------------ #
    # Runtime contract
    # ------------------------------------------------------------------ #
    def attach(self, go: "GameObject", kind: str, obj: dict) -> None:
        """Give ``go`` (an authored object of ``kind``) its in-game visual."""
        self._ids[go] = int(obj["id"])
        if kind == "cube":
            model = load_cube_model(self._app.loader)
            if model is not None:
                model.reparent_to(self._app.render)
                model.set_light_off()
                self._nodes[go] = model
                self._node_scale[go] = CUBE_MODEL_SCALE
        elif kind == "sphere":
            node = GeomNode(f"scene_sphere_{obj['id']}")
            node.add_geom(build_sphere_geom(0.5))
            np_ = self._app.render.attach_new_node(node)
            np_.set_light_off()
            self._nodes[go] = np_
            self._node_scale[go] = 1.0
        elif kind == "light":
            if self._pipeline is None:
                if not self._warned_no_pipeline:
                    log.info("authored lights skipped: no GPU lighting pipeline")
                    self._warned_no_pipeline = True
            else:
                from fire_engine.lighting.lights import PointLight
                p = go.transform.position
                lid = self._pipeline.lights.add(PointLight(
                    position=(p.x, p.y, p.z),
                    color=LIGHT_COLOR,
                    intensity=LIGHT_INTENSITY,
                    radius=LIGHT_RADIUS_M,
                ))
                self._light_ids[go] = lid
        # empty / spawn: no visual.

        if self._overlay is not None:
            half = 0.5 if kind in ("cube", "sphere") else 0.25
            self._overlay.manager.add_selectable(go, Vec3(half, half, half))

    def teardown(self) -> None:
        """Synchronously remove every visual, light and pickable registration.

        Called by the runtime before a rebuild (and safe to call repeatedly);
        must not wait on the engine's deferred GameObject destroy, or a double
        load within one frame would leak NodePaths/lights.
        """
        for np_ in self._nodes.values():
            np_.remove_node()
        self._nodes.clear()
        self._node_scale.clear()
        if self._pipeline is not None:
            for lid in self._light_ids.values():
                self._pipeline.lights.remove(lid)
        self._light_ids.clear()
        if self._overlay is not None:
            for go in self._ids:
                self._overlay.manager.remove_selectable(go)
        self._ids.clear()
        self._last_synced.clear()

    # ------------------------------------------------------------------ #
    # Per-frame sync
    # ------------------------------------------------------------------ #
    def _sync_task(self, task):
        """Mirror GameObject transforms onto visuals; write edits to the store.

        The F1 overlay's gizmo moves the GameObject's Transform directly; this
        task keeps the rendered model / light glued to it and pushes the new
        LOCAL transform back into the authored store (via the runtime), so a
        subsequent F5 persists the gizmo edit instead of the stale authored
        value.
        """
        lights_dirty = False
        for go, sid in self._ids.items():
            t = go.transform
            p, q, s = t.position, t.rotation, t.local_scale
            key = (p.x, p.y, p.z, q.w, q.x, q.y, q.z, s.x, s.y, s.z)
            last = self._last_synced.get(go)
            if last is not None and all(
                    abs(a - b) <= _MOVE_EPS for a, b in zip(key, last)):
                continue
            self._last_synced[go] = key

            np_ = self._nodes.get(go)
            if np_ is not None:
                f = self._node_scale[go]
                np_.set_pos(p.x, p.y, p.z)
                np_.set_quat(LQuaternionf(q.w, q.x, q.y, q.z))
                np_.set_scale(f * s.x, f * s.y, f * s.z)

            lid = self._light_ids.get(go)
            if lid is not None and self._pipeline is not None:
                light = self._pipeline.lights.get(lid)
                if light is not None:
                    light.position = (p.x, p.y, p.z)
                    lights_dirty = True

            if self.runtime is not None and last is not None:
                # Skip the first sync (authored values are already in the store).
                lp, lq, ls = t.local_position, t.local_rotation, s
                self.runtime.store.set_transform(
                    sid,
                    position=(lp.x, lp.y, lp.z),
                    rotation=(lq.w, lq.x, lq.y, lq.z),
                    scale=(ls.x, ls.y, ls.z),
                )
        if lights_dirty:
            self._pipeline.lights.notify_changed()
        return task.cont
