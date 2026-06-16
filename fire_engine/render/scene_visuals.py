"""
world/scene_visuals.py — panda3d visuals for editor-authored scene objects.

The render half of :class:`fire_engine.scene.runtime.SceneRuntime` (which is
headless). The runtime instantiates GameObjects from a loaded scene and calls
:meth:`SceneVisualFactory.attach` per object; this factory walks the object's
**component list** (the source of truth — ``kind`` is only a creation archetype)
and gives each built-in component its in-game representation:

    Mesh{primitive=cube}    1 m stock-cube model (scales with local_scale)
    Mesh{primitive=sphere}  1 m procedural UV-sphere
    Light{color,intensity,radius}
                            a real ``PointLight`` on the GPU lighting pipeline,
                            using the component's own params (skipped with a log
                            line on the CPU backend)
    SpawnPoint / (no component)
                            nothing (the runtime exposes spawn_position from the
                            first kind=="spawn" object; main.py applies it)

Because visuals are component-driven, an ``empty`` given a Light component emits
light, and a ``cube`` whose Mesh is removed becomes an invisible transform.
Param hot-reload inside a running game session is a NON-goal: authored params
are read once at :meth:`attach` time (a fresh game load reflects edits); the
per-frame sync task only mirrors transforms.

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

Docs: docs/systems/render.md
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from panda3d.core import GeomNode, LQuaternionf, NodePath

from fire_engine.core.math3d import Vec3
from fire_engine.render.primitives import (
    CUBE_MODEL_SCALE,
    build_sphere_geom,
    load_cube_model,
)
from fire_engine.scene.components import (
    default_components_for_kind,
    default_params,
)

if TYPE_CHECKING:
    from fire_engine.render.gameobject import GameObject
    from fire_engine.scene.runtime import SceneRuntime

log = logging.getLogger(__name__)

_MOVE_EPS = 1e-4  # transform-change threshold for light re-injection/write-back


class SceneVisualFactory:
    """Creates and synchronises panda3d visuals for authored scene objects.

    Args:
        app: The App/ShowBase (owns ``render``, ``loader``, ``taskMgr``).
        lighting_pipeline: ``GpuLightingPipeline`` or ``None`` (CPU backend —
            authored lights are skipped with one log line).
        dev_overlay: ``DevOverlay`` or ``None``; when present, authored objects
            register as click-pickable in the F1 overlay.

    Docs: docs/systems/render.md
    """

    def __init__(self, app: Any, lighting_pipeline: Any = None, dev_overlay: Any = None) -> None:
        self._app = app
        self._pipeline = lighting_pipeline
        self._overlay = dev_overlay
        self.runtime: SceneRuntime | None = None  # set by main.py for write-back
        self._nodes: dict[GameObject, NodePath] = {}
        self._node_scale: dict[GameObject, float] = {}  # model-unit fixup
        self._light_ids: dict[GameObject, int] = {}
        self._ids: dict[GameObject, int] = {}  # go -> scene object id
        self._last_synced: dict[GameObject, tuple[float, ...]] = {}
        self._warned_no_pipeline = False
        app.taskMgr.add(self._sync_task, "scene-visuals-sync")

    # ------------------------------------------------------------------ #
    # Runtime contract
    # ------------------------------------------------------------------ #
    def attach(self, go: GameObject, kind: str, obj: dict[str, Any]) -> None:
        """Give ``go`` its in-game visuals by walking its component list.

        ``kind`` is only a fallback for pre-component data; the components list
        (Mesh, Light, ...) drives what is built. Disabled components are skipped.

        Docs: docs/systems/render.md
        """
        self._ids[go] = int(obj["id"])
        components = obj.get("components")
        if components is None:  # pre-component data: synthesise from kind
            components = default_components_for_kind(kind)

        has_mesh = False
        for comp in components:
            if not comp.get("enabled", True):
                continue
            ctype = comp.get("type")
            params = comp.get("params", {})
            if ctype == "Mesh" and go not in self._nodes:
                self._attach_mesh(go, obj, str(params.get("primitive", "cube")))
                has_mesh = go in self._nodes
            elif ctype == "Light" and go not in self._light_ids:
                self._attach_light(go, params)
        # SpawnPoint / empty / disabled: no visual.

        if self._overlay is not None:
            half = 0.5 if has_mesh else 0.25
            self._overlay.manager.add_selectable(go, Vec3(half, half, half))

    def _attach_mesh(self, go: GameObject, obj: dict[str, Any], primitive: str) -> None:
        if primitive == "sphere":
            node = GeomNode(f"scene_sphere_{obj['id']}")
            node.add_geom(build_sphere_geom(0.5))
            np_ = self._app.render.attach_new_node(node)
            np_.set_light_off()
            self._nodes[go] = np_
            self._node_scale[go] = 1.0
        else:  # cube (default)
            model = load_cube_model(self._app.loader)
            if model is not None:
                model.reparent_to(self._app.render)
                model.set_light_off()
                self._nodes[go] = model
                self._node_scale[go] = CUBE_MODEL_SCALE

    def _attach_light(self, go: GameObject, params: dict[str, Any]) -> None:
        if self._pipeline is None:
            if not self._warned_no_pipeline:
                log.info("authored lights skipped: no GPU lighting pipeline")
                self._warned_no_pipeline = True
            return
        from fire_engine.lighting.lights import PointLight

        defaults = default_params("Light")
        raw_color = [float(c) for c in params.get("color", defaults["color"])]
        color: tuple[float, float, float] = (raw_color[0], raw_color[1], raw_color[2])
        intensity = float(params.get("intensity", defaults["intensity"]))
        radius = float(params.get("radius", defaults["radius"]))
        p = go.transform.position
        lid = self._pipeline.lights.add(
            PointLight(
                position=(p.x, p.y, p.z),
                color=color,
                intensity=intensity,
                radius=radius,
            )
        )
        self._light_ids[go] = lid

    def teardown(self) -> None:
        """Synchronously remove every visual, light and pickable registration.

        Called by the runtime before a rebuild (and safe to call repeatedly);
        must not wait on the engine's deferred GameObject destroy, or a double
        load within one frame would leak NodePaths/lights.

        Docs: docs/systems/render.md
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
    def _sync_task(self, task: Any) -> Any:
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
                abs(a - b) <= _MOVE_EPS for a, b in zip(key, last, strict=True)
            ):
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
