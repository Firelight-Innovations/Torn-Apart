"""SceneRuntime — game-side loader for editor-authored scenes.

The Saveable the GAME registers under save_key ``"editor_scene"`` (the same key
the Fire Editor's :class:`~fire_engine.scene.objects.SceneObjectStore` writes),
so a scene saved in the editor materialises as live GameObjects on load:

    kind        runtime mapping
    ----        ---------------
    empty       bare GameObject (a transform/grouping node)
    cube        GameObject + 1 m cube visual (via the visual factory)
    sphere      GameObject + 1 m sphere visual (via the visual factory)
    light       GameObject + a real PointLight emitter (via the visual factory)
    spawn       GameObject; the FIRST one (DFS order) is the player start —
                read :attr:`SceneRuntime.spawn_position` after load

This module is headless (zero panda3d imports): all rendering/lighting is
delegated to a ``visual_factory`` constructed in ``world/`` (see
``fire_engine.render.scene_visuals.SceneVisualFactory``). With
``visual_factory=None`` (tests, dedicated servers) objects still instantiate
with correct transforms — they just have no visuals.

Example::

    from fire_engine.scene import SceneRuntime
    runtime = SceneRuntime(visual_factory=factory)   # factory may be None
    save_manager.register(runtime)                   # claims "editor_scene"
    save_manager.load("scenes/ambush.ta")            # → GameObjects exist now
    if runtime.spawn_position is not None:
        app.camera_go.transform.position = runtime.spawn_position
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from fire_engine.core.math3d import Quat, Vec3
from fire_engine.scene.objects import SceneObjectStore

if TYPE_CHECKING:
    from fire_engine.render.gameobject import GameObject

# NOTE: fire_engine.render is imported INSIDE rebuild(), not at module level.
# Importing any world submodule executes the world package __init__, which
# pulls panda3d when it is installed — and this module reaches the editor
# daemon via the fire_editor.scene_objects shim, where panda3d is forbidden
# (tests/editor/test_no_panda3d.py). The daemon only ever touches `store`;
# rebuild() runs in the game, where panda3d is fine.

log = logging.getLogger(__name__)

# Tag stamped on every GameObject this runtime creates — lets tools and the
# dev overlay distinguish authored content from procedural/debug objects.
SCENE_TAG = "editor_scene"


class SceneRuntime:
    """Instantiates an authored scene's objects as live GameObjects.

    Implements the ``Saveable`` protocol. Owns a :class:`SceneObjectStore` by
    composition (the game registers THIS object, never a bare store — only one
    system per ``save_key`` may register with a SaveManager).

    Attributes:
        store: The authored-scene data (ids, kinds, local TRS).
        objects: ``{scene object id: GameObject}`` for the current build.
        visual_factory: Optional object with ``attach(go, kind, obj_dict)`` and
            ``teardown()`` — the panda3d half (``SceneVisualFactory``); ``None``
            keeps everything headless.
    """

    save_key: str = "editor_scene"

    def __init__(self, visual_factory: Any | None = None,
                 on_rebuilt: Callable[[], None] | None = None) -> None:
        self.store = SceneObjectStore()
        self.objects: "dict[int, GameObject]" = {}
        self.visual_factory = visual_factory
        self.on_rebuilt = on_rebuilt

    # ------------------------------------------------------------------ #
    # Saveable protocol
    # ------------------------------------------------------------------ #
    def get_delta(self) -> dict:
        """The authored scene as saved by the editor (empty dict when empty)."""
        return self.store.get_delta()

    def apply_delta(self, delta: dict) -> None:
        """Restore the authored scene and (re)build its GameObjects."""
        self.store.apply_delta(delta)
        self.rebuild()

    # ------------------------------------------------------------------ #
    # Build / teardown
    # ------------------------------------------------------------------ #
    def rebuild(self) -> None:
        """Tear down any previous build, then instantiate the store's objects.

        Visuals are torn down synchronously (NodePaths/lights must not leak on
        a double load); GameObject destruction is the engine's normal deferred
        end-of-frame flush — harmless, the stale objects have no components or
        visuals by then.

        ``store.tree()`` is DFS (parents before children), so each object's
        parent GameObject already exists when it is reached. Parenting uses
        ``keep_world=False`` and THEN writes local TRS — the store's transforms
        are local to the parent, exactly like ``scene.set_transform``.
        """
        from fire_engine.render.registry import destroy, instantiate

        if self.visual_factory is not None:
            self.visual_factory.teardown()
        for go in self.objects.values():
            destroy(go)
        self.objects.clear()

        for obj in self.store.tree():
            go = instantiate()
            go.name = str(obj["name"])
            go.tag = SCENE_TAG
            parent_id = obj["parent"]
            if parent_id is not None:
                parent_go = self.objects.get(int(parent_id))
                if parent_go is not None:  # tree() guarantees this; belt+braces
                    go.transform.set_parent(parent_go.transform, keep_world=False)
            px, py, pz = obj["position"]
            rw, rx, ry, rz = obj["rotation"]
            sx, sy, sz = obj["scale"]
            go.transform.local_position = Vec3(float(px), float(py), float(pz))
            go.transform.local_rotation = Quat(float(rw), float(rx), float(ry), float(rz))
            go.transform.local_scale = Vec3(float(sx), float(sy), float(sz))
            self.objects[int(obj["id"])] = go
            if self.visual_factory is not None:
                self.visual_factory.attach(go, str(obj["kind"]), obj)

        log.info("scene runtime: built %d authored object(s)", len(self.objects))
        if self.on_rebuilt is not None:
            self.on_rebuilt()

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    @property
    def spawn_position(self) -> Vec3 | None:
        """World position of the first ``spawn`` object (DFS order), or None.

        World, not local — a spawn parented under a moved group composes its
        parents' transforms.
        """
        for obj in self.store.tree():
            if obj["kind"] == "spawn":
                go = self.objects.get(int(obj["id"]))
                if go is not None:
                    return go.transform.position
        return None
