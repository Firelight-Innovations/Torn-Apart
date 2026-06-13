"""SceneService — authoring hierarchy RPC (EDITOR_PRD Phase E2).

Registers the ``scene.*`` methods that create, rename, reparent, transform,
delete and (add/remove/edit components on) placeable objects in the open
world's :class:`SceneObjectStore`, plus ``scene.catalog`` (the static built-in
component catalog the inspector renders from). Every mutation broadcasts a
``scene.changed`` notification carrying the full object list, so the sidebar
tree and the 3D viewport stay in sync without each having to diff. Pure
control-channel JSON — no binary frames here.

Every mutation also pushes a :class:`~fire_editor.commands.SceneCommand` onto
the daemon's single undo stack (shared with terrain brushes), so Ctrl+Z walks
scene and terrain edits chronologically. Rapid same-object ``set_transform``
calls coalesce into one command — a throttled gizmo drag undoes in one step.
"""
from __future__ import annotations

import logging
import time

from fire_engine.scene.components import catalog_payload

from .._generated import ErrorCode, Method, Notification
from ..commands import SceneCommand
from ..rpc import RpcError
from ..scene_objects import SceneError
from ..session import EditorSession

log = logging.getLogger("fire_editor.scene")

# set_transform commands with the same label arriving within this window merge
# into the previous command (drag coalescing).
_COALESCE_WINDOW_S = 1.0


class SceneService:
    """Owns the ``scene.*`` methods for one daemon."""

    def __init__(self, daemon) -> None:
        self.daemon = daemon
        self._register()

    def _register(self) -> None:
        d = self.daemon.dispatcher
        d.register(Method.SCENE_TREE, self.tree)
        d.register(Method.SCENE_CREATE, self.create)
        d.register(Method.SCENE_RENAME, self.rename)
        d.register(Method.SCENE_REPARENT, self.reparent)
        d.register(Method.SCENE_SET_TRANSFORM, self.set_transform)
        d.register(Method.SCENE_DELETE, self.delete)
        d.register(Method.SCENE_CATALOG, self.catalog)
        d.register(Method.SCENE_ADD_COMPONENT, self.add_component)
        d.register(Method.SCENE_REMOVE_COMPONENT, self.remove_component)
        d.register(Method.SCENE_SET_COMPONENT, self.set_component)

    # ------------------------------------------------------------------ #
    # Methods
    # ------------------------------------------------------------------ #
    async def tree(self, params: dict) -> dict:
        store = self._require_session().scene
        return {"objects": store.tree()}

    async def create(self, params: dict) -> dict:
        store = self._require_session().scene
        pos = (
            float(params.get("x") or 0.0),
            float(params.get("y") or 0.0),
            float(params.get("z") or 0.0),
        )
        parent = params.get("parent")
        before = store.get_delta()
        try:
            obj = store.create(
                str(params["kind"]),
                parent=None if parent is None else int(parent),
                name=params.get("name"),
                position=pos,
            )
        except SceneError as e:
            raise RpcError(ErrorCode.INVALID_PARAMS, str(e))
        self._push_history(f"create {obj['kind']}", before, store.get_delta())
        await self._broadcast_changed(store)
        return {"ok": True, "object": obj}

    async def rename(self, params: dict) -> dict:
        store = self._require_session().scene
        before = store.get_delta()
        try:
            obj = store.rename(int(params["id"]), str(params["name"]))
        except SceneError as e:
            raise RpcError(ErrorCode.INVALID_PARAMS, str(e))
        self._push_history(f"rename {obj['id']}", before, store.get_delta())
        await self._broadcast_changed(store)
        return {"ok": True, "object": obj}

    async def reparent(self, params: dict) -> dict:
        store = self._require_session().scene
        parent = params.get("parent")
        before = store.get_delta()
        try:
            obj = store.reparent(int(params["id"]), None if parent is None else int(parent))
        except SceneError as e:
            raise RpcError(ErrorCode.INVALID_PARAMS, str(e))
        self._push_history(f"reparent {obj['id']}", before, store.get_delta())
        await self._broadcast_changed(store)
        return {"ok": True, "object": obj}

    async def set_transform(self, params: dict) -> dict:
        store = self._require_session().scene
        position = _vec3(params, "px", "py", "pz")
        rotation = _quat(params, "rw", "rx", "ry", "rz")
        scale = _vec3(params, "sx", "sy", "sz")
        before = store.get_delta()
        try:
            obj = store.set_transform(
                int(params["id"]), position=position, rotation=rotation, scale=scale
            )
        except SceneError as e:
            raise RpcError(ErrorCode.INVALID_PARAMS, str(e))
        self._push_history(f"transform {obj['id']}", before, store.get_delta(),
                           coalesce=True)
        await self._broadcast_changed(store)
        return {"ok": True, "object": obj}

    async def delete(self, params: dict) -> dict:
        store = self._require_session().scene
        before = store.get_delta()
        try:
            removed = store.delete(int(params["id"]))
        except SceneError as e:
            raise RpcError(ErrorCode.INVALID_PARAMS, str(e))
        self._push_history(f"delete {params['id']}", before, store.get_delta())
        await self._broadcast_changed(store)
        return {"ok": True, "removed": removed}

    # ------------------------------------------------------------------ #
    # Components
    # ------------------------------------------------------------------ #
    async def catalog(self, params: dict) -> dict:
        """The built-in component catalog (static; no open world required)."""
        return catalog_payload()

    async def add_component(self, params: dict) -> dict:
        store = self._require_session().scene
        before = store.get_delta()
        try:
            obj = store.add_component(int(params["id"]), str(params["type"]))
        except SceneError as e:
            raise RpcError(ErrorCode.INVALID_PARAMS, str(e))
        self._push_history(f"add component {obj['id']}", before, store.get_delta())
        await self._broadcast_changed(store)
        return {"ok": True, "object": obj}

    async def remove_component(self, params: dict) -> dict:
        store = self._require_session().scene
        before = store.get_delta()
        try:
            obj = store.remove_component(int(params["id"]), int(params["index"]))
        except SceneError as e:
            raise RpcError(ErrorCode.INVALID_PARAMS, str(e))
        self._push_history(f"remove component {obj['id']}", before, store.get_delta())
        await self._broadcast_changed(store)
        return {"ok": True, "object": obj}

    async def set_component(self, params: dict) -> dict:
        store = self._require_session().scene
        index = int(params["index"])
        enabled = params.get("enabled")
        before = store.get_delta()
        try:
            obj = store.set_component(
                int(params["id"]),
                index,
                params=params.get("params"),
                enabled=None if enabled is None else bool(enabled),
            )
        except SceneError as e:
            raise RpcError(ErrorCode.INVALID_PARAMS, str(e))
        # Per-(object,index) label so a slider drag coalesces but editing a
        # different component starts a new undo step.
        self._push_history(
            f"component {obj['id']}.{index}", before, store.get_delta(),
            coalesce=True,
        )
        await self._broadcast_changed(store)
        return {"ok": True, "object": obj}

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _push_history(self, label: str, before: dict, after: dict,
                      coalesce: bool = False) -> None:
        """Record a scene mutation on the daemon's shared undo stack.

        With ``coalesce=True`` a command whose label matches the stack top
        within :data:`_COALESCE_WINDOW_S` merges into it (keep the oldest
        ``before``, take the newest ``after``) so a stream of throttled gizmo
        ``set_transform`` calls undoes as one step.
        """
        history = self.daemon.chunks.history
        now = time.monotonic()
        if coalesce:
            top = history.peek()
            if (isinstance(top, SceneCommand) and top.label == label
                    and now - top.timestamp <= _COALESCE_WINDOW_S):
                history.replace_top(SceneCommand(label, top.before_delta, after, now))
                return
        history.push(SceneCommand(label, before, after, now))

    async def _broadcast_changed(self, store) -> None:
        await self.daemon.server.broadcast_notification(
            Notification.SCENE_CHANGED, {"objects": store.tree()}
        )

    def _require_session(self) -> EditorSession:
        s = self.daemon.session
        if s is None:
            raise RpcError(ErrorCode.APP_ERROR, "no world open; call world.open first")
        return s


def _vec3(params: dict, *keys: str) -> tuple[float, float, float] | None:
    """Build a 3-tuple from params if *any* component is present, else None.

    A missing component defaults to 0 so a partial update still produces a full
    vector; callers that want to leave a whole channel untouched simply omit all
    three keys.
    """
    if not any(k in params and params[k] is not None for k in keys):
        return None
    return tuple(float(params.get(k) or 0.0) for k in keys)  # type: ignore[return-value]


def _quat(params: dict, *keys: str) -> tuple[float, float, float, float] | None:
    if not any(k in params and params[k] is not None for k in keys):
        return None
    # Default to identity (w=1) when only some components are supplied.
    w = params.get(keys[0])
    return (
        float(w if w is not None else 1.0),
        float(params.get(keys[1]) or 0.0),
        float(params.get(keys[2]) or 0.0),
        float(params.get(keys[3]) or 0.0),
    )
