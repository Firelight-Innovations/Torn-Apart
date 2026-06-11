"""SceneService — authoring hierarchy RPC (EDITOR_PRD Phase E2).

Registers the ``scene.*`` methods that create, rename, reparent, transform and
delete placeable objects in the open world's :class:`SceneObjectStore`. Every
mutation broadcasts a ``scene.changed`` notification carrying the full object
list, so the sidebar tree and the 3D viewport stay in sync without each having
to diff. Pure control-channel JSON — no binary frames here.
"""
from __future__ import annotations

import logging

from .._generated import ErrorCode, Method, Notification
from ..rpc import RpcError
from ..scene_objects import SceneError
from ..session import EditorSession

log = logging.getLogger("fire_editor.scene")


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
        try:
            obj = store.create(
                str(params["kind"]),
                parent=None if parent is None else int(parent),
                name=params.get("name"),
                position=pos,
            )
        except SceneError as e:
            raise RpcError(ErrorCode.INVALID_PARAMS, str(e))
        await self._broadcast_changed(store)
        return {"ok": True, "object": obj}

    async def rename(self, params: dict) -> dict:
        store = self._require_session().scene
        try:
            obj = store.rename(int(params["id"]), str(params["name"]))
        except SceneError as e:
            raise RpcError(ErrorCode.INVALID_PARAMS, str(e))
        await self._broadcast_changed(store)
        return {"ok": True, "object": obj}

    async def reparent(self, params: dict) -> dict:
        store = self._require_session().scene
        parent = params.get("parent")
        try:
            obj = store.reparent(int(params["id"]), None if parent is None else int(parent))
        except SceneError as e:
            raise RpcError(ErrorCode.INVALID_PARAMS, str(e))
        await self._broadcast_changed(store)
        return {"ok": True, "object": obj}

    async def set_transform(self, params: dict) -> dict:
        store = self._require_session().scene
        position = _vec3(params, "px", "py", "pz")
        rotation = _quat(params, "rw", "rx", "ry", "rz")
        scale = _vec3(params, "sx", "sy", "sz")
        try:
            obj = store.set_transform(
                int(params["id"]), position=position, rotation=rotation, scale=scale
            )
        except SceneError as e:
            raise RpcError(ErrorCode.INVALID_PARAMS, str(e))
        await self._broadcast_changed(store)
        return {"ok": True, "object": obj}

    async def delete(self, params: dict) -> dict:
        store = self._require_session().scene
        try:
            removed = store.delete(int(params["id"]))
        except SceneError as e:
            raise RpcError(ErrorCode.INVALID_PARAMS, str(e))
        await self._broadcast_changed(store)
        return {"ok": True, "removed": removed}

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
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
