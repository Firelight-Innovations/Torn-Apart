"""SceneService — RPC round-trips over the daemon dispatcher (Phase E2).

Mirrors test_edit.py's pattern: build a Daemon with a live session and drive the
async service methods directly (no socket). Confirms the scene.* methods are
registered, mutate the session's store, and broadcast scene.changed.
"""
from __future__ import annotations

import asyncio

from fire_engine.core import load_config

from fire_editor import Daemon, EditorSession
from fire_editor._generated import Method, Notification


def _run(coro):
    return asyncio.run(coro)


def _daemon_with_world():
    daemon = Daemon()
    daemon.session = EditorSession(load_config())
    return daemon


def test_scene_methods_are_registered():
    daemon = Daemon()
    for m in (
        Method.SCENE_TREE,
        Method.SCENE_CREATE,
        Method.SCENE_RENAME,
        Method.SCENE_REPARENT,
        Method.SCENE_SET_TRANSFORM,
        Method.SCENE_DELETE,
    ):
        assert m in daemon.dispatcher._handlers, f"{m} not registered"


def test_create_and_tree_round_trip():
    daemon = _daemon_with_world()
    res = _run(daemon.scene.create({"kind": "cube", "name": "Crate", "x": 4.0, "z": 2.0}))
    assert res["ok"] and res["object"]["name"] == "Crate"
    assert res["object"]["position"] == [4.0, 0.0, 2.0]

    tree = _run(daemon.scene.tree({}))
    assert [o["id"] for o in tree["objects"]] == [res["object"]["id"]]


def test_reparent_and_delete_cascade():
    daemon = _daemon_with_world()
    parent = _run(daemon.scene.create({"kind": "empty"}))["object"]
    child = _run(daemon.scene.create({"kind": "sphere"}))["object"]
    _run(daemon.scene.reparent({"id": child["id"], "parent": parent["id"]}))

    tree = _run(daemon.scene.tree({}))
    assert next(o for o in tree["objects"] if o["id"] == child["id"])["parent"] == parent["id"]

    removed = _run(daemon.scene.delete({"id": parent["id"]}))["removed"]
    assert set(removed) == {parent["id"], child["id"]}
    assert _run(daemon.scene.tree({}))["objects"] == []


def test_scene_changed_is_broadcast(monkeypatch):
    daemon = _daemon_with_world()
    sent: list[tuple[str, dict]] = []

    async def fake_broadcast(method, params):
        sent.append((method, params))

    monkeypatch.setattr(daemon.server, "broadcast_notification", fake_broadcast)
    _run(daemon.scene.create({"kind": "light"}))
    assert sent and sent[-1][0] == Notification.SCENE_CHANGED
    assert len(sent[-1][1]["objects"]) == 1
