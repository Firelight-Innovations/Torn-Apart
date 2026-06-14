"""Scene-op undo/redo tests (EDITOR_PRD scene undo).

The daemon keeps ONE chronological undo stack for terrain brushes and scene
edits. These tests cover: set_transform→undo restores the prior TRS, create→undo
removes the object, rapid gizmo-style transforms coalesce into a single undo
step, and terrain + scene edits interleave and undo in reverse chronological
order.
"""

from __future__ import annotations

import asyncio

import numpy as np

from fire_engine.core import load_config

from fire_editor import Daemon, EditorSession

BRUSH = {"shape": "sphere", "x": 0.0, "y": 0.0, "z": 7.5, "mode": "remove", "radius": 2.0}
SURFACE = (0, 0, 0)


def _run(coro):
    return asyncio.run(coro)


def _daemon():
    daemon = Daemon()
    daemon.session = EditorSession(load_config())
    return daemon


def _obj(store, oid):
    return next(o for o in store.tree() if o["id"] == oid)


class TestSceneUndo:
    def test_set_transform_undo_restores(self):
        async def scenario():
            d = _daemon()
            store = d.session.scene
            created = await d.scene.create({"kind": "cube", "x": 1, "y": 2, "z": 3})
            oid = created["object"]["id"]
            await d.scene.set_transform({"id": oid, "px": 9, "py": 9, "pz": 9})
            assert _obj(store, oid)["position"] == [9.0, 9.0, 9.0]

            res = await d.chunks.undo({})
            assert res["ok"] and res["label"] == f"transform {oid}"
            assert _obj(store, oid)["position"] == [1.0, 2.0, 3.0]

            res = await d.chunks.redo({})
            assert res["ok"]
            assert _obj(store, oid)["position"] == [9.0, 9.0, 9.0]

        _run(scenario())

    def test_create_undo_removes(self):
        async def scenario():
            d = _daemon()
            store = d.session.scene
            created = await d.scene.create({"kind": "sphere"})
            oid = created["object"]["id"]
            assert any(o["id"] == oid for o in store.tree())

            await d.chunks.undo({})
            assert not any(o["id"] == oid for o in store.tree())

            await d.chunks.redo({})
            assert any(o["id"] == oid for o in store.tree())

        _run(scenario())

    def test_rapid_transforms_coalesce(self):
        async def scenario():
            d = _daemon()
            store = d.session.scene
            created = await d.scene.create({"kind": "cube"})
            oid = created["object"]["id"]
            # Simulate a throttled gizmo drag: many set_transform calls in a row.
            for i in range(1, 6):
                await d.scene.set_transform({"id": oid, "px": float(i)})
            assert _obj(store, oid)["position"][0] == 5.0

            # One undo should revert the WHOLE drag back to the create state (0),
            # not just the last increment.
            res = await d.chunks.undo({})
            assert res["label"] == f"transform {oid}"
            assert _obj(store, oid)["position"][0] == 0.0
            # And the create is still one more undo away.
            assert d.chunks.history.can_undo

        _run(scenario())

    def test_terrain_and_scene_interleave(self):
        async def scenario():
            d = _daemon()
            s = d.session
            store = s.scene
            baseline = s.cm.get_or_create(SURFACE).materials.copy()

            await d.chunks.brush(BRUSH)  # terrain edit
            after_brush = s.cm.chunks[SURFACE].materials.copy()
            created = await d.scene.create({"kind": "light"})  # scene edit
            oid = created["object"]["id"]

            # Undo #1 reverts the most recent (scene create).
            await d.chunks.undo({})
            assert not any(o["id"] == oid for o in store.tree())
            assert np.array_equal(s.cm.chunks[SURFACE].materials, after_brush)

            # Undo #2 reverts the terrain brush.
            await d.chunks.undo({})
            assert np.array_equal(s.cm.chunks[SURFACE].materials, baseline)
            assert not d.chunks.history.can_undo

        _run(scenario())
