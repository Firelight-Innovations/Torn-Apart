"""Per-object component model — catalog, store mutators, migration, runtime.

Covers the Unity-style component stack added on top of the authoring scene
graph: ``kind`` seeds default components, the inspector edits them via
add/remove/set, old (pre-component) saves migrate forward, and a Light
component's authored params reach the game's visual factory.
"""

from __future__ import annotations

import asyncio

import pytest

from fire_engine.scene.components import (
    COMPONENT_CATALOG,
    catalog_payload,
    coerce_params,
    default_components_for_kind,
    make_component,
)
from fire_engine.scene.objects import SceneError, SceneObject, SceneObjectStore

from fire_editor import Daemon, EditorSession
from fire_engine.core import load_config


def _run(coro):
    return asyncio.run(coro)


def _daemon():
    d = Daemon()
    d.session = EditorSession(load_config())
    return d


def _obj(store, oid):
    return next(o for o in store.tree() if o["id"] == oid)


# --------------------------------------------------------------------------- #
# Catalog (pure data)
# --------------------------------------------------------------------------- #
class TestCatalog:
    def test_payload_lists_builtins_with_fields(self):
        payload = catalog_payload()
        by_type = {t["type"]: t for t in payload["types"]}
        assert set(by_type) == {"Mesh", "Light", "SpawnPoint"}
        light = by_type["Light"]
        names = [f["name"] for f in light["fields"]]
        assert names == ["color", "intensity", "radius"]
        assert by_type["Mesh"]["fields"][0]["choices"] == ["cube", "sphere"]
        assert by_type["SpawnPoint"]["fields"] == []

    def test_payload_is_deterministic(self):
        assert catalog_payload() == catalog_payload()

    def test_default_components_per_kind(self):
        assert [c["type"] for c in default_components_for_kind("cube")] == ["Mesh"]
        assert default_components_for_kind("cube")[0]["params"]["primitive"] == "cube"
        assert default_components_for_kind("sphere")[0]["params"]["primitive"] == "sphere"
        assert [c["type"] for c in default_components_for_kind("light")] == ["Light"]
        assert [c["type"] for c in default_components_for_kind("spawn")] == ["SpawnPoint"]
        assert default_components_for_kind("empty") == []

    def test_coerce_clamps_and_drops_unknown_keys(self):
        out = coerce_params("Light", {"intensity": 999.0, "bogus": 1, "radius": -5})
        assert out["intensity"] == COMPONENT_CATALOG["Light"].fields[1].max
        assert out["radius"] == 0.0  # clamped to min
        assert "bogus" not in out

    def test_coerce_snaps_enum_to_valid_choice(self):
        assert coerce_params("Mesh", {"primitive": "pyramid"})["primitive"] == "cube"
        assert coerce_params("Mesh", {"primitive": "sphere"})["primitive"] == "sphere"


# --------------------------------------------------------------------------- #
# Store mutators
# --------------------------------------------------------------------------- #
class TestStoreComponents:
    def test_create_seeds_components_from_kind(self):
        s = SceneObjectStore()
        assert [c["type"] for c in s.create("light")["components"]] == ["Light"]
        assert s.create("empty")["components"] == []

    def test_add_component_to_empty(self):
        s = SceneObjectStore()
        e = s.create("empty")
        out = s.add_component(e["id"], "Light")
        assert [c["type"] for c in out["components"]] == ["Light"]
        assert out["components"][0]["params"]["intensity"] == 8.0

    def test_add_component_rejects_unknown_and_duplicate_singleton(self):
        s = SceneObjectStore()
        c = s.create("cube")  # already has a Mesh
        with pytest.raises(SceneError):
            s.add_component(c["id"], "Banana")
        with pytest.raises(SceneError):
            s.add_component(c["id"], "Mesh")  # singleton

    def test_remove_component_by_index(self):
        s = SceneObjectStore()
        c = s.create("cube")
        out = s.remove_component(c["id"], 0)
        assert out["components"] == []
        with pytest.raises(SceneError):
            s.remove_component(c["id"], 0)  # now out of range

    def test_set_component_merges_and_coerces(self):
        s = SceneObjectStore()
        lt = s.create("light")
        out = s.set_component(lt["id"], 0, params={"intensity": 999, "color": [0.1, 0.2, 0.3]})
        p = out["components"][0]["params"]
        assert p["intensity"] == 64.0  # clamped
        assert p["color"] == [0.1, 0.2, 0.3]
        assert p["radius"] == 16.0  # untouched

    def test_set_component_toggles_enabled(self):
        s = SceneObjectStore()
        lt = s.create("light")
        out = s.set_component(lt["id"], 0, enabled=False)
        assert out["components"][0]["enabled"] is False

    def test_set_component_bad_index_raises(self):
        s = SceneObjectStore()
        lt = s.create("light")
        with pytest.raises(SceneError):
            s.set_component(lt["id"], 5, params={"intensity": 1})


# --------------------------------------------------------------------------- #
# Migration of pre-component data
# --------------------------------------------------------------------------- #
class TestMigration:
    def test_from_dict_synthesises_components_when_absent(self):
        # A pre-component object dict (no "components" key).
        old = {
            "id": 7,
            "name": "Old Light",
            "kind": "light",
            "parent": None,
            "position": [1, 2, 3],
            "rotation": [1, 0, 0, 0],
            "scale": [1, 1, 1],
        }
        obj = SceneObject.from_dict(old)
        assert [c["type"] for c in obj.components] == ["Light"]

    def test_from_dict_keeps_explicit_empty_components(self):
        # An explicit empty list is NOT the same as "absent" — keep it empty.
        d = {
            "id": 1,
            "name": "Bare",
            "kind": "cube",
            "parent": None,
            "position": [0, 0, 0],
            "rotation": [1, 0, 0, 0],
            "scale": [1, 1, 1],
            "components": [],
        }
        assert SceneObject.from_dict(d).components == []

    def test_apply_delta_migrates_old_save(self):
        old_delta = {
            "objects": [
                {
                    "id": 1,
                    "name": "Crate",
                    "kind": "cube",
                    "parent": None,
                    "position": [0, 0, 0],
                    "rotation": [1, 0, 0, 0],
                    "scale": [1, 1, 1],
                }
            ],
            "next_id": 2,
        }
        s = SceneObjectStore()
        s.apply_delta(old_delta)
        assert [c["type"] for c in _obj(s, 1)["components"]] == ["Mesh"]

    def test_components_survive_save_round_trip(self):
        s = SceneObjectStore()
        lt = s.create("light")
        s.set_component(lt["id"], 0, params={"intensity": 20.0})
        restored = SceneObjectStore()
        restored.apply_delta(s.get_delta())
        assert restored.tree() == s.tree()
        assert _obj(restored, lt["id"])["components"][0]["params"]["intensity"] == 20.0


# --------------------------------------------------------------------------- #
# Daemon service + undo
# --------------------------------------------------------------------------- #
class TestComponentService:
    def test_catalog_rpc_without_open_world(self):
        async def scenario():
            d = Daemon()  # no session — catalog is static
            res = await d.scene.catalog({})
            assert {t["type"] for t in res["types"]} == {"Mesh", "Light", "SpawnPoint"}

        _run(scenario())

    def test_add_then_undo(self):
        async def scenario():
            d = _daemon()
            store = d.session.scene
            e = (await d.scene.create({"kind": "empty"}))["object"]
            await d.scene.add_component({"id": e["id"], "type": "Light"})
            assert [c["type"] for c in _obj(store, e["id"])["components"]] == ["Light"]

            res = await d.chunks.undo({})
            assert res["ok"] and res["label"] == f"add component {e['id']}"
            assert _obj(store, e["id"])["components"] == []

        _run(scenario())

    def test_set_component_coalesces(self):
        async def scenario():
            d = _daemon()
            store = d.session.scene
            lt = (await d.scene.create({"kind": "light"}))["object"]
            for v in (4.0, 8.0, 12.0):
                await d.scene.set_component(
                    {"id": lt["id"], "index": 0, "params": {"intensity": v}}
                )
            assert _obj(store, lt["id"])["components"][0]["params"]["intensity"] == 12.0

            # One undo reverts the whole slider drag to the create default.
            res = await d.chunks.undo({})
            assert res["label"] == f"component {lt['id']}.0"
            assert _obj(store, lt["id"])["components"][0]["params"]["intensity"] == 8.0
            # The create is still one undo away (not coalesced into it).
            assert d.chunks.history.can_undo

        _run(scenario())

    def test_set_component_invalid_index_is_rpc_error(self):
        from fire_editor.rpc import RpcError

        async def scenario():
            d = _daemon()
            lt = (await d.scene.create({"kind": "light"}))["object"]
            with pytest.raises(RpcError):
                await d.scene.set_component({"id": lt["id"], "index": 9, "params": {}})

        _run(scenario())


# --------------------------------------------------------------------------- #
# Runtime: authored params reach the visual factory
# --------------------------------------------------------------------------- #
class _RecordingFactory:
    def __init__(self):
        self.attached: list[dict] = []
        self.teardowns = 0

    def attach(self, go, kind, obj):
        self.attached.append(obj)

    def teardown(self):
        self.teardowns += 1


class TestRuntimeComponents:
    def test_light_params_reach_factory(self):
        from fire_engine.scene import SceneRuntime
        from fire_engine.render.registry import ComponentRegistry

        ComponentRegistry.clear()
        store = SceneObjectStore()
        lt = store.create("light")
        store.set_component(lt["id"], 0, params={"intensity": 25.0, "color": [1.0, 0.0, 0.0]})

        factory = _RecordingFactory()
        runtime = SceneRuntime(visual_factory=factory)
        runtime.apply_delta(store.get_delta())

        attached = factory.attached[0]
        light_comp = next(c for c in attached["components"] if c["type"] == "Light")
        assert light_comp["params"]["intensity"] == 25.0
        assert light_comp["params"]["color"] == [1.0, 0.0, 0.0]

    def test_runtime_migrates_old_delta(self):
        from fire_engine.scene import SceneRuntime
        from fire_engine.render.registry import ComponentRegistry

        ComponentRegistry.clear()
        old_delta = {
            "objects": [
                {
                    "id": 1,
                    "name": "Crate",
                    "kind": "cube",
                    "parent": None,
                    "position": [0, 0, 0],
                    "rotation": [1, 0, 0, 0],
                    "scale": [1, 1, 1],
                }
            ],
            "next_id": 2,
        }
        factory = _RecordingFactory()
        runtime = SceneRuntime(visual_factory=factory)
        runtime.apply_delta(old_delta)
        assert [c["type"] for c in factory.attached[0]["components"]] == ["Mesh"]
