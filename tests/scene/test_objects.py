"""Mirror tests for fire_engine/scene/objects.py.

Covers: KINDS constant, SceneObjectStore hierarchy operations (create, rename,
reparent, set_transform, delete, clear), component mutators (add, remove, set),
tree DFS ordering, Saveable protocol (get_delta / apply_delta round-trip), and
determinism (same sequence of edits → same ids and output).

Categories: CORRECTNESS, DETERMINISM, ROUND-TRIP.
"""

from __future__ import annotations

import pytest

from fire_engine.scene.objects import KINDS, SceneError, SceneObjectStore

# ---------------------------------------------------------------------------
# KINDS
# ---------------------------------------------------------------------------


class TestKinds:
    def test_expected_kinds_present(self):
        assert {"empty", "cube", "sphere", "light", "spawn"} <= KINDS

    def test_kinds_is_frozenset(self):
        assert isinstance(KINDS, frozenset)


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


class TestCreate:
    def test_monotonic_ids_start_at_one(self):
        s = SceneObjectStore()
        a = s.create("cube")
        b = s.create("sphere")
        assert a["id"] == 1
        assert b["id"] == 2

    def test_default_names_by_kind(self):
        s = SceneObjectStore()
        assert s.create("empty")["name"] == "GameObject"
        assert s.create("cube")["name"] == "Cube"
        assert s.create("sphere")["name"] == "Sphere"
        assert s.create("light")["name"] == "Light"
        assert s.create("spawn")["name"] == "Spawn Point"

    def test_explicit_name_used(self):
        s = SceneObjectStore()
        assert s.create("cube", name="Crate")["name"] == "Crate"

    def test_default_transform(self):
        s = SceneObjectStore()
        obj = s.create("cube")
        assert obj["position"] == [0.0, 0.0, 0.0]
        assert obj["rotation"] == [1.0, 0.0, 0.0, 0.0]
        assert obj["scale"] == [1.0, 1.0, 1.0]

    def test_position_kwarg(self):
        s = SceneObjectStore()
        obj = s.create("cube", position=(3.0, 4.0, 5.0))
        assert obj["position"] == [3.0, 4.0, 5.0]

    def test_parent_stored(self):
        s = SceneObjectStore()
        root = s.create("empty")
        child = s.create("cube", parent=root["id"])
        assert child["parent"] == root["id"]

    def test_unknown_kind_raises(self):
        s = SceneObjectStore()
        with pytest.raises(SceneError):
            s.create("banana")

    def test_missing_parent_raises(self):
        s = SceneObjectStore()
        with pytest.raises(SceneError):
            s.create("cube", parent=999)

    def test_all_kinds_creatable(self):
        s = SceneObjectStore()
        for k in KINDS:
            d = s.create(k)
            assert d["kind"] == k

    def test_returns_dict_form(self):
        s = SceneObjectStore()
        obj = s.create("light")
        assert isinstance(obj, dict)
        assert "id" in obj and "name" in obj and "kind" in obj


# ---------------------------------------------------------------------------
# len
# ---------------------------------------------------------------------------


class TestLen:
    def test_empty_store(self):
        assert len(SceneObjectStore()) == 0

    def test_grows_with_creates(self):
        s = SceneObjectStore()
        s.create("cube")
        s.create("sphere")
        assert len(s) == 2


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


class TestGet:
    def test_get_known_id(self):
        s = SceneObjectStore()
        obj = s.create("cube")
        retrieved = s.get(obj["id"])
        assert retrieved.id == obj["id"]

    def test_get_unknown_id_raises(self):
        s = SceneObjectStore()
        with pytest.raises(SceneError):
            s.get(9999)


# ---------------------------------------------------------------------------
# tree ordering
# ---------------------------------------------------------------------------


class TestTree:
    def test_empty_store_returns_empty_list(self):
        assert SceneObjectStore().tree() == []

    def test_depth_first_roots_first(self):
        s = SceneObjectStore()
        root = s.create("empty", name="Root")
        cube = s.create("cube", parent=root["id"])
        child = s.create("sphere", parent=cube["id"])
        other = s.create("light")
        ids = [o["id"] for o in s.tree()]
        assert ids == [root["id"], cube["id"], child["id"], other["id"]]

    def test_tree_objects_are_dicts(self):
        s = SceneObjectStore()
        s.create("cube")
        for item in s.tree():
            assert isinstance(item, dict)


# ---------------------------------------------------------------------------
# rename
# ---------------------------------------------------------------------------


class TestRename:
    def test_name_updated(self):
        s = SceneObjectStore()
        obj = s.create("cube")
        out = s.rename(obj["id"], "Renamed")
        assert out["name"] == "Renamed"
        assert s.get(obj["id"]).name == "Renamed"


# ---------------------------------------------------------------------------
# reparent
# ---------------------------------------------------------------------------


class TestReparent:
    def test_promote_to_root(self):
        s = SceneObjectStore()
        root = s.create("empty")
        child = s.create("cube", parent=root["id"])
        out = s.reparent(child["id"], None)
        assert out["parent"] is None

    def test_reparent_to_new_parent(self):
        s = SceneObjectStore()
        p1 = s.create("empty", name="P1")
        p2 = s.create("empty", name="P2")
        child = s.create("cube", parent=p1["id"])
        out = s.reparent(child["id"], p2["id"])
        assert out["parent"] == p2["id"]

    def test_self_parent_raises(self):
        s = SceneObjectStore()
        obj = s.create("cube")
        with pytest.raises(SceneError):
            s.reparent(obj["id"], obj["id"])

    def test_cycle_raises(self):
        s = SceneObjectStore()
        root = s.create("empty")
        child = s.create("cube", parent=root["id"])
        with pytest.raises(SceneError):
            s.reparent(root["id"], child["id"])

    def test_missing_parent_raises(self):
        s = SceneObjectStore()
        obj = s.create("cube")
        with pytest.raises(SceneError):
            s.reparent(obj["id"], 9999)


# ---------------------------------------------------------------------------
# set_transform
# ---------------------------------------------------------------------------


class TestSetTransform:
    def test_position_partial_update(self):
        s = SceneObjectStore()
        obj = s.create("cube")
        out = s.set_transform(obj["id"], position=(1.0, 2.0, 3.0))
        assert out["position"] == [1.0, 2.0, 3.0]
        assert out["rotation"] == [1.0, 0.0, 0.0, 0.0]  # unchanged
        assert out["scale"] == [1.0, 1.0, 1.0]  # unchanged

    def test_rotation_partial_update(self):
        s = SceneObjectStore()
        obj = s.create("cube")
        out = s.set_transform(obj["id"], rotation=(0.707, 0.0, 0.707, 0.0))
        assert out["rotation"] == [0.707, 0.0, 0.707, 0.0]
        assert out["position"] == [0.0, 0.0, 0.0]  # unchanged

    def test_scale_partial_update(self):
        s = SceneObjectStore()
        obj = s.create("cube")
        out = s.set_transform(obj["id"], scale=(2.0, 1.0, 3.0))
        assert out["scale"] == [2.0, 1.0, 3.0]

    def test_all_channels_at_once(self):
        s = SceneObjectStore()
        obj = s.create("cube")
        s.set_transform(
            obj["id"],
            position=(1.0, 2.0, 3.0),
            rotation=(0.707, 0.0, 0.707, 0.0),
            scale=(2.0, 2.0, 2.0),
        )
        stored = s.get(obj["id"])
        assert stored.position == (1.0, 2.0, 3.0)
        assert stored.scale == (2.0, 2.0, 2.0)

    def test_missing_id_raises(self):
        s = SceneObjectStore()
        with pytest.raises(SceneError):
            s.set_transform(9999, position=(0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# Component mutators
# ---------------------------------------------------------------------------


class TestAddComponent:
    def test_add_to_empty(self):
        s = SceneObjectStore()
        obj = s.create("empty")
        out = s.add_component(obj["id"], "Light")
        assert [c["type"] for c in out["components"]] == ["Light"]

    def test_unknown_type_raises(self):
        s = SceneObjectStore()
        obj = s.create("empty")
        with pytest.raises(SceneError):
            s.add_component(obj["id"], "Banana")

    def test_singleton_duplicate_raises(self):
        s = SceneObjectStore()
        obj = s.create("cube")  # already has Mesh
        with pytest.raises(SceneError):
            s.add_component(obj["id"], "Mesh")


class TestRemoveComponent:
    def test_remove_by_index(self):
        s = SceneObjectStore()
        obj = s.create("cube")
        out = s.remove_component(obj["id"], 0)
        assert out["components"] == []

    def test_out_of_range_raises(self):
        s = SceneObjectStore()
        obj = s.create("cube")
        with pytest.raises(SceneError):
            s.remove_component(obj["id"], 5)

    def test_negative_index_raises(self):
        s = SceneObjectStore()
        obj = s.create("cube")
        with pytest.raises(SceneError):
            s.remove_component(obj["id"], -1)


class TestSetComponent:
    def test_merge_params(self):
        s = SceneObjectStore()
        obj = s.create("light")
        out = s.set_component(obj["id"], 0, params={"intensity": 20.0})
        assert out["components"][0]["params"]["intensity"] == 20.0
        assert out["components"][0]["params"]["radius"] == 16.0  # untouched

    def test_clamps_params(self):
        s = SceneObjectStore()
        obj = s.create("light")
        out = s.set_component(obj["id"], 0, params={"intensity": 9999.0})
        assert out["components"][0]["params"]["intensity"] == 64.0

    def test_toggle_enabled(self):
        s = SceneObjectStore()
        obj = s.create("light")
        out = s.set_component(obj["id"], 0, enabled=False)
        assert out["components"][0]["enabled"] is False

    def test_bad_index_raises(self):
        s = SceneObjectStore()
        obj = s.create("light")
        with pytest.raises(SceneError):
            s.set_component(obj["id"], 99, params={})


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_removes_object(self):
        s = SceneObjectStore()
        obj = s.create("cube")
        removed = s.delete(obj["id"])
        assert obj["id"] in removed
        assert len(s) == 0

    def test_delete_cascades_to_descendants(self):
        s = SceneObjectStore()
        root = s.create("empty")
        child = s.create("cube", parent=root["id"])
        grandchild = s.create("sphere", parent=child["id"])
        removed = s.delete(root["id"])
        assert set(removed) == {root["id"], child["id"], grandchild["id"]}
        assert len(s) == 0

    def test_delete_sibling_unaffected(self):
        s = SceneObjectStore()
        a = s.create("cube")
        b = s.create("sphere")
        s.delete(a["id"])
        assert len(s) == 1
        assert s.get(b["id"]).id == b["id"]

    def test_delete_unknown_id_raises(self):
        s = SceneObjectStore()
        with pytest.raises(SceneError):
            s.delete(9999)


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_empties_store(self):
        s = SceneObjectStore()
        s.create("cube")
        s.create("sphere")
        s.clear()
        assert len(s) == 0
        assert s.tree() == []

    def test_clear_resets_id_counter(self):
        s = SceneObjectStore()
        s.create("cube")
        s.create("sphere")
        s.clear()
        assert s.create("cube")["id"] == 1


# ---------------------------------------------------------------------------
# Saveable protocol — get_delta / apply_delta
# ---------------------------------------------------------------------------


class TestSaveableProtocol:
    def test_empty_store_delta_is_empty_dict(self):
        assert SceneObjectStore().get_delta() == {}

    def test_round_trip_preserves_tree(self):
        s = SceneObjectStore()
        root = s.create("empty", name="Root")
        cube = s.create("cube", parent=root["id"], name="Crate", position=(4.0, 0.0, 2.0))
        s.create("sphere", parent=cube["id"], name="Ball")
        delta = s.get_delta()

        restored = SceneObjectStore()
        restored.apply_delta(delta)
        assert restored.tree() == s.tree()

    def test_apply_delta_restores_transforms(self):
        s = SceneObjectStore()
        obj = s.create("cube")
        s.set_transform(obj["id"], position=(5.0, 6.0, 7.0), scale=(2.0, 2.0, 2.0))
        delta = s.get_delta()

        r = SceneObjectStore()
        r.apply_delta(delta)
        loaded = r.get(obj["id"])
        assert loaded.position == (5.0, 6.0, 7.0)
        assert loaded.scale == (2.0, 2.0, 2.0)

    def test_apply_delta_continues_id_counter(self):
        """After apply_delta, new ids must not collide with loaded ones."""
        s = SceneObjectStore()
        for _ in range(3):
            s.create("cube")
        delta = s.get_delta()

        r = SceneObjectStore()
        r.apply_delta(delta)
        new_id = r.create("sphere")["id"]
        assert new_id == 4  # monotonic past the 3 loaded ids

    def test_apply_delta_empty_dict_gives_empty_store(self):
        s = SceneObjectStore()
        s.create("cube")
        s.apply_delta({})
        assert len(s) == 0

    def test_round_trip_components(self):
        s = SceneObjectStore()
        obj = s.create("light")
        s.set_component(obj["id"], 0, params={"intensity": 25.0})
        delta = s.get_delta()

        r = SceneObjectStore()
        r.apply_delta(delta)
        comp = r.get(obj["id"]).components[0]
        assert comp["params"]["intensity"] == 25.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_ops_same_ids_and_tree(self):
        def run():
            s = SceneObjectStore()
            a = s.create("empty", name="Root")
            s.create("cube", parent=a["id"], position=(1.0, 2.0, 3.0))
            s.create("sphere")
            return s.tree()

        assert run() == run()

    def test_get_delta_same_each_time(self):
        s = SceneObjectStore()
        s.create("cube")
        s.create("light")
        assert s.get_delta() == s.get_delta()
