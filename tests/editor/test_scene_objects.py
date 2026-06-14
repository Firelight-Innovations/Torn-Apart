"""SceneObjectStore — hierarchy ops, determinism, and save round-trip."""

from __future__ import annotations

import pytest

from fire_editor.scene_objects import KINDS, SceneError, SceneObjectStore


def _build() -> tuple[SceneObjectStore, dict, dict, dict]:
    s = SceneObjectStore()
    root = s.create("empty", name="Root")
    cube = s.create("cube", parent=root["id"], name="Crate")
    child = s.create("sphere", parent=cube["id"], name="Ball")
    return s, root, cube, child


def test_create_assigns_monotonic_ids_and_defaults():
    s = SceneObjectStore()
    a = s.create("cube")
    b = s.create("sphere")
    assert (a["id"], b["id"]) == (1, 2)
    assert a["name"] == "Cube" and b["name"] == "Sphere"
    assert a["parent"] is None
    assert a["position"] == [0.0, 0.0, 0.0]
    assert a["rotation"] == [1.0, 0.0, 0.0, 0.0]
    assert a["scale"] == [1.0, 1.0, 1.0]


def test_create_rejects_unknown_kind_and_missing_parent():
    s = SceneObjectStore()
    with pytest.raises(SceneError):
        s.create("banana")
    with pytest.raises(SceneError):
        s.create("cube", parent=999)


def test_tree_is_depth_first_roots_first():
    s, root, cube, child = _build()
    other = s.create("light", name="Sun")
    ids = [o["id"] for o in s.tree()]
    # Root subtree (root -> cube -> child) is emitted before the later root.
    assert ids == [root["id"], cube["id"], child["id"], other["id"]]


def test_reparent_rejects_self_and_cycles():
    s, root, cube, child = _build()
    with pytest.raises(SceneError):
        s.reparent(root["id"], root["id"])  # self-parent
    with pytest.raises(SceneError):
        s.reparent(root["id"], child["id"])  # parent under own descendant
    # Promoting to a root is allowed.
    assert s.reparent(child["id"], None)["parent"] is None


def test_delete_cascades_to_descendants():
    s, root, cube, child = _build()
    removed = s.delete(cube["id"])
    assert set(removed) == {cube["id"], child["id"]}
    assert len(s) == 1  # only root remains
    assert [o["id"] for o in s.tree()] == [root["id"]]


def test_set_transform_partial_update_leaves_other_channels():
    s = SceneObjectStore()
    o = s.create("cube")
    s.set_transform(o["id"], position=(1.0, 2.0, 3.0))
    updated = s.get(o["id"]).to_dict()
    assert updated["position"] == [1.0, 2.0, 3.0]
    assert updated["rotation"] == [1.0, 0.0, 0.0, 0.0]  # untouched


def test_determinism_same_ops_same_ids():
    def run() -> list[dict]:
        s = SceneObjectStore()
        a = s.create("empty")
        s.create("cube", parent=a["id"])
        s.create("sphere")
        return s.tree()

    assert run() == run()


def test_save_round_trip():
    s, *_ = _build()
    s.set_transform(2, position=(4.0, -1.0, 2.5))
    delta = s.get_delta()

    restored = SceneObjectStore()
    restored.apply_delta(delta)
    assert restored.tree() == s.tree()
    # Counter continues past loaded ids (no id reuse).
    assert restored.create("cube")["id"] == s.create("cube")["id"]


def test_empty_store_saves_nothing():
    assert SceneObjectStore().get_delta() == {}


def test_all_kinds_creatable():
    s = SceneObjectStore()
    for k in KINDS:
        assert s.create(k)["kind"] == k
