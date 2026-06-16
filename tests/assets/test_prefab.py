"""Tests for fire_engine.assets.prefab — Prefab model + SceneObjectStore interop."""

from __future__ import annotations

from typing import Any

import pytest

from fire_engine.assets.enums import AssetType
from fire_engine.assets.prefab import Prefab
from fire_engine.assets.types import AssetError, AssetSource, Transform
from fire_engine.scene import SceneObjectStore


def _sample_store() -> tuple[SceneObjectStore, int]:
    """A cube root with an empty child carrying a Light; returns (store, root_id)."""
    store = SceneObjectStore()
    root = store.create("cube", name="Crate")
    child = store.create("empty", parent=root["id"], name="Pivot")
    store.set_transform(child["id"], position=(1.0, 2.0, 3.0))
    store.add_component(child["id"], "Light")
    return store, root["id"]


def test_from_store_snapshots_subtree_with_local_ids() -> None:
    store, root_id = _sample_store()
    prefab = Prefab.from_store(store, root_id)

    assert prefab.root == 1
    assert prefab.asset_type == "prefab"
    assert len(prefab.objects) == 2

    root_obj = next(o for o in prefab.objects if o["id"] == 1)
    child_obj = next(o for o in prefab.objects if o["id"] == 2)
    assert root_obj["kind"] == "cube"
    assert root_obj["parent"] is None
    assert child_obj["parent"] == 1
    assert child_obj["position"] == [1.0, 2.0, 3.0]
    assert child_obj["components"][0]["type"] == "Light"


def test_from_store_missing_root_raises() -> None:
    store = SceneObjectStore()
    with pytest.raises(AssetError):
        Prefab.from_store(store, 999)


def test_instantiate_into_empty_store() -> None:
    store, root_id = _sample_store()
    prefab = Prefab.from_store(store, root_id)

    dst = SceneObjectStore()
    new_root = prefab.instantiate_into(dst, at_transform=Transform(position=(5.0, 0.0, 0.0)))

    assert len(dst) == 2
    root_obj = dst.get(new_root)
    assert root_obj.kind == "cube"
    assert root_obj.parent is None
    assert root_obj.position == (5.0, 0.0, 0.0)  # at_transform replaced the root TRS

    kids = [o for o in dst.tree() if o["parent"] == new_root]
    assert len(kids) == 1
    assert kids[0]["position"] == [1.0, 2.0, 3.0]
    assert kids[0]["components"][0]["type"] == "Light"


def test_instantiate_remaps_ids_into_occupied_store() -> None:
    store, root_id = _sample_store()
    prefab = Prefab.from_store(store, root_id)

    dst = SceneObjectStore()
    existing = dst.create("sphere", name="Existing")
    first = prefab.instantiate_into(dst)
    second = prefab.instantiate_into(dst)

    # Three independent objects' worth, no id collisions.
    assert first != second != existing["id"]
    assert len({existing["id"], first, second}) == 3
    assert len(dst) == 5  # 1 existing + 2 prefab objects x2
    assert dst.get(existing["id"]).name == "Existing"  # untouched


def test_instantiate_under_explicit_parent() -> None:
    store, root_id = _sample_store()
    prefab = Prefab.from_store(store, root_id)

    dst = SceneObjectStore()
    anchor = dst.create("empty", name="Anchor")
    new_root = prefab.instantiate_into(dst, parent=anchor["id"])
    assert dst.get(new_root).parent == anchor["id"]


def test_instantiate_under_missing_parent_raises() -> None:
    store, root_id = _sample_store()
    prefab = Prefab.from_store(store, root_id)
    with pytest.raises(AssetError):
        prefab.instantiate_into(SceneObjectStore(), parent=999)


def test_envelope_round_trip_is_stable() -> None:
    store, root_id = _sample_store()
    src = AssetSource(def_name="cube_def", params={"a": 1}, seed=7)
    prefab = Prefab.from_store(store, root_id, asset_type=AssetType.BUILDING, source=src)

    env = prefab.to_envelope()
    again = Prefab.from_envelope(env)
    assert again.to_envelope() == env
    assert again.asset_type == "building"
    assert again.source == src
    assert env["guid"] is None  # reserved, never generated in v1


def test_from_envelope_rejects_missing_root() -> None:
    env: dict[str, Any] = {
        "fire_asset": 1,
        "asset_type": "prefab",
        "root": 5,
        "objects": [
            {
                "id": 1,
                "name": "x",
                "kind": "empty",
                "parent": None,
                "position": [0, 0, 0],
                "rotation": [1, 0, 0, 0],
                "scale": [1, 1, 1],
                "components": [],
            }
        ],
        "blobs": {},
    }
    with pytest.raises(AssetError):
        Prefab.from_envelope(env)


def test_arbitrary_kinds_and_components_survive_verbatim() -> None:
    # The format is GENERIC: a "building" kind + opaque "Building" component
    # (neither registered in the scene catalog) must round-trip and instantiate
    # without coercion or stripping — proving assets/ does not depend on the
    # scene catalog or buildings/.
    env: dict[str, Any] = {
        "fire_asset": 1,
        "asset_type": "building",
        "root": 1,
        "objects": [
            {
                "id": 1,
                "name": "Farmhouse",
                "kind": "building",
                "parent": None,
                "position": [0, 0, 0],
                "rotation": [1, 0, 0, 0],
                "scale": [1, 1, 1],
                "components": [
                    {
                        "type": "Building",
                        "enabled": True,
                        "params": {"storeys": 2, "walls": [[0, 0], [4, 0]]},
                    }
                ],
            }
        ],
        "blobs": {},
    }
    prefab = Prefab.from_envelope(env)
    dst = SceneObjectStore()
    rid = prefab.instantiate_into(dst)
    obj = dst.get(rid)
    assert obj.kind == "building"
    assert obj.components == [
        {"type": "Building", "enabled": True, "params": {"storeys": 2, "walls": [[0, 0], [4, 0]]}}
    ]
