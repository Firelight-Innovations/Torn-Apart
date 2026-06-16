"""Tests for fire_engine.assets.asset_file — versioned .asset IO."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fire_engine.assets.asset_file import load_asset, save_asset
from fire_engine.assets.prefab import Prefab
from fire_engine.assets.types import AssetError, AssetVersionError
from fire_engine.scene import SceneObjectStore


def _sample_prefab() -> Prefab:
    store = SceneObjectStore()
    root = store.create("cube", name="Crate")
    store.create("empty", parent=root["id"], name="Pivot")
    return Prefab.from_store(store, root["id"])


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    prefab = _sample_prefab()
    path = tmp_path / "crate.asset"
    save_asset(path, prefab)
    loaded = load_asset(path)
    assert loaded.to_envelope() == prefab.to_envelope()


def test_save_is_byte_stable(tmp_path: Path) -> None:
    prefab = _sample_prefab()
    p1, p2 = tmp_path / "a.asset", tmp_path / "b.asset"
    save_asset(p1, prefab)
    save_asset(p2, load_asset(p1))  # save -> load -> save
    assert p1.read_bytes() == p2.read_bytes()


def test_save_uses_sorted_keys_and_trailing_newline(tmp_path: Path) -> None:
    path = tmp_path / "c.asset"
    save_asset(path, _sample_prefab())
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    top_keys = list(json.loads(text).keys())
    assert top_keys == sorted(top_keys)


def test_load_missing_file_raises_asset_error(tmp_path: Path) -> None:
    with pytest.raises(AssetError):
        load_asset(tmp_path / "nope.asset")


def test_load_newer_version_raises_version_error(tmp_path: Path) -> None:
    env = _sample_prefab().to_envelope()
    env["fire_asset"] = 999
    path = tmp_path / "future.asset"
    path.write_text(json.dumps(env), encoding="utf-8")
    with pytest.raises(AssetVersionError):
        load_asset(path)


def test_load_missing_version_field_raises(tmp_path: Path) -> None:
    env = _sample_prefab().to_envelope()
    del env["fire_asset"]
    path = tmp_path / "nover.asset"
    path.write_text(json.dumps(env), encoding="utf-8")
    with pytest.raises(AssetError):
        load_asset(path)


def test_load_non_object_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "arr.asset"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(AssetError):
        load_asset(path)


def test_load_garbage_raises_asset_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.asset"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(AssetError):
        load_asset(path)
