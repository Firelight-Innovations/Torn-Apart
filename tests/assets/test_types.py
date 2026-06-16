"""Tests for fire_engine.assets.types."""

from __future__ import annotations

from fire_engine.assets.types import (
    AssetError,
    AssetSource,
    AssetVersionError,
    Transform,
)


def test_transform_defaults_are_identity() -> None:
    t = Transform()
    assert t.position == (0.0, 0.0, 0.0)
    assert t.rotation == (1.0, 0.0, 0.0, 0.0)
    assert t.scale == (1.0, 1.0, 1.0)


def test_asset_source_round_trip() -> None:
    src = AssetSource(def_name="building_farmhouse", params={"storeys": 2}, seed=1337)
    d = src.to_dict()
    assert d == {"def": "building_farmhouse", "params": {"storeys": 2}, "seed": 1337}
    assert AssetSource.from_dict(d) == src


def test_asset_source_seed_may_be_none() -> None:
    src = AssetSource(def_name="cube")
    assert src.to_dict() == {"def": "cube", "params": {}, "seed": None}
    assert AssetSource.from_dict(src.to_dict()) == src


def test_exception_hierarchy() -> None:
    # AssetVersionError is a kind of AssetError is a kind of ValueError.
    assert issubclass(AssetVersionError, AssetError)
    assert issubclass(AssetError, ValueError)
