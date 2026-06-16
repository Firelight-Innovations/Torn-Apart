"""Tests for fire_engine.assets.enums."""

from __future__ import annotations

from fire_engine.assets.enums import AssetType


def test_values() -> None:
    assert AssetType.PREFAB.value == "prefab"
    assert AssetType.BUILDING.value == "building"


def test_str_enum_compares_equal_to_its_value() -> None:
    # str-valued enum so envelope code can treat asset_type as an open string.
    assert AssetType.BUILDING == "building"
    assert AssetType("prefab") is AssetType.PREFAB
