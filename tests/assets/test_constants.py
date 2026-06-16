"""Tests for fire_engine.assets.constants."""

from __future__ import annotations

from fire_engine.assets.constants import FIRE_ASSET_VERSION, PREFAB_INSTANCE_COMPONENT


def test_version_is_a_positive_int() -> None:
    assert isinstance(FIRE_ASSET_VERSION, int)
    assert not isinstance(FIRE_ASSET_VERSION, bool)
    assert FIRE_ASSET_VERSION >= 1


def test_prefab_instance_component_name_is_stable() -> None:
    # Producers and consumers agree on this exact spelling — guard against drift.
    assert PREFAB_INSTANCE_COMPONENT == "PrefabInstance"
