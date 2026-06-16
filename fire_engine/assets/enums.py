"""Enumerations for the ``.asset`` prefab file format.

Docs: docs/systems/assets.md
"""

from __future__ import annotations

from enum import StrEnum


class AssetType(StrEnum):
    """Canonical ``asset_type`` values for an .asset envelope.

    The on-disk ``asset_type`` field is an **open string** (a consumer may add
    its own kinds), but these are the engine-known values. ``str``-valued so
    ``AssetType.BUILDING == "building"`` and ``AssetType.BUILDING.value`` both
    hold.

    Members:
        PREFAB: a generic GameObject subtree (the universal case).
        BUILDING: a building authored from a ``BuildingDef`` (consumer #1).

    Docs: docs/systems/assets.md
    """

    PREFAB = "prefab"
    BUILDING = "building"
