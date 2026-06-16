"""fire_engine.assets — the ``.asset`` GameObject/prefab file format.

A general, human-readable (JSON) serialisation of a GameObject subtree (a
"prefab") as a standalone, reusable asset, decoupled from any world save.
Buildings are the first consumer, but the format is generic: any
:class:`~fire_engine.scene.SceneObject` subtree can be snapshotted, saved,
loaded, edited, and instantiated into multiple scenes. Headless (no panda3d).

Docs: docs/systems/assets.md
"""

from fire_engine.assets.asset_file import load_asset, save_asset
from fire_engine.assets.blobs import decode_array, encode_array
from fire_engine.assets.constants import FIRE_ASSET_VERSION, PREFAB_INSTANCE_COMPONENT
from fire_engine.assets.enums import AssetType
from fire_engine.assets.prefab import Prefab
from fire_engine.assets.types import (
    AssetError,
    AssetSource,
    AssetVersionError,
    Transform,
)

__all__ = [
    "FIRE_ASSET_VERSION",
    "PREFAB_INSTANCE_COMPONENT",
    "AssetError",
    "AssetSource",
    "AssetType",
    "AssetVersionError",
    "Prefab",
    "Transform",
    "decode_array",
    "encode_array",
    "load_asset",
    "save_asset",
]
