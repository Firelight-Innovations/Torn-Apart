"""Constants for the ``.asset`` prefab file format.

Docs: docs/systems/assets.md
"""

from __future__ import annotations

# Spec version written into every .asset envelope's ``"fire_asset"`` field. Bump
# on any backward-incompatible envelope change and add a migration step in
# :func:`fire_engine.assets.asset_file._migrate` (same discipline as the editor
# protocol schema).
FIRE_ASSET_VERSION: int = 1

# Reserved scene-component type name for the linked-instance layer: a scene
# object carrying a component of this type references an .asset by path and is
# materialised by instantiating that asset's subtree under the object's
# transform (linked, not baked — editing the .asset updates every scene that
# references it). Registering the component in the scene catalog and the
# scene-load resolver is the consuming branch's job; the NAME is fixed here so
# producers and consumers agree. See docs/systems/assets.md "Cross-scene reference".
PREFAB_INSTANCE_COMPONENT: str = "PrefabInstance"
