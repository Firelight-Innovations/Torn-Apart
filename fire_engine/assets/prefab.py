"""Prefab — an in-memory ``.asset`` (header + a SceneObject subtree + blobs).

A :class:`Prefab` is the runtime model of a serialised GameObject subtree: the
same wire-dict shape :class:`fire_engine.scene.SceneObject` uses, snapshotted
out of a live :class:`~fire_engine.scene.SceneObjectStore` and re-materialisable
into any scene with its asset-local ids remapped into that scene's id space.

The model is **generic**: kinds and component types are opaque strings/dicts
copied verbatim, so any GameObject subtree round-trips — buildings are just the
first consumer. This package never imports ``buildings/`` (buildings depend on
it, never the reverse).

Docs: docs/systems/assets.md
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from fire_engine.assets.constants import FIRE_ASSET_VERSION
from fire_engine.assets.enums import AssetType
from fire_engine.assets.types import AssetError, AssetSource, Transform

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fire_engine.scene import SceneObjectStore


class Prefab:
    """A standalone, reusable GameObject subtree (a "prefab").

    Attributes:
        asset_type: open ``asset_type`` string (see :class:`AssetType` for the
            engine-known values).
        objects: the subtree as a list of SceneObject wire dicts
            (``{id, name, kind, parent, position, rotation, scale, components}``)
            with **asset-local** ids; the root is :attr:`root` and a root
            object's ``parent`` is ``None``.
        root: asset-local id of the subtree root.
        source: optional provenance (which def/params/seed produced it).
        blobs: name -> Base64 blob dict (see :mod:`fire_engine.assets.blobs`);
            empty for buildings.
        guid: reserved for a future rename-safe identity layer; always ``None``
            in v1 (path is identity).

    Example::

        prefab = Prefab.from_store(store, cube_id)         # snapshot a subtree
        save_asset("assets/prefabs/crate.asset", prefab)   # -> a .asset file
        new_root = prefab.instantiate_into(other_store, at_transform=Transform(position=(4, 0, 0)))

    Docs: docs/systems/assets.md
    """

    def __init__(
        self,
        *,
        objects: list[dict[str, Any]],
        root: int,
        asset_type: str | AssetType = AssetType.PREFAB,
        source: AssetSource | None = None,
        blobs: dict[str, dict[str, Any]] | None = None,
        guid: str | None = None,
    ) -> None:
        self.asset_type: str = (
            asset_type.value if isinstance(asset_type, AssetType) else str(asset_type)
        )
        self.objects: list[dict[str, Any]] = objects
        self.root: int = int(root)
        self.source: AssetSource | None = source
        self.blobs: dict[str, dict[str, Any]] = {} if blobs is None else blobs
        self.guid: str | None = guid

    # ------------------------------------------------------------------ #
    # SceneObjectStore interop
    # ------------------------------------------------------------------ #
    @classmethod
    def from_store(
        cls,
        store: SceneObjectStore,
        root_id: int,
        *,
        asset_type: str | AssetType = AssetType.PREFAB,
        source: AssetSource | None = None,
    ) -> Prefab:
        """Snapshot the subtree rooted at ``root_id`` out of ``store``.

        Asset-local ids are assigned depth-first (root = 1); the root's parent is
        recorded as ``None``. Transforms and components are deep-copied verbatim.

        Raises:
            AssetError: if ``root_id`` is not present in ``store``.

        Docs: docs/systems/assets.md
        """
        flat: dict[int, dict[str, Any]] = {int(d["id"]): d for d in store.tree()}
        rid = int(root_id)
        if rid not in flat:
            raise AssetError(f"no scene object with id {root_id} to snapshot")

        children: dict[int | None, list[dict[str, Any]]] = {}
        for d in flat.values():
            children.setdefault(d.get("parent"), []).append(d)

        order: list[int] = []

        def _walk(node: int) -> None:
            order.append(node)
            for child in children.get(node, []):
                _walk(int(child["id"]))

        _walk(rid)

        id_map = {old: new for new, old in enumerate(order, start=1)}
        objects: list[dict[str, Any]] = []
        for old in order:
            d = copy.deepcopy(flat[old])
            d["id"] = id_map[old]
            parent = d.get("parent")
            d["parent"] = None if (old == rid or parent is None) else id_map[int(parent)]
            objects.append(d)
        return cls(objects=objects, root=1, asset_type=asset_type, source=source)

    def instantiate_into(
        self,
        store: SceneObjectStore,
        *,
        at_transform: Transform | None = None,
        parent: int | None = None,
    ) -> int:
        """Materialise this prefab into ``store``; returns the new root's id.

        Asset-local ids are remapped onto fresh store ids (so the same prefab can
        be instantiated many times into one scene). The root is parented under
        ``parent`` (a scene root if ``None``); ``at_transform``, if given,
        replaces the root's local TRS. Components are written **verbatim** (no
        catalog coercion), so arbitrary component data survives the round trip.

        Raises:
            AssetError: if ``parent`` is given but absent from ``store``.

        Docs: docs/systems/assets.md
        """
        delta = store.get_delta()
        existing: list[dict[str, Any]] = list(delta.get("objects", []))
        if parent is not None and int(parent) not in {int(o["id"]) for o in existing}:
            raise AssetError(f"cannot instantiate under missing parent id {parent}")

        next_id = int(delta.get("next_id", 1))
        id_map = {int(o["id"]): next_id + i for i, o in enumerate(self.objects)}
        new_root_id = id_map[int(self.root)]

        new_objs: list[dict[str, Any]] = []
        for o in self.objects:
            d = copy.deepcopy(o)
            d["id"] = id_map[int(o["id"])]
            op = o.get("parent")
            if op is None:
                d["parent"] = None if parent is None else int(parent)
            else:
                d["parent"] = id_map[int(op)]
            if d["id"] == new_root_id and at_transform is not None:
                d["position"] = list(at_transform.position)
                d["rotation"] = list(at_transform.rotation)
                d["scale"] = list(at_transform.scale)
            new_objs.append(d)

        store.apply_delta({"objects": existing + new_objs, "next_id": next_id + len(self.objects)})
        return new_root_id

    # ------------------------------------------------------------------ #
    # Envelope <-> model
    # ------------------------------------------------------------------ #
    def to_envelope(self) -> dict[str, Any]:
        """Serialise to the JSON-friendly .asset envelope dict.

        Docs: docs/systems/assets.md
        """
        return {
            "fire_asset": FIRE_ASSET_VERSION,
            "asset_type": self.asset_type,
            "guid": self.guid,
            "source": None if self.source is None else self.source.to_dict(),
            "root": int(self.root),
            "objects": copy.deepcopy(self.objects),
            "blobs": copy.deepcopy(self.blobs),
        }

    @classmethod
    def from_envelope(cls, env: dict[str, Any]) -> Prefab:
        """Build a :class:`Prefab` from a (current-version) envelope dict.

        Raises:
            AssetError: if required keys are missing or the root id is absent
                from ``objects``.

        Docs: docs/systems/assets.md
        """
        try:
            asset_type = str(env["asset_type"])
            root = int(env["root"])
            objects = [copy.deepcopy(o) for o in env["objects"]]
        except (KeyError, TypeError, ValueError) as e:
            raise AssetError(f"malformed asset envelope: {e}") from e
        if root not in {int(o["id"]) for o in objects}:
            raise AssetError(f"asset root id {root} not present in objects")
        src = env.get("source")
        source = None if src is None else AssetSource.from_dict(src)
        blobs = {str(k): dict(v) for k, v in env.get("blobs", {}).items()}
        return cls(
            objects=objects,
            root=root,
            asset_type=asset_type,
            source=source,
            blobs=blobs,
            guid=env.get("guid"),
        )
