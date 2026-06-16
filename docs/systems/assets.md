# assets — System Doc
keywords: asset, .asset, prefab, Prefab, prefab file, GameObject file, serialise, serialize, save_asset, load_asset, from_store, instantiate_into, to_envelope, from_envelope, envelope, fire_asset, asset_type, AssetType, AssetSource, AssetError, AssetVersionError, Transform, encode_array, decode_array, blob, base64, numpy blob, migration, versioned loader, PrefabInstance, linked instance, prefab instance, path as identity, guid, building asset, reusable asset, scene object subtree, id remap

> One doc per code package; filename matches the package exactly (`docs/systems/assets.md` ↔ `fire_engine/assets/`).

## Role

`assets/` is the **`.asset` GameObject/prefab file format** (a headless Layer-2 package — zero panda3d, zero RNG). It serialises a `GameObject` (and its child hierarchy + component data) as a standalone, reusable, human-readable text file — a "prefab" — decoupled from any world save. Buildings are consumer #1, but the package is **deliberately generic**: any `SceneObject` subtree can be snapshotted, saved, loaded, edited, and instantiated into multiple scenes.

It owns:

- **`Prefab`** — the in-memory model of one `.asset`: a header (`asset_type`, optional provenance, reserved `guid`), the subtree as a list of `SceneObject` **wire dicts** with asset-local ids, and a `blobs` map. Reuses the existing `SceneObject` dict shape rather than inventing a parallel object representation.
- **`save_asset` / `load_asset`** — byte-stable JSON IO with a versioned, migrating loader.
- **`encode_array` / `decode_array`** — the Base64 numpy codec for binary payloads.
- **`AssetSource` / `Transform` / `AssetType`** — provenance, a TRS placement value, and the engine-known `asset_type` values.

`assets/` deliberately does **NOT**:
- Import `panda3d` (Hard Rule 1 — fully headless-testable).
- Import `buildings/` or any consumer (buildings depend on assets, never the reverse). Kinds and component types are opaque strings/dicts; the package never inspects them.
- Import `scene/` at runtime — `Prefab` operates on a `SceneObjectStore` passed in, via its public `tree()` / `get_delta()` / `apply_delta()` API (the type is imported only under `TYPE_CHECKING`). This keeps the runtime dependency direction `scene/buildings → assets`, with no cycle.
- Use `pickle` (Hard Rule 3) — pure JSON of primitives + Base64.

## Public API

All symbols below are re-exported from `fire_engine.assets` (`__init__.py`).

| Symbol | Description |
|---|---|
| `FIRE_ASSET_VERSION` | `int` — current `.asset` spec version written into every envelope's `"fire_asset"`. |
| `PREFAB_INSTANCE_COMPONENT` | `str` — reserved scene-component type name (`"PrefabInstance"`) for the linked-instance layer. |
| `AssetType` | `str`-valued `Enum` of engine-known `asset_type`s: `PREFAB` (`"prefab"`), `BUILDING` (`"building"`). The on-disk field is an open string. |
| `AssetSource` | `@dataclass(frozen=True)` — provenance `{def, params, seed}`; `to_dict()` / `from_dict(d)`. |
| `Transform` | `@dataclass(frozen=True)` — a local TRS placement `(position, rotation, scale)`; identity by default. |
| `AssetError` | `ValueError` subclass — malformed/missing/unreadable `.asset` or blob. |
| `AssetVersionError` | `AssetError` subclass — file's `fire_asset` is newer than this build supports. |
| `Prefab(*, objects, root, asset_type, source, blobs, guid)` | In-memory model of a `.asset`. |
| `Prefab.from_store(store, root_id, *, asset_type, source) -> Prefab` | Snapshot the subtree rooted at `root_id` out of a `SceneObjectStore` (asset-local ids, root = 1). |
| `Prefab.instantiate_into(store, *, at_transform, parent) -> int` | Materialise into `store`, remapping ids; returns the new root's id. Components written verbatim. |
| `Prefab.to_envelope() -> dict` | Serialise to the JSON-friendly envelope dict. |
| `Prefab.from_envelope(env) -> Prefab` | Build a `Prefab` from a (current-version) envelope dict. |
| `save_asset(path, prefab) -> None` | Write a byte-stable `.asset` file (UTF-8, `indent=2`, `sort_keys=True`, trailing newline). |
| `load_asset(path) -> Prefab` | Read + migrate + parse a `.asset` file. |
| `encode_array(arr) -> dict` | numpy array → `{"dtype", "shape", "base64"}` (little-endian, contiguous, deterministic). |
| `decode_array(d) -> ndarray` | Inverse of `encode_array`; returns a fresh writable array. |

### Envelope anatomy

```json
{
  "fire_asset": 1,
  "asset_type": "building",
  "guid": null,
  "source": { "def": "building_farmhouse", "params": { "storeys": 2 }, "seed": 1337 },
  "root": 1,
  "objects": [
    { "id": 1, "name": "Farmhouse", "kind": "building", "parent": null,
      "position": [0, 0, 0], "rotation": [1, 0, 0, 0], "scale": [1, 1, 1],
      "components": [ { "type": "Building", "enabled": true, "params": { } } ] }
  ],
  "blobs": {}
}
```

A trivial cube prefab is the same envelope with one `kind:"cube"` object and a `Mesh` component — the format is general, not building-specific. `source` is `null` for hand-authored prefabs. `guid` is reserved and always `null` in v1.

## Imports Allowed

`numpy` (blob codec). `fire_engine.scene` **only under `TYPE_CHECKING`** (the `SceneObjectStore` type hint). No other `fire_engine` package, no `panda3d`, no `pickle`. Per ARCHITECTURE.md §4a.2 the runtime edges are `assets → numpy` and (consumer) `buildings → assets`.

## Events

Published: none.
Subscribed: none. Asset IO is a direct call from a consumer (editor, buildings, scene loader); it never plumbs per-frame or per-event data.

## Units & Invariants

- Transforms: position in **meters** (Z-up), rotation a **quaternion** `(w, x, y, z)`, scale unitless `(x, y, z)` — identical to `SceneObject`.
- **Asset-local ids** are 1-based and dense within a prefab (root = 1, depth-first). `instantiate_into` remaps them onto the destination store's fresh ids, so the same prefab instantiates many times into one scene without collision.
- **Identity = path** (relative to `assets/`). No GUIDs generated in v1 (sidesteps the determinism/RNG rule); the `guid` envelope field is reserved for a future rename-safe layer.
- **Byte-stable output**: `save_asset` uses `sort_keys=True`, fixed `indent=2`, and a trailing newline; `encode_array` normalises to little-endian contiguous bytes. Re-saving an unchanged asset is a no-op git diff (the determinism ethos applied to files).
- **Versioning**: `load_asset` reads `fire_asset`, runs `_migrate` (forward-only; newer-than-known → `AssetVersionError`), then parses. Bump `FIRE_ASSET_VERSION` and add a migration step on any incompatible envelope change.
- **Fidelity**: kinds and component dicts are copied **verbatim** on both snapshot and instantiate — no catalog coercion, no kind validation — so arbitrary/opaque component data (e.g. a future `Building` blob, or a component type not yet registered in the scene catalog) survives a round trip.

## Examples

Snapshot a subtree, save it, and instantiate it elsewhere:

```python
from fire_engine.assets import Prefab, Transform, save_asset, load_asset
from fire_engine.scene import SceneObjectStore

store = SceneObjectStore()
crate = store.create("cube", name="Crate")
store.create("empty", parent=crate["id"], name="Pivot")

prefab = Prefab.from_store(store, crate["id"])
save_asset("assets/prefabs/crate.asset", prefab)

# ...later, in another scene:
scene = SceneObjectStore()
root_id = load_asset("assets/prefabs/crate.asset").instantiate_into(
    scene, at_transform=Transform(position=(4.0, 0.0, 2.0))
)
```

Binary payload (only when content is genuinely binary — buildings use none):

```python
import numpy as np
from fire_engine.assets import encode_array, decode_array

blob = encode_array(np.arange(6, dtype="<f4").reshape(2, 3))
prefab.blobs["mesh_positions"] = blob          # carried in the envelope
np.array_equal(decode_array(blob), np.arange(6).reshape(2, 3))   # True
```

## Gotchas

- **Cross-scene reference is linked, not baked.** The chosen design (owner-confirmed 2026-06-16) is a `PrefabInstance` scene component `{asset_path, overrides: {}}` on an empty object; on scene load the runtime loads the `.asset` and instantiates its subtree under that object's transform — so editing the `.asset` updates **every** scene that references it. v1 ships the foundation (codec + `Prefab` + `instantiate_into`); registering `PrefabInstance` in the scene catalog and wiring the scene-load resolver land with the consuming editor/buildings branch. `PREFAB_INSTANCE_COMPONENT` fixes the type name so producers and consumers agree. Per-instance `overrides` are stubbed (`{}`) and deferred.
- **`instantiate_into` rebuilds the whole destination store** (it merges via `get_delta`/`apply_delta`), so any externally held `SceneObject` *references* into that store go stale — re-`get` by id afterwards. Ids are stable; object identities are not.
- **`Prefab.from_store` requires a single rooted subtree.** It walks children from `root_id`; a node whose parent lies outside the subtree is not included.
- **`save_asset` does not create parent directories.** Ensure `assets/prefabs/` or `assets/buildings/` exists first.
- **Directory convention vs the "hand-crafted only" note.** CLAUDE.md says `assets/` is hand-crafted only; generated-then-hand-edited buildings count as authored content, so they live at `assets/buildings/*.asset` (generic prefabs at `assets/prefabs/*.asset`). The on-disk `assets/` tree is excluded from the standards gate (`pyproject` `[tool.firelight] exclude`), so `.asset` files there are never linted as code.
- **`assets/` (the package) ≠ `assets/` (the directory).** The Python package is `fire_engine/assets/`; the on-disk asset files live in the repo-root `assets/` tree. Only the latter is in the standards `exclude` list.
