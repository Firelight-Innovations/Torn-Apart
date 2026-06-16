# scene — System Doc
keywords: scene, scene object, SceneObject, SceneObjectStore, SceneRuntime, editor scene, authored scene, placed object, scene graph, hierarchy, parent, reparent, transform, TRS, position, rotation, scale, quaternion, component, ComponentSpec, FieldSpec, COMPONENT_CATALOG, Mesh, Light, SpawnPoint, catalog, coerce_params, make_component, default_components_for_kind, catalog_payload, is_known, default_params, KINDS, SceneError, save_key, editor_scene, get_delta, apply_delta, visual_factory, spawn_position, rebuild, SCENE_TAG, to_dict, from_dict, tree, create, delete, clear, rename, reparent, set_transform, add_component, remove_component, set_component

> One doc per code package; filename matches the package exactly (`docs/systems/scene.md` ↔ `fire_engine/scene/`).

## Role

`scene/` is the **authored-scene data model** package (Layer 2 — Structure).  It owns:

- **`SceneObject`** — a single node in the authoring hierarchy: an integer id, a display name, a kind (one of `KINDS`), an optional parent id, a local TRS transform (position/rotation/scale), and a list of built-in components.
- **`SceneObjectStore`** — the mutable, ordered store of `SceneObject` instances with full hierarchy operations: `create`, `rename`, `reparent`, `set_transform`, `add_component`, `remove_component`, `set_component`, `delete`, `clear`, plus a DFS `tree()` query.  Implements the `Saveable` protocol (`save_key = "editor_scene"`).
- **`SceneRuntime`** — the game-side `Saveable` that the game registers at boot.  On `apply_delta` it restores the authored scene and calls `rebuild()`, which instantiates every `SceneObject` as a live `GameObject` (using an optional `visual_factory` for panda3d visuals).  Owns a `SceneObjectStore` by composition; only this object is registered with `SaveManager`.
- **`COMPONENT_CATALOG` / `ComponentSpec` / `FieldSpec`** — the single authoritative catalog of built-in component types (`Mesh`, `Light`, `SpawnPoint`) that the Fire Editor's inspector renders and the game's `SceneVisualFactory` materialises.  Functions `make_component`, `coerce_params`, `default_components_for_kind`, `catalog_payload` operate on this catalog.
- **`SceneError`** — the exception class for invalid scene operations (unknown id/kind, cycle-detection in reparent).

`scene/` deliberately does NOT:
- Import panda3d (Hard Rule 1 — headless-testable; all rendering is delegated to a `visual_factory` passed at construction time).
- Import `core.rng` (ids come from a monotonic counter so the same sequence of edits always yields the same ids — no RNG needed).
- Export re-render commands or touch the scene graph directly.

## Public API

All symbols below are re-exported from `fire_engine.scene` (`__init__.py`).

| Symbol | Description |
|---|---|
| `KINDS` | `frozenset[str]` of valid object kinds: `{"empty", "cube", "sphere", "light", "spawn"}`. |
| `SceneError` | `ValueError` subclass for invalid scene operations (bad id/kind, reparent cycle). |
| `SceneObject` | `@dataclass` — one authoring node: `id`, `name`, `kind`, `parent`, `position`, `rotation`, `scale`, `components`. |
| `SceneObject.to_dict() -> dict` | Wire/save form: plain JSON-friendly primitives. |
| `SceneObject.from_dict(d) -> SceneObject` | Deserialise from wire/save dict; migrates pre-component saves. |
| `SceneObjectStore()` | Mutable ordered store with hierarchy operations. Implements `Saveable` (`save_key = "editor_scene"`). |
| `SceneObjectStore.create(kind, *, parent, name, position) -> dict` | Create a new object; returns its dict form. |
| `SceneObjectStore.rename(obj_id, name) -> dict` | Rename an existing object. |
| `SceneObjectStore.reparent(obj_id, parent) -> dict` | Move `obj_id` under `parent` (None = root). Rejects cycles. |
| `SceneObjectStore.set_transform(obj_id, *, position, rotation, scale) -> dict` | Set local TRS. |
| `SceneObjectStore.add_component(obj_id, type_name) -> dict` | Attach a component; rejects singletons already present. |
| `SceneObjectStore.remove_component(obj_id, index) -> dict` | Remove the component at `index` (0-based). |
| `SceneObjectStore.set_component(obj_id, index, *, params, enabled) -> dict` | Edit component params/enabled at `index`. |
| `SceneObjectStore.delete(obj_id) -> list[int]` | Delete `obj_id` and all descendants; returns removed ids. |
| `SceneObjectStore.clear()` | Drop all objects; resets the id counter. |
| `SceneObjectStore.tree() -> list[dict]` | Flat, depth-first list of object dicts (roots first, DFS order). |
| `SceneObjectStore.get(obj_id) -> SceneObject` | Look up an object by id; raises `SceneError` if missing. |
| `SceneObjectStore.get_delta() -> dict` | Deviation from empty baseline: full object list or `{}`. |
| `SceneObjectStore.apply_delta(delta)` | Restore objects saved by `get_delta` onto an empty baseline. |
| `SceneRuntime(visual_factory, on_rebuilt)` | Game-side `Saveable` that materialises the authored scene as `GameObject` instances. |
| `SceneRuntime.store` | The `SceneObjectStore` of authored-scene data. |
| `SceneRuntime.objects` | `{scene object id: GameObject}` for the current build. |
| `SceneRuntime.rebuild()` | Tear down then reinstantiate all authored objects as live `GameObject` instances. |
| `SceneRuntime.spawn_position -> Vec3 \| None` | World position of the first `spawn` object (DFS), or `None`. |
| `SceneRuntime.get_delta() -> dict` | Delegates to `store.get_delta()`. |
| `SceneRuntime.apply_delta(delta)` | Restores the store and calls `rebuild()`. |

From `fire_engine.scene.components`:

| Symbol | Description |
|---|---|
| `COMPONENT_CATALOG` | `dict[str, ComponentSpec]` — built-in component types: `"Mesh"`, `"Light"`, `"SpawnPoint"`. |
| `ComponentSpec` | Frozen dataclass: `type`, `label`, `multiple`, `fields`. |
| `FieldSpec` | Frozen dataclass: `name`, `ui_type`, `default`, `min`, `max`, `choices`, `label`. |
| `is_known(type_name) -> bool` | True if `type_name` is a registered component type. |
| `default_params(type_name) -> dict` | Fresh default `params` dict for `type_name` (deep-copied). |
| `make_component(type_name, *, enabled=True) -> dict` | Build a component dict with default params. |
| `default_components_for_kind(kind) -> list[dict]` | Components a freshly created object of `kind` carries. |
| `coerce_params(type_name, params) -> dict` | Validate+coerce a partial params dict against the catalog. |
| `catalog_payload() -> dict` | JSON-friendly catalog dict for the `scene.catalog` RPC. |

## Imports Allowed

Per ARCHITECTURE.md §4a.2, `scene/` may import:
- Python standard library (`copy`, `dataclasses`, `logging`, `typing`, ...)
- `fire_engine.core` (math3d `Vec3`/`Quat`, `logging`)
- `fire_engine.render` — only for TYPE_CHECKING (for `GameObject` type hint); the actual `render` import in `SceneRuntime.rebuild()` is a deferred local import to avoid panda3d at import time.

**No panda3d imports.  No imports from world/, simulation/, or any layer above core at module level.**

## Events

### Published
None.  `scene/` does not publish events.  The editor daemon reads and writes `SceneObjectStore` directly; `SceneRuntime` is registered as a `Saveable` with `SaveManager` and never emits events.

### Subscribed
None.  `SceneRuntime.rebuild()` is called from `apply_delta` and directly from the runtime when needed.

## Units & Invariants

- Positions are in **meters**, Z-up, local to the parent transform (same coordinate space as the engine: world space = metres, 1 voxel = 0.5 m, 1 chunk = 16 m).
- Rotations are **quaternions** `(w, x, y, z)`.  The identity quaternion is `(1.0, 0.0, 0.0, 0.0)`.  Euler angles are never stored here.
- The `SceneObjectStore.save_key` and `SceneRuntime.save_key` are both `"editor_scene"`.  Only ONE of them is registered with `SaveManager` per process: the game registers `SceneRuntime`; the editor daemon registers a bare `SceneObjectStore` under the same key (via the `fire_editor.scene_objects` shim).
- Object ids come from a monotonic counter seeded at 1 (resets to 1 on `clear()`), so the same sequence of edits always yields the same ids — deterministic, no RNG.
- `KINDS = frozenset({"empty", "cube", "sphere", "light", "spawn"})`.  The `kind` field is the creation archetype only; the `components` list is the source of truth thereafter (an `empty` can be given a `Light`; a `cube`'s `Mesh` can be removed).
- Pre-component saves (no `components` key in the object dict) are migrated forward on load by `SceneObject.from_dict`, which synthesises the kind's default components.
- The `SCENE_TAG = "editor_scene"` is stamped on every `GameObject` `SceneRuntime` creates — lets the dev overlay and other tools distinguish authored content from procedural or debug objects.

## Examples

### Build a small authored scene

```python
from fire_engine.scene import SceneObjectStore, SceneError

store = SceneObjectStore()
cube = store.create("cube", name="Crate")           # -> {"id": 1, "kind": "cube", ...}
child = store.create("empty", parent=cube["id"], name="Pivot")
store.set_transform(cube["id"], position=(4.0, 0.0, 2.0))
store.reparent(child["id"], parent=None)            # promote to root
print(store.tree())                                  # flat DFS list of dicts
```

### Add and edit a component

```python
# Add a Light component to an "empty" node
store.add_component(cube["id"], "Light")
# Edit the Light's intensity and color
store.set_component(cube["id"], 1, params={"intensity": 4.0, "color": [1.0, 0.8, 0.4]})
```

### Wire SceneRuntime in the game

```python
from fire_engine.scene import SceneRuntime
from fire_engine.save import SaveManager

runtime = SceneRuntime(visual_factory=factory)   # factory may be None for headless
save_manager.register(runtime)                   # claims "editor_scene"
save_manager.load("scenes/ambush.ta")            # -> GameObjects exist now

if runtime.spawn_position is not None:
    app.player.transform.position = runtime.spawn_position
```

### Save and reload a scene

```python
delta = store.get_delta()        # {"objects": [...], "next_id": N}
store2 = SceneObjectStore()
store2.apply_delta(delta)        # restore from delta
assert len(store2) == len(store)
```

### Fetch the catalog for the inspector

```python
from fire_engine.scene.components import catalog_payload

payload = catalog_payload()
# -> {"types": [{"type": "Mesh", "label": "Mesh", "multiple": False,
#                "fields": [{"name": "primitive", "ui_type": "enum", ...}]}, ...]}
```

## Gotchas

1. **Only register ONE of `SceneObjectStore` or `SceneRuntime` with `SaveManager`.**  Both share `save_key = "editor_scene"`.  The game registers `SceneRuntime` (which owns a `SceneObjectStore` internally); the editor daemon registers a bare `SceneObjectStore` (via its shim).  Registering both would raise a duplicate-key conflict or double-apply the delta.

2. **`SceneRuntime.rebuild()` is an eager teardown — call it only when ready.**  It destroys all previously created `GameObjects` synchronously (via `destroy()`), then creates fresh ones.  This is safe to call multiple times but do not call it mid-frame from a hot path.

3. **Deferred panda3d import in `rebuild()`.**  `fire_engine/scene/runtime.py` does NOT import `fire_engine.render` at module level — the import is deferred to inside `rebuild()`.  This keeps the module headless-importable (the editor daemon imports `scene` and `panda3d` is not installed there, or forbidden in tests via `test_no_panda3d.py`).

4. **`KINDS` is the creation archetype set, not the component gating.**  After creation, `kind` is frozen, but any `COMPONENT_CATALOG` type can be added to any object.  An `empty` can have a `Light`; a `cube` can have its `Mesh` removed.  Do not rely on `kind` to infer current visuals — walk the `components` list instead.

5. **`coerce_params` silently drops unknown keys.**  Unknown parameter keys in the dict passed to `coerce_params` or `set_component` are dropped without error (the inspector may send stale keys when the catalog is updated).  If a parameter doesn't appear in the result, the key is not in the catalog.

6. **`reparent` raises on cycles.**  Parenting an object beneath any of its own descendants raises `SceneError`.  The check walks the full subtree, so it is O(subtree size) — avoid in tight loops.

7. **`apply_delta` calls `clear()` first.**  Calling `apply_delta` unconditionally wipes the store before restoring.  Do not call it if you only want to overlay new data — use `create` / `set_transform` instead.
