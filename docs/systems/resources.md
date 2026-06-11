# resources — System Doc
keywords: resource manager, asset loading, cache, refcount, reference count, handle, loader, loaders, egg, bam, gltf, glb, ogg, wav, png, jpg, model, audio, texture, hand-crafted, landmark, inversion of control, panda3d adapter, unload, evict, dispatch, register_loader, UnknownResourceFormatError, LoaderCallable, default_manager, acquire, release, unload_unreferenced, stats

> One doc per code package; filename matches the package exactly (`docs/systems/resources.md` ↔ `fire_engine/resources/`).

## Role

`resources/` is the **single gateway for all hand-crafted asset file I/O**: landmark 3D models, player hands, audio files, and static PNG/JPG textures authored by hand. It provides an in-memory reference-counted cache so assets are loaded once and shared.

`resources/` deliberately does NOT:
- Import panda3d (hard rule — see below).
- Load procedural environment textures (those come from `procedural/` via `world/texture_bridge.py`).
- Generate any content procedurally.
- Own the actual Panda3D loader code (injected via inversion of control from `world/`).

### Panda3D-free / Inversion of Control

`resources/` contains **zero** panda3d imports. Real loader functions are injected at boot by `world/resource_adapter.register_panda_loaders(manager)`, which calls `register_loader(suffix, fn)` for each supported format. This keeps `resources/` fully headless-testable and free of the render-layer dependency.

## Public API

All symbols below are re-exported from `fire_engine.resources` (`__init__.py`).

### Error

| Symbol | Description |
|---|---|
| `UnknownResourceFormatError(path, suffix)` | Raised by `dispatch()` when no loader is registered for the file suffix. Has `.path` and `.suffix` attributes. |

### Loader Registry (`resources/loaders.py`)

| Symbol | Description |
|---|---|
| `LoaderCallable` | Type alias: `Callable[[str], Any]`. Every loader must match this signature. |
| `register_loader(suffix: str, loader: LoaderCallable)` | Register (or replace) the callable for a file suffix (e.g. `".egg"`). Called by `world/resource_adapter` at boot. |
| `dispatch(path: str) -> Any` | Look up and invoke the loader for `path`'s suffix. Raises `UnknownResourceFormatError` if absent. |
| `registered_suffixes() -> list[str]` | Returns sorted list of suffixes that have a non-None loader. |

### Handle (`resources/manager.py`)

| Symbol | Description |
|---|---|
| `Handle` | Wraps a loaded resource. Attributes: `.resource` (the raw object), `.path` (normalised cache key), `.refcount` (int). |

### ResourceManager (`resources/manager.py`)

| Symbol | Description |
|---|---|
| `ResourceManager(loaders_module=None)` | Cache + dispatcher. Accepts optional fake loaders module for testing. |
| `manager.load(path: str) -> Handle` | Cache-miss: dispatch → wrap in Handle (refcount=0) → cache. Cache-hit: return same Handle. |
| `manager.acquire(handle: Handle) -> Handle` | Increment refcount by 1. Returns the same handle (chainable). |
| `manager.release(handle: Handle)` | Decrement refcount by 1; never below 0. |
| `manager.unload_unreferenced() -> int` | Evict all zero-refcount handles. Calls `handle.resource.cleanup()` if present. Returns eviction count. |
| `manager.stats() -> dict` | Returns `cache_size`, `total_handles`, `zero_ref`, `nonzero_ref`, `max_refcount`, `total_refcount`. |

### Module-level convenience (`resources/manager.py`)

| Symbol | Description |
|---|---|
| `default_manager` | Global `ResourceManager` instance. Used by convenience functions below. |
| `load(path: str) -> Handle` | `default_manager.load(path)` |
| `acquire(handle: Handle) -> Handle` | `default_manager.acquire(handle)` |
| `release(handle: Handle)` | `default_manager.release(handle)` |
| `unload_unreferenced() -> int` | `default_manager.unload_unreferenced()` |

## Imports Allowed

`resources/` may only import:
- Python standard library (`os`, `typing`, ...)
- `fire_engine.core` (logging etc. if needed — currently not imported)

**No panda3d imports, ever.** Real loaders are injected at runtime by `world/resource_adapter.py`.
Per ARCHITECTURE.md §4a.2: `resources → procedural → core`.

## Events

### Published
None. `resources/` is a pure service layer; it emits no events.

### Subscribed
None. Cache management is on-demand (explicit `load`/`release`/`unload_unreferenced` calls).

## Units & Invariants

### Refcount Semantics
- `load(path)` returns a Handle at **refcount 0** ("you requested it; you must claim it").
- `acquire(handle)` increments refcount by 1; paired with `release`.
- `release(handle)` decrements refcount by 1; clamped to 0 (never negative).
- A handle is eligible for eviction when `refcount == 0`.
- `unload_unreferenced()` must be called explicitly — handles are not auto-evicted.

### Cache Key
Paths are normalised via `os.path.normcase(os.path.normpath(path))` for the cache key so that `assets\models\foo.egg` and `assets/models/foo.egg` share one Handle. The loader receives `os.path.normpath(path)` (case preserved) to avoid Windows filesystem mismatches.

### Supported Suffixes
`.egg`, `.bam`, `.gltf`, `.glb` — 3D models (Panda3D Loader, registered by world/).
`.ogg`, `.wav` — audio (Panda3D audio, registered by world/).
`.png`, `.jpg` — static hand-crafted textures (Pillow, registered by world/).

### What Does NOT Route Here
Procedural environment textures generated by `fire_engine.procedural` (e.g. `wasteland_ground`) — those are numpy arrays bridged to panda3d via `world/texture_bridge.py`. Only **hand-crafted** files on disk go through `ResourceManager`.

## Examples

### Boot (world/app.py)
```python
from fire_engine.resources import default_manager
from fire_engine.world.resource_adapter import register_panda_loaders

# Called once after ShowBase is created:
register_panda_loaders(default_manager)
```

### Loading a landmark model
```python
from fire_engine.resources import load, acquire, release, unload_unreferenced

# Load and claim ownership
handle = acquire(load("assets/models/landmark_church.egg"))
nodepath = handle.resource   # Panda3D NodePath

# Later, when the model is no longer needed:
release(handle)
unload_unreferenced()        # evicts the handle (and any other zero-ref handles)
```

### Multiple owners sharing one Handle
```python
# Both callers get the same Handle instance (cache hit):
h1 = acquire(load("assets/audio/ambient_wind.ogg"))
h2 = acquire(load("assets/audio/ambient_wind.ogg"))
assert h1 is h2              # same Handle
# h1.refcount == 2

release(h1)                  # refcount → 1
release(h2)                  # refcount → 0
unload_unreferenced()        # evicted
```

### Headless testing with a fake loader
```python
class FakeLoaders:
    def __init__(self):
        self._loaders = {}
    def register_loader(self, suffix, fn):
        self._loaders[suffix] = fn
    def dispatch(self, path):
        suffix = path[path.rfind("."):]
        if suffix not in self._loaders:
            from fire_engine.resources.loaders import UnknownResourceFormatError
            raise UnknownResourceFormatError(path, suffix)
        return self._loaders[suffix](path)

fake = FakeLoaders()
fake.register_loader(".fake", lambda p: {"data": p})
manager = ResourceManager(loaders_module=fake)
handle = manager.load("test.fake")
assert handle.resource == {"data": "test.fake"}
```

### stats() output
```python
s = manager.stats()
# {
#   "cache_size": 3,
#   "total_handles": 3,
#   "zero_ref": 1,       # eligible for eviction
#   "nonzero_ref": 2,
#   "max_refcount": 4,
#   "total_refcount": 5,
# }
```

## Gotchas

1. **`load()` returns refcount=0** — callers must call `acquire(handle)` if they want to keep the asset alive past the next `unload_unreferenced()`. Forgetting `acquire` is a common bug: the manager will evict the handle on cleanup even though the caller holds a Python reference to it (the Handle object won't be GC'd, but a subsequent `load()` of the same path will re-load from disk and return a **new** Handle).

2. **No auto-eviction** — `unload_unreferenced()` must be called explicitly. A typical pattern is to call it once per frame or at level-transition boundaries. Until then, zero-refcount handles stay cached (acts as a free warm-cache for recently-used assets).

3. **`default_manager` uses the global `resources.loaders` module** — registering a loader via `register_loader(".egg", fn)` affects ALL `ResourceManager` instances that use the default loaders module (including `default_manager`). Use an isolated `FakeLoadersModule` in tests.

4. **Window test requires ShowBase** — `test_load_triangle_egg_with_panda3d` is marked `@pytest.mark.window` and excluded from the default headless run. It creates an offscreen `ShowBase` instance. Run explicitly: `pytest tests/test_resources.py -m window`.

5. **`cleanup()` convention** — `unload_unreferenced()` calls `handle.resource.cleanup()` if the method exists. Panda3D `NodePath`s don't have a `cleanup()` method by default; you should call `nodepath.remove_node()` before releasing if you want it out of the scene graph.

6. **Path normalisation on Windows** — the cache key uses `normcase` (lowercases on Windows) for case-insensitive deduplication. The actual path passed to the loader uses `normpath` only (case preserved) to avoid Panda3D's internal Filename system getting confused by an all-lowercase Windows path.
