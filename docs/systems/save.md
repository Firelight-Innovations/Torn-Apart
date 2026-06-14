# save — System Doc
keywords: save, load, delta, msgpack, zlib, Saveable, header, F5, F9, atomic, SaveIncompatibleError, save_key, get_delta, apply_delta, format_version, world_seed, config_digest, game_clock, clock, pickle, no-pickle, tuple-key, numpy, compression, dump_save, on-disk layout, registration order, kv_pairs, ndarray, blake2b, incompatible, round-trip, baseline, deviation, persistence, serialisation, editor_scene, scene save, authored scene, --load, scenes folder

> Registered save keys today (registration order in main.py): `terrain`
> (ChunkManager), `weather`, `zones`, `editor_scene`
> (`fire_engine.scene.SceneRuntime` — placed objects authored in the Fire
> Editor; the editor daemon registers terrain + the bare store under the same
> key). A key absent from a save file keeps its baseline (skipped, logged
> debug); unknown keys in the file are ignored — so editor saves load in the
> game and vice versa. Only ONE registered system may claim a given save_key.
> `python main.py --load PATH` loads a save/scene at boot and retargets F5/F9.
> The `editor_scene` delta is just the object list; each object now carries a
> `components` list (`{type, enabled, params}`) inside its dict — no save-format
> bump needed (the layer stores the delta verbatim). Pre-component saves have no
> `components` key and migrate forward on load via `SceneObject.from_dict`
> (synthesises the kind's defaults), so old `.ta` files keep loading.

> One doc per code package; filename matches the package exactly (`docs/systems/save.md` ↔ `fire_engine/save/`).

## Role

`save/` is the **delta persistence** package (Foundation — callable from any layer, per ARCHITECTURE.md §4a.4).  It owns:

- **`Saveable` protocol** — the structural interface any system implements to participate in delta saves: a `save_key` string attribute, `get_delta() -> dict`, and `apply_delta(dict)`.
- **`SaveIncompatibleError`** — raised on load when the save file is incompatible (wrong seed, wrong config digest, or newer format version).
- **`SaveManager`** — registers Saveables at boot, orchestrates `save()` and `load()`, encodes deltas as msgpack+zlib, validates headers, and applies deltas in registration order.

`save/` deliberately does NOT:
- Import panda3d (Hard Rule 1 — headless-testable).
- Import any layer above `core` (systems *register into* it; it never imports them).
- Use pickle anywhere (Hard Rule 3 — owner decision 2026-06-09).
- Store live object references in deltas (only primitives + numpy arrays).

The design principle: `world = regenerate_from_seed() + apply_delta(saved_delta)`.  Only deviations from the procedural baseline are stored.  An untouched world costs ~0 bytes of delta storage (e.g. < 1 KB for the terrain blob with zero edited chunks).

## Public API

All symbols below are re-exported from `fire_engine.save` (`__init__.py`).

| Symbol | Description |
|---|---|
| `Saveable` | `@runtime_checkable Protocol`: `save_key: str`, `get_delta() -> dict`, `apply_delta(dict) -> None`. |
| `SaveIncompatibleError` | Exception raised by `SaveManager.load` on header mismatch. No partial load when raised. |
| `SaveManager(config, clock)` | Main persistence manager. |
| `SaveManager.register(saveable)` | Register a `Saveable` for persistence. Call at boot, in order. Raises `TypeError` if not a Saveable. |
| `SaveManager.save(path)` | Write header + per-system compressed delta blobs atomically. |
| `SaveManager.load(path)` | Validate header, reset clock, apply deltas in registration order. |

Internal helpers (used by tests and `tools/dump_save.py`, not part of the public API):

| Symbol | Description |
|---|---|
| `save_manager._encode_delta(delta) -> bytes` | Encode a system delta dict to raw msgpack bytes. |
| `save_manager._decode_delta(bytes) -> dict` | Decode a system delta from raw msgpack bytes. |
| `save_manager._encode_value(obj) -> Any` | Recursively make `obj` msgpack-serialisable (handles numpy arrays + tuple keys). |
| `save_manager._decode_value(obj) -> Any` | Inverse of `_encode_value`. |
| `save_manager._compute_config_digest(config) -> str` | Compute the blake2b config-hash stored in the header. |

## Imports Allowed

Per ARCHITECTURE.md §4a.2, `save/` may import:
- Python standard library (`hashlib`, `os`, `zlib`, `pathlib`, `typing`, ...)
- `numpy`
- `msgpack` (third-party; in `requirements.txt`)
- `fire_engine.core` (Config, Clock, get_logger)

**No panda3d imports.  No imports from any layer above core.**  Systems register *into* SaveManager; SaveManager never imports them.  This is inversion of control — adding a new saveable system never touches save code.

## On-Disk Layout

One file is written per save operation.  The file is a single msgpack-encoded outer envelope:

```
{
    "header": {
        "format_version": 1,           # int — increment when layout changes
        "world_seed":     <int>,        # must match Config.world_seed on load
        "config_digest":  <hex str>,    # blake2b-16 of canonical config fields
        "game_clock":     <dict>        # Clock.get_state() plain dict
    },
    "systems": {
        "<save_key>": <bytes>,          # zlib(msgpack(encoded_delta))
        ...                             # one entry per registered Saveable
    }
}
```

The outer dict is NOT zlib-compressed (the header stays cheaply readable).  Each per-system blob is `zlib.compress(msgpack.packb(encoded_delta))`.

### Config Digest

`config_digest` is a lowercase hex `blake2b` (digest_size=16, → 32 hex chars) of:

```
f"{world_seed}:{voxel_size}:{chunk_size}:{light_grid_scale}"
```

Fields included are those whose change would make the save geometrically invalid: `world_seed`, `voxel_size`, `chunk_size`, `light_grid_scale`.  Debug flags (`show_fps`, `show_chunk_borders`, `show_light_grid`) and `view_distance_chunks` are deliberately excluded — changing them does not invalidate a save.

### Numpy & Tuple-Key Encoding

msgpack does not natively support numpy arrays or tuple dict-keys.  Two transforms are applied before packing and reversed after unpacking:

**1. Numpy arrays** → `["__ndarray__", dtype_str, shape_list, raw_bytes]`

```python
# Encode:
["__ndarray__", "uint8", [32, 32, 32], b"<32768 raw bytes>"]

# Decode:
np.frombuffer(raw_bytes, dtype="uint8").reshape([32, 32, 32])
```

**2. Dicts with non-string keys** (e.g. terrain's `{(cx,cy,cz): uint8[32,32,32]}`) → wrapped as kv_pairs:

```python
# Encode:
{
    "__delta_type__": "kv_pairs",
    "pairs": [
        [[0, 0, 0], <encoded_array>],
        [[1, -2, 3], <encoded_array>],
        ...
    ]
}

# Decode:
{(0, 0, 0): array1, (1, -2, 3): array2, ...}
```

Only top-level delta dicts use kv_pairs encoding; dicts with all-string keys encode directly (no wrapper).  On decode, each key-list is reconstructed as `tuple(int(x) for x in key_list)`.

### Header Validation Rules (load)

`SaveManager.load` raises `SaveIncompatibleError` (with a clear message, **no partial load**) if any of these fail:

| Check | Raises if |
|---|---|
| `format_version` | Saved value > `_FORMAT_VERSION` (engine too old for this save) |
| `world_seed` | Saved value ≠ `config.world_seed` (wrong world) |
| `config_digest` | Saved value ≠ current digest (geometry-affecting config changed) |

### Registration Order (apply_delta)

`apply_delta` is called on registered Saveables in **registration order** (the order `register()` was called at boot).  Per ARCHITECTURE.md §4a.4:

1. `"terrain"` — ChunkManager (always first; geometry must exist before AI/economy can reference it)
2. `"ai"` — Phase 8+ (not yet wired)
3. `"economy"` / `"politics"` — Phase 9+ (not yet wired)

If a system's `save_key` is absent from the file, `apply_delta` is NOT called and the system retains its freshly-generated state.

### Atomic Write

`save(path)` writes to `<path>.tmp` then calls `os.replace(<path>.tmp, path)`.  The destination is never left in a half-written state.  If the write fails, the `.tmp` file is removed.

### No-Pickle Rule

**No pickle anywhere in the codebase** (Hard Rule 3, owner decision 2026-06-09).  `tests/test_save.py::TestNoPickle::test_no_pickle_imports` walks all `fire_engine/` and `tools/` `.py` files and fails if any contain `import pickle`, `import cPickle`, `from pickle`, `from cPickle`, or `pickle.` (regex).  This test runs as part of the normal headless suite.

## Events

### Published
None.  `save/` does not publish events.

### Subscribed
None.  `save/` is called directly (F5 → `sm.save(path)`, F9 → `sm.load(path)`).

## Units & Invariants

- Save files use the extension `.ta` by convention (`saves/quick.ta` for F5/F9).
- The `saves/` directory is gitignored — never commit save files.
- `format_version` starts at `1`; increment (with migration code) only when the on-disk layout changes incompatibly.
- An unmodified terrain (zero edited chunks) produces a terrain blob < 1 KB after zlib compression.
- A 1-chunk crater (1 × uint8[32,32,32] = 32 768 bytes raw) compresses to ~1 270 bytes (~96% compression ratio).
- The outer envelope (header + empty or small systems dict) is typically < 200 bytes for a fresh world.

## Examples

### Wire up SaveManager at boot (main.py)

```python
from fire_engine.core import load_config, Clock, EventBus
from fire_engine.core.rng import set_world_seed
from fire_engine.world.terrain import ChunkManager
from fire_engine.save import SaveManager

cfg = load_config()
set_world_seed(cfg.world_seed)
bus = EventBus()
clock = Clock(fixed_dt=cfg.fixed_dt, bus=bus)
cm = ChunkManager(cfg, bus)

sm = SaveManager(cfg, clock)
sm.register(cm)               # "terrain" — always first
# sm.register(ai_manager)     # "ai" — Phase 8+
# sm.register(economy)        # "economy" — Phase 9+
```

### F5 (quick save)

```python
# Bind in world/app.py input handler:
sm.save("saves/quick.ta")     # atomic: .tmp → rename
```

### F9 (quick load)

```python
from fire_engine.save import SaveIncompatibleError

try:
    sm.load("saves/quick.ta")
except SaveIncompatibleError as exc:
    show_hud_error(f"Cannot load: {exc}")
    return

# After load, the terrain ChunkManager has apply_delta'd all edited chunks.
# Those chunks are marked dirty=True and edited=True.  The streaming pipeline
# will remesh them on the next stream_frame() call.  If you need immediate
# geometry, flush like this:
for coord, chunk in cm.chunks.items():
    if chunk.dirty:
        cm.mesh_chunk(coord, light_sampler=sunlight.sample)
```

### Implement a custom Saveable

```python
from fire_engine.save import Saveable   # runtime_checkable Protocol

class EconomyManager:
    save_key: str = "economy"

    def get_delta(self) -> dict:
        # Return only prices that differ from the procedural baseline
        return {good_id: price for good_id, price in self._prices.items()
                if price != self._baseline_prices[good_id]}

    def apply_delta(self, delta: dict) -> None:
        # Baseline was regen'd from seed first; overlay saved prices
        for good_id, price in delta.items():
            self._prices[good_id] = price

assert isinstance(EconomyManager(), Saveable)  # True at runtime
sm.register(EconomyManager())
```

### Inspect a save file

```
python tools/dump_save.py saves/quick.ta
```

Output:
```
Save file: saves/quick.ta  (1,471 bytes on disk)

=== HEADER ===
  format_version : 1
  world_seed     : 1337
  config_digest  : 4013a3445925f82f616eaea083e599b3
  game_clock:
    game_day             : 0
    game_time_of_day     : 300.0
    total_real_time      : 5.0
    accumulator          : 5.0

=== SYSTEMS ===

  [terrain]
    delta entries  : 1
    compressed     :      1,270 bytes
    uncompressed   :     32,831 bytes
    compression    : 96.13 %
```

## Gotchas

1. **No pickle, ever.**  The no-pickle test (`TestNoPickle`) scans the entire source tree.  Importing pickle anywhere — even transitively — will fail CI.  Use msgpack + the `_encode_value`/`_decode_value` helpers for any non-primitive serialisation need.

2. **Tuple keys in terrain deltas.**  `ChunkManager.get_delta()` returns `{(cx,cy,cz): uint8[32,32,32]}`.  msgpack cannot represent tuple keys — they are encoded as kv_pairs lists.  If you write a custom Saveable with non-string dict keys, the same encoding is applied automatically by `_encode_value`.

3. **Clock is in the header, not a registered Saveable.**  Per ARCHITECTURE.md §4a.4, the clock state is stored in the header and is restored from it on load (authoritative).  The clock is NOT registered as a Saveable; `SaveManager.load` calls `clock.set_state(header["game_clock"])` directly.  Do not also register the clock — you would double-apply the state.

4. **apply_delta happens AFTER baseline regen.**  `SaveManager.load` does NOT re-seed or re-generate the world.  The registered Saveables' `apply_delta` methods are responsible for generating the baseline first and then overlaying the delta.  `ChunkManager.apply_delta` calls `generate_chunk` on any coord not yet in `self.chunks` before overwriting materials.

5. **Absent save_key = keep baseline.**  If a system was registered but its `save_key` is absent from the save file (e.g., a newly-added system whose save was created before it existed), `apply_delta` is NOT called for that system.  It silently keeps its procedurally-generated state.  This makes new systems backward-compatible with old saves.

6. **Atomic write — parent directory must exist.**  `save(path)` creates the parent directory with `mkdir(parents=True, exist_ok=True)`, but the `saves/` directory is gitignored.  Ensure it exists at runtime (or let `save()` create it automatically).

7. **format_version on load.**  A save file whose `format_version` equals `_FORMAT_VERSION` or is *lower* loads fine.  Only a version *higher* than the engine supports raises `SaveIncompatibleError`.  Lower versions load without migration for now (no migration code exists yet); add migration logic before bumping the version.
