# DECISIONS — Torn Apart

A dated log of implementation decisions that `docs/ARCHITECTURE.md` did **not** already pin.
Per CLAUDE.md, prefer the smallest decision that doesn't close doors, and record it here
(date · question · choice · one-line why). ARCHITECTURE.md remains the design authority; this
file captures the choices made *underneath* it during implementation.

---

## 2026-06-09 — Session 1 implementation

### Quaternion storage order
- **Q:** How is a `Quat` laid out in memory?
- **Choice:** Scalar-first `[w, x, y, z]` float32 numpy array (`core/math3d.py`).
- **Why:** Matches scipy `Rotation` / common quaternion literature; one unambiguous convention spelled out once so no module guesses x-first.

### Quaternion multiplication semantics
- **Q:** What does `q1 * q2` mean?
- **Choice:** Hamilton product where `q1 * q2` **applies `q2` first, then `q1`** (so `yaw * pitch` yaws in world space, then pitches in the yawed frame).
- **Why:** Matches Unity's `Quaternion.*` and scipy; lets the mouse-look composition read naturally and stay roll-free.

### Euler (HPR) convention
- **Q:** What do the three Euler angles mean and in what order do they compose?
- **Choice:** `from_euler(h, p, r)` / `as_euler()` use **H = heading about world +Z, P = pitch about +X, R = roll about +Y**, composed **H then P then R** (`qH * qP * qR`, applied R-first). Euler is a presentation view only — never stored state.
- **Why:** Z-up Panda3D-native axes; quaternion-only storage avoids gimbal lock (ARCHITECTURE §5.4), so Euler exists purely as a convenience at the API edge.

### RNG key digest
- **Q:** How are `for_domain(*keys)` keys hashed into a deterministic seed?
- **Choice:** `hashlib.blake2b` (digest_size=8) over the canonical repr of the keys, mixed into `np.random.SeedSequence` as a `spawn_key`. Never Python's built-in `hash()`.
- **Why:** `hash()` is salted per process since Python 3.3 and would silently break cross-run/cross-machine determinism — the foundation of delta saves and bug repro.

### Save envelope encoding
- **Q:** How are delta saves serialised without pickle?
- **Choice:** A single **msgpack** outer envelope (`{header, systems}`); each per-system delta blob is **zlib-compressed msgpack**. The outer dict is uncompressed so the header stays cheaply readable. Numpy arrays are encoded as `["__ndarray__", dtype_str, shape_list, raw_bytes]` triples; dicts with non-string keys (e.g. terrain's `(cx,cy,cz)` keys) are wrapped as `{"__delta_type__": "kv_pairs", "pairs": [...]}` and reconstructed to int tuples on decode.
- **Why:** No pickle anywhere (Hard Rule 3) — msgpack+zlib is compact, inspectable (`tools/dump_save.py`), and refactor-safe (no live object refs).

### Config digest fields
- **Q:** Which config fields go into the save's `config_digest` (the load-compatibility hash)?
- **Choice:** blake2b (16-byte) of `f"{world_seed}:{voxel_size}:{chunk_size}:{light_grid_scale}"`. Debug flags and `view_distance_chunks` are deliberately excluded.
- **Why:** Only fields whose change makes a save *geometrically* invalid belong in the digest; changing debug flags or view distance must not refuse a valid save.

### Sunlight light levels
- **Q:** What are the discrete sunlight values before blur?
- **Choice:** `LIGHT_FULL = 255` (no solid above in the column), `LIGHT_AMBIENT = 40` (shadowed). The 3×3×3 box blur produces intermediate penumbra values but preserves the [40, 255] range.
- **Why:** A non-zero ambient floor keeps shadowed undersides readable rather than pure black; 255 ceiling keeps lit tops at full brightness. Cheap, vectorised, GPU-portable.

### texture_bridge channel order (BGRA)
- **Q:** What byte order does a Panda3D `F_rgba` RAM image expect?
- **Choice:** `to_panda_texture` reorders **RGBA → BGRA** (`arr[..., [2,1,0,3]]`) and vertically flips before `set_ram_image`.
- **Why:** Panda3D's native `F_rgba` RAM layout is BGRA and its UV origin is bottom-left; without both transforms every texture renders blue-for-brown and/or upside-down. Documented so nobody "fixes" it back.

### Mesher emits world-space vertices
- **Q:** Are mesher vertex positions chunk-local or world-space?
- **Choice:** `MeshArrays.positions` are **absolute world meters**. Consequently each chunk's NodePath is attached under `terrain_root` at the origin with **no per-chunk offset**.
- **Why:** One coordinate space end-to-end (lighting samples the same world positions); offsetting the NodePath as well would double the world position.

### ResourceManager.load refcount
- **Q:** What refcount does `load(path)` return a handle at?
- **Choice:** `load()` returns a `Handle` at **refcount 0**; callers must `acquire()` to claim ownership. `unload_unreferenced()` evicts zero-ref handles (called explicitly, no auto-eviction).
- **Why:** "You requested it; you must claim it." Makes ownership explicit and lets `load()` double as a warm-cache probe without forcing a lifetime.

### reset_to_baseline() for F9 revert
- **Q:** How does F9 undo craters dug *after* the save?
- **Choice:** Added `ChunkManager.reset_to_baseline()` — regenerate every loaded edited chunk from seed, clear `edited`, mark `dirty`. F9 calls it *before* `SaveManager.load()`, which then re-applies only the saved craters.
- **Why:** `apply_delta` only touches chunks present in the saved delta; without a baseline wipe first, post-save edits would survive a load. This restores true "revert to save" semantics.

### World-floor padding is SOLID
- **Q:** How does the mesher pad the −Z world boundary when the chunk below is absent?
- **Choice:** Absent neighbours pad **AIR** (open/visible edge), **except the −Z world floor**, which pads **SOLID** via the `WORLD_FLOOR_SOLID` sentinel (`ChunkManager` supplies it for `cz <= -2` when the below-chunk is unloaded).
- **Why:** Air padding everywhere would leave the bottom of the world see-through; a solid floor closes it without generating an infinite column of chunks downward.

### Flat, seed-independent baseline terrain + finite world footprint (2026-06-09)
- **Q:** Should terrain stay fully procedurally generated from `world_seed`?
- **Choice:** No. `generate_chunk` now emits **flat, seed-independent** baseline terrain: solid below `config.ground_height_m`, clamped to a square `config.world_size_m` footprint **centred on the origin** (default 1000 m = 1 km). No hills, noise, or caves. `world_seed` is retained but now drives **other** procedural systems (textures, ambient noise, NPC behaviour), not terrain. Added config fields `world_size_m` and `ground_height_m`.
- **Why:** The world is authored *semi-procedurally* (humans + rule-based / AI content agents working from parameters) on top of a blank flat canvas, rather than emerging from a heightmap. This gives the owner a deterministic, controllable starting state to lead development from. The old value-noise heightmap + 3-D carve pass were removed from `generation.py`; terrain determinism/save-delta guarantees are unchanged (still a pure function, just of `(coord, config)`).

## 2026-06-09 — Fire Editor (EDITOR_PRD)

### Editor repo placement
- **Q:** Where do the editor daemon and extension live?
- **Choice:** New top-level `editor/` in the game repo: `editor/fire_editor/` (Python daemon, uses the repo `.venv`), `editor/extension/` (TypeScript VS Code/Cursor extension), `editor/protocol/` (single-source `schema.json` + `codegen.py`). Editor docs at `docs/systems/editor.md`; daemon tests at `tests/editor/` (run in the headless `pytest` suite); the extension has its own `npm test` (excluded from pytest).
- **Why:** Matches EDITOR_PRD §2. Keeping the daemon in-repo lets it bind to `torn_apart` public APIs and share `.venv`; one codegen source keeps the two languages' bindings from drifting.

### WebSocket library for the daemon transport
- **Q:** How does the daemon serve the protocol?
- **Choice:** `websockets==13.1` (added to `requirements.txt`). The daemon binds `127.0.0.1` only and announces its OS-assigned port as a `{"event":"listening","port":N}` JSON line on **stdout**; logs go to **stderr** (the extension pipes stderr to its output channel).
- **Why:** A robust async WebSocket implementation handles binary frames natively (mesh/texture payloads, EDITOR_PRD hard rule 5). A hand-rolled RFC-6455 server would be more code to test for no benefit. Fresh-clone bootstrap installs it via `pip install -r requirements.txt`.

### Extension host owns the single daemon connection (webviews relay via postMessage)
- **Q:** Do webviews connect to the daemon directly over WebSocket, or through the extension host?
- **Choice:** The **extension host** (Node) owns the one WebSocket connection to the daemon; webviews (Scene View, Inspector, Texture Lab…) will receive data via `postMessage` with transferable `ArrayBuffer`s rather than opening their own sockets.
- **Why:** Avoids webview CSP/network restrictions and keeps a single authoritative connection + reconnect path. A minor deviation from the EDITOR_PRD §2 diagram (which sketches webviews on the socket); transferable ArrayBuffers keep binary mesh data zero-copy across the postMessage boundary. Revisit only if relay overhead shows up in the §8 budgets.

### Binary frame endianness + magic
- **Q:** Byte order and magic for `[u32 magic][u32 schema_id][u32 payload_id][payload]`?
- **Choice:** **Little-endian** throughout (Python `struct '<III'`, JS `DataView(..., true)`); magic = `0x46495245` ("FIRE"). `schema_id` ∈ `SchemaId` (MESH=1, TEXTURE=2); `payload_id` correlates a frame with the JSON-RPC message that announced it.
- **Why:** Little-endian matches x86 and typed-array native order, so mesh buffers map straight into three.js `BufferGeometry` with no byte swapping.

### Mouse-look uses relative mode + raw pixel deltas; auto-captured at boot
- **Q:** Why did free-look feel locked to one axis and require hunting for ESC?
- **Choice:** `App` now captures the mouse on startup and uses **relative** mouse mode (`M_relative`) with the cursor hidden, reading raw pixel deltas from `win.get_pointer(0)` relative to window centre and recentring each frame (first post-capture delta skipped). ESC toggles capture off (cursor shown, absolute mode).
- **Why:** The old path used confined mode + normalised `mouseWatcher` deltas, which clamped the pointer at a screen edge and could freeze an axis. Raw-pixel + relative mode keeps both yaw and pitch symmetric and never edge-clamps. Auto-capture removes the "press ESC to look" surprise.
