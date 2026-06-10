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

### MESH payload layout + world-space positions (E1)
- **Q:** How is a chunk mesh serialised, and where does the viewport place it?
- **Choice:** A self-describing MESH payload (`i32 cx,cy,cz`, `u32 N`, `u32 M`, then `f32` positions/normals/colors(RGBA)/uvs and `u32` indices). Positions are **absolute world meters** (engine mesher already emits world-space verts), so the webview attaches every chunk mesh at the origin with no per-chunk offset. Codec: `meshcodec.py` ↔ `meshPayload.ts`.
- **Why:** Self-describing frames let the client route by coord without depending on notification ordering; world-space positions keep one coordinate space end-to-end and match the game's `geometry_bridge` behaviour.

### Viewport shades with baked vertex colours (MeshBasicMaterial)
- **Q:** How does the three.js viewport light chunks?
- **Choice:** `MeshBasicMaterial({ vertexColors: true })`. The engine's CPU sunlight pass is already baked into the mesh's RGBA vertex colours (greyscale × light), so the viewport must **not** re-light — a basic (unlit) material shows the engine's lighting faithfully. A full-strength `AmbientLight` is added only so any future lit materials are visible.
- **Why:** Re-lighting with a Lambert/standard material would double-shade and diverge from the game. Parity comes from reusing the engine's baked light, not from re-deriving it client-side.

### Chunk streaming is cooperative-async, not threaded (E1)
- **Q:** How does the daemon stream a region without blocking the control channel?
- **Choice:** `chunks.set_center` cancels any in-flight stream and launches an `asyncio` task that generates → relights → meshes coords nearest-first, `await asyncio.sleep(0)` every 8 chunks, broadcasting MESH frames as it goes. No worker threads.
- **Why:** Chunk gen/mesh are small numpy ops (sub-ms each); cooperative yielding keeps `hello`/`ping`/`raycast` responsive within the §8 budget without the complexity of cross-thread sharing of the engine session. Revisit (worker process) only if profiling shows meshing starving RPC.

### Protocol version bumps every schema change (E0→E1: v1→v2)
- **Q:** When does `protocol_version` increment?
- **Choice:** On **any** `schema.json` change, even additive ones (E1 added methods → bumped 1→2). The `hello` handshake requires exact equality; daemon and extension always regenerate from the same commit, so exact-match + per-change bump is the honest, drift-proof contract (hard rule 6).
- **Why:** Exact-match handshake means an old extension against a new daemon must fail fast with a clear "rebuild" message rather than silently calling absent methods.

### Brush undo via before/after material snapshots over the AABB (E3)
- **Q:** How does editor undo/redo restore exact voxels for a brush edit?
- **Choice:** Each `terrain.brush` snapshots the `uint8` material arrays of every chunk overlapping the brush's AABB **before** and **after** the engine `apply_brush` call (`commands.EditCommand` / `UndoStack`, bounded to 200 entries, LRU-drop). Undo writes `before`, redo writes `after`; `restore()` recomputes each chunk's `edited` flag by comparing to `generate_chunk` so save deltas stay correct. Affected chunks **and their loaded neighbours** remesh+relight and restream.
- **Why:** AABB snapshots are local and cheap (a few 32 KB arrays), give byte-exact restore (the E3 acceptance), and keep undo entirely editor-side (no engine change). Neighbours must remesh because a boundary edit exposes faces on the adjacent chunk.

### Brush aimed at the viewport crosshair (E3)
- **Q:** How does the user aim a brush while the fly camera owns the mouse?
- **Choice:** Pointer-lock = look mode; a left-click **while locked** casts a ray from the camera through screen-centre (the crosshair) — the webview sends the ray, the host `terrain.raycast`s it and `terrain.brush`es at the hit point. First click (unlocked) just captures the mouse.
- **Why:** Avoids a cursor-vs-pointer-lock conflict; FPS-style "look at it, click to carve" needs no separate cursor and reuses the existing fly controls.

### Crater round-trip is the headline integration test (E3)
- **Q:** How is "edit in editor → game shows it" verified without a window?
- **Choice:** `tests/editor/test_edit.py::TestCraterRoundTrip` carves via the daemon, `world.save`s, then loads the file through the **engine's own** `SaveManager`+`ChunkManager` (the `python main.py --load` path) and asserts the loaded chunk materials equal the editor's and deviate from `generate_chunk` baseline.
- **Why:** Proves the editor produces standard engine saves (terrain deltas), not an editor-only format — the whole point of binding to the engine's `Saveable` path.

### Mouse-look uses relative mode + raw pixel deltas; auto-captured at boot
- **Q:** Why did free-look feel locked to one axis and require hunting for ESC?
- **Choice:** `App` now captures the mouse on startup and uses **relative** mouse mode (`M_relative`) with the cursor hidden, reading raw pixel deltas from `win.get_pointer(0)` relative to window centre and recentring each frame (first post-capture delta skipped). ESC toggles capture off (cursor shown, absolute mode).
- **Why:** The old path used confined mode + normalised `mouseWatcher` deltas, which clamped the pointer at a screen edge and could freeze an axis. Raw-pixel + relative mode keeps both yaw and pitch symmetric and never edge-clamps. Auto-capture removes the "press ESC to look" surprise.

## 2026-06-09 — In-game developer overlay (devtools)

### In-game dev overlay is a separate system from the Fire Editor
- **Q:** The owner described an in-game debug menu (noclip cam, perf stats, click-to-select + outline, live value editing, spawn/fire-event buttons) — is that the `EDITOR_PRD.md` Fire Editor?
- **Choice:** No — it is a **new, distinct in-game system** (`torn_apart/devtools/` + `world/devtools_overlay.py`), toggled with **F1**, that runs *inside the live Panda3D window*. The Fire Editor (`editor/`) is the *external* VS Code/Cursor tool that runs with the game *closed*. Both coexist; ARCHITECTURE.md §6 explicitly anticipates in-game debug overlays for Session 1.
- **Why:** They serve different workflows (live in-engine tweaking vs. offline content authoring). Conflating them would have forced the daemon/webview architecture onto something that just needs to draw over the running game.

### Renderer: Panda3D DirectGUI now, not Dear ImGui (but swappable)
- **Q:** The "common debug-menu UI" the owner named is Dear ImGui. Build a real ImGui-in-Panda3D integration, or use Panda3D's native GUI?
- **Choice:** **DirectGUI** for v1 (owner-approved). The dev-tools *logic* is fully decoupled behind a declarative `Panel`/`Section`/`Field`/`Button` model (`devtools/fields.py`); the renderer only consumes that model. A real Dear ImGui backend can replace `world/devtools_overlay.py` later without touching `torn_apart/devtools/`.
- **Why:** ImGui has no first-class Panda3D binding — a custom draw-list backend is a fragile native dependency and slow to a first working version. DirectGUI is zero-dependency, solid on Windows/Panda3D, and ships a working stats/inspector/spawn overlay today. The panel-model indirection keeps the ImGui door open.

### New headless `devtools/` package (logic) + renderer in `world/`
- **Q:** Where does the dev-overlay code live given hard rule 1 (panda3d only in `world/`/`lighting/`)?
- **Choice:** A new **headless** package `torn_apart/devtools/` (selection, CPU picking, GameObject introspection, tools, manager) imports **`core` only** — never panda3d, never `world` at runtime (TYPE_CHECKING duck-typing). The single panda3d-touching file is `world/devtools_overlay.py` (DirectGUI + mouse→ray + outline + spawn visuals). `tests/test_devtools.py` runs in the headless suite.
- **Why:** Keeps the editor logic unit-testable without a window and obeys the import rule; the renderer is a thin, replaceable presentation layer.

### Object picking via CPU ray/AABB, not a Panda3D collision graph
- **Q:** How does click-to-select find the object under the cursor?
- **Choice:** The overlay extrudes a world-space ray from the mouse through the camera lens and hands it to a **headless ray/AABB slab test** (`devtools/picking.py`) over registered `Selectable` boxes (world-axis-aligned, derived from transform position ± half-extents × scale; rotation ignored for v1). Nearest hit wins.
- **Why:** Standing up a `CollisionTraverser`/`CollisionRay` graph just to click a few dev props is overkill; CPU ray/AABB is deterministic, unit-testable, and keeps the picking math headless.

## 2026-06-09 — Procedural sky + weather (session 2)

### Sky lives in a new headless Layer-1 package; rendering stays in world/
- **Q:** Where does the sky/weather system live given hard rule 1 (panda3d only in `world/`/`lighting/`)?
- **Choice:** A new **headless** package `torn_apart/sky/` (Layer 1 — Services, peer of `lighting/`): celestial math (`celestial.py`), the weather state machine (`weather.py`), and the per-frame `SkyState` aggregate (`sky_state.py`). All panda3d rendering (dome shader, clouds, rain, fog) lives in `world/sky_renderer.py` + `world/sky_shaders.py`, driven by a `SkyRendererComponent` on a "Sky" GameObject — so the system is authored through the World API object model, per the owner's request.
- **Why:** Same split as lighting and devtools: simulation is headless-testable and deterministic; `world/` only reads a frozen `SkyState` dataclass and writes uniforms/scene state in bulk.

### Day/night + weather lighting integration is a global colour-scale (v0)
- **Q:** "The lighting system should also be affected" — relight the voxel grid per time-of-day?
- **Choice:** No relight. `SkyState.terrain_light_scale` (RGB ≈ (1,1,1) clear noon → (0.16,0.19,0.30) night, warm-tinted at dawn/dusk, dimmed by overcast/storm) is applied as **one `terrain_root.set_color_scale(...)` write per frame** on top of the baked vertex sunlight.
- **Why:** The baked sun/shadow contrast already lives in vertex colours; modulating it globally gives a convincing day/night/weather response for one scene-graph write — no per-voxel work, no light-grid recompute. A real sun-angle-aware relight is a later upgrade (would need directional column passes per sun elevation).

### Weather is a day-anchored Markov chain; the save delta is ~0 bytes
- **Q:** How is weather made deterministic and saveable without storing a schedule?
- **Choice:** Time is divided into 2-game-hour segments (12/day). Each segment's state is sampled via `rng.for_domain("weather", game_day, segment_index)`; **each day's segment 0 draws from a fixed (≈stationary) initial distribution** instead of chaining across midnight, so any segment is recomputable from `(world_seed, day, segment)` in ≤12 steps. Parameters blend over 20 game-minutes at transitions. `WeatherSystem.get_delta()` returns `{}` unless a `force_weather` override is active — the natural schedule is pure function of seed + clock and costs nothing to save.
- **Why:** Matches the engine's delta-save philosophy (baseline regenerates from seed); the midnight hand-off discontinuity is hidden by the standard parameter blend.

### Night sky is ONE baked equirect procedural texture
- **Q:** Stars/galaxy as point geometry, or baked?
- **Choice:** A single registered `ProceduralTextureDef` `"night_sky"` (1024×512 equirect): galaxy band on a tilted great circle (noise filaments + subtractive dust lanes + warm core ramp) and the full star field, alpha = luminance. The dome shader samples it by view direction, additively by `star_visibility`, and adds per-pixel hash twinkle; shooting stars are shader streaks driven by uniforms scheduled via `for_domain("sky", "shooting_stars", day, slot)`.
- **Why:** Fits "100% procedural environment textures", costs one texture fetch instead of thousands of points, and keeps the whole night sky deterministic from the world seed.

### Environment (day/night + weather) controls live in the overlay
- **Q:** Where do the day/night-cycle / weather controls the owner wants go?
- **Choice:** An **Environment** panel registered in the overlay (`CallbackTool`) that edits `clock.game_time_of_day` / `game_time_scale` and cycles `sky_system.weather.force_weather(...)`, reading the live `SkyState`. It is registered only when `app.sky_system` is present and is bound defensively (`getattr`/`try`) against the concurrent sky feature.
- **Why:** The owner explicitly wanted day/night editable "in the game world with the same system." A generic `CallbackTool` surfaces it without coupling `devtools/` to the in-flight `sky` API.
