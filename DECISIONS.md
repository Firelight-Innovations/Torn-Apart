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
- **Why:** Matches EDITOR_PRD §2. Keeping the daemon in-repo lets it bind to `fire_engine` public APIs and share `.venv`; one codegen source keeps the two languages' bindings from drifting.

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
- **Choice:** No — it is a **new, distinct in-game system** (`fire_engine/devtools/` + `world/devtools_overlay.py`), toggled with **F1**, that runs *inside the live Panda3D window*. The Fire Editor (`editor/`) is the *external* VS Code/Cursor tool that runs with the game *closed*. Both coexist; ARCHITECTURE.md §6 explicitly anticipates in-game debug overlays for Session 1.
- **Why:** They serve different workflows (live in-engine tweaking vs. offline content authoring). Conflating them would have forced the daemon/webview architecture onto something that just needs to draw over the running game.

### Renderer: Panda3D DirectGUI now, not Dear ImGui (but swappable)
- **Q:** The "common debug-menu UI" the owner named is Dear ImGui. Build a real ImGui-in-Panda3D integration, or use Panda3D's native GUI?
- **Choice:** **DirectGUI** for v1 (owner-approved). The dev-tools *logic* is fully decoupled behind a declarative `Panel`/`Section`/`Field`/`Button` model (`devtools/fields.py`); the renderer only consumes that model. A real Dear ImGui backend can replace `world/devtools_overlay.py` later without touching `fire_engine/devtools/`.
- **Why:** ImGui has no first-class Panda3D binding — a custom draw-list backend is a fragile native dependency and slow to a first working version. DirectGUI is zero-dependency, solid on Windows/Panda3D, and ships a working stats/inspector/spawn overlay today. The panel-model indirection keeps the ImGui door open.

### New headless `devtools/` package (logic) + renderer in `world/`
- **Q:** Where does the dev-overlay code live given hard rule 1 (panda3d only in `world/`/`lighting/`)?
- **Choice:** A new **headless** package `fire_engine/devtools/` (selection, CPU picking, GameObject introspection, tools, manager) imports **`core` only** — never panda3d, never `world` at runtime (TYPE_CHECKING duck-typing). The single panda3d-touching file is `world/devtools_overlay.py` (DirectGUI + mouse→ray + outline + spawn visuals). `tests/test_devtools.py` runs in the headless suite.
- **Why:** Keeps the editor logic unit-testable without a window and obeys the import rule; the renderer is a thin, replaceable presentation layer.

### Object picking via CPU ray/AABB, not a Panda3D collision graph
- **Q:** How does click-to-select find the object under the cursor?
- **Choice:** The overlay extrudes a world-space ray from the mouse through the camera lens and hands it to a **headless ray/AABB slab test** (`devtools/picking.py`) over registered `Selectable` boxes (world-axis-aligned, derived from transform position ± half-extents × scale; rotation ignored for v1). Nearest hit wins.
- **Why:** Standing up a `CollisionTraverser`/`CollisionRay` graph just to click a few dev props is overkill; CPU ray/AABB is deterministic, unit-testable, and keeps the picking math headless.

## 2026-06-09 — Procedural sky + weather (session 2)

### Sky lives in a new headless Layer-1 package; rendering stays in world/
- **Q:** Where does the sky/weather system live given hard rule 1 (panda3d only in `world/`/`lighting/`)?
- **Choice:** A new **headless** package `fire_engine/world/sky/` (Layer 1 — Services, peer of `lighting/`): celestial math (`celestial.py`), the weather state machine (`weather.py`), and the per-frame `SkyState` aggregate (`sky_state.py`). All panda3d rendering (dome shader, clouds, rain, fog) lives in `world/sky_renderer.py` + `world/sky_shaders.py`, driven by a `SkyRendererComponent` on a "Sky" GameObject — so the system is authored through the World API object model, per the owner's request.
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

### Clouds are GPU-raymarched boxes, not geometry
- **Q:** How are the Minecraft-style boxy clouds rendered — instanced box meshes or a shader?
- **Choice:** Two static quads bracketing the cloud slab (`sky_cloud_altitude_m` … `+sky_cloud_thickness_m`); the fragment shader **2-D-DDA raymarches** the slab through a grid of `sky_cloud_cell_m` cells (≤48 steps, early-out), occupancy from hash-noise seeded by `for_domain("sky", "clouds")`, per-cell height variation, flat-face shading (lit tops / dark bottoms), wind-offset uniform. Coverage is mapped through a CPU-computed **noise quantile table** so `cloud_coverage` is the actual fill fraction, not a raw threshold.
- **Why:** Zero per-frame geometry churn (the camera-follow quad is one `set_pos`), works from below/inside/above the layer, and the raw-threshold alternative produced almost no clouds below coverage 0.3 because the noise is bell-distributed.

### Environment (day/night + weather) controls live in the overlay
- **Q:** Where do the day/night-cycle / weather controls the owner wants go?
- **Choice:** An **Environment** panel registered in the overlay (`CallbackTool`) that edits `clock.game_time_of_day` / `game_time_scale` and cycles `sky_system.weather.force_weather(...)`, reading the live `SkyState`. It is registered only when `app.sky_system` is present and is bound defensively (`getattr`/`try`) against the concurrent sky feature.
- **Why:** The owner explicitly wanted day/night editable "in the game world with the same system." A generic `CallbackTool` surfaces it without coupling `devtools/` to the in-flight `sky` API.

## 2026-06-11 — Faceted terrain + pixel-art ground textures (session 3)

### Terrain mesher is flat-shaded naive surface nets ("faceted"), cubes kept behind a config switch
- **Q:** The owner wants terrain between Minecraft-blocky and marching-cubes-smooth (Daggerfall Unity feel) — which algorithm?
- **Choice:** **Naive surface nets** over the binary voxel grid (`terrain/surface_nets.py`, `build_mesh_faceted`): one vertex per surface-straddling 2×2×2 cell at the centroid of its crossing-edge midpoints, one quad per exposed voxel face (the *same* exposure mask as the cube mesher), emitted as two independent flat triangles. No vertex smoothing passes (1-voxel neighbour padding stays sufficient; seams stay byte-identical). `config.mesh_style` selects `"faceted"` (default) or `"blocky"` (old mesher, kept for fixtures/regression).
- **Why:** Surface nets with binary voxels naturally produces chamfered 45°-ish facets — smooth silhouettes with clearly visible polygons — while keeping flat ground *exactly* planar. Marching cubes would over-smooth; bevelled cubes would not handle crater walls. Sharing the exposure mask keeps the `light_sampler` contract and `face_count` invariants unchanged.

### Facet accent: a fixed normal-based shade term baked into vertex colours
- **Q:** With scene lighting off (texture × vertex-colour pipeline) adjacent coplanar-ish facets are indistinguishable — how do facets stay readable?
- **Choice:** The faceted mesher multiplies `(1-s) + s*clamp(n·accent_dir, 0, 1)` (s = `config.facet_shade_strength` = 0.25, accent_dir ≈ high noon SE, a fixed art-direction constant) into the baked light, per triangle.
- **Why:** Cheap (pure numpy at mesh time), deterministic, and sells the low-poly look. It is NOT the sun: real sunlight still arrives via `light_sampler`; the accent is subtle enough not to fight the day/night colour scale. Revisit when a real sun-angle relight lands.

### Ground materials: grass skin (2) on the baseline's top voxel layer, dirt (1) below
- **Q:** Separate grass and dirt textures — how does the terrain know which face is which?
- **Choice:** `generate_chunk` writes `MATERIAL_GRASS` (2) into the topmost solid layer (pure function of world Z) and `MATERIAL_DIRT` (1) below; the faceted mesher tags each face with its solid voxel's material (`MeshArrays.face_materials`); `world/geometry_bridge.to_geom_node` splits each chunk into one Geom per material with that material's texture in a Geom-level RenderState. Brush ADD default material stays dirt.
- **Why:** Uses the existing `uint8` material storage (no new arrays, saves unchanged), digging naturally exposes dirt, and per-Geom RenderStates avoid texture atlas UV-wrapping headaches at 1 m tiling.

### Ground textures are low-res pixel-art defs; the "blur" was bilinear noise, not the sampler
- **Q:** Owner: "turn off the bilinear filter so textures are pixelated."
- **Choice:** The sampler was already nearest-neighbour (`texture_bridge`); the smoothness came from `value_noise`'s bilinear octave upsampling at 256 px/m. Added `pixel_noise` (nearest-upsampled octaves) and two new 64×64 defs `"grass_ground"` / `"dirt_ground"` with hard-threshold palette quantisation (8/6 colours). `"wasteland_ground"` remains as the node-level fallback texture.
- **Why:** Crisp square texels at 64 px per 1 m tile read correctly through the existing nearest-neighbour pipeline; quantised palettes give the retro look the owner asked for.

## 2026-06-11 — GPU volumetric lighting + physical sky (session 4)

### Lighting goes fully GPU: camera-centered cascaded radiance volumes
- **Q:** The owner wants Minecraft-shader-style volumetric lighting (GI, bounce, AO, volumetric fog, god rays, point/area lights, voxel shadows) computed on the GPU. What is the data model?
- **Choice:** Two **camera-centered cascaded 3-D textures** ("radiance cascades"): cascade 0 at **0.25 m texels, 128³ (32 m box)** — the owner's "8 light pixels per 0.5 m voxel" (2× per axis) — and cascade 1 at **1.0 m texels, 128³ (128 m box)**. CPU (numpy, headless `lighting/volume.py`) assembles occupancy/albedo/emission windows from chunk material arrays; windows recenter with hysteresis when the camera crosses half-cell boundaries. ARCHITECTURE §2's "8 terrain voxels per light cell" reading is superseded by the owner's explicit 8-texels-per-voxel request; the old CPU `LightGrid`/`SunlightComputer` (1 m cells, baked vertex colours) is kept as the `lighting_backend = "cpu"` fallback.
- **Why:** A scrolling window over the camera bounds GPU memory regardless of world size (per-chunk 3-D textures would not), and two cascades give fine light pixelation near the camera with full view-distance coverage at ~75 MB VRAM.

### GI is GPU flood-fill propagation; shadows are voxel raymarches (no shadow maps)
- **Q:** How are bounce light, AO, and shadows computed?
- **Choice:** GLSL **430 compute shaders** dispatched via `GraphicsEngine.dispatch_compute` each frame: (1) an *injection* pass writes direct radiance — sun/moon via short occupancy raymarch toward the light, point/area lights with distance falloff + occupancy march, emissive voxels from the palette; (2) an iterative *propagation* pass (ping-pong, N iterations/frame) spreads radiance through air cells and tints bounces by surface albedo — flood-fill GI, which also darkens corners (AO comes free from the same volume plus a 3³ occupancy term at shading). Sun/moon shadows at surfaces are per-fragment occupancy raymarches through the cascades — **no shadow maps anywhere**.
- **Why:** Flood-fill in a voxel volume is the established "Minecraft shader" GI shape: converges over a few frames, costs a fixed budget independent of light count, and is exactly representable in our occupancy windows. Shadow maps would add an entire second render path for a look the voxel march already gives in the same pixelated aesthetic.

### Volumetric fog is a froxel volume sampled in-shader (no post-process chain)
- **Q:** How do volumetric fog and god rays composite without a screen-space post pipeline?
- **Choice:** A camera-frustum-aligned **froxel 3-D texture** (160×90×64, exponential Z out to the fog far range) filled by compute: weather height-fog density, sun scattering with per-froxel occupancy shadow march (→ god rays), point-light and ambient/GI scattering; a second pass integrates along Z into accumulated (inscatter, transmittance). The **terrain and sky fragment shaders sample the integrated froxel texture directly** at their own depth and composite there.
- **Why:** Panda3D 1.10 has no post chain in this repo; in-shader sampling avoids depth-texture plumbing and a fullscreen pass, and the sky dome (drawn at far depth) gets exactly the same fog for free, so god rays cross the horizon seamlessly.

### Sky upgraded to a physical atmosphere that FEEDS the lighting system
- **Q:** The owner wants a physically simulated atmosphere (real-looking sunsets, scattered ambient) driving scene lighting, bigger textured sun/moon, fully procedural per seed.
- **Choice:** New headless `sky/atmosphere.py` (numpy Rayleigh+Mie single scattering) computes per-frame `SkyState` additions — `sun_radiance`, `moon_radiance`, `sky_ambient` (linear HDR RGB) — consumed by the lighting volume as the sun/moon injection colour and the sky-visible-cell ambient term. The dome fragment shader raymarches the same model per pixel (GLSL 330, no LUT buffers). Sun/moon discs ~2.5× larger with procedural textures (moon craters seeded from `world_seed`, phases kept). `SkyRendererComponent(external_lighting=True)` stops writing `terrain_root` colour-scale/Fog — the GPU pipeline owns surface light; `SkyState` remains the only sky↔lighting contract.
- **Why:** One scattering model evaluated twice (numpy for light values, GLSL for pixels) keeps the sky picture and the scene light physically consistent — sunset turns the *light* orange, not just the backdrop — while staying headless-testable and deterministic from seed.

### Normal/emission maps derive from the existing procedural textures
- **Q:** Normal mapping and emissive surfaces without hand-authored maps?
- **Choice:** `procedural` gains height→normal derivation (numpy Sobel on texture luminance) and an optional per-def emission map; the terrain shader builds its TBN analytically from the flat face normal's dominant axis (matching the planar UV projection) and emissive materials inject into the radiance volume via a material→(albedo, emission) palette sampled from each def's average colour.
- **Why:** Keeps "environment textures are 100 % procedural" intact — no asset files — and flat axis-aligned-ish faces make analytic tangents exact where it matters.

### stream_frame remeshes dirty chunks BEFORE loading missing ones
- **Q:** Craters stayed invisible for minutes — why?
- **Choice:** The 2-chunk/frame budget was consumed by missing-chunk loads first (desired set ≈ 1.2k chunks ⇒ ~600 frames of loading), starving dirty remeshes. Reordered: dirty remesh first, then loads.
- **Why:** Dirty means "an edit or relight the player is looking at"; the docs always promised edits remesh "within a frame or two". Also required for the new brush border-dirty propagation (neighbours of an edit remesh promptly).

---

## 2026-06-11 — GPU grass + zone volumes (session 5)

### Grass volumes are tagged boxes in a new `zones` package
- **Q:** Where does "grass grows here" live? (Also the foundation for biome regions later.)
- **Choice:** New foundation-layer package `fire_engine/zones/`: frozen `ZoneVolume` AABBs (tag `"grass"` now, `"biome"` reserved) in a `ZoneStore` registry that is `Saveable` (`save_key="zones"`, full-list delta vs a `mark_baseline()` snapshot). Matches the ZoneVolume concept in ARCHITECTURE §5.2.
- **Why:** Smallest shape that covers grass today and biome/snow regions next; volumes are a handful of dicts, so full-snapshot deltas beat diff machinery.

### Grass is GPU-only: blades derive from gl_InstanceID, CPU stores none
- **Q:** Owner wants Daggerfall/Morrowind-style swaying grass as real geometry, "rendered completely on the GPU — the CPU has no idea it exists".
- **Choice:** One shared 3-crossed-quad tuft Geom per volume drawn with `set_instance_count(density × area)`; the vertex shader hashes `gl_InstanceID` (lowbias32 chain, per-volume seed from `for_domain("zones","grass",id)`) into position/yaw/scale/sway-phase. `zones/grass_placement.py` mirrors the hash line-for-line so placement is headless-testable. The only CPU artifact is a tiny per-volume height-field texture (R = surface Z in the volume's z-window, 255 = no ground → shader-culled), re-baked on `TerrainEditedEvent`/`ChunkLoadedEvent` — craters cull grass. Weather drives sway uniforms (storm = bigger, faster). Distance fade shrinks blades over `[grass_fade_start_m, grass_fade_end_m]`.
- **Why:** ~10k blades cost one draw call and zero per-frame CPU; the hash mirror keeps Hard-Rule-2 determinism provable without a GPU.

### Grass inherits the terrain's lighting contract by scene-graph parenting
- **Q:** How does grass get radiance-cascade light + froxel fog without new per-frame plumbing?
- **Choice:** The grass root is parented under `App.terrain_root`, where `GpuLightingPipeline` already binds/refreshes every cascade/fog/celestial shader input; the grass fragment shader declares the same uniform names and samples GI + voxel-shadowed sun at the blade base, quantised to `light_quant_m`. GPU lighting backend only (component disables itself on "cpu").
- **Why:** Zero new uniform-sync paths to keep in step; grass light pixels match the terrain's by construction.

---

## 2026-06-11 — World-space procedural ground (non-repeating pixel-art)

### Ground albedo is generated in the shader from world-space noise, not a tiled texture
- **Q:** The baked 64×64 `grass_ground`/`dirt_ground` textures tiled every 1 m visibly repeat across the 1 km map and shimmer at distance (nearest filtering, no mipmaps). How do we get the same pixel-art look without repetition?
- **Choice:** The GPU terrain fragment shader (`world/shaders/terrain.frag`) now computes albedo in **world space**: a 2-octave integer-hash value noise of the dominant-axis-planar world coords, snapped to a `config.ground_texels_per_m` (≈16 → 0.0625 m) virtual texel grid for crisp pixels, posterised through a per-material **palette LUT** (`u_ground_lut`). The tiled `p3d_Texture0` albedo is no longer sampled (normal/emission maps still are). The face material id reaches the shader packed into **vertex-colour alpha** (`surface_nets.py`, `id/255`) so one NodePath-level shader needs no per-Geom uniforms; `extra_materials` adds flat LUT rows for debug/test materials (GI room).
- **Why:** Pattern is `O(1)` per fragment, never repeats anywhere, and stays pixel-art crisp. Packing material into alpha (terrain is opaque, alpha was unused) avoids per-Geom shader inputs and any vertex-format change.

### The shader's palette LUT is baked from the texture defs' own ramps (single source of truth)
- **Q:** How do we guarantee the procedural ground matches the hand-tuned `grass_ground`/`dirt_ground` art instead of drifting?
- **Choice:** `procedural/textures/ground_lut.build_ground_lut` bakes each material's `(PALETTE, THRESHOLDS)` — the very constants the defs export (`GRASS_PALETTE`, `DIRT_PALETTE`, …) — into a `(rows,256,4)` LUT via the same `searchsorted(side="right")` posterise rule, uploaded with `to_field_texture` (nearest, clamp). The shader reads `lut[material][noise]`.
- **Why:** Both the baked preview and the in-shader ground go through one posterise definition, so they agree bucket-for-bucket; a palette tweak updates both with no GLSL edit.

### Distant shimmer is killed with a derivative fade, not mipmaps
- **Q:** Hard pixels alias badly when a texel shrinks below a screen pixel at distance.
- **Choice:** The shader measures `fwidth(world_planar) × texels_per_m` and lerps the noise value toward the dominant mid bucket (0.5) as texels drop below ~1 pixel, so far ground resolves to a flat mean colour. No mipmap chain needed for the procedural ground (the LUT has none).
- **Why:** A procedural pattern has no prefiltered mip pyramid; collapsing toward the mean is the cheap analytic equivalent and matches the "limited palette" aesthetic. **(Superseded 2026-06-11 below — the collapse-to-mean made distant ground a flat "sea of green"; replaced by per-octave LOD.)**

---

## 2026-06-11 — Lighting resolution, far cascade, edit responsiveness, ground LOD

### Ground distance handling: per-octave LOD, not collapse-to-mean (supersedes the entry above)
- **Q:** The derivative fade above lerped *all* ground noise toward one mid colour once a pixel spanned >2 texels, so the ground went from crisp pixel-art to flat green only ~3–5 m ahead (worst on grazing ground, where `fwidth` is large).
- **Choice:** `terrain.frag::groundNoise` now sums **three hash octaves** (fine 1×, mid 4×, macro 16× larger texels); each octave fades toward 0.5 by its **own** screen footprint (`smoothstep(1.0, 2.5, mpp*texels)`, `mpp` = world m/pixel). Fine detail drops first, then mid, leaving the macro colour patches — distant ground stays varied. Each octave's mean is 0.5 so the posterise buckets stay balanced. `ground_texels_per_m` restored to 16 (crisp near pixels; the LOD now prevents the distant shimmer that previously forced it down to 8).
- **Why:** Real mip-like band fade degrades detail gracefully toward the horizon instead of an all-or-nothing collapse — needed for the eventual 1 km render distance.

### A third, coarse, FAR radiance cascade instead of a flat-ambient cutoff
- **Q:** Beyond cascade 1 (~96 m) surfaces fell back to flat sky ambient with full sun visibility (no shadows/GI). While moving, the leading edge of newly-streamed chunks — and the GI test room as you backed away from it — popped to flat/unlit ("the lighting breaks when far away").
- **Choice:** Added **cascade 2** (`light_c2_cells=64`, `light_c2_cell_m=8.0` → 512 m box). It reuses the existing camera-centered `VolumeWindow`, the off-thread `CascadeAssemblyWorker`, and the inject/propagate loops verbatim (they iterate `self.cascades`), so "bake far chunks on a separate thread at a lower resolution" added **no new subsystem** — just a third cascade entry, a `u_c2_*` uniform block, and a third branch in `sampleCascades`. Chose 64³ @ 8 m (not 96³) to keep the assembly chunk-span (~33k vs ~110k coords) and VRAM modest given the flat world's wasted vertical extent.
- **Why:** The documented upgrade path (lighting.md gotcha #7). Graceful low-res far lighting matches the owner's stated solution and lays groundwork for `view_distance_chunks` → 1 km. At the current 96 m streaming radius it is reached only by geometry the trailing (hysteresis-lagged) cascade-1 window hasn't caught up to.

### Light-pixel granularity: `light_quant_m` 0.25 → 0.0625 (8×8×8 per voxel)
- **Q:** The owner wanted finer "lighting voxels" (up from 2×2 per face), found changing config seemed to have no effect, then found 4×4 (0.125) still looked too blocky up close.
- **Choice:** Settled on `light_quant_m = 0.0625` (8×8 per 0.5 m face). The value was already plumbed (`u_quant_m`); the apparent "no effect" is that the radiance **data** lives at the 0.5 m cascade cell with trilinear filtering, so re-sampling it on a finer grid makes the light-pixel blocks smaller/smoother but does not add lighting detail. Recorded that the real data-resolution lever is the **cascade cell size** (`light_c0_cell_m`), which `light_quant_m` cannot exceed in true detail.
- **Why:** Smallest change that makes the up-close light-pixel grid as fine as the owner wanted without a VRAM-heavy finer cascade; the cascade-cell lever is documented for when crisper near lighting (more actual detail) is wanted.

### Brush edits relight synchronously the same frame (kills the "black then lit" crater)
- **Q:** Explosions/digging showed the new crater black for a frame or two before it lit up.
- **Choice:** `GpuLightingPipeline._apply_edits_sync` re-slices + uploads + re-injects the hit cascades **synchronously** the frame a `TerrainEditedEvent` arrives, instead of waiting for the 1–2-frame async reassembly (during which the stale occupancy still marks the crater solid → shadowed → black). Edits are discrete events, so the synchronous gather is affordable; cascades already mid-flight fall back to the batched `_pending_coords` async path.
- **Why:** The async assembly worker exists to smooth *continuous* fly-around recenters, not one-shot edits — for an edit, same-frame correctness beats anti-stutter latency that doesn't apply.

---

## 2026-06-11 — Terrain "z-fighting" root cause: quantise-after-filter, fixed by posterising per tap

### The shimmer was the palette LUT re-hardening the filtered noise, not the normal map or the light grid
- **Q:** Owner reported persistent "z-fighting"/shimmer in the terrain textures while moving, after the previous session's albedo Nyquist-LOD work did not cure it. The standing hypotheses were the nearest-filtered normal map (A) and the 0.0625 m light-quant grid (B).
- **Choice:** Built a *working* motion-shimmer meter first (`tools/shimmer_probe.py`: sub-pixel yaw sweep written through `FlyController.yaw`, multi-frame settle per pose, `RTM_copy_ram` capture, static + positive controls so a broken harness self-reports — the prior session's diff attempts failed silently). Measurements: disabling the normal map changed nothing (A refuted); LOD-clamping the quant grid changed nothing on open ground (B not the cause there); a constant-albedo run dropped even the 8 px positive control to zero → all motion contrast lived in the albedo path. Root cause: `terrain.frag` averaged the band-limited `groundNoise` over 4 supersample taps and *then* pushed the single averaged value through the hard posterising palette LUT — a quantiser after the filter, so pixels near a palette-bucket edge popped a full palette step on every sub-pixel camera move. Fix: run the LUT lookup **inside** the 4-tap loop and average the resulting colours. Far-ground flip fraction 0.00805 → 0.00028 (~27×); near field unchanged (all taps share one texel up close).
- **Why:** Filtering must come after quantisation or the quantiser undoes it — now recorded as `world.md` gotcha 21 for every future palette/posterise/dither stage. Measure-first beat the plausible-but-wrong fix list: both inherited hypotheses were innocent on open ground.

### Residual horizon twinkle is geometric edge aliasing, deliberately left to an owner call
- **Q:** After the fix the probe's only hot band is the terrain-vs-sky silhouette line.
- **Choice:** Left `render.set_antialias(M_none)` (the explicit "retro look" choice in `App.__init__`) untouched; documented MSAA (`framebuffer-multisample` PRC + `M_multisample`) as the lever if the owner wants the silhouette smoothed. MSAA only touches polygon edges, so the pixel-art interiors would stay crisp.
- **Why:** Aesthetic default flips belong to the owner; the measured texture shimmer — the actual complaint — is gone.

---

## 2026-06-11 — Lighting overhaul: recenter pops, load latency, GI room, fog reach

### Radiance shift on cascade recenter (kills the worst recenter pop)
- **Q:** When a cascade window recenters (camera flies past the hysteresis margin), the geom texture and origin uniform are committed together, but the two radiance ping-pong textures still hold the *previous* window's converged field at the *old* origin — so for the many frames `light_prop_iters`=2/frame needs to re-converge, the GI is read at the new origin while holding old, spatially-misaligned light. That misalignment was the most visible fly-around pop.
- **Choice:** New `shift.comp` compute pass (`SHIFT_COMPUTE`, exported like the others via `core.shader_source.load_glsl`). On commit, when the origin moves, copy the current radiance (read side = `ping`) into the other ping-pong texture **shifted by the integer cell delta `new_origin − old_origin`** (`dst[c] = src[c + u_shift]`, `vec4(0)` for source cells off the previous window), then swap `ping` so the next propagate reads the spatially-aligned field. Also set a per-cascade `boost_frames = 4` so the cascade runs +6 propagate iterations for 4 frames, re-converging the newly-exposed border band fast.
- **Why:** Shifting the already-converged field is nearly free (one 16f texture copy at recenter granularity) and the converged GI now *follows* the window instead of being thrown away and rebuilt over ~½ s. rgba16f image bindings match inject/propagate; cascade 0 stays 1 cell = 1 voxel binary — the shift is origin-delta only and never touches geometry.

### Boot warmup burst (kills the 1–2 s dark load-in)
- **Q:** On boot the world brightened over ~1 s as the GI flood-fill filled in at 2 iters/frame.
- **Choice:** On the boot frame only, after the synchronous assemble + first inject + normal propagate, run a one-shot **48-iteration propagate burst across all cascades** before the first rendered frame.
- **Why:** The field converges in one frame instead of over dozens; 48 iters is a one-time cost paid while the world is already assembling, invisible to steady-state perf.

### Cascade-2 keepalive: ChunkBlockCache + wider recenter hysteresis
- **Q:** The coarse far cascade (8 m cells, 512 m box, ~33k chunk coords) recenters every 64 m but its gather takes longer than that at flight speed, so its committed volume permanently lagged — c2 was *always* mid-assembly.
- **Choice:** (1) Wire the parallel agents' `ChunkBlockCache` (owned by the worker) through every assembly path — async jobs get it automatically; the synchronous boot/edit paths pass `worker.block_cache` so boot warms it; terrain edits call `worker.invalidate_chunk(coord)` so a stale pre-edit mini-block can't keep a crater dark. (2) Widen c2's recenter hysteresis to `margin_cells=16` (128 m, vs the default 8 cells = 64 m) so it recenters half as often.
- **Why:** The cache restores throughput (cache hits skip the per-chunk downsample, the dominant cost of the 33k-chunk gather); the wider margin halves the recenter *rate*. Together c2 keeps up instead of perpetually lagging. The cache is palette-independent and skips cascade-0 (k==1, no downsample to amortise).

### Cascade-0 reassembles immediately on a near chunk load (kills the 0.25 s unshadowed-then-pop)
- **Q:** `_LOAD_REASSEMBLE_INTERVAL_S = 0.25` batches newly-streamed chunks, so a chunk loading inside the small near cascade rendered unshadowed for up to 0.25 s, then popped to lit.
- **Choice:** A pending coord that intersects **cascade 0** triggers an immediate c0 reassembly (its ~27-chunk gather is cheap); cascades 1/2 keep the 0.25 s batch. The shared `_pending_coords` clear stays gated on the batch interval so a c0-immediate pass can't make the mid/far cascades miss a coord.
- **Why:** Near terrain is what the player is looking at; far-cascade relight lags invisibly so it can stay batched. Smallest change that special-cases only the cheap, visible cascade.

### fog_far_m 160 → 192 (fog covers the cascade-1 range)
- **Q:** Fog cut off at 160 m while cascade 1 covers 192 m, so the fog edge read as a pop at the cascade-1 boundary.
- **Choice:** `fog_far_m = 192.0` (config.toml + `core/config.py` default). Froxel counts unchanged.
- **Why:** Align the fog far plane with the cascade-1 box so there is no visible fog cutoff inside the lit range; keeping the froxel count fixed means the same slices stretch slightly, negligible cost.

### GI room exposure: lower the panel light + emission so the coloured bounce reads
- **Q:** The Cornell room interior washed out to flat gray (AreaLight intensity 6 + emissive panel (8,7.2,5.6) in a closed white box + auto-exposure → blowout that hid the red/green bounce).
- **Choice:** `_GI_PANEL_INTENSITY` 6.0 → 2.0 and `_GI_GLOW_RADIANCE` (8,7.2,5.6) → (4,3.6,2.8) (main.py only). The lower white direct-fill lets the red/green wall inter-reflection through; measured the warm-left / cool-right bounce gradient on the `--inside` shot (left wall R−G ≈ +7 vs uniform ≈ +3 before).
- **Why:** Auto-exposure normalises absolute brightness, so the lever is the *ratio* of coloured bounce to white direct fill — cutting the white AreaLight (not just total brightness) is what surfaces the bounce. NOTE/risk: the `tools/screenshot.py --inside` framing jams the camera near the back wall and its right side catches sky through the doorway/roof openings, which auto-exposure still meters on — the red-wall bounce is now clearly visible but the green wall is partly washed by that sky light. A cleaner verification framing (square-on to a coloured wall, no sky opening in frame) would show both walls; left as an owner call since it is a tooling-framing limit, not a lighting one.

### HDR offscreen buffer + post-processing (the sun finally blooms)
- **Q:** The sun read as a hard white disc, the sunrise as a flat gradient, and fog as a grey wall over the sun. Root cause: every surface shader ACES-tonemapped + clamped to [0,1] internally, destroying all >1.0 radiance before anything could use it.
- **Choice:** Render the scene into a linear **RGBA16F** float buffer via `FilterManager` and move tonemapping into a post-process composite (`world/post_process.py`). A single `u_hdr_output` shader-input on `render` switches every surface shader (terrain/sky_dome/grass/cloud) between **emit-linear-HDR-with-exposure-applied** (post on) and the legacy in-shader tonemap (post off). Exposure stays in the surface shaders (not the composite) so bloom operates on the exposed signal. Requires `textures-power-2 none` (loaded only when post is on) or the full-window NPOT render target is padded to a power-of-two and the scene renders into a sub-rectangle.
- **Why:** HDR is the foundation for bloom/flare/god-rays and for the atmosphere reading correctly — the physical Rayleigh+Mie scatter only looked like a gradient because it was clamped. The `u_hdr_output` gate keeps the exact legacy look as a one-flag fallback for weak GPUs (and on float-buffer allocation failure the pipeline self-disables to it).

### Volumetric clouds replace the boxy DDA clouds
- **Q:** Owner wanted true volumetric clouds (and the sun to punch through them), not the Minecraft-style 2D-DDA box clouds.
- **Choice:** A second camera-centred inverted "cloud dome" sphere (reuses the sky-dome geometry for a per-pixel world view direction) whose fragment shader analytically intersects the horizontal cloud slab and raymarches it sampling baked tileable 3-D noise (`sky/cloud_noise.py`: Perlin-Worley base + Worley FBM erosion + detail volume). Self-shadow light-march (Beer + powder), HG forward-scatter phase; output premultiplied `(scattered, transmittance)` with a `src + dst·srcAlpha` over-blend (bin background:15) so a bright sun bleeds through thin cloud and thick cloud occludes it; terrain (opaque) draws over and occludes clouds behind it. The cloud sun term is **2×-boosted** to compensate `SkyState.sun_radiance` already being cloud-dimmed at the ground (the cloud tops see the undimmed sun). Coverage→density threshold `mix(0.95, 0.55, coverage)` tuned via a debug coverage view so partial weather leaves real blue gaps.
- **Why:** Analytic slab-intersection on a dome covers the whole sky to the horizon with no slab-quad extent limit. The noise bake is numpy/headless + deterministic (`for_domain`), so it is disk-cached under `saves/cloud_cache/` (keyed by seed+size) — the ~1.7 s 64³ bake happens once per seed, not per boot.

### Graphics quality presets ([graphics] table)
- **Q:** The HDR buffer + bloom pyramid + volumetric raymarch are too heavy for the integrated-GPU dev machine at full quality; owner wanted a config to dial it down or off.
- **Choice:** A `[graphics]` config table with `off/low/medium/high` presets (`core.config.resolve_graphics_preset`) expanding into flat `gfx_*` fields; explicit `gfx_*` keys override the preset. `off` = legacy path (no HDR buffer, no clouds); `low` = HDR+bloom+cheap clouds, no flare/god-rays/FXAA; defaults == `high`. Every effect pass is individually gated and drops its composite contribution to 0 when disabled.
- **Why:** One knob (`preset`) covers the common case; per-field overrides cover tuning. Verified all three presets render without crashing and degrade as intended.

---

## 2026-06-11 — Crater shimmer round 2: texel-coverage albedo filtering, analytic footprint, MSAA

### Crater dirt walls still boiled after the posterise-per-tap fix; cause was camera-dependent sample positions
- **Q:** Owner: flat ground and the GI room were fixed, but the dirt walls of a fresh blast crater still "z-fight". Probe isolation (open surface crater 10 m ahead): light quantisation innocent (no-quant identical), constant albedo collapsed the band → albedo again. Crater walls face the camera (cos i ≈ 1 → small footprint → no octave fade), so they render full-contrast hash texels — and the 4 supersample taps SLID continuously through that field with the camera, popping a quarter palette step at every texel crossing. Flat ground had hidden this behind grazing-angle octave fade.
- **Choice:** Replaced sliding-tap supersampling with **analytic texel-coverage filtering**: evaluate the noise stack only at the 4 nearest fine-texel centres (fixed world points → each corner's posterised colour is camera-invariant), posterise per corner, blend the colours by the pixel footprint's coverage of each texel. Output is a continuous function of surface position — popping is impossible by construction; texel edges become ~1 px AA ramps; interiors saturate to one flat palette colour (pixel art intact, verified by stills). Same cost (4 noise+LUT evaluations). Crater band 0.0663 → 0.0379 (threshold 0.04), flat ground 0.0000.
- **Why:** Two stacked quantisers (hash texels, palette LUT) can only be temporally stable if every quantiser input is anchored to fixed world points and all camera dependence lives in continuous blend weights.

### fwidth() banned in the terrain shader; footprint is analytic (dist × px-angle / cos i)
- **Q:** Screen-space derivatives are computed on 2×2 quads; on the faceted mesh, quads straddling facet edges extrapolate the wrong plane and the derivative explodes — every fwidth-driven LOD/AA term popped along the dense small triangles of crater rims.
- **Choice:** New per-frame uniform `u_px_rad` (lens FOV / window width, set in `update_surface_inputs`); the shader computes `mpp = dist * u_px_rad / max(|dot(view, n)|, 0.18)` — exact for planar facets, stable everywhere. The light-quant LOD snaps to power-of-two multiples of `u_quant_m` from the same `mpp` (nested, world-anchored lattices; a continuously varying cell size re-seats every boundary = its own shimmer).
- **Why:** The faceted-mesh art style guarantees pathological derivative quads; analytic geometry is exact and free.

### Geometry-edge AA via config `msaa_samples = 4` (overrides the earlier blanket "AA off for retro look")
- **Q:** With surfaces filtered, the probe's residual was facet-silhouette / horizon twinkle — rasterisation aliasing, even with constant albedo.
- **Choice:** `msaa_samples` config (default 4, 0 = off): `framebuffer-multisample` PRC before window creation + `AntialiasAttrib.M_multisample`. Edge-only — interiors are single-sample, crop comparison pixel-identical, no measured fps cost at 720p. Crater pops (threshold 0.12) 0.0123 → 0.0090.
- **Why:** MSAA is the only cheap fix for silhouette aliasing and provably does not soften the pixel-art interiors — the original "AA off" intent (crisp texels) survives. Owner can set 0 to compare.

---

## 2026-06-11 — Wind field system (fire_engine/world/wind/)

### Spectral seeded gust modes over an accumulated Brownian random walk
- **Q:** The owner asked for a Brownian-motion wind field driving grass/flags/cloth/particles/physics/audio. A literal accumulated random walk would need its full grid state in every save, desync on reload, be unreproducible in bug reports, and could not recenter analytically as the player moves.
- **Choice:** The field is a sum of 12 seeded spectral "Brownian-band" modes (wavelengths 20–120 m, amplitudes ∝ 1/λ red-noise) whose phases advance with game time and **advect downwind** — a pure function of (world_seed, game_time, world_position), drawn once from `for_domain("wind", "gusts")`. No Saveable anywhere in `wind/`; the venturi correction is likewise a pure function of (terrain snapshot, region origin).
- **Why:** Bit-reproducible across processes and save/load with **zero save bytes** (the `sky/weather.py` ethos), free analytic recenter, and visually indistinguishable from true Brownian gusting at these wavelengths — the quasi-periodicity of a 12-mode red-noise sum is imperceptible. Tested: in-process + subprocess determinism, crest advection ≈ `mean·dt`.

### 2.5-D wind field (2-D grid + analytic vertical profile) over a 3-D volume
- **Q:** Store wind as a coarse full-3-D volume (e.g. 64×64×16) or a 2-D horizontal field?
- **Choice:** 64×64 × 4 m horizontal grid (256 m region; channels vx/vy/turbulence) + analytic boundary-layer profile `clamp(((z−z_ground)/z_ref)^0.18, 0.35, 1.6)` + analytic venturi updraft. Uploads as one 32 KB RGBA16F texture per frame (`T_half_float` — `T_float` asserts on fp16 buffers).
- **Why:** Covers every current consumer (ground grass, motes, leaves, ball-on-plane, future tall flags via the profile) at ~1/16 the memory/eval/upload cost of 3-D; nothing samples mid-air detail that a 4 m-per-layer Z axis could resolve anyway. A future 3-D corrector registers as a `WindModifier` without changing the `sample()`/texture contracts — the same seam reserved for volumetric-weather gust fronts.

## 2026-06-11 - Flora system (flowers / bushes / trees)

### Sprite-billboard trees over voxel trees
- **Q:** Trees could be carved into the voxel terrain as blocks (Minecraft / Vintage Story logs + leaf voxels) or rendered as instanced crossed-quad sprites (Daggerfall billboards).
- **Choice:** GPU-instanced crossed-quad sprites with seeded procedural atlases (`tree_sprite`, 3 condition variants), placed by the grass hash-chain idiom inside `"trees"` zone volumes; bushes (`"bushes"`, `bush_sprite`) and wildflowers (`"flowers"`, `flower_sprite`) the same way. One table-driven `FloraRendererComponent` renders all three kinds; `flora.vert` = the grass chain + an h5 atlas-variant link (mirror: `zones/flora_placement.py`).
- **Why:** Daggerfall billboards are literally in the art direction; zero terrain/mesher/save coupling (voxel trees would dirty chunks, deltas and the mesher); zero CPU per-plant state and zero save bytes (pure function of seed + volume); the wind texture sways canopies per-plant with a two-uniform shape change (`u_sway_gain`/`u_sway_pivot`); and `"trees"` volumes already feed the wind system's leaf litter, so a forest gets falling leaves for free. Voxel/destructible trees can land later behind the same volume-registration seam without touching zones/wind contracts.

---

## 2026-06-11 — Lighting: rendered shadow resolution + visible GI

### Boxy 2 m shadows fixed by a penumbra-gated per-fragment refinement march, not by growing cascade 0
- **Q:** Owner: shadows render as soft ~2 m (4×4-voxel) boxes even though the lighting is computed at 0.5 m — rendered resolution must match computed resolution. GPU readback (`tools/light_probe.py`) proved cascade-0 `u_vis` data is crisp (1.00 → 0.00 across one cell); the boxiness is render-side: beyond ~17 m (cascade-0's cross-fade band, and ALL ground when flying high since the c0 box centres on the camera in 3-D) the surface samples cascade 1's 2 m cells and trilinear filtering smears the edge over a full cell. Options: enlarge c0 / add cascades (memory + assembly cost, only moves the boundary), shadow maps (parallel pipeline, against the voxel-light design), or re-resolve per fragment.
- **Choice:** `terrain.frag::refineVis` — only when the sampled celestial visibility is in the penumbra band (`vis ∈ (0.02, 0.98)`), march occupancy from the quantized light-pixel probe through the c0→c1→c2 chain (28/24/12 single-cell steps, nearest-cell taps via `occCell`) × the analytic dynamic-occluder `boxVis` (mirror of `inject.comp`; the box uniforms are now also bound to the surface shader, zero-filled when empty — Panda asserts on unbound GLSL arrays).
- **Why:** The data was never wrong — only the reconstruction. Refining only penumbra pixels makes the cost proportional to shadow-edge screen area (fps unchanged, 41–50 across scenes); shadow edges resolve at the light-pixel grid at ANY distance, which is literally the owner's acceptance criterion. Verified: crisp cube-shadow parallelogram from 45 m altitude, voxel-stepped crater rim shadows at 16:00.

### Invisible GI fixed by splitting bounce into a full-strength direct texture + gain-compensated flood forcing
- **Q:** First-bounce GI was implemented but invisible (~4 % of ambient; bounce on/off image diff 0.7/255). Root cause: the propagate fixed point passes BROAD fields (skylight) at full strength but squashes LOCALIZED sources by ≈ (1−decay) — 3–8× at c0 — so the bounce was computed, then drowned. Raising `light_bounce_strength` past 1 would be unphysical and still loses the squash-shape; lowering decay changes GI reach everywhere.
- **Choice:** Two-fold: (a) new per-cascade `bounce_direct` rgba16f volume — INJECT writes the un-gained localized sources (first bounce + emissive leak + dynamic lights), `terrain.frag` samples it (`u_c0_bounce`/`u_c1_bounce`, cross-faded like the cascades) for crisp contact GI; (b) `u_gi_gain = 0.6/(1−decay)` pre-amplifies the GI terms (NOT skylight, NOT dynamic lights) inside the flood-fill forcing so spread GI converges near physical strength. The gain scales the forcing, not the diffusion operator — contraction (spectral radius < 1) and therefore stability are untouched.
- **Why:** The squash is a property of the diffusion equilibrium, so the only stable lever is the source term; the direct texture restores the high-frequency detail no amount of forcing gain can (diffusion blurs by design). Verified by radiance readback: ground-air bounce contribution red +40 %, green +30 %, blue ~0 with bounce on vs off (was ~+9 %); night torch shows the warm pool + green grass-bounce ambiance. NOTE: auto-exposure normalizes broad ambient shifts, so bright-scene A/B screenshots stay subtle even when the field moves +40 % — judge GI work with `tools/light_probe.py`, not eyeballs.

---

## 2026-06-12 — 3-D skeleton trees replace billboard trees

### Real instanced 3-D trees/bushes (Dynamic-Trees style); sprites demoted to far-LOD impostors
- **Q:** Owner: 2-D billboards are wrong for anything bigger than flowers/grass — trees and bushes need real 3-D geometry in the style of Minecraft's *Dynamic Trees* mod (tapering trunk, branches at script-controlled near-90° angle sets, procedural leaves on branch tips), with billboarding kept ONLY for far-distance LOD. New species must be authorable as plain Python scripts calling a helper library ("Unreal's node graph, but in code") so AI agents can build the asset catalogue.
- **Choice:** New `procedural/flora/` subpackage: `SkeletonBuilder` (`trunk`/`branches`/`skeleton` with `pitch_set`, `yaw_mode`, `length_scale_by_height`, droop/upturn knobs) → validated `TreeSkeleton` + `LeafClusters` → square-prism + crossed-quad `TreeMesh` (per-vertex sway weight in `color.a`) → 64×64 bark/leaf atlas + headless software-rasterised impostor strip, all bundled per species into a cached `TreeVariantSet` **pool of unique meshes per world seed** (oak 8, others 6) by `TreeSpeciesDef.generate`. Species = one script each (`gnarled_oak`, `dead_tree`, `scrub_bush`, `berry_bush`; guide: `docs/content/tree_species_authoring.md`). Placement moved CPU-side (`zones/tree_placement.py`): jittered grid with guaranteed ≥ 0.3·cell spacing, height-field Z, weighted `species_mix` params, packed into an RGBA32F data texture read by `texelFetch` (`world/tree_renderer.py` + `tree.vert`/`tree.frag`/`tree_impostor.vert`). Mesh ↔ impostor crossfade over config windows (trees 110–140 m, impostors out 300–380 m). `FloraRendererComponent` shrank to flowers-only; `tree_sprite`/`bush_sprite` defs deleted.
- **Why:** Variant pools give visible 3-D parallax and per-species silhouettes at instanced cost (oak ≈ 1.2 k verts × shared Geom per variant); CPU placement kills the GLSL hash-chain mirror discipline for trees, guarantees no twin trunks, and gives the game knowable trunk positions for future collision/forage — the remaining CPU↔GPU contract is one pinned texel layout. Authoring-as-Python is the strategic point: species scripts are data the owner's AI agents can write, review and preview headlessly (`tools/preview_tree.py`) without touching engine code. A bush is a tree with a 0.15 m trunk — one system, both plants. Sprites survive exactly where they're correct: past 110 m, where a billboard is indistinguishable and 500× cheaper.

---

## 2026-06-12 — Lighting: flood-fill GI deprecated; ray-marched gather (voxel-realistic lighting)

### Replace the flood-fill propagate pass with a per-cell ray-marched GATHER
- **Q:** Owner: deprecate flood-fill lighting entirely — the engine goes solely voxel-based realistic lighting. Symptoms driving it: (a) flat sky-blue ambient filled interior floors (the flood diffuses skylight through any opening into every cell, walls only slow it); (b) bounce GI was visible but far too weak, with no red/green wall bleed in the Cornell test room; (c) the gain machinery (`u_gi_gain`) needed to keep localized sources alive was a fudge fighting the diffusion's contractive squash.
- **Choice:** `gather.comp` replaces `propagate.comp`. INJECT now writes `u_source` (surface radiosity proxies: celestial first bounce × **1/π** + emissive leak) and `u_lit` (dynamic-light direct in air) — no skylight, no `u_direct`, no `u_gi_gain`. GATHER: per air cell, `light_gi_rays` (16) fibonacci-sphere rays march occupancy with transmittance; escapes gather `sky_ambient × skyW(z) × 7/3` (mean of skyW over the sphere is 3/7 → open cell ≡ sky_ambient, magnitude parity with the old skylight); hits gather `u_source` at the last air cell + feedback `u_prev × albedo × light_bounce_strength` (multi-bounce + colour bleed). Own cell adds `u_source + u_lit` at full strength (crisp contact GI); the fan skips its first marched cell to avoid double-counting. Runs ONLY on inject (2 ping-pong iterations, `light_gi_iters`); the per-frame propagate loop, boot 48-iter warmup and recenter boost-frames machinery are deleted — steady-state per-frame lighting GPU cost is now froxel fog alone.
- **Why:** Sky reaching a cell only through real openings makes the blue-floor fill impossible by construction instead of tuned away; bounce strength comes from actual visible-surface solid angles instead of a diffusion equilibrium, so the Cornell bleed exists at room scale; and the result is a pure function of the injected fields — instant response, no convergence delay, fewer moving parts. Energy accounting that made it stable (found via `tools/_gi_room_probe.py` readback, room was 2.3× hot): the 1/π on `u_source` (the old ×0.45 was this in disguise) and `u_lit` added once per cell, never re-gathered off surfaces (in-air irradiance is not surface emission; its wall bounce arrives via the albedo-tinted feedback term).

### Soft penumbra via a cone of refinement rays
- **Q:** Owner: the refined shadow edges don't transition smoothly — hard voxel stairs on the test-room wall.
- **Choice:** `refineVisSoft` replaces the single-ray penumbra refinement: four `refineVis` marches jittered inside a cone of half-angle `light_penumbra_deg` (2.5°, bound as `u_penumbra_tan`), averaged, marching from the UNQUANTISED surface probe (the light-pixel snap stays for everything else).
- **Why:** The voxel stairs are the projected silhouette of 0.5 m occupancy — exact but harsh; averaging a small direction cone is the physically-shaped fix (penumbra width grows with occluder distance) and costs 4× marches only inside the penumbra band. The unquantised probe keeps the gradient continuous across light pixels; determinism is untouched (fixed offsets, no per-frame jitter).

---

## 2026-06-12 — Trees: individual CA-grown leaves replace billboard leaf clusters

### One leaf card per leaf, grown by a cellular automaton, still one mesh per variant
- **Q:** Owner: the crossed-quad foliage-blob billboards on the 3-D trees read as big flat cards — leaves should be individual, procedurally generated and procedurally placed (the Dynamic-Trees cellular idea), while staying batched as one big mesh on the GPU for speed.
- **Choice:** `leaves_at_tips` (replaces `leaf_clusters_at_tips`): branch tips seed hydration `rounds` into a coarse cell grid; each CA round spreads `max(self, neighbours − 1)` over the 6 axis neighbours (vectorized roll/maximum, sway field propagated alongside); surviving cells sprout 1–2 jittered leaf cards with rim-thinned probability and a deterministic `max_leaves` cap. `mesh_leaves` (replaces `mesh_leaf_clusters`): ONE small quad per leaf with an upward-biased random normal (15–70° off vertical), merged into the variant mesh — hundreds of leaves, unchanged one-draw-per-variant instancing, no shader changes. `atlas.leaf_texture` now draws a single pixel-art teardrop leaf (midrib, side shading, ragged `hole_thresh` edge, berries near the base); the impostor's leaf pass became a vectorized point-scatter + diamond dilation of the actual leaf cloud. Budgets: oak ≤ 420 leaves ≈ 2.5 k verts/variant; dead snags ≤ 2 micro-tufts (some bare).
- **Why:** The canopy SHAPE now emerges from the branch structure cell by cell (a snag tufts, an oak domes) instead of from blob radii, per-leaf normals give the dappled Lambert a single billboard can't, and the wind path is untouched (per-vertex sway weights, now per leaf). Generating leaves AS baked mesh data rather than on-GPU geometry keeps the owner's batching goal literal: the whole canopy is static vertices in one instanced Geom. Gotcha encoded in the authoring guide: `rounds=1` cannot spread (seeds start at `rounds`, neighbours get `rounds − 1`) — single-cell tufts; use `rounds=2` + small `cell_m`.

### GI gather de-noised by a phase-tiled ray fan + air-masked smooth pass
- **Q:** Owner: the gathered lighting is noisy — blotchy patches on the GI-room walls/roof and rainbow confetti on the ground at night. Cause: 16 rays/cell with per-cell random rotation → adjacent cells' fans disagree wherever visible sources have high contrast. Options: more rays (√N gain only — 32 rays buys 1.4× for 2× cost), origin jitter (doesn't fix direction-set disagreement), or structure the noise so a cheap filter can remove it exactly.
- **Choice:** Two matched halves (background-agent work, verified by readback): (a) `gather.comp` replaces the free hash rotation with `phase8` — 8 phases on a 2×2×2 world-cell tile offsetting both azimuth and fibonacci polar stratum, so the 8 neighbouring fans interleave into one stratified 8×16-direction set; (b) new `smooth.comp` (`light_gi_smooth_passes` = 1) box-filters ONLY the ray-gathered component over the 3³ air neighbourhood (own contact term `u_source+u_lit` subtracted before, re-added after; solids excluded; radius-1 kernel can't cross a ≥2-cell wall).
- **Why:** Any 3³ block contains all 8 phases, so the blur *completes* the stratified sequence rather than losing information — measured 5–7× hf-noise reduction on the wall bands (9× open ground) for one extra cheap pass, with interior mean within 0.5% and the red/green bleed ratios bit-identical. Metric gotcha recorded in lighting.md 16: measure noise on the ray component minus local mean; raw radiance stddev is dominated by the crisp contact term and real gradients.

---

## 2026-06-12 — Rendering: one shared lit-surface GLSL contract (lit_surface.glsl)

### Extract the lighting contract into an included library instead of hand-copied shader code
- **Q:** Foliage looked washed out vs terrain: grass/flora/tree fragments carried three hand-copied, simplified, drifted versions of the terrain lighting code (C0/C1 hard switch with no far cascade or cross-fade, no shadow refinement, no AO — and tree.frag/flora.frag lacked the `u_hdr_output` gate, so the HDR pipeline double-tonemapped them). Buildings and NPCs are coming and need the same detailed lighting; a fourth/fifth copy would drift the same way.
- **Choice:** `world/shaders/lit_surface.glsl` — extracted verbatim from `terrain.frag` (cascade uniforms + `sampleCascades` cross-fade + `refineVis`/`refineVisSoft` + `litQuantSize`/`litQuantPos`/`litAo`/`litFog`/`litFinish`) — included by every lit-surface fragment via a new `//#include "<file>"` directive expanded in `core/shader_source.py::load_glsl` (one level, same `shaders/` dir, begin/end markers, no `#line`). The directive is a valid GLSL comment so sidecars lint standalone. The expensive refinement march compiles only under `#define LIT_REFINE` and is runtime-gated by a per-object-root `u_refine` uniform (terrain pins 1.0; foliage binds `gfx_foliage_shadow_refine`, preset-wired, so the iGPU can turn the march off without shader variants). `tests/test_lit_surface.py` pins the contract: exactly one canonical `sampleCascades` per composed fragment, hdr gate present, no local redefinitions, fade bands / march steps, and the 15-of-16 fragment-sampler budget on terrain.
- **Why:** Drift is the disease (the double-tonemap bug proved it) — sharing the text makes terrain and objects incapable of diverging, and "light a new object" becomes a 6-line recipe instead of a 250-line copy. Compile-time tiering (define) keeps cheap shaders free of the march's uniform/sampler budget; runtime gating (uniform) avoids shader-variant management and lets graphics presets flip it. GL 3.3 guarantees only 16 fragment samplers and terrain sits at 15 — the budget pin keeps the library from ever pushing a consumer over on the owner's iGPU.

---

## 2026-06-12 — Lighting: trees cast into the light grid (static occluder splats)

### Splat baked tree placements into the cascade geometry volumes instead of voxelising trees
- **Q:** Trees received light but did not block it — noon-bright ground under every canopy, no crown self-shadow. Options: voxelise tree meshes into the terrain field (heavy, couples meshes to terrain, breaks on sway), per-tree dynamic-occluder AABBs (16-box cap, boxes read wrong for crowns), or splat analytic shapes into the cascade volumes at assembly time.
- **Choice:** `lighting/occluders.py` — `TreeOccluderSet` (struct-of-arrays from the zone placements) + `splat_tree_occluders`: trunk = near-opaque column (`light_tree_trunk_occ` 0.85), canopy = FRACTIONAL ellipsoid (`light_tree_canopy_occ` 0.30 — leaves attenuate; 1.0 reads pitch-black). Hooked after the chunk gather in `assemble_geometry` (max-combine: terrain solids win; albedo written only where occupancy rises, so bounce colour comes from the species atlas means), threaded through `AssemblyJob`, pushed by `tree_renderer` via `GpuLightingPipeline.set_static_occluders` after every placement (re)bake; stale cascades re-splat asynchronously at their committed origins. Coarse cells scale contributions by shape-volume / cell-volume (a bush is a wisp in an 8 m cell).
- **Why:** The splat rides the existing assembly path end-to-end — INJECT sun march, GATHER bounce, the lit_surface refinement march and voxel AO all see trees with ZERO shader changes (the payoff of the unified lit-surface contract), and it stays deterministic and headless-testable (`tests/test_tree_occluders.py`). Fractional canopy occupancy is what makes dappled shade instead of a black disc: the vis march multiplies (1 - occ) per cell, so light decays through leaves the way the fractional-occupancy contract already intended.

## 2026-06-12 — Editor scenes load in the game (fire_engine/scene + SceneRuntime)

### The placed-object schema moved into the engine; the game grew a scene loader
- **Q:** The Fire Editor saved placeable objects (save_key `editor_scene`) but the game never read them — `SceneObjectStore` lived only under `editor/`, so authored scenes silently dropped their objects in-game. Where should the schema live, and what do the kinds mean at runtime?
- **Choice:** `SceneObjectStore`/`SceneObject` moved verbatim to `fire_engine/scene/objects.py`; `fire_editor.scene_objects` became a re-export shim (editor imports engine, never the reverse — one schema definition, drift impossible; guarded by `tests/editor/test_scene_roundtrip.py`). New headless `fire_engine/scene/runtime.py::SceneRuntime` registers as the game's `editor_scene` Saveable and on `apply_delta` instantiates GameObjects (DFS, parent-first, `set_parent(keep_world=False)` then local TRS). Visuals delegate to `fire_engine/render/scene_visuals.py::SceneVisualFactory`: cube/sphere → 1 m primitives (shared `world/primitives.py`, sphere extracted from wind_debug), `light` → a real `PointLight` (torch defaults 1.0/0.62/0.28 × 8.0 @ 16 m; skipped+logged on the CPU backend), `spawn` → the FIRST spawn (DFS) sets `camera_go` position on every successful load, `empty` → bare transform. Placed objects register as F1-overlay selectables, and the per-frame sync task WRITES GIZMO EDITS BACK into the store, so in-game F5 persists moved objects. `python main.py --load PATH` opens a save/scene at boot and retargets F5/F9 to that path for the session.
- **Why:** SceneRuntime must import the world object model lazily (inside `rebuild()`): importing any `fire_engine.render` submodule executes the package `__init__`, which pulls panda3d when installed — and this module reaches the daemon through the shim (`test_no_panda3d.py` regression caught it). Fixed light params are the smallest decision: `SceneObject` has no params field yet; adding one later is backward-compatible because `from_dict` ignores unknown keys.

### Authored scenes live in scenes/ (committed), not saves/ (gitignored)
- **Q:** Where does the editor's "Save Scene" write so the game can load it?
- **Choice:** `scenes/` at the repo root, same `.ta` delta format; `saves/` stays player state.
- **Why:** Authored scenes are content — they belong in version control and code review; a save's seed must match `config.toml` either way, which `--load` reports clearly on mismatch.

### Canopy occupancy is a per-METER extinction medium, not a per-cell opacity (2026-06-12, same day)
- **Q:** Owner: ground near/under trees renders completely black. Cause: the splat stored a flat per-CELL opacity (0.30) while the light marches multiply (1 - occ) per cell CROSSED, so total extinction depended on cascade cell size - a 5 m canopy was ~10+ cells at cascade 0 (0.7^10 = 3% light = black) but 3 cells at cascade 1, and there was no light gradient through the crown.
- **Choice:** Each instance carries `canopy_sigma`, a per-meter extinction coefficient derived from the species' REAL leaf thickness (`procedural.flora.mesh_leaf_area_m2` - sum of actual leaf-card triangle areas, identified by the atlas-right-half UV contract - divided by canopy ellipsoid volume, x0.5 random-orientation projection, /instance scale). A cell stores the Beer-Lambert opacity over ONE cell of path: occ = 1 - exp(-sigma * gain * rim_falloff * cell_m), with rim_falloff = sqrt(1 - d^2) thinning the medium toward the canopy edge. Config `light_tree_canopy_occ` replaced by `light_tree_canopy_extinction_gain` (default 1.0 = the species own leaf density). Test pins the invariant: transmittance through the same canopy marched at 0.5 m and 2 m cells agrees.
- **Why:** Marching (1-occ) per cell of a Beer-Lambert per-cell opacity composes to exp(-sigma * meters) regardless of cell size - light passes through leaves and decays gradually with canopy depth, which is the physical behaviour the owner asked for ("not a zero or one - a gradient, set by how thick the leaves are"). Deriving sigma from the meshes means a dense oak shades hard while a two-tuft snag barely dims the ground, with zero per-species authoring. The owner plans a later lighting iteration; per-meter extinction is the representation that survives it (any future march that respects path length composes correctly).

## 2026-06-13 — Editor agent harness (Python client + CLI + browser viewport)

### An agent drives the editor through the SAME protocol the extension uses, via one shared JSON-RPC core
- **Q:** Claude needed to operate and visually verify the Fire Editor without a human clicking in VS Code (the editor analogue of `tools/screenshot.py`). How to do that without forking the protocol or the viewport into a second, drift-prone implementation?
- **Choice:** Three additive pieces, zero new RPC methods. (1) `fire_editor.EditorClient` + `spawn_daemon()` — an async websockets client mirroring the extension's `FireEditorClient`. (2) `tools/editor_client.py` — a CLI mapping subcommands 1:1 to RPC methods plus `serve` (a long-lived daemon + stdlib `http.server` hosting the browser harness) and `rpc`/`watch` escape hatches. (3) A browser harness: the transport-agnostic `protocol/rpcSession.ts` (now shared by `client.ts` AND the harness), a `webview/host.ts` seam (`acquireVsCodeApi()` in the panel, `window.__fireEditorHost` in the browser), `webview/viewportMarkup.ts` (CSS/body shared by panel + harness), and `webview/harnessBoot.ts` (a browser port of the extension relay that runs the same `media/sceneView.js` bundle). The harness exposes `window.fireHarness` + `window.__fireSceneDebug.snapshot()` and logs `[harness]` lines for Chrome MCP reads.
- **Why:** Reusing one `RpcSession` and one `sceneView.js` means the harness speaks byte-for-byte the same protocol and renders pixel-for-pixel the same viewport as VS Code — an agent's screenshots and a human's editor can never silently diverge, and any new viewport UI (gizmos, ground shader) is harness-testable for free. Building on `serve`'s broadcast-to-all-clients, the CLI and the browser page drive the same world at once, so scripted edits render live.

### `chunks.set_center {resend}` fixes the daemon-global sent-chunk cache for a 2nd client
- **Q:** `ChunkService._client_chunks` is daemon-global, so a harness attaching to an already-running daemon received zero meshes (the daemon thought they were already sent).
- **Choice:** `set_center` honours an optional `resend: true` that clears the sent-chunk cache before streaming; the harness always boots with it. Tested both ways in `test_client.py` (2nd client: 0 frames without, full set with).
- **Why:** Smallest protocol-compatible fix (one optional param, version already bumped to 5 for the ground LUT) — no per-client cache rework, and existing single-client streaming is unchanged.

---

## 2026-06-12 — Buildings: free-form floorplan model (fire_engine/buildings/)

ARCHITECTURE.md §5.7 reserved `fire_engine/buildings/` as a "blocks + primitives" Building Manager stub. The owner's brief reframed it: buildings are deliberately **not** voxels — free-form walls in any direction, arbitrary building rotation, curved walls, windows/doors, variable thickness, foundations, with rooms as first-class objects (future systems procedurally generate buildings from tags and furnish each room). This rewrites §5.7 to a free-form floorplan model and records the sub-decisions made underneath it. (Commit 1 of a multi-commit build; the model layer only.)

### Floorplan walls, not free 3-D surfaces or block kits
- **Q:** What geometric primitive backs a building — voxel sub-grid, CSG of solid blocks, or 2-D floorplan extruded per storey?
- **Choice:** Per-storey 2-D floorplans (`Storey` holds plan-space walls/rooms; `Building` stacks storeys under one world transform). Sims/Paralives-style. Walls extrude between slab top and storey height; arbitrary building rotation lives entirely in the `Building.position`/`rotation` node transform, never baked into plan coords.
- **Why:** Floorplans are what a tag→building generator actually reasons about (rooms, adjacencies, openings) and what furnishing needs; extrusion + a single node transform keeps every wall headless and numpy-friendly without a voxel coupling.

### D1 — One `Wall` class, DXF bulge arcs (no separate curve type)
- **Q:** How are curved walls represented without a second wall class?
- **Choice:** `Wall(a, b, bulge=0.0, ...)`; `|bulge| = tan(included_angle/4)` (DXF convention). `bulge=0` ⇒ straight; **positive bows LEFT of a→b**, negative right; `|bulge|=1` ⇒ semicircle. `kind` is derived; `arc_params()` returns signed sweep `-4·atan(bulge)`; `tessellate()` emits a centerline polyline with exact endpoints.
- **Why:** One scalar captures the full straight↔arc range, room topology needs only endpoints, and arc geometry is derived once in tessellation/meshing — no parallel class to keep in sync.

### D3 — Building-local elevation contract
- **Q:** Where is z=0 and how do storeys stack?
- **Choice:** Local z=0 = top of the foundation slab (foundation occupies `[-depth, 0]`). Storey i: `base_z = Σ` lower storey heights; floor slab `[base_z, base_z+slab_m]`; walls `[base_z+slab_m, base_z+height_m]` (a wall's own `height_m` measures from slab top). Flat roof slab caps `total_height_m`. World = `position + rotation.rotate(local)`.
- **Why:** A single unambiguous datum lets meshing, room detection and the future lighting voxelization all agree without per-call origin negotiation.

### Numbers from Config, dimensions never hardcoded
- **Q:** Where do default storey height / wall thickness / slab / foundation / tessellation / snap tolerance live?
- **Choice:** A `[buildings]` config table → `BuildingDefaults.from_config(cfg)` (single number source). `building_arc_segments_per_quarter=8`, `building_snap_eps_m=0.01`.
- **Why:** Hard Rule "config values from core.config"; one place to retune, and `BuildingDefaults` travels with each building's save payload so old saves keep their authored dimensions.

### Save/serialize: plain-primitive dicts, per-building element ids
- **Q:** How are buildings serialized given no-pickle, and how do element ids stay stable?
- **Choice:** Every model type has `to_dict()`/`from_dict()` over primitives only (Vec3/Quat→list, enums→str, no numpy in the dict). Element ids are a per-building monotonic int (`next_eid`) serialized in the payload; building ids are assigned later by the manager.
- **Why:** Round-trippable, inspectable saves with no live refs (Hard Rule 3); serializing `next_eid` keeps ids stable across save/load so deltas and references survive reload.

### Deferred to later commits (recorded so the seam is intentional, not forgotten)
- **D4** room auto-detection (planar half-edge minimal cycles, endpoint-snap; v1 requires walls to meet at endpoints — no mid-span T-split) → commit 2 (`rooms.py`).
- **D5** meshing by partition not CSG (arc→chords, centerline ±t/2 miter, opening-rect face partition, ear-clipped slabs → `terrain.meshing.MeshArrays`) → commit 3.
- **D6** mesh emitted in building-local space; renderer applies the node transform (move/rotate = transform write, no remesh; `building.vert` must compute `v_world` via `p3d_ModelMatrix`) → commit 7.

---

## 2026-06-13 — Weather M8 (summon API + save delta + gust-front coupling)

### Summoned cells reuse the natural StormCell path instead of a parallel "summon" type
- **Q:** A summoned storm must drive coverage/rain/fog, the `.cells` readout, the weather-map raster, and (for M7) the strike schedule. Make it a new bespoke object, or a plain `StormCell` like the natural ones?
- **Choice:** `WeatherSystem.summon_cell` appends a normal `StormCell` (id `"s:{n}"`) to `self._summoned`; `_active_cells` already unions natural ∪ summoned, so every downstream sampler picks it up with **zero** new branches. Placement is **upwind**: the cell spawns `weather_summon_upwind_m` opposite the synoptic blow direction at `time_abs`, so it drifts over the player on the steering current. `drift_bias=(0,0)` — it rides the raw synoptic flow exactly like a natural cell.
- **Why:** The closed-form `StormCell` is already the unit of weather; reusing it means a summon is a first-class participant everywhere for free and the M7 strike scheduler needs no special case. Upwind placement makes "summon a storm" read as a storm *arriving*, not popping onto your head.

### Save delta = summoned-cell param list + suppressed-id list (no live refs)
- **Q:** How do summons/suppressions persist with the no-pickle, seed+delta save model and the load-resume invariant (identical future incl. would-be strikes)?
- **Choice:** `get_delta()` is `{}` for pure natural weather; otherwise `{summoned: [~80-byte primitive cell dicts], summon_seq: int, suppressed: [ids], ...legacy override keys}`. `apply_delta` rebuilds the cells from params (a malformed entry is skipped, never fatal) and bumps `summon_seq` past any restored id. Because a `StormCell` is a pure fn of its params, the reconstructed cell reproduces the identical track + footprint → identical future samples AND would-be strike positions. `clear_all` suppresses the natural cells active *now* (current-weather clear, not all-future).
- **Why:** Params, not state, is the only thing a closed-form cell needs to round-trip — keeps the delta tiny and pickle-free while guaranteeing determinism. Legacy Markov override deltas still load (their keys map onto the retained `force_weather` shim), so old saves don't break.

### Gust-front coupling lives in `update()` behind an injected wind-field handle
- **Q:** Where does the "storm's leading edge kicks the grass" coupling live without `wind/` importing `weather/` or `weather/` pulling panda3d into the headless suite?
- **Choice:** `attach_wind_field(field)` (world layer calls once) stores the field; `update()` → `_update_gust_fronts` registers one `GustFront` per cell whose leading edge is within `weather_gustfront_range_m`, removes it when the cell passes/decays (balanced; tracked in `_active_fronts`). The `wind` import is **lazy/local**, so importing `weather` never imports `wind`, and with no field attached the whole thing is a silent no-op (headless tests use a tiny fake field).
- **Why:** One-way dependency (weather → wind modifier seam) matches the existing `WindModifier` seam design; keeping the field injected (not imported at module scope) preserves the headless-testable guarantee and the zero-save-bytes property (`GustFront` is itself pure in (seed_key, t)).

## 2026-06-13 — Editor: per-object component stack (Unity-style inspector)
- **Q:** How does the inspector show/edit "components and scripts" on a scene object when `SceneObject` was only `kind` + TRS?
- **Choice:** Add a `components: list[{type, enabled, params}]` to `SceneObject`. The **Transform stays intrinsic** (the TRS fields, rendered as a synthetic non-removable section) — it is NOT in the list. A pure-data catalog (`fire_engine/scene/components.py`: `Mesh`, `Light`, `SpawnPoint`) is the single source of truth for built-in types + their editable fields; the inspector fetches it via a new `scene.catalog` RPC instead of hardcoding field lists in TypeScript. Protocol bumps **5 → 6** with `scene.add_component/remove_component/set_component/catalog`. The visual factory (`world/scene_visuals.py`) becomes **component-driven** (walks the list, not `kind`), so a `Light`'s authored color/intensity/radius actually drives the in-game `PointLight` and an `empty`+`Light` emits light. Custom **Python scripts are deferred** (built-in components only this pass).
- **Why:** Components independent of `kind` is the Unity model the owner asked for, and it makes authored light params real (they were hardcoded torch defaults before). One Python catalog avoids cross-language drift.
- **Sub-decisions:**
  - **`kind` = creation archetype, components = truth.** `create(kind)` seeds defaults via `default_components_for_kind`; thereafter the list rules. `kind` still drives the hierarchy icon and **spawn detection stays kind-based** (`SceneRuntime.spawn_position` filters `kind=="spawn"`; `SpawnPoint` is a no-param marker only) — moving spawn onto a component added migration surface for no near-term gain.
  - **Singleton Mesh/Light** (`multiple:false`): `add_component` rejects a duplicate; `remove_component` is index-based. Avoids ambiguous double-visuals.
  - **One migration seam:** `SceneObject.from_dict` synthesises `default_components_for_kind` when `components` is absent (pre-component `.ta` saves). NOT duplicated in `apply_delta`/`tree`/`to_dict`. No save-format bump (the delta stores the list verbatim).
  - **Live light-param hot-reload is a non-goal:** `attach()` reads params once at build time; a fresh game load reflects edits. The per-frame sync stays transform-only.
  - **`set_component` undo coalesces per `(id,index)`** (reusing `SceneCommand` snapshots) so a slider drag is one undo step; add/remove are discrete steps.

## 2026-06-13 — Standards gate: machine-enforced code-quality / structure / docs / testing
- **Q:** How do we keep a repo built largely by parallel AI agents clean as it 10–20×'s toward millions of lines, without relying on anyone remembering to be tidy?
- **Choice:** Install an enforced **standards gate** wired into the headless pytest suite (`tests/standards/`) and `.pre-commit-config.yaml`, so a standards violation fails the build exactly like a failing test. Off-the-shelf core: **Ruff** (lint + the sole formatter), **mypy --strict** (typing; Ruff `ANN` left off so the two don't double-report), **pylint** (narrow: `duplicate-code` + `too-many-lines`/500), **vulture** (cross-module dead code). Custom AST/tree walks cover what no tool does: `tools/check_repo_structure.py` (≤5 sub-folders, ≤10 modules/folder, 1 public class/module, test mirror) and `tools/check_docs.py` (`Docs:` pointer in every public docstring + per-package `docs/systems/` schema). `mkdocs build --strict` doubles as the dead-link/nav checker. All limits live in one place: `pyproject.toml [tool.firelight]`.
- **Why:** Cleanliness that isn't machine-enforced will not survive parallel agents. Each failure prints a **delegate-to-sub-agent** message naming the offending paths, so the orchestrator never bulk-fixes inline (which would blow its context at scale).
- **Sub-decisions:**
  - **Coverage = ratchet, not big-bang.** `coverage_fail_under` is a floor that only ever rises; standard 17 (every module ships a test) keeps *new* code honest while the floor climbs. Backfilling branch coverage across ~100k existing lines as a flat high number would stall the gate for weeks. **Floor initialised to `0.0`**, with measurement deferred to the first CI baseline run (`pytest -m coverage`) rather than run here — a full-suite coverage pass is heavy and was out of scope for installing the gate. The coverage test carries the `coverage` pytest marker and is **deselected by default** (pytest.ini), so routine `pytest -q` / `pytest -q tests/standards/` stay fast; CI/nightly runs `pytest -m coverage`.
  - **One-public-class-per-module exemptions.** Trivial tightly-coupled support types — `@dataclass`, `Enum`, `Protocol`, and the `*Event` frozen dataclasses CLAUDE.md mandates — may be grouped in a dedicated `events.py`/`types.py`/`enums.py`/`protocols.py`/`constants.py` (configurable `grouping_modules`). Private `_`-prefixed helper classes are always allowed beside the public class they serve. The intent is one *responsibility* per file, not dogmatic one-symbol-per-file.
  - **Ambiguous-unicode lints off (`RUF001/002/003`).** The docstrings are product (CLAUDE.md) and deliberately use em-dashes/ellipses/arrows; those rules would fight the house style tree-wide without catching a real defect.
  - **Panda3D typing override is narrow.** `mypy --strict` stays global; only `panda3d.*`/`direct.*` get `ignore_missing_imports` (no usable stubs, and they're import-restricted to `world/`/`lighting/` anyway). The gate itself is never weakened to silence Panda3D.
  - **Test mirror accepts legacy flat layout.** Canonical is `tests/<path-under-source-root>/test_<stem>.py`; the existing flat `tests/test_<stem>.py` is also accepted so the present suite is recognised while new code goes deep-and-narrow.
  - **Rollout is hard-fail, fixes deferred.** The gate was installed and verified to *detect* violations (structure 149, docs 849, plus lint/type findings) but the existing violations were **not** fixed in this change — per the owner's instruction a later dedicated agent does the per-package cleanup once new features have landed. The checks are layout-driven, so they keep working across the in-flight package reorg without edits.

## 2026-06-13 — Performance profiler (headless core + overlay + PStats + harness)

### `frame_ms` is wall-clock between successive `begin_frame()` calls (full frame, incl. render)
- **Q:** Should the headline frame time measure only the CPU/Python loop body, or the true full frame including the GPU render/flip/vsync after the task returns?
- **Choice:** Full frame = delta between two successive `begin_frame()` calls; the previous frame is committed at the next `begin_frame()` once its duration is known. `end_frame()` also records the loop-body (CPU) time as the `frame_cpu_ms` counter, so the CPU-vs-total split is visible. The gap `frame_ms − Σ(top-level scopes)` ≈ render + overhead (also in PStats).
- **Why:** The 200+ FPS / 5 ms target is a *total*-frame budget; measuring only the CPU body would under-report. Cost: the last frame of a run isn't committed (no following `begin_frame`); `tools/profile_run.py` steps one extra frame. An injectable `time_source` keeps tests exact.

### Module-level singleton (`get_profiler`/`init_profiler`), not threaded context
- **Q:** Reachable from the app loop, the registry, and `world/sky/` without threading a handle everywhere. Global or DI?
- **Choice:** A process-wide singleton in `core/profiler.py`, mirroring the existing `ComponentRegistry` singleton. `init_profiler(config)` mutates it in place; `get_profiler()` returns it; direct `Profiler(...)` for headless tests.
- **Why:** Matches the established registry pattern, not a rogue global. In-place mutation lets `tools/profile_run.py` force the profiler ON after `build_demo()` (config defaults OFF) without re-plumbing boot.

### Config is flat `profiler_*` fields + a `[profiler]` table
- **Choice:** Flat `profiler_*` fields on `Config` + `[profiler]` in the table-flatten list, matching every other subsystem (`fog_*`, `gfx_*`, `wind_*`).
- **Why:** Nested config objects exist nowhere else here.

### Per-component-type scopes in the registry; explicit `Weather:Update` in `world/sky/`
- **Choice:** `ComponentRegistry.run_frame` (in `render/registry.py`) wraps each per-type bucket in `Update:<Type>` / `LateUpdate:<Type>` / `FixedUpdate:<Type>` scopes (cached names; no-op when disabled). `world/sky/sky_state.py::SkySystem.update` adds an explicit `Weather:Update` scope around the weather sim advance. Both stay panda3d-free (import only `core.profiler`).
- **Why:** Automatic, exhaustive per-component attribution surfaces the weather render component and the headless weather sim by name, with zero per-frame event traffic (Hard Rule 5). A first windowed run already flagged `LateUpdate:LightningRendererComponent` at ~20 ms.

### Layout: built on `perf-profiler` (pre-reorg), then reconciled onto the post-reorg `render/` layout for merge
- **Q:** The profiler was developed on a branch off the pre-reorg `world/` layout; master meanwhile landed the `world/`→`render/` reorg (+ standards-gate, editor).
- **Choice:** Re-applied the profiler changes onto master's new layout as a single linear commit: panda3d mirrors live in `render/profiler_bridge.py` + `render/profiler_overlay.py`; loop instrumentation in `render/app.py` + `render/registry.py`; weather scope in `world/sky/sky_state.py`; core unchanged location (`core/profiler.py`). Doc kept as the flat `docs/systems/profiler.md` (the profiler spans `core/` + `render/`, not one package).
- **Why:** Cleaner and more reviewable than a rename-conflict-laden 3-way merge; ships as a fast-forwardable commit on master.

---

## 2026-06-15 — Standards-gate remediation (branch `refactor/standards-remediation`)

Bringing the 6 failing `tests/standards/` gates to green honestly (no shortcuts;
CLAUDE.md Hard Rules + the 2660-logic-test tripwire outrank passing any gate).
See `docs/sessions/standards-remediation-spec.md` and the session note.

### Exclude `tools/out/` from ruff lint
- **Q:** `ruff check .` linted `tools/out/diag/*` (ad-hoc probe/dump scripts) — ~111 of 484 hits — that were never meant as maintained source.
- **Choice:** Add `tools/out` to `[tool.ruff] extend-exclude` (the dir is already git-tracked scratch/output; left tracked, just unlinted).
- **Why:** Scratch diagnostic output, not source. Sanctioned by the remediation spec; does not hide any `fire_engine` problem (the limit and every engine path stay linted). Dropped ruff to 373 real hits.

### Resolve `[6] fire_engine/ has 12 sub-folders (max 5)` via source-root exemption — NOT a top-level reorg
- **Q:** The structure gate counts `fire_engine/`'s own 12 immediate sub-packages against `max_subdirs=5`. But CLAUDE.md's documented Repo Layout *prescribes* exactly those 12 top-level subsystems (core/ render/ world/ simulation/ …), and `max_subdirs` is a real limit the goal forbids loosening. The two conflict.
- **Choice (option B, default):** Treat the **source-root package directory itself** (`fire_engine/`) as a namespace aggregator exempt from the `[6]` sub-folder count — mirroring `check_docs.py`, which *already* exempts the source root (`if pkg == root: continue`, "it only re-exports"). Every real sub-package still enforces ≤5 sub-folders and ≤10 modules; `max_subdirs` stays 5. Implemented as a one-line guard in `check_repo_structure.py`, not a `pyproject` limit change.
- **Why:** Lowest-risk honest fix that does **not** weaken the deep-&-narrow guarantee for any actual code folder, is consistent with the docs-checker's existing root exemption, and matches CLAUDE.md's prescribed layout. The alternative (**option A**: regroup the 12 into ≤5 super-packages, e.g. `foundation/{core,save,resources}`) rewrites every import across `fire_engine/`, `tests/`, `tools/`, `editor/`, `main.py` and contradicts the documented layout — high blast-radius for an unattended run. **Flagged for owner review:** if option A is preferred, the delta is a mechanical top-level move + path rewrite.

### Vec3 world-space constants typed as `ClassVar`
- **Q:** `Vec3.ZERO/ONE/UP/FORWARD/RIGHT` are assigned after the class body (a Vec3 can't be built inside its own definition), so `mypy --strict` reported ~20 `attr-defined` errors tree-wide.
- **Choice:** Annotation-only `ClassVar[Vec3]` declarations inside the class body; values still assigned just below it.
- **Why:** Pure typing fix, zero runtime change (`from __future__ import annotations` makes the annotations strings); clears the errors at the source instead of per-call-site.

### Rule `[17]` (test mirror) exempts modules that import panda3d
- **Q:** The structure gate demanded a headless test mirror for ~41 `render/`/`lighting/` modules. But Hard Rule 1 confines panda3d to those two packages, and the headless suite excludes anything importing panda3d (`docs/sessions/standards-remediation-spec.md`: *"do not write a panda3d-importing test into tests/"*). A module that cannot be imported headlessly cannot have a headless mirror — the rule contradicted the testing philosophy for exactly these files.
- **Choice:** In `check_repo_structure.py`, exempt a module from `[17]` **iff it directly imports `panda3d`/`direct`** (AST check `_imports_panda3d`). Such modules are the real render bridges (`app.py`, `*_renderer.py`, `gpu.py`, the texture/geometry bridges); they are integration-verified by launching the app (`tools/screenshot.py` / `main.py`), not unit-mirrored. The criterion is deliberately **import-based, not a blanket `render/`/`lighting/` carve-out**: the headless halves of those packages — the GLSL-string builders (`*_shaders.py`, `lighting/glsl.py`), the pure object model (`render/{component,gameobject,registry,transform}.py`), and the lighting math/data (`lighting/{lights,volume,palette,sunlight,light_grid,occluders}.py`, which `render/__init__`/`lighting/__init__` import unconditionally and which existing headless tests already exercise) do **not** import panda3d and therefore **still require real test mirrors**, written in this remediation.
- **Why:** Honest and minimal — it removes a self-contradictory requirement (headless mirror for non-headless code) without weakening coverage of anything that *can* be headless-tested. Not a `pyproject` limit change; a precise guard in the checker, documented in its module docstring. **Flagged for owner review** alongside the `[6]` exemption.

### Rule `[6]`/`[7]` sub-folder cap counts only *packages*, not data directories
- **Q:** After sub-packaging `render/` (bridges/sky/vegetation/overlay/_impl) the structure gate flagged `[6] render/ has 6 sub-folders (max 5)` — but the 6th "sub-folder" is `render/shaders/`, a directory of `.vert`/`.frag`/`.glsl` GLSL source files with **no `__init__.py`**. It is not an importable Python sub-package; it is data the renderers read via `core.shader_source.load_glsl`.
- **Choice:** `check_repo_structure._subdirs` now counts a child directory toward the sub-folder cap **only if it is a Python package** (contains `__init__.py`). Data directories (`shaders/`, `__pycache__`, …) are not sub-packages and no longer count.
- **Why:** The deep-&-narrow standard is explicitly about nesting of CODE packages ("one idea per file", ≤5 sub-packages); a sibling GLSL data folder is not a sub-package, so counting it was a checker bug, not a real violation. Fixing it lets `render/` keep its 5 genuine code sub-packages. Module/`__init__` checks are unaffected (a data dir has no `*.py` modules). Not a `pyproject` change; a one-line guard, documented in the checker.
