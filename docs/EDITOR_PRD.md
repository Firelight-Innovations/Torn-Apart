# Fire Editor — PRD & Implementation Plan
keywords: editor, fire editor, vscode, cursor, viewport, scene view, hierarchy, inspector, gizmo, webview, three.js, daemon, websocket, texture lab, model workspace, brush editing, undo, introspection

*For: implementing agent (Claude Opus + subagents) | Date: 2026-06-09 | Owner-approved scope*
*Read `docs/ARCHITECTURE.md` and `CLAUDE.md` first. ARCHITECTURE.md is authoritative on engine design; this file on editor design and sequencing. Conflicts → ARCHITECTURE.md wins; log in DECISIONS.md.*

---

## 1. Product Overview

**Fire Editor** is a Unity-Editor-style visual editor for Fire Engine (the Torn Apart engine), running **inside VS Code/Cursor** as an extension. It lets the owner, **while the game is not running**:

- Fly a camera through the generated game world (scene view) and see it approximately as the game renders it.
- Browse the GameObject hierarchy and inspect/edit components and transforms (hierarchy + inspector).
- Edit the world: terrain brushes, ZoneVolume placement, save/load via the engine's seed+delta save system.
- Author procedural textures with live parameter preview and hot reload (texture lab).
- Assemble 3D models from primitives with procedural textures, and preview imported gltf models (model workspace).

This turns Cursor into the complete development environment for the engine: code + the one missing piece, the visual editor.

### Why this architecture is cheap here
Fire Engine is headless-by-design: only `torn_apart/world/` and `torn_apart/lighting/` may import panda3d (CLAUDE.md hard rule 1). Terrain generation, meshing (vertex positions/normals/colors), the Unity-clone object model (`Transform`, `GameObject`, `Component`), procedural textures, RNG, and saves are all pure Python/numpy. The editor therefore **does not need Panda3D at all** — it runs the headless engine in a Python daemon and renders the resulting mesh/texture arrays in a WebGL viewport.

### Non-goals (v1)
- **No live attach to a running game.** Designed for (versioned protocol, daemon embeddable later), built later.
- No play-in-editor, no animation tools, no vertex-level mesh modeling, no multi-user.
- No pixel-perfect render parity. Geometry, vertex lighting, and textures are produced by the same engine code, so parity is high; shader-level differences are acceptable. A later "render in engine" button can open a Panda3D window for ground truth.

---

## 2. Architecture

```
┌─ VS Code / Cursor ────────────────────────────────────────────┐
│  Fire Editor extension (TypeScript)                           │
│  ├─ Scene View      webview: three.js WebGL viewport          │
│  ├─ Hierarchy       tree view (native VS Code TreeView)       │
│  ├─ Inspector       webview: generated property forms         │
│  ├─ Texture Lab     webview: preview + param sliders          │
│  ├─ Model Workspace webview: three.js + part list             │
│  └─ daemon lifecycle: spawn/respawn, status bar, output log   │
└───────────────▲───────────────────────────────────────────────┘
                │ WebSocket (localhost): JSON-RPC control
                │ + length-prefixed binary frames (meshes/textures)
┌───────────────▼───────────────────────────────────────────────┐
│  fire_editor daemon (Python, in the game repo's .venv)        │
│  ├─ EditorSession: world seed/config/save → headless world    │
│  ├─ imports torn_apart: core, procedural, terrain, save,      │
│  │   world object model (gameobject/component/transform),     │
│  │   lighting CPU pass — NEVER panda3d (hard rule, see §6)    │
│  ├─ ChunkService: generate → mesh → vertex-light → stream     │
│  ├─ SceneService: hierarchy snapshot, introspection, edits    │
│  ├─ EditService: apply_brush, ZoneVolumes, undo/redo stack    │
│  ├─ TextureService: render defs, watch files, hot reload      │
│  └─ ModelService: primitive models, gltf passthrough          │
└───────────────────────────────────────────────────────────────┘
```

### Components
- **`editor/extension/`** — TypeScript VS Code extension. Spawns the daemon (`python -m fire_editor --port <p>`), owns panels, reconnects on daemon restart. three.js viewport with fly camera (same bindings as the game: WASD + mouse-look, Shift = 5×), `THREE.NearestFilter` on all textures (retro look), Z-up (`THREE.Object3D.DEFAULT_UP = (0,0,1)`).
- **`editor/fire_editor/`** — Python package, the daemon. Lives in the game repo, uses the repo `.venv`. All engine access goes through public APIs documented in `docs/systems/`.
- **Protocol** — see §4.

### Repo placement (record in DECISIONS.md)
New top-level `editor/` directory in the game repo: `editor/fire_editor/` (Python) + `editor/extension/` (TS). Editor docs at `docs/systems/editor.md` (from `_TEMPLATE.md`). Daemon tests live in `tests/editor/` and run in the headless suite. The extension has its own `npm test`; it is excluded from `pytest`.

---

## 3. Features (v1)

### F1 — Scene View
- Open a world three ways: **seed** (generate from `config.toml`/override), **save file** (`saves/*` via SaveManager: regen baseline + `apply_delta`), or **current editor session**.
- Chunks stream around the editor camera using the engine's generation + culled-face meshing + sunlight v0 vertex colors. View distance configurable (default: engine `view_distance_chunks`).
- Procedural ground textures rendered via the same `TextureGenerator` output (RGBA numpy → texture atlas → three.js).
- Overlays (toggleable): chunk borders, ZoneVolume bounds, light-grid heat view, wireframe, stats (chunks loaded, verts, ms).
- Picking: terrain raycast through daemon (`terrain/raycast.py`); object picking client-side via AABBs included in scene snapshots.

### F2 — Hierarchy
- Tree of all GameObjects in the editor session (name, tag, active state icons). Search/filter by name/tag.
- Select → highlights in scene view + opens in Inspector. Create empty GameObject, reparent (drag), delete, toggle active.

### F3 — Inspector
- Shows selected GameObject: name/tag/layer/active + Transform (position, rotation as euler-view-of-quaternion, scale) + each component with its editable properties.
- Property forms generated from the engine **introspection API** (§5.1): typed fields → typed widgets (float drag, Vec3 triple, bool, enum, color, string).
- Edits round-trip: widget change → daemon applies via public setters → confirmation → scene view updates (e.g. transform gizmo moves).
- Transform gizmos in viewport: translate/rotate/scale handles (three.js `TransformControls`), writing back through the same path. Rotation edits compose quaternions; eulers are display-only (ARCHITECTURE.md §5.4).

### F4 — World Editing
- Brush palette: Sphere/Box/Cylinder × ADD/REMOVE × material picker. Click/drag in viewport → `terrain.apply_brush(...)` in daemon → dirty chunks remesh + relight → updated meshes pushed to viewport. Brush preview ghost at cursor raycast hit.
- ZoneVolume tool: place/move/resize tagged volumes; tags from a registry-provided list.
- **Undo/redo** (editor-side, §5.4): brush ops, transform edits, property edits, create/delete, ZoneVolume ops.
- Save/Load: writes standard saves through `SaveManager` — terrain edits persist as deltas exactly like in-game edits. Authored ZoneVolumes/placed objects persist via the world manifest (§5.2). Dirty indicator + save prompt on close.

### F5 — Texture Lab
- Lists all registered `ProceduralTextureDef`s (`procedural.register`/`get` registry).
- Select def → rendered preview (256² default, 512² toggle), zoom with nearest-neighbor, tiled 3×3 view for seam checking.
- Param panel: def parameters (from introspection) → sliders/inputs; changing one re-renders live. Seed scrubber (renders via `core.rng.for_domain`).
- **Hot reload**: daemon watches the def's source file; on save in the editor, re-import the module, re-register, re-render. Import errors surface in the panel, never crash the daemon.
- "Open source" button jumps to the def's `.py` file in the editor. Export preview to `tools/out/` (compatible with `tools/preview_texture.py` fixtures).

### F6 — Model Workspace
- **Primitive assembly**: build models as a list of parts (box, cylinder, arch, ramp, sphere segment) each with local TRS + material/procedural-texture assignment — the same blocks-and-primitives philosophy as `BuildingDef` (ARCHITECTURE.md §5.7). Add/select/transform parts with the same gizmos as F3.
- Persisted as **`.model.json`** part lists under `assets/models/primitive/`, loaded by a new `PrimitiveModelDef(ProceduralDef)` in the engine (§5.3). JSON is data authored by a tool — consistent with "hand-crafted assets in assets/"; the Python def interprets it.
- **Imported model viewer**: load `.gltf`/`.glb` from `assets/models/` into the viewport (three.js GLTFLoader, client-side; daemon just serves the file). `.egg`/`.bam` are out of scope for viewing (panda3d-only formats) — list them, show metadata only.
- Place instances of either kind into the world session (records into the world manifest).

---

## 4. Protocol

WebSocket on `localhost:<port>` (port chosen by extension, passed to daemon).

- **Control channel**: JSON-RPC 2.0 messages. Every request has `id`; daemon replies result or structured error. Daemon → client notifications for: chunk ready, scene changed, selection sync, watch events, long-op progress, log lines.
- **Binary channel**: same socket, binary frames: `[u32 magic][u32 schema_id][u32 payload_id][payload]`. Used for mesh buffers (positions f32, normals f32, colors u8, uvs f32, indices u32 — one frame per chunk) and texture payloads (raw RGBA8 + width/height header). JSON messages reference payloads by `payload_id`. Never base64 meshes through JSON.
- **Versioning**: `hello` handshake exchanges `protocol_version`; mismatch → extension shows "rebuild daemon" error. All schemas in one shared spec file `editor/protocol/SCHEMA.md` + generated TS types and Python dataclasses (single source: a JSON Schema or a small codegen script — implementer's choice, but both sides must be generated from one source).

Core methods (sketch — final list defined in Phase E0):
`hello`, `world.open {seed|save_path}`, `world.save {path}`, `chunks.subscribe {center, radius}`, `chunks.set_center`, `scene.snapshot`, `scene.select`, `object.create/destroy/reparent/set_active/set_property`, `component.add/remove/set_property`, `introspect.component_types`, `introspect.object {id}`, `terrain.brush {shape, params, center, mode, material}`, `terrain.raycast {origin, dir}`, `zone.create/update/delete`, `undo`, `redo`, `texture.list/render/set_params/watch`, `model.list/load/save/instantiate`.

---

## 5. Engine-Side Prerequisites (gaps the plan must build)

These are engine changes, in the game repo, following all CLAUDE.md rules (docs/systems updates in the same commit, headless tests, type hints, docstrings-as-product).

### 5.1 Introspection API — `torn_apart/world/introspect.py` (headless)
The inspector needs to enumerate component types and properties without hardcoding.
- `component_types() -> list[type[Component]]` — from the registry's type buckets + an import-scan registration hook.
- `describe(obj: GameObject | Component) -> ObjectDescription` — fields with name, type, value, units, readonly flag, and UI hints. Source of truth: type hints on public attributes + an optional class-level `__editor_fields__` override for hints (range, step, enum choices). Same mechanism reused for `ProceduralDef` params (texture lab) — put the shared core in `core/introspect.py` if cleaner.
- `set_field(target, name, value)` — validated, typed application through public setters.

### 5.2 World manifest (authored content persistence) — record in DECISIONS.md
Saves hold *runtime deltas*; the editor also produces *authored* content: ZoneVolumes, hand-placed GameObjects/model instances. Persist these as a **world manifest** — `assets/world/<name>.manifest.json` (plain JSON: primitives only, no pickle — consistent with hard rule 3) loaded during worldgen before deltas apply. Engine change: worldgen reads manifest → instantiates ZoneVolumes/objects as part of the baseline. Editing the manifest changes the baseline; saves stay delta-only on top.

### 5.3 `PrimitiveModelDef` — `torn_apart/procedural/models.py`
`ProceduralDef` subclass: loads a `.model.json` part list, emits merged vertex arrays (numpy) with per-part procedural texture assignment. Headless; `world/` uploads it via the existing geometry bridge when the *game* uses it; the *editor* renders the same arrays in three.js. Determinism test required like any `ProceduralDef`.

### 5.4 Editor undo/redo — `fire_editor/commands.py` (editor-side, not engine)
Command stack with inverse ops. Brush inverse = pre-edit snapshot of affected voxel regions (bounded: brushes are local). Property/transform inverse = previous value. Create inverse = destroy (and vice versa, with serialized state). Cap memory (config; default 256 MB of voxel snapshots, LRU-drop oldest history).

### 5.5 Headless lighting CPU pass
Sunlight v0 (column pass → vertex colors) must be importable without panda3d so the daemon can light meshes. If Session 1 put it in `lighting/` next to GPU code, split it (e.g. `lighting/sunlight_cpu.py` clean of panda3d imports, re-exported) — this respects the spirit of hard rule 1 and needs a one-line DECISIONS.md note.

### 5.6 Engine version check
Daemon asserts engine compatibility at boot (read a `torn_apart.__version__`; add one if missing).

---

## 6. Hard Rules (editor-specific; violations are bugs)

1. **`fire_editor` never imports panda3d.** The editor's whole premise is running with the game closed; panda3d in the daemon is a regression. CI/test: an import-graph test like the engine's.
2. **All engine access through public APIs** (`docs/systems/` documented). No reaching into private state; if the editor needs something private, add a public API to the engine with docs.
3. **All editor-side persistence is JSON/msgpack of primitives** — no pickle (hard rule 3 applies to the editor too).
4. **Determinism preserved**: the daemon sets the world seed via `core.rng.set_world_seed` and never introduces unseeded randomness. Editor preview of seed N must match game world of seed N.
5. **Bulk data is bulk** (mirrors engine rules 4/7): meshes/textures cross the wire as binary arrays, one frame per chunk/texture; never per-voxel/per-vertex JSON.
6. **Protocol schema changes bump `protocol_version`** and regenerate both language bindings in the same commit.
7. **Engine changes made for the editor follow all CLAUDE.md rules**, including same-commit `docs/systems/*.md` updates.

---

## 7. Implementation Plan

Phased like DEVELOPMENT_PLAN.md; one commit per phase minimum, prefixed `editor phase N:`. Headless `pytest -q` (including `tests/editor/`) green before every commit. Suggested subagent split per phase: one agent on daemon/engine (Python), one on extension/webview (TS), with the protocol schema frozen between them at phase start.

**Dependency note:** Phases E1+ require the engine through game-phase "terrain + save" (Session 1 scope). E0 can start immediately.

### Phase E0 — Scaffold + Protocol (foundation)
- `editor/` layout, extension scaffold (esbuild/yo-generator equivalent, activation event, status bar, output channel), daemon skeleton (`python -m fire_editor`), spawn + handshake + reconnect, log streaming to output channel.
- Protocol spec `editor/protocol/SCHEMA.md` + codegen for TS types and Python dataclasses; binary framing; `hello`/version check.
- **Acceptance:** extension activates in Cursor, spawns daemon from repo `.venv`, handshake succeeds, daemon crash → auto-respawn with status-bar indication. `tests/editor/test_protocol.py`: JSON-RPC round-trip, binary frame encode/decode round-trip, version-mismatch rejection.
- **Commit:** `editor phase 0: extension scaffold + daemon + protocol`

### Phase E1 — Scene View (read-only world)
- Daemon: `EditorSession` (open by seed or save path → SaveManager load), `ChunkService` (generate → mesh → CPU sunlight → binary frames, centered on editor camera, prioritized by distance, off-thread so RPC stays responsive). Engine prereq §5.5 (headless lighting) and §5.6 land here.
- Extension: three.js viewport (Z-up, fly camera WASD+mouse, Shift 5×), chunk mesh management (add/remove by streamed frames), procedural texture atlas with `NearestFilter`, overlays: chunk borders, wireframe, stats.
- **Acceptance:** open world by seed → fly over the same terrain the game shows for that seed; open a game save → craters visible. Camera move streams chunks in/out without UI freeze. Determinism test: chunk mesh bytes for seed S from daemon == direct engine call. Visual: commit a screenshot fixture to `tools/out/`.
- **Commit:** `editor phase 1: scene view + chunk streaming + save loading`

### Phase E2 — Hierarchy + Inspector + Gizmos
- Engine prereq §5.1 (introspection API) with its own headless tests + `docs/systems/world.md` update.
- Daemon: `SceneService` — hierarchy snapshots (incremental: changed-subtree notifications), selection, `object.*`/`component.*` ops.
- Extension: Hierarchy TreeView (search, drag-reparent, context menu), Inspector webview (generated forms incl. Vec3/Quat-as-euler widgets), `TransformControls` gizmos bound to selection, selection sync both directions (click in viewport ↔ tree).
- **Acceptance:** select object in viewport → tree + inspector follow; edit position in inspector → object moves in viewport; drag gizmo → inspector updates; add/remove a component; lifecycle hooks fire correctly on editor-driven create/destroy (test against the registry's recorded call order). Round-trip test: `set_property` then `introspect.object` returns the new value.
- **Commit:** `editor phase 2: hierarchy + inspector + introspection + gizmos`

### Phase E3 — World Editing + Undo + Persistence
- Daemon: `EditService` — brush ops via `terrain.apply_brush`, dirty-chunk remesh/relight push; ZoneVolume CRUD; undo/redo stack (§5.4); `world.save` via SaveManager. Engine prereq §5.2 (world manifest) lands here with worldgen integration + tests.
- Extension: brush palette + ghost preview at raycast hit, material picker, drag-to-paint; ZoneVolume tool with resize handles; undo/redo keybindings (`ctrl+z/y` scoped to editor panels); dirty-state indicator and save prompts.
- **Acceptance:** carve a crater in the editor → save → `python main.py` shows the crater (the headline integration test of the whole project — automate the daemon half: save from editor, load delta with engine directly, assert voxels). Undo restores exact voxel content (byte-compare). ZoneVolume placed in editor appears in manifest and affects next worldgen. 100 sequential brush ops keep viewport interactive.
- **Commit:** `editor phase 3: brush editing + zone volumes + undo + saves`

### Phase E4 — Texture Lab
- Daemon: `TextureService` — registry listing, param introspection (reuses §5.1 mechanism), render-on-change (debounced), seed scrubbing, file watcher + module re-import with error capture.
- Extension: Texture Lab panel — def list, preview (zoom, 3×3 tile mode), param widgets, seed scrubber, error surface, "open source", export to `tools/out/`.
- **Acceptance:** edit a def's `.py`, save → preview updates without daemon restart; introduce a syntax error → panel shows traceback, daemon stays alive, fixing the file recovers. Exported PNG identical to `tools/preview_texture.py` output for same def+seed (parity test).
- **Commit:** `editor phase 4: texture lab + hot reload`

### Phase E5 — Model Workspace
- Engine prereq §5.3 (`PrimitiveModelDef`) with determinism test + `docs/systems/procedural.md` + `docs/content/` authoring guide update.
- Daemon: `ModelService` — `.model.json` CRUD, part-list → mesh arrays, gltf file serving, instantiate-into-world (records to manifest).
- Extension: Model Workspace panel — part list UI, add-primitive menu, per-part gizmo editing, material/texture assignment from the registry, gltf preview via GLTFLoader, "place in world" handoff to scene view.
- **Acceptance:** assemble a model from ≥3 primitives with 2 different procedural textures, save, reload editor → identical; instantiate into world, save world, manifest contains it; engine loads the def headlessly (test). gltf in `assets/models/` previews.
- **Commit:** `editor phase 5: model workspace + PrimitiveModelDef`

### Phase E6 — Verification + Polish (do not cut)
- Perf pass against budgets (§8); error-path audit (daemon death mid-stream, malformed save, watcher storms); keyboard/UX pass; `docs/systems/editor.md` completed (template headings); `docs/sessions/` handoff note; README section "Running Fire Editor".
- **Final checklist:** full `pytest -q` green · extension `npm test` green · fresh-clone bootstrap works (`npm install && code .` → F5 → editor opens a world) · seed determinism editor==game verified · save round-trip game↔editor verified both directions · no panda3d in `fire_editor` import graph · protocol bindings regenerated and committed.
- **Commit:** `editor phase 6: verification + docs`

**If behind schedule:** cut in this order — gltf viewer (E5), 3×3 tile view + seed scrubber (E4), ZoneVolume resize handles (E3, keep create/delete), rotate/scale gizmos (E2, keep translate). Never cut: tests, undo for brushes, the E3 crater round-trip, Phase E6.

---

## 8. Performance Budgets
- Viewport ≥ 30 fps at `view_distance_chunks=6` on the dev machine (the *game* targets 60; the editor tolerates 30).
- Chunk mesh frame ≤ 1 MB typical; daemon meshes ≥ 20 chunks/s; camera-move re-center latency ≤ 100 ms to first new chunk.
- Brush edit → updated mesh visible ≤ 150 ms for a ≤ 4-chunk brush.
- Texture re-render ≤ 200 ms at 256²; param-drag debounce 50 ms.
- Inspector edit round-trip ≤ 50 ms. Daemon RSS ≤ 2 GB at default view distance including undo cap.

## 9. Risks
- **Engine code may not exist yet / diverges from ARCHITECTURE.md** — the daemon binds only to documented public APIs; where the engine is missing a needed API, build it engine-side per §5 rather than working around it. If Session 1 isn't complete, E0 still proceeds; E1+ blocks on engine terrain+save phases.
- **Webview perf ceiling** — mitigated by binary transport, typed arrays straight into three.js BufferGeometry, and budgets in §8; fallback is reduced default view distance.
- **Hot reload fragility** (module re-import state leaks) — registry must support re-registration by name; texture defs are pure (def + seed + params → array), which makes reload safe; test the error paths explicitly (E4 acceptance).
- **Two-language schema drift** — single-source codegen + version handshake (hard rule 6).
- **Render-look drift vs. game** — geometry/lighting/textures come from engine code; only shading differs. E1/E6 screenshot fixtures catch regressions; "render in engine" ground-truth button is future scope.
