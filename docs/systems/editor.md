# editor — System Doc
keywords: editor, fire editor, daemon, fire_editor, extension, vscode, cursor, websocket, json-rpc, protocol, schema, codegen, binary frame, handshake, hello, scene view, hierarchy, inspector, properties panel, components, component stack, add component, component catalog, scene.catalog, scene.add_component, scene.set_component, light component, mesh component, gizmo, transform controls, texture lab, model workspace, ground lut, textured terrain, scene undo, save scene, scenes folder, resend

> Documents the `editor/` tree: the headless Python daemon `editor/fire_editor/`,
> the TypeScript VS Code/Cursor extension `editor/extension/`, and the shared
> protocol `editor/protocol/`. The editor is **not** part of the runtime engine —
> it imports `fire_engine` public APIs to drive an offline visual editor.
> Authoritative design: `docs/EDITOR_PRD.md`.

## Role
The **Fire Editor** is a Unity-Editor-style visual editor for the Torn Apart
("Fire") engine, running inside VS Code / Cursor *while the game is not running*.
A Python **daemon** (`fire_editor`) runs the headless engine and serves a
WebSocket protocol; a TypeScript **extension** spawns the daemon from the repo
`.venv`, manages its lifecycle, and (from Phase E1) renders the engine's mesh and
texture output in webview panels.

It deliberately does **not**: import panda3d (hard rule 1 — the editor runs with
the game closed), attach to a running game (v1), or do pixel-perfect render
parity. Geometry/lighting/textures come from the same engine code, so parity is
high; shader-level differences are accepted.

**Phase status:** E0 complete (scaffold + protocol + handshake + lifecycle).
E1 complete (scene view: chunk streaming + CPU sunlight + three.js viewport +
save loading). E3 complete (brush editing + undo/redo + crater round-trip +
delta saves). E2 **in progress** — the scene-hierarchy foundation is in: an
editable authoring `SceneObjectStore` (placeable GameObjects, persisted in the
save), the `scene.*` protocol, a native sidebar **Hierarchy** TreeView, viewport
gizmos, and two-way selection sync. (The `world/` object model turned out to be
headless-importable, so no panda3d split was needed.) Inspector/components, E4
(texture lab), E5 (model workspace), E6 (verification) remain per EDITOR_PRD §7.

## Public API
The editor is a standalone tool, not an importable engine package. Its surfaces:

**Daemon — `fire_editor` (importable for tests):**
- `Daemon` — builds the JSON-RPC dispatcher + WebSocket server, registers core + service methods, holds the open `EditorSession`, `run(port)`.
- `Dispatcher` / `RpcError` — transport-agnostic JSON-RPC 2.0 dispatch; handlers are `async (params) -> result`.
- `EditorSession` — one open world: terrain `ChunkManager`, `LightGrid` + `SunlightComputer`, `SaveManager`, and the authoring `scene` (`SceneObjectStore`). `from_seed`, `from_save`, `region_coords`, `ensure_loaded`, `relight`, `mesh`, `raycast`, `save`.
- `scene_objects` — a re-export **shim**: the authoring hierarchy now lives in the ENGINE at `fire_engine/scene/objects.py` (`SceneObjectStore` / `SceneObject`, kinds `empty|cube|sphere|light|spawn`) so the game's `SceneRuntime` consumes the identical schema (DECISIONS.md 2026-06-12). Deterministic integer ids (monotonic counter, no RNG); `create`, `rename`, `reparent` (cycle-rejecting), `set_transform`, `delete` (cascades), `tree` (flat DFS). Implements `Saveable` (`save_key="editor_scene"`) so the scene persists as a delta — an empty scene saves nothing.
- **Components** — each `SceneObject` carries a `components` list (`{type, enabled, params}`) on top of its intrinsic Transform; `kind` only *seeds* it (`default_components_for_kind`), after which the list is the source of truth (an `empty` can get a Light; a `cube`'s Mesh can be removed — Unity-style). The catalog of built-in types (`Mesh`, `Light`, `SpawnPoint`) lives in `fire_engine/scene/components.py` (pure data, single source — the inspector fetches it via `scene.catalog` rather than hardcoding field lists). Store ops: `add_component(id,type)` (singleton-checked), `remove_component(id,index)`, `set_component(id,index,params?,enabled?)` (params validated+clamped to the catalog field types). Pre-component saves migrate forward in `SceneObject.from_dict` (the single migration seam).
- `encode_frame` / `decode_frame` — protocol binary framing.
- `encode_mesh_payload(coord, mesh)` / `decode_mesh_payload(bytes)` — MESH payload codec.
- `texturecodec.encode_texture_payload(rgba)` / `decode_texture_payload(bytes)` — TEXTURE payload codec (`[u32 width][u32 height][rgba8]`); used by `world.ground_lut`.
- `EditorServer` — `websockets` transport; `broadcast_binary`, `broadcast_notification`.
- `services.chunks.ChunkService` — registers `world.open/save`, `world.ground_lut`, `chunks.set_center`, `scene.stats`, `terrain.raycast`, `terrain.brush`, `edit.undo/redo`; streams MESH frames and drives the undo stack. `EditorSession.ground_seed` (same `for_domain("terrain","ground")` derivation as main.py) and `ground_texels_per_m` ride in the `world.open` result config.
- `services.scene.SceneService` — registers `scene.tree/create/rename/reparent/set_transform/delete`, the component ops `scene.add_component/remove_component/set_component`, and the static `scene.catalog`; mutates `session.scene`, pushes a `SceneCommand` onto the shared undo stack, and broadcasts `scene.changed` (full object list) after every change. `set_component` coalesces per `(id,index)` so a slider drag undoes in one step.
- `commands.UndoStack` / `EditCommand` / `SceneCommand` — ONE chronological undo/redo stack for both edit types: `EditCommand` snapshots before/after material arrays over the brush AABB chunks (EDITOR_PRD §5.4); `SceneCommand` snapshots the full scene delta (tiny dicts). Consecutive `transform <id>` scene commands within ~1 s coalesce, so a throttled gizmo drag undoes in one step.
- Generated constants in `fire_editor._generated` (`PROTOCOL_VERSION`, `BINARY_MAGIC`, `SchemaId`, `ErrorCode`, `Method`, `Notification`, typed param/result `TypedDict`s).

**Methods (protocol_version 6):** `hello`, `ping`, `world.open {seed|save_path}`,
`world.save {path}`, `world.ground_lut {}` (announces a TEXTURE binary frame
carrying the procedural-ground palette LUT; result also returns
`ground_seed`/`ground_texels_per_m` for the client-side ground shader),
`chunks.set_center {x,y,z,radius?,resend?}` (`resend: true` clears the daemon's
sent-chunk cache so a freshly attached client gets the full region), `scene.stats`,
`terrain.raycast {o*,d*,max_distance?}`, `terrain.brush {shape,x,y,z,mode,…}`,
`edit.undo`, `edit.redo` (terrain AND scene ops — one chronological stack),
`scene.tree` (object dicts now include `components`), `scene.catalog {}` (built-in
component types + field specs), `scene.create {kind,parent?,name?,x?,y?,z?}`,
`scene.rename {id,name}`, `scene.reparent {id,parent?}`,
`scene.set_transform {id,p*?,r*?,s*?}`, `scene.delete {id}`,
`scene.add_component {id,type}`, `scene.remove_component {id,index}`,
`scene.set_component {id,index,params?,enabled?}`. Notifications: `log`,
`chunk.ready`, `chunk.unload`, `stream.done`, `edit.state`, `scene.changed`. Full
table in `editor/protocol/SCHEMA.md`.

**Extension webview (Scene View, F1/F2):** `editor/extension/src/webview/sceneView.ts`
— three.js viewport, Z-up, **Unity-style editor camera** (right-drag look +
WASD/QE flythrough, middle-drag pan, scroll dolly, Alt+left orbit; the cursor is
free otherwise). Builds a `BufferGeometry` per chunk from MESH frames. **Ground
is textured, not lit** — `src/webview/groundMaterial.ts` is a full-bright GLSL3
`ShaderMaterial` port of the game's `terrain.frag` ground albedo (same hash /
octaves / 4-corner texel-coverage filtering / palette LUT), fed by the
`world.ground_lut` TEXTURE frame (`texturePayload.ts` decode) and `ground_seed`/
`ground_texels_per_m` config; it composes `alb * v_color.rgb` with NO cascades/
fog/tonemap (the owner wants no lighting in the editor). A boot-time
`MeshBasicMaterial` (vertex colours) shows until the LUT arrives, then every
chunk swaps. Gotchas (mirror `terrain.frag` HEAD): `mpp` is **analytic** (never
`fwidth`); the vec4 `color` attribute is self-declared in the vertex shader with
`vertexColors:false` so the alpha material-id survives; the LUT row is clamped so
blocky-mesher `alpha==1.0` degrades to the last palette row. Overlays (wireframe
`G`, chunk borders `B`, stats). Brush palette (shape/mode/size/material) with a
wireframe **brush preview gizmo** that tracks the hovered terrain point; left-click
selects an object gizmo if one is under the cursor, else carves
(`terrain.raycast` → `terrain.brush`). Placeable objects render as coloured
gizmos parented to mirror the hierarchy; the selected one wears a yellow box and
a three.js **`TransformControls`** move/rotate/scale gizmo (`W`/`E`/`R` or the
palette buttons; local space; axis + plane + uniform handles). Gizmo drags send
throttled `transform` messages → `scene.set_transform`; the dragged id is echo-
suppressed so daemon round-trips don't snap it back, and `dragging-changed`
sends one final un-throttled transform (so the drag coalesces to a single undo
step). Carve/select is suppressed while `tc.dragging || tc.axis !== null`. `F`
frames the selection, `Esc` deselects. `Ctrl+Z`/`Ctrl+Y` undo/redo, dirty
indicator from `edit.state`. Host side: `sceneViewPanel.ts` relays MESH + TEXTURE
(`postTexture`) frames + object/select/frame messages down and camera/focus/edit/
select/transform messages up.

**Extension hierarchy (F2):** `editor/extension/src/hierarchyView.ts` — a native
sidebar `TreeView` (`HierarchyProvider`, activity-bar container `fireEditor`)
backed by the `scene.tree` cache, refreshed on `scene.changed`. Drag-and-drop
reparents (`scene.reparent`); context menu creates/renames/deletes; selection is
synced both ways with the viewport (tree select → highlight gizmo; gizmo click →
`reveal` the node).

**Extension inspector (properties panel):** `editor/extension/src/inspectorViewProvider.ts`
(a `WebviewView` below the Hierarchy) + `src/webview/inspector.ts` (plain-DOM
form bundled to `media/inspector.js`). Shows the selected object's name,
kind/id, the **Transform** (position (m), rotation as XYZ Euler **degrees** —
converted to/from the wire's scalar-first quaternion by
`src/webview/inspectorMath.ts`, unit-tested, with a uniform-scale lock), and a
Unity-style **component stack** below it: one section per `components` entry
(enable checkbox + remove ✕) with fields generated from the `scene.catalog`
descriptors (number / color picker / enum / bool), plus a `[+ Add]` dropdown of
catalog types not already present (singletons hidden once added). Edits commit on
change/Enter/blur → `scene.rename` / `scene.set_transform` /
`scene.add_component` / `scene.remove_component` / `scene.set_component`; the host
fetches `scene.catalog` once after `world.open` and forwards it (`postCatalog`,
cached for lazy view resolve). Incoming `scene.changed` refreshes the form but
never clobbers the field being typed in (echo guard); component sections rebuild
only on a structural change (component count/type), values patch in place.
Selection is centralised in `extension.ts::setSelection` — tree click ↔ viewport
click ↔ inspector always agree.

**Save Scene:** command `fireEditor.saveScene` (+ `saveSceneAs`), bound to
**Ctrl+S** when the Scene View panel or Hierarchy view is focused. First save
opens a dialog defaulting to `scenes/` at the repo root (committed authored
content — DECISIONS.md 2026-06-12); `Open World from Save` primes the same path
so Ctrl+S round-trips the opened file. Load the result in the game with
`python main.py --load scenes/<file>.ta`.

**CLI:** `python -m fire_editor --port <p> [--host 127.0.0.1] [--log-level info]`
— announces `{"event":"listening","port":N}` on stdout; logs to stderr.

**Agent harness:** an agent can drive *and* visually verify the editor headlessly
(no VS Code) via `fire_editor.EditorClient`/`spawn_daemon`, the
`tools/editor_client.py` CLI (`serve` hosts a browser viewport that runs the same
`sceneView.js` bundle), and `window.fireHarness`/`__fireSceneDebug` in that page.
See `docs/systems/editor_harness.md`.

**Protocol — `editor/protocol/` (single source):**
- `schema.json` — the one source of truth for the wire protocol.
- `codegen.py` — regenerates `fire_editor/_generated.py` and `extension/src/protocol/generated.ts`. Run `python editor/protocol/codegen.py` after any `schema.json` change (hard rule 6).
- `SCHEMA.md` — human-readable protocol reference.

**Extension — `editor/extension/`:** activates on startup; commands
`Fire Editor: Restart Daemon`, `Fire Editor: Show Daemon Log`,
`Fire Editor: Show Status`, `Open Scene View`, `Open World by Seed/Save`, and the
hierarchy commands (`Create Empty/Cube/Sphere/Light/Spawn`, `Rename`, `Delete`,
`Frame in Scene View`, `Refresh`); settings `fireEditor.pythonPath`,
`fireEditor.autoStart`, `fireEditor.autoOpenSceneView`, `fireEditor.logLevel`.
The repo root is resolved from an open workspace folder **or** the extension's own
install path, so the daemon starts even when the Extension Development Host opens
with no folder. The Scene View auto-opens once the daemon connects.

## Imports Allowed
The daemon may import **any headless `fire_engine` public API** (`core`,
`terrain`, `save`, `procedural`, `lighting`, and the `world/` object model —
`Transform`, `Component`, `GameObject`, `ComponentRegistry`, `instantiate`,
etc.) plus `websockets`, `msgpack`, `numpy`. It may **never** import `panda3d`
or the panda3d-bound `world/` bridges (`app`, `camera`, `geometry_bridge`,
`texture_bridge`, `resource_adapter`) — enforced by
`tests/editor/test_no_panda3d.py`. Engine access goes through documented public
APIs only (hard rule 2); if something private is needed, add a public engine API
with docs rather than reaching in.

## Events
The editor does not use the engine Event Bus across the process boundary.
Daemon → client communication uses **JSON-RPC notifications** over the socket
(e.g. `log`, and from later phases: chunk-ready, scene-changed, selection-sync,
watch events, progress). Inside the daemon, services may subscribe to engine
events (e.g. `TerrainEditedEvent`) to know which chunks to remesh.

## Units & Invariants
- Transport: WebSocket on `127.0.0.1:<port>` (port OS-assigned via `--port 0`).
- Control channel: JSON-RPC 2.0 (text frames). Binary channel: same socket,
  frames `[u32 magic][u32 schema_id][u32 payload_id][payload]`, **little-endian**,
  magic `0x46495245`. Bulk data (meshes/textures) is binary, never base64 JSON
  (hard rule 5).
- `PROTOCOL_VERSION` is exchanged in the `hello` handshake; mismatch →
  `VERSION_MISMATCH` error and the extension prompts to rebuild. Any schema
  change bumps `protocol_version` and regenerates both bindings in the same
  commit (hard rule 6).
- Determinism: the daemon sets the world seed via the engine's RNG service and
  introduces no unseeded randomness — editor preview of seed N matches the game
  world of seed N (hard rule 4).
- Versions: the daemon reports `fire_engine.__version__` in the handshake; the
  extension surfaces it.

## Examples
Start the daemon and shake hands (the path the extension automates):

```python
import asyncio, json, websockets
from fire_editor import Daemon
from fire_editor._generated import PROTOCOL_VERSION

async def main():
    d = Daemon()
    port = await d.server.start(0)                 # OS-assigned port
    async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
        await ws.send(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "hello",
                                  "params": {"protocol_version": PROTOCOL_VERSION,
                                             "client": "example"}}))
        print(json.loads(await ws.recv())["result"])   # {'ok': True, 'engine_version': ...}
    await d.server.close()

asyncio.run(main())
```

Encode/decode a binary payload frame:

```python
from fire_editor import encode_frame, decode_frame
from fire_editor._generated import SchemaId
frame = encode_frame(SchemaId.TEXTURE, 7, rgba_bytes)
schema_id, payload_id, payload = decode_frame(frame)   # (2, 7, rgba_bytes)
```

Regenerate the bindings after editing `schema.json`:

```
python editor/protocol/codegen.py
```

## Gotchas
- **Never hand-edit `_generated.py` / `generated.ts`** — they are codegen output.
  `tests/editor/test_protocol.py::TestCodegenConsistency` fails the build if the
  committed Python binding drifts from `schema.json`.
- The daemon prints exactly **one** machine-readable line to stdout (the
  listening port). Everything else must go to stderr/logging, or the extension's
  stdout parser will choke.
- `python -m fire_editor` needs both the repo root (for `fire_engine`) and
  `editor/` (for `fire_editor`) on `PYTHONPATH`; the extension sets this when it
  spawns the daemon, and `tests/editor/conftest.py` sets it for tests.
- The extension auto-respawns the daemon up to 5 times with backoff; a genuine
  crash loop ends in the `crashed` status and an error toast — check
  **Fire Editor: Show Daemon Log**.
- Binary frames use `max_size=None` on the server so large mesh payloads are not
  rejected by the default 1 MiB websockets frame cap.
