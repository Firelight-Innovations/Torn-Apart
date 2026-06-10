# editor — System Doc
keywords: editor, fire editor, daemon, fire_editor, extension, vscode, cursor, websocket, json-rpc, protocol, schema, codegen, binary frame, handshake, hello, scene view, hierarchy, inspector, gizmo, texture lab, model workspace

> Documents the `editor/` tree: the headless Python daemon `editor/fire_editor/`,
> the TypeScript VS Code/Cursor extension `editor/extension/`, and the shared
> protocol `editor/protocol/`. The editor is **not** part of the runtime engine —
> it imports `torn_apart` public APIs to drive an offline visual editor.
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
delta saves). E2 (hierarchy/inspector — blocked on an engine `world/` panda3d
split), E4 (texture lab), E5 (model workspace), E6 (verification) remain per
EDITOR_PRD §7.

## Public API
The editor is a standalone tool, not an importable engine package. Its surfaces:

**Daemon — `fire_editor` (importable for tests):**
- `Daemon` — builds the JSON-RPC dispatcher + WebSocket server, registers core + service methods, holds the open `EditorSession`, `run(port)`.
- `Dispatcher` / `RpcError` — transport-agnostic JSON-RPC 2.0 dispatch; handlers are `async (params) -> result`.
- `EditorSession` — one open world: terrain `ChunkManager`, `LightGrid` + `SunlightComputer`, `SaveManager`. `from_seed`, `from_save`, `region_coords`, `ensure_loaded`, `relight`, `mesh`, `raycast`, `save`.
- `encode_frame` / `decode_frame` — protocol binary framing.
- `encode_mesh_payload(coord, mesh)` / `decode_mesh_payload(bytes)` — MESH payload codec.
- `EditorServer` — `websockets` transport; `broadcast_binary`, `broadcast_notification`.
- `services.chunks.ChunkService` — registers `world.open/save`, `chunks.set_center`, `scene.stats`, `terrain.raycast`, `terrain.brush`, `edit.undo/redo`; streams MESH frames and drives the undo stack.
- `commands.UndoStack` / `EditCommand` — editor-side undo/redo: per-edit before/after material snapshots over the brush AABB chunks (EDITOR_PRD §5.4).
- Generated constants in `fire_editor._generated` (`PROTOCOL_VERSION`, `BINARY_MAGIC`, `SchemaId`, `ErrorCode`, `Method`, `Notification`, typed param/result `TypedDict`s).

**Methods (protocol_version 3):** `hello`, `ping`, `world.open {seed|save_path}`,
`world.save {path}`, `chunks.set_center {x,y,z,radius?}`, `scene.stats`,
`terrain.raycast {o*,d*,max_distance?}`, `terrain.brush {shape,x,y,z,mode,…}`,
`edit.undo`, `edit.redo`. Notifications: `log`, `chunk.ready`, `chunk.unload`,
`stream.done`, `edit.state`. Full table in `editor/protocol/SCHEMA.md`.

**Extension webview (Scene View, F1):** `editor/extension/src/webview/sceneView.ts`
— three.js viewport, Z-up, fly camera (WASD/QE, mouse-look, Shift 5×), builds a
`BufferGeometry` per chunk from MESH frames, `MeshBasicMaterial` with vertex
colours (baked sunlight), overlays (wireframe `G`, chunk borders `B`, stats).
Brush palette (shape/mode/size/material) + crosshair; left-click while in
look-mode carves at the crosshair (`terrain.raycast` → `terrain.brush`),
`Ctrl+Z`/`Ctrl+Y` undo/redo, dirty indicator from `edit.state`.
Host side: `sceneViewPanel.ts` relays MESH frames down and camera/edit/undo
messages up.

**CLI:** `python -m fire_editor --port <p> [--host 127.0.0.1] [--log-level info]`
— announces `{"event":"listening","port":N}` on stdout; logs to stderr.

**Protocol — `editor/protocol/` (single source):**
- `schema.json` — the one source of truth for the wire protocol.
- `codegen.py` — regenerates `fire_editor/_generated.py` and `extension/src/protocol/generated.ts`. Run `python editor/protocol/codegen.py` after any `schema.json` change (hard rule 6).
- `SCHEMA.md` — human-readable protocol reference.

**Extension — `editor/extension/`:** activates on startup; commands
`Fire Editor: Restart Daemon`, `Fire Editor: Show Daemon Log`,
`Fire Editor: Show Status`; settings `fireEditor.pythonPath`,
`fireEditor.autoStart`, `fireEditor.logLevel`.

## Imports Allowed
The daemon may import **any headless `torn_apart` public API** (`core`,
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
- Versions: the daemon reports `torn_apart.__version__` in the handshake; the
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
- `python -m fire_editor` needs both the repo root (for `torn_apart`) and
  `editor/` (for `fire_editor`) on `PYTHONPATH`; the extension sets this when it
  spawns the daemon, and `tests/editor/conftest.py` sets it for tests.
- The extension auto-respawns the daemon up to 5 times with backoff; a genuine
  crash loop ends in the `crashed` status and an error toast — check
  **Fire Editor: Show Daemon Log**.
- Binary frames use `max_size=None` on the server so large mesh payloads are not
  rejected by the default 1 MiB websockets frame cap.
