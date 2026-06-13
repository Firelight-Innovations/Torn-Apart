# editor_harness — System Doc
keywords: editor harness, agent access, editor client, EditorClient, editor_client.py, serve, browser harness, viewport harness, headless editor, drive the editor, fireHarness, __fireSceneDebug, chrome mcp, screenshot the editor, spawn_daemon, rpcSession, host shim, cameraPose

> How an agent (or human, or CI) drives and visually verifies the Fire Editor
> without clicking in VS Code — the editor analogue of `tools/screenshot.py` for
> the game. See `docs/systems/editor.md` for the daemon/protocol itself.

## Role
Gives an automated agent two ways to operate the editor headlessly: a **Python
client + CLI** that speaks the daemon's WebSocket protocol (open worlds, stream
chunks, carve terrain, place/transform objects, undo/redo, save), and a
**browser viewport harness** — a plain web page that runs the *exact same*
three.js viewport bundle the VS Code panel uses, so Chrome screenshots match the
editor pixel-for-pixel. It deliberately does NOT add new RPC methods (it reuses
the existing protocol) and does NOT replace the VS Code extension (it's a
parallel, scriptable front door to the same daemon).

## Public API
- `fire_editor.EditorClient` — async WebSocket client: `connect(port)`, `hello()`,
  `request(method, params)`, `notify()`, `wait_notification(method)`,
  `drain_until_stream_done(trigger)` (collect MESH frames until `stream.done`),
  `close()`. Logs every notification (`.notifications`) and binary frame
  (`.binary_frames`); optional `on_notification`/`on_binary` hooks.
- `fire_editor.spawn_daemon()` — async context manager: spawns `python -m
  fire_editor --port 0` with the right cwd/PYTHONPATH, yields `(proc, port)`,
  terminates on exit.
- `fire_editor.RpcRemoteError` / `fire_editor.BinaryFrame` — error + decoded-frame types.
- `tools/editor_client.py` — CLI; subcommands map 1:1 to RPC methods plus `serve`
  (persistent daemon + HTTP host for the browser harness) and `rpc`/`watch`.
- `editor/extension/src/protocol/rpcSession.ts` — `RpcSession`, the
  transport-agnostic JSON-RPC core shared by the extension `client.ts` and the
  browser `harnessBoot.ts` (one protocol implementation, no drift).
- `editor/extension/src/webview/host.ts` — `host.post()`, the transport seam the
  viewport uses (VS Code API in the panel; `window.__fireEditorHost` in the harness).
- `editor/extension/src/webview/viewportMarkup.ts` — `VIEWPORT_CSS` /
  `VIEWPORT_BODY_HTML`, the shared viewport markup (panel + harness).
- Browser globals (in the harness page): `window.fireHarness`
  (`rpc/open/setCamera/select/waitForStreamDone/state/events`) and
  `window.__fireSceneDebug.snapshot()` (chunks/verts/tris/objects/selected/
  hasGround/fps/camera) for numeric assertions and Chrome MCP console reads.

## Imports Allowed
- `fire_editor.client` imports `websockets`, `fire_editor._generated`,
  `fire_editor.binary` — **never panda3d** (hard rule 1; guarded by
  `tests/editor/test_no_panda3d.py`).
- `tools/editor_client.py` imports `fire_editor` + stdlib (`http.server`,
  `asyncio`, `argparse`) only.
- Browser TS: `harnessBoot.ts`/`host.ts`/`viewportMarkup.ts` import the shared
  `protocol/` + `generated.ts`; no `vscode`/`ws`.

## Events
This package emits no engine `*Event`s. It consumes the daemon's JSON-RPC
**notifications** (`chunk.ready`, `chunk.unload`, `stream.done`, `edit.state`,
`scene.changed`, `log`) and **binary frames** (`SchemaId.MESH`, `SchemaId.TEXTURE`).
`drain_until_stream_done` keys off `stream.done`; the browser harness relays each
notification/frame into the viewport via a synthetic `window` `message` event.

## Units & Invariants
- Ports: daemon WebSocket default `8123`, harness HTTP default `8770` (CLI flags).
- The daemon broadcasts to **all** connected clients, so the CLI and the browser
  harness can drive the same world simultaneously — CLI edits render live.
- **Resend invariant:** a second client attaching to a running daemon gets chunk
  meshes only with `chunks.set_center {resend: true}` (the sent-chunk cache is
  daemon-global). The harness always boots with `resend: true`.
- Camera poses via `cameraPose` are deterministic (explicit position + look-at
  target or yaw/pitch), so screenshots are reproducible.
- Determinism: the editor ground pattern for a seed matches the game's (same
  `for_domain("terrain","ground")` derivation) — verify by eye across editor and
  `python main.py` at the same seed.

## Examples
Persistent server + scripted edits + a live browser view:
```bash
# Terminal 1 — daemon + harness host (prints the harness URL):
python tools/editor_client.py serve --port 8123 --http-port 8770 --seed 1337

# Terminal 2 — drive that same daemon (edits appear live in the browser):
python tools/editor_client.py --port 8123 open --seed 1337
python tools/editor_client.py --port 8123 set-center --x 0 --y 0 --z 24 --resend --wait
python tools/editor_client.py --port 8123 brush --x 0 --y 0 --z 7.5 --mode remove --radius 2 --wait
python tools/editor_client.py --port 8123 create cube --x 2 --y 0 --z 8
python tools/editor_client.py --port 8123 save --path scenes/demo.ta
```
Chrome MCP visual check (textured terrain, gizmo, crater):
```
navigate  http://127.0.0.1:8770/harness/?port=8123&seed=1337&cam=20,-20,24
# read console for "[harness] boot complete"; then:
javascript  window.__fireSceneDebug.snapshot()          // numeric assertions
javascript  await window.fireHarness.waitForStreamDone() // settle before screenshot
javascript  window.fireHarness.select(1)                 // attach the gizmo to object 1
screenshot
```
Python, one-shot (throwaway daemon — fine for stateless checks):
```python
import asyncio
from fire_editor import EditorClient, spawn_daemon

async def main():
    async with spawn_daemon() as (_proc, port):
        c = EditorClient(); await c.connect(port); await c.hello("agent")
        await c.request("world.open", {"seed": 1337})
        frames = await c.drain_until_stream_done(
            lambda: c.request("chunks.set_center", {"x":0,"y":0,"z":24,"resend":True}))
        print("meshes:", sum(f.schema_id == 1 for f in frames))
        await c.close()

asyncio.run(main())
```

## Gotchas
- **One-shot CLI calls are stateless.** Without `--port`, each invocation spawns
  a fresh daemon (new world), so `open` then `tree` in two calls won't share
  state. Use `serve` + `--port` for any multi-step sequence.
- **Script load order in the harness HTML matters.** `harnessBoot.js` must load
  *before* `sceneView.js`: it injects the viewport markup (which `sceneView.ts`
  dereferences at module load) and installs `window.__fireEditorHost` before the
  viewport posts its first message. `test_harness_files.py` guards the order.
- **`acquireVsCodeApi()` may be called once per webview.** `host.ts` is the only
  call site; never call it elsewhere or the panel throws. In the harness it's
  absent, so `host` falls back to `window.__fireEditorHost`.
- **`drain_until_stream_done` waits for the NEXT `stream.done`,** not a
  historical one — register-then-trigger is safe, but a bare wait with a stream
  already finished will block until the next stream.
- **The daemon has no origin check** (localhost dev tool). The HTTP host in
  `serve` is `SimpleHTTPRequestHandler` over `editor/extension/` — don't expose
  either port beyond `127.0.0.1`.
- **Browser bundles must be built** (`npm run compile` in `editor/extension`)
  before `serve` can host them; `test_harness_files.py` skips with a hint when
  `media/` is missing.
- GLSL/viewport changes land in the harness automatically because both the panel
  and the harness consume `viewportMarkup.ts` + the same `sceneView.js` — keep
  new viewport UI in those shared modules, not inlined in `sceneViewPanel.ts`.
