# Fire Editor — Iteration 3 Handoff

> Brief for the next agent picking up Fire Editor work. Iteration 2 (this
> session, 2026-06-16) collapsed the editor to a **single UI** and gave AI agents
> a real **offscreen screenshot** path. Read this, then `docs/systems/editor.md`
> (the editor system doc) and `docs/systems/render.md` (`offscreen.py` section)
> before changing anything.

PR branch for iter-2 work: `editor/single-ui-and-screenshot` (4 commits off
`master`, NOT yet merged). The owner's flora work lives on its own branch and is
intentionally **not** in this PR.

---

## 0. Where the editor stands now

The Fire Editor has **one** UI: the VS Code extension (`editor/extension/`). It
spawns a headless Python daemon (`python -m fire_editor`, panda3d-free) and talks
to it over WebSocket JSON-RPC; the Scene View webview renders mesh/texture frames
with three.js.

AI agents do **not** use a browser UI anymore. They drive the world through:
- **Python calls** — `fire_editor.EditorClient` / `spawn_daemon`, or the
  `tools/editor_client.py` CLI (every subcommand ≈ one RPC method).
- **Screenshots** — `world.screenshot` renders the current live-edited world
  **offscreen on a GPU** and returns a PNG path the agent reads back.

This replaced the deleted standalone browser viewport harness (`harnessBoot.ts`,
`harness/index.html`, the HTTP host in `editor_client.py serve`).

Wire protocol is at **version 7** (`editor/protocol/schema.json`); the bindings
in `editor/fire_editor/_generated.py` and
`editor/extension/src/protocol/generated.ts` are codegen output — never hand-edit,
always `python editor/protocol/codegen.py` and commit all three together.

---

## 1. What iteration 2 did (this session)

Commits (oldest→newest) on the PR branch:
1. `fix(editor): make F5 launch self-install extension deps` — `.vscode/tasks.json`
   self-installs `editor/extension/node_modules` before bundling (the `Could not
   resolve "ws"` F5 failure). See `[[fire-editor-f5-deps]]` memory.
2. `fix(editor): give Scene View canvas keyboard focus in VS Code webview` —
   `sceneView.ts` makes the canvas focusable (`tabIndex`, focus-on-load /
   pointerdown) so WASD/keys reach the viewport inside the webview iframe.
   **Owner still to verify via F5** (was pending at session end).
3. `refactor(editor): remove browser harness, single VS Code editor` — deleted the
   browser twin; `host.ts` calls `acquireVsCodeApi()` unconditionally; `serve` is a
   long-lived headless daemon (no HTTP host / `--http-port/--seed/--cam`).
4. `feat(editor): world.screenshot offscreen render RPC` — the new agent "see the
   world" path (details below).

DECISIONS.md has the dated rationale for #3/#4.

---

## 2. `world.screenshot` architecture (read before touching it)

The daemon is panda3d-free (Hard Rule 1), so it **cannot render in-process**. The
flow:

1. `ChunkService.screenshot` (`editor/fire_editor/services/chunks.py`) temp-saves
   the live `EditorSession` into a **fresh temp dir** (a dir, not an open handle —
   `SaveManager.save` does an atomic `os.replace` that fails on open files on
   Windows).
2. It spawns a separate render subprocess
   `python -m fire_engine.render._impl.offscreen` with `PYTHONPATH`/`cwd` wired
   like `client.spawn_daemon` (repo root + `editor/`). The single subprocess seam
   is `ChunkService._run_offscreen` (tests stub it).
3. `render_offscreen()` (`fire_engine/render/_impl/offscreen.py`) sets
   `window-type offscreen` + `win-size W H` via `loadPrcFileData` **before**
   ShowBase, calls `build_demo(load_path=, seed=, headless=True)`, poses the
   camera, steps `frames`, captures the framebuffer, writes the PNG, prints
   `SCREENSHOT_RESULT {json}` and `os._exit(0)`.
4. The daemon returns `{ok, path, width, height}` and rmtree's the temp save dir.

CLI: `editor_client.py --port <p> screenshot --px 0 --py -20 --pz 12 [--yaw
--pitch --width --height --frames --out]`.

### Landmines (each cost real time — don't relearn them)
- **Seed trap.** `SaveManager.load` rejects a save whose
  `world_seed`/`config_digest` differs from config. The subprocess MUST load the
  save with the **same seed it was written with** → the daemon passes
  `session.seed` as `--seed`, and `build_demo(seed=N)` does
  `dataclasses.replace(cfg, world_seed=N)` (mirrors `EditorSession.from_seed`). If
  you change how sessions pick seeds, fix both sides or you get
  `SaveIncompatibleError`.
- **Module cap.** `render/` is at the 10-module limit
  (`pyproject.toml [tool.firelight] max_modules_per_dir`), so the offscreen
  renderer had to go in `render/_impl/offscreen.py`, hence the `_impl` in the
  `python -m` path. Don't "move it up" into `render/` without sub-packaging.
- **GPU required.** Offscreen still needs a real GL context. No GPU → clear
  `RpcError(APP_ERROR)`, never a hang (subprocess `os._exit`s; daemon has a 180 s
  timeout). The first render is slow (cold GL + shader compiles).
- **`os._exit(0)` is mandatory** in the subprocess — a normal return leaves the
  Panda3D buffer / OpenAL device alive and the daemon's `proc.wait()` blocks until
  timeout.

---

## 3. How to verify (owner's Windows + GPU box)

```
# 1. Extension builds + TS tests
cd editor/extension && npm run compile && npm test

# 2. Protocol bindings have no drift
python editor/protocol/codegen.py            # must produce no git diff

# 3. Headless gates + orchestration (no GPU)
.venv/Scripts/python.exe -m pytest -q tests/editor tests/render -m "not window and not coverage"
.venv/Scripts/python.exe -m pytest -q tests/standards/test_repo_structure.py tests/standards/test_docs.py

# 4. REAL offscreen render (needs the GPU) — the one thing CI can't run
.venv/Scripts/python.exe -m pytest tests/render/test_offscreen.py -m window -q

# 5. End-to-end agent flow
python tools/editor_client.py serve --port 8123 &
python tools/editor_client.py --port 8123 open --seed 1337
python tools/editor_client.py --port 8123 brush --x 0 --y 0 --z 8 --mode remove --radius 3
python tools/editor_client.py --port 8123 create cube --x 3 --y 0 --z 9
python tools/editor_client.py --port 8123 screenshot --px 0 --py -20 --pz 12 --yaw 0 --pitch -20 --out shot.png
# open shot.png → crater + cube should be visible
```

As of this handoff, steps 1–3 are green locally; **step 4 (real GPU render) and
the in-VS-Code keyboard fix are UNVERIFIED on hardware** — do these first.

---

## 4. Repo facts that bite newcomers
- **`editor/` is NOT under `[tool.firelight] source_roots` (`["fire_engine"]`).**
  Editor modules are exempt from the line-count / module-count / test-mirror
  structure gates. `chunks.py` at ~509 lines is fine. Only `fire_engine/**` is
  gated. (So if you add render-side code, it IS gated — watch the caps.)
- **The standards code-quality gate is pre-existing RED repo-wide** on these
  branches: `main.py` has ~46 mypy `--strict` errors + ruff/format issues that
  predate this work. `test_repo_structure` / `test_docs` / protocol-drift are
  green. Don't fix `main.py` inline to make the gate pass — that's a separate
  delegated cleanup.
- **The `[17]` test-mirror exemption is AST-import based** (`_imports_panda3d`
  uses `ast.walk`, so it catches lazy imports inside functions). A render module
  that imports panda3d — even lazily inside a function, like `offscreen.py` — is
  exempt from the headless mirror requirement; it's integration-verified via a
  `@pytest.mark.window` test instead.
- **The main working tree is usually on a feature branch, not `master`**, and the
  owner runs **parallel worktrees/sessions** (flora, buildings, graphics-perf).
  `git worktree list` and `git status` before committing; commit only your own
  files (use explicit pathspecs), never bulk `git add -A`.

---

## 5. Candidate scope for iteration 3

Grounded in what exists; pick with the owner. Roughly ordered by leverage:

1. **Land iter-2 + verify on hardware.** Merge `editor/single-ui-and-screenshot`,
   then run §3 step 4 + the keyboard fix in-VS-Code. Everything else builds on a
   confirmed screenshot path.
2. **Screenshot ergonomics.** `world.screenshot` is v1: single GPU, serialized,
   look-at-origin default. Likely asks: a `look_at` target param (vs raw
   yaw/pitch), framing helpers (fit-object-in-view), time-of-day / weather
   overrides (the in-process `tools/screenshot.py` already supports these — factor
   the shared knobs), and returning image bytes/base64 for agents that can't read
   the daemon host's filesystem.
3. **Authoring breadth in the editor.** Buildings (`fire_engine/buildings/`) and
   the proc-gen content (flora/zones/biomes) are not yet placeable/editable from
   the editor. A `building.*` / content-placement RPC surface + inspector UI is the
   natural next authoring layer. (The buildings memory flagged "iter3 =
   editor/harness/proc-gen".)
4. **Editor scene → screenshot fidelity.** Confirm authored scene objects (cube/
   sphere/light/spawn) and component-driven visuals render in the offscreen shot
   exactly as in the live VS Code viewport; lights need the GPU lighting backend
   (the repo default) — document/guard the CPU-backend case.
5. **Daemon robustness.** Concurrent screenshots serialize on the GPU; consider a
   queue + explicit "busy" result, and surface cold-start latency to the client.

See the `[[fire-editor-one-ui-screenshot]]` and `[[editor-harness-session]]`
memories for prior editor context.
