# Fire Editor

A Unity-Editor-style visual editor for the Torn Apart ("Fire") engine, running
inside **VS Code / Cursor**. It runs the headless engine in a Python daemon and
renders mesh/texture output in webview panels — no Panda3D, the game stays
closed. Design: [`docs/EDITOR_PRD.md`](../docs/EDITOR_PRD.md). System reference:
[`docs/systems/editor.md`](../docs/systems/editor.md).

## Layout
```
editor/
  fire_editor/      Python daemon (uses the repo .venv; never imports panda3d)
  extension/        TypeScript VS Code / Cursor extension
  protocol/         schema.json (single source) + codegen.py + SCHEMA.md
```

## Quick start
```bash
# 1. daemon deps (in the repo .venv)
pip install -r requirements.txt

# 2. open the REPO ROOT in VS Code / Cursor, press F5
#    -> the "fire-editor: build" preLaunchTask runs `npm install`
#       (deps live in editor/extension/node_modules, gitignored) then
#       `npm run compile`, the Extension Development Host opens, the
#       daemon auto-spawns, status bar shows "Fire Editor: connected"
```

Pressing **F5** is enough on a fresh checkout — the launch task installs the
extension's npm deps (`three`, `ws`) before bundling. To build by hand instead:
```bash
cd editor/extension && npm install && npm run compile
```

## Tests
```bash
pytest -q tests/editor              # daemon (part of the headless suite)
cd editor/extension && npm test     # extension protocol unit tests
```

## Protocol changes
Edit `protocol/schema.json`, then regenerate both bindings and commit together:
```bash
python editor/protocol/codegen.py
```
Bump `protocol_version` on any incompatible change (EDITOR_PRD hard rule 6).
