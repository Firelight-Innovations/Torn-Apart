# Fire Editor (VS Code / Cursor extension)

Visual editor for the Torn Apart (Fire) engine. Spawns the headless
`fire_editor` Python daemon from the repo `.venv` and renders the engine's
mesh/texture output in webview panels. See `docs/EDITOR_PRD.md` and
`docs/systems/editor.md`.

## Develop

```bash
cd editor/extension
npm install
npm run check     # tsc type-check
npm run compile   # bundle to dist/extension.js
npm test          # pure-node protocol unit tests
```

Then open the **repo root** in VS Code / Cursor and press `F5` to launch an
Extension Development Host. With `fireEditor.autoStart` on (default), the
daemon spawns automatically and the status bar shows `Fire Editor: connected`.

## Commands

- **Fire Editor: Restart Daemon**
- **Fire Editor: Show Daemon Log**
- **Fire Editor: Show Status**

## Settings

- `fireEditor.pythonPath` — interpreter override (default: autodetect `.venv`).
- `fireEditor.autoStart` — spawn the daemon on workspace open (default `true`).
- `fireEditor.logLevel` — daemon verbosity (`debug|info|warning|error`).
