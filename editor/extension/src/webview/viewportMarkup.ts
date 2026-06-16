// Shared viewport markup — the CSS + body HTML for the Scene View, injected by
// the VS Code panel (`sceneViewPanel.ts`). Keeping it in one module means any
// new viewport UI (gizmo buttons, palette fields, hints) has a single home.
//
// IMPORTANT: every element id `sceneView.ts` dereferences must live here. The
// `viewportMarkup.test.ts` test guards that contract.

export const VIEWPORT_CSS = `
  html, body { margin: 0; padding: 0; overflow: hidden; background: #101418; height: 100%; }
  #stats {
    position: fixed; top: 8px; left: 10px; z-index: 10;
    font: 12px/1.4 var(--vscode-editor-font-family, monospace);
    color: #cfe3f2; text-shadow: 0 1px 2px #000; pointer-events: none;
  }
  #hint {
    position: fixed; bottom: 8px; left: 10px; z-index: 10;
    font: 11px var(--vscode-editor-font-family, monospace); color: #7da3ba;
  }
  #palette {
    position: fixed; top: 8px; right: 10px; z-index: 10;
    background: rgba(16,20,24,0.85); border: 1px solid #2a3a48; border-radius: 6px;
    padding: 8px 10px; color: #cfe3f2;
    font: 12px var(--vscode-editor-font-family, monospace); display: flex; flex-direction: column; gap: 6px;
  }
  #palette label { display: flex; justify-content: space-between; gap: 8px; align-items: center; }
  #palette select, #palette input { background: #1a2430; color: #cfe3f2; border: 1px solid #2a3a48; }
  #dirty { color: #e0b341; }
  #gizmoModes { display: flex; gap: 4px; }
  #gizmoModes button {
    flex: 1; background: #1a2430; color: #cfe3f2; border: 1px solid #2a3a48;
    border-radius: 3px; padding: 2px 0; cursor: pointer;
    font: 11px var(--vscode-editor-font-family, monospace);
  }
  #gizmoModes button.active { background: #2d4a66; border-color: #4f9fe0; }
`;

export const VIEWPORT_BODY_HTML = `
  <div id="stats">waiting for daemon…</div>
  <div id="palette">
    <div id="gizmoModes">
      <button id="gizmoMove" title="Move (W)">Move</button>
      <button id="gizmoRotate" title="Rotate (E)">Rotate</button>
      <button id="gizmoScale" title="Scale (R)">Scale</button>
    </div>
    <strong>Brush <span id="dirty"></span></strong>
    <label>shape
      <select id="brushShape">
        <option value="sphere">sphere</option>
        <option value="box">box</option>
        <option value="cylinder">cylinder</option>
      </select>
    </label>
    <label>mode
      <select id="brushMode">
        <option value="remove">remove</option>
        <option value="add">add</option>
      </select>
    </label>
    <label>size <input id="brushSize" type="range" min="0.5" max="8" step="0.5" value="2" /></label>
    <label>material <input id="brushMaterial" type="number" min="0" max="255" value="1" style="width:48px" /></label>
  </div>
  <div id="hint">Right-drag look + WASD/QE fly · Middle-drag pan · Scroll zoom · Alt+Left orbit · Left-click select/carve · W/E/R move/rotate/scale gizmo · F frame · Esc deselect · Ctrl+Z/Y undo · G wire · B borders</div>
`;
