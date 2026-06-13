// Scene View panel (extension host side, EDITOR_PRD F1).
// Owns the WebviewPanel, builds its HTML (CSP + nonce + bundled three.js app),
// relays mesh/unload/config down to the webview and camera moves back up.
import * as vscode from "vscode";

export type MessageHandler = (msg: Record<string, unknown>) => void;

export class SceneViewPanel {
  static current: SceneViewPanel | undefined;
  private readonly disposables: vscode.Disposable[] = [];
  private readyResolved = false;

  private constructor(
    private readonly panel: vscode.WebviewPanel,
    private readonly extensionUri: vscode.Uri,
    private readonly onMessage: MessageHandler,
    private readonly onReady: () => void
  ) {
    this.panel.webview.html = this.html();
    this.panel.onDidDispose(() => this.dispose(), null, this.disposables);
    this.panel.webview.onDidReceiveMessage(
      (msg) => {
        if (msg?.type === "ready" && !this.readyResolved) {
          this.readyResolved = true;
          this.onReady();
        } else if (msg) {
          this.onMessage(msg);
        }
      },
      null,
      this.disposables
    );
  }

  static createOrShow(
    extensionUri: vscode.Uri,
    onMessage: MessageHandler,
    onReady: () => void
  ): SceneViewPanel {
    if (SceneViewPanel.current) {
      SceneViewPanel.current.panel.reveal(vscode.ViewColumn.One);
      return SceneViewPanel.current;
    }
    const panel = vscode.window.createWebviewPanel(
      "fireEditor.sceneView",
      "Fire Editor — Scene View",
      vscode.ViewColumn.One,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [vscode.Uri.joinPath(extensionUri, "media")],
      }
    );
    SceneViewPanel.current = new SceneViewPanel(panel, extensionUri, onMessage, onReady);
    return SceneViewPanel.current;
  }

  postEditState(state: unknown): void {
    this.panel.webview.postMessage({ type: "editState", state });
  }

  postMesh(payload: Uint8Array): void {
    this.panel.webview.postMessage({ type: "mesh", payload });
  }
  /** Relay a TEXTURE binary frame (currently the procedural-ground LUT). */
  postTexture(payload: Uint8Array): void {
    this.panel.webview.postMessage({ type: "groundLut", payload });
  }
  postUnload(coord: [number, number, number]): void {
    this.panel.webview.postMessage({ type: "unload", coord });
  }
  postConfig(config: unknown): void {
    this.panel.webview.postMessage({ type: "config", config });
  }
  /** Replace the viewport's object gizmos from a scene.tree payload. */
  postObjects(objects: unknown): void {
    this.panel.webview.postMessage({ type: "objects", objects });
  }
  /** Highlight an object in the viewport (null clears selection). */
  postSelect(id: number | null): void {
    this.panel.webview.postMessage({ type: "select", id });
  }
  /** Move the camera to frame an object. */
  postFrame(id: number): void {
    this.panel.webview.postMessage({ type: "frame", id });
  }
  reset(): void {
    this.panel.webview.postMessage({ type: "reset" });
  }

  private html(): string {
    const webview = this.panel.webview;
    const scriptUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this.extensionUri, "media", "sceneView.js")
    );
    const nonce = makeNonce();
    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta http-equiv="Content-Security-Policy"
  content="default-src 'none'; img-src ${webview.cspSource} data:; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<style>
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
</style>
</head>
<body>
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
  <script nonce="${nonce}" src="${scriptUri}"></script>
</body>
</html>`;
  }

  dispose(): void {
    SceneViewPanel.current = undefined;
    this.panel.dispose();
    while (this.disposables.length) this.disposables.pop()?.dispose();
  }
}

function makeNonce(): string {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let s = "";
  for (let i = 0; i < 32; i++) s += chars.charAt(Math.floor(Math.random() * chars.length));
  return s;
}
