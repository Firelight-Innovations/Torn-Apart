// Scene View panel (extension host side, EDITOR_PRD F1).
// Owns the WebviewPanel, builds its HTML (CSP + nonce + bundled three.js app),
// relays mesh/unload/config down to the webview and camera moves back up.
import * as vscode from "vscode";

export type CameraHandler = (x: number, y: number, z: number) => void;

export class SceneViewPanel {
  static current: SceneViewPanel | undefined;
  private readonly disposables: vscode.Disposable[] = [];
  private readyResolved = false;

  private constructor(
    private readonly panel: vscode.WebviewPanel,
    private readonly extensionUri: vscode.Uri,
    private readonly onCamera: CameraHandler,
    private readonly onReady: () => void
  ) {
    this.panel.webview.html = this.html();
    this.panel.onDidDispose(() => this.dispose(), null, this.disposables);
    this.panel.webview.onDidReceiveMessage(
      (msg) => {
        if (msg?.type === "camera") this.onCamera(msg.x, msg.y, msg.z);
        else if (msg?.type === "ready" && !this.readyResolved) {
          this.readyResolved = true;
          this.onReady();
        }
      },
      null,
      this.disposables
    );
  }

  static createOrShow(
    extensionUri: vscode.Uri,
    onCamera: CameraHandler,
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
    SceneViewPanel.current = new SceneViewPanel(panel, extensionUri, onCamera, onReady);
    return SceneViewPanel.current;
  }

  postMesh(payload: Uint8Array): void {
    this.panel.webview.postMessage({ type: "mesh", payload });
  }
  postUnload(coord: [number, number, number]): void {
    this.panel.webview.postMessage({ type: "unload", coord });
  }
  postConfig(config: unknown): void {
    this.panel.webview.postMessage({ type: "config", config });
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
</style>
</head>
<body>
  <div id="stats">waiting for daemon…</div>
  <div id="hint">click to capture mouse · WASD move · Q/E down/up · Shift fast · G wireframe · B borders</div>
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
