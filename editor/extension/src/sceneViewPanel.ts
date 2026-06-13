// Scene View panel (extension host side, EDITOR_PRD F1).
// Owns the WebviewPanel, builds its HTML (CSP + nonce + bundled three.js app),
// relays mesh/unload/config down to the webview and camera moves back up.
import * as vscode from "vscode";
import { VIEWPORT_CSS, VIEWPORT_BODY_HTML } from "./webview/viewportMarkup";

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
<style>${VIEWPORT_CSS}</style>
</head>
<body>
${VIEWPORT_BODY_HTML}
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
