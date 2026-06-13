// Inspector view (extension host side) — a WebviewView in the fireEditor
// sidebar, below the Hierarchy. Hosts media/inspector.js (the properties
// form); relays the selected object down and rename/setTransform edits up.
import * as vscode from "vscode";

import { SceneObjectDTO } from "./hierarchyView";

export type InspectorMessageHandler = (msg: Record<string, unknown>) => void;

export class InspectorViewProvider implements vscode.WebviewViewProvider {
  static readonly viewId = "fireEditor.inspector";

  private view: vscode.WebviewView | undefined;
  private lastObject: SceneObjectDTO | null = null;
  private lastCatalog: unknown[] | null = null;

  constructor(
    private readonly extensionUri: vscode.Uri,
    private readonly onMessage: InspectorMessageHandler
  ) {}

  resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view;
    view.webview.options = {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this.extensionUri, "media")],
    };
    view.webview.html = this.html(view.webview);
    view.webview.onDidReceiveMessage((msg) => {
      if (msg?.type === "ready") {
        // The view resolves lazily (first reveal) — replay cached state.
        if (this.lastCatalog) this.postCatalog(this.lastCatalog);
        this.postObject(this.lastObject);
      } else if (msg) {
        this.onMessage(msg);
      }
    });
    view.onDidDispose(() => {
      if (this.view === view) this.view = undefined;
    });
  }

  /** Show an object's properties (null = "no selection"). Cached for lazy resolve. */
  postObject(obj: SceneObjectDTO | null): void {
    this.lastObject = obj;
    void this.view?.webview.postMessage({ type: "object", object: obj });
  }

  /** Provide the built-in component catalog (scene.catalog). Cached for lazy resolve. */
  postCatalog(catalog: unknown[]): void {
    this.lastCatalog = catalog;
    void this.view?.webview.postMessage({ type: "catalog", catalog });
  }

  private html(webview: vscode.Webview): string {
    const scriptUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this.extensionUri, "media", "inspector.js")
    );
    const nonce = makeNonce();
    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta http-equiv="Content-Security-Policy"
  content="default-src 'none'; img-src ${webview.cspSource} data:; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
</head>
<body>
  <script nonce="${nonce}" src="${scriptUri}"></script>
</body>
</html>`;
  }
}

function makeNonce(): string {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let s = "";
  for (let i = 0; i < 32; i++) s += chars.charAt(Math.floor(Math.random() * chars.length));
  return s;
}
