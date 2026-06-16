// Transport seam for the Scene View webview.
//
// The viewport talks to the extension host via `acquireVsCodeApi()`. Routing
// every message through this single module keeps the call contract small —
// `sceneView.ts` only ever calls `host.post(...)` — and honours the VS Code
// rule that `acquireVsCodeApi()` may be called AT MOST ONCE per webview.

interface VsCodeApi {
  postMessage(msg: unknown): void;
  getState(): unknown;
  setState(s: unknown): void;
}

declare function acquireVsCodeApi(): VsCodeApi;

const vscodeApi: VsCodeApi = acquireVsCodeApi();

export const host = {
  /** Send a message up to the VS Code extension host. */
  post(msg: unknown): void {
    vscodeApi.postMessage(msg);
  },
};
