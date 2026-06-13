// Transport seam for the Scene View webview.
//
// In VS Code the viewport talks to the extension host via `acquireVsCodeApi()`.
// In the standalone browser harness there is no VS Code API, so `harnessBoot.ts`
// installs `window.__fireEditorHost` instead and relays the same messages over a
// raw WebSocket. `sceneView.ts` only ever calls `host.post(...)`, so the exact
// same viewport bundle runs in both places — an agent's harness screenshots are
// pixel-identical to the VS Code panel.
//
// `acquireVsCodeApi()` may be called AT MOST ONCE per webview, so this module is
// the single call site; everything else imports `host`.

interface VsCodeApi {
  postMessage(msg: unknown): void;
  getState(): unknown;
  setState(s: unknown): void;
}

interface FireEditorHost {
  post(msg: unknown): void;
}

declare function acquireVsCodeApi(): VsCodeApi;
declare global {
  interface Window {
    __fireEditorHost?: FireEditorHost;
  }
}

let vscodeApi: VsCodeApi | undefined;
try {
  vscodeApi = acquireVsCodeApi();
} catch {
  vscodeApi = undefined; // not running inside a VS Code webview (browser harness)
}

export const host = {
  /** Send a message up to whichever host owns this viewport. */
  post(msg: unknown): void {
    if (vscodeApi) {
      vscodeApi.postMessage(msg);
    } else if (window.__fireEditorHost) {
      window.__fireEditorHost.post(msg);
    }
    // else: no host attached yet — the harness queues until handshake, so a
    // dropped early message here is harmless (the viewport re-sends camera/etc).
  },
  /** True when running inside VS Code (vs. the standalone browser harness). */
  get inVsCode(): boolean {
    return vscodeApi !== undefined;
  },
};
