// Browser viewport harness — the standalone, agent-drivable twin of the VS Code
// extension relay (EDITOR_PRD agent access). Loaded by editor/extension/harness/
// index.html as the FIRST script (before sceneView.js), it:
//
//   1. injects the shared viewport markup (so sceneView.ts finds its DOM ids),
//   2. installs `window.__fireEditorHost` so the same sceneView bundle posts to
//      us instead of VS Code (see webview/host.ts),
//   3. opens a raw WebSocket to the daemon and wraps it in the shared
//      `RpcSession` — byte-for-byte the same protocol the extension speaks,
//   4. runs the boot sequence (hello → world.open → config → scene.tree →
//      ground_lut → set_center) and relays daemon frames into the viewport,
//   5. exposes `window.fireHarness` (rpc/open/setCamera/select/waitForStreamDone/
//      state) and logs `[harness]` lines for Chrome MCP console reads.
//
// Query params: ?port=8123&seed=1337&save=foo.ta&cam=20,-20,24
import { RpcSession, RpcTransport } from "../protocol/rpcSession";
import { PROTOCOL_VERSION, Method, Notification, SchemaId } from "../protocol/generated";
import { VIEWPORT_CSS, VIEWPORT_BODY_HTML } from "./viewportMarkup";

interface FireHarness {
  rpc(method: string, params?: Record<string, unknown>): Promise<unknown>;
  open(opts: { seed?: number; save?: string }): Promise<unknown>;
  setCamera(x: number, y: number, z: number, target?: [number, number, number]): void;
  select(id: number | null): void;
  waitForStreamDone(timeoutMs?: number): Promise<{ sent: number; removed: number }>;
  state(): unknown;
  events: Array<{ method: string; params: unknown }>;
}

declare global {
  interface Window {
    __fireEditorHost?: { post(msg: unknown): void };
    __fireSceneDebug?: { snapshot(): unknown };
    fireHarness?: FireHarness;
  }
}

function log(...args: unknown[]): void {
  console.log("[harness]", ...args);
}

// --- 1. inject the shared markup synchronously (before sceneView.js runs) ---
(function injectMarkup() {
  const style = document.createElement("style");
  style.textContent = VIEWPORT_CSS;
  document.head.appendChild(style);
  const wrap = document.createElement("div");
  wrap.innerHTML = VIEWPORT_BODY_HTML;
  while (wrap.firstChild) document.body.appendChild(wrap.firstChild);
})();

// --- query params ---
const qs = new URLSearchParams(location.search);
const PORT = Number(qs.get("port") || 8123);
const HOST = qs.get("host") || "127.0.0.1";
const SEED = qs.get("seed");
const SAVE = qs.get("save");
const CAM = parseVec(qs.get("cam"), [20, -20, 24]);

function parseVec(s: string | null, fallback: [number, number, number]): [number, number, number] {
  if (!s) return fallback;
  const p = s.split(",").map(Number);
  return p.length === 3 && p.every((n) => Number.isFinite(n)) ? [p[0], p[1], p[2]] : fallback;
}

// --- relay plumbing ---
const events: Array<{ method: string; params: unknown }> = [];
let session: RpcSession | undefined;
const streamWaiters: Array<(p: { sent: number; removed: number }) => void> = [];

/** Push a message DOWN into the viewport (sceneView listens on window 'message'). */
function toViewport(msg: unknown): void {
  window.dispatchEvent(new MessageEvent("message", { data: msg }));
}

/** Relay viewport → daemon (mirror of extension.ts openSceneView onMessage). */
function handleViewportMessage(msg: Record<string, unknown>): void {
  switch (msg.type) {
    case "ready":
      // The viewport finished loading; the boot sequence already ran (or will on
      // ws open). Nothing to do — config/objects/meshes are pushed by boot().
      break;
    case "camera":
      void session?.request(Method.CHUNKS_SET_CENTER,
        { x: Number(msg.x), y: Number(msg.y), z: Number(msg.z) });
      break;
    case "focus":
      break; // placement focus — tracked by the extension; unused headless
    case "selectObject":
      // Echo the selection straight back so the gizmo attaches in the viewport.
      toViewport({ type: "select", id: msg.id ?? null });
      break;
    case "transform": {
      const p = msg.position as number[];
      const r = msg.rotation as number[]; // (w, x, y, z)
      const s = msg.scale as number[];
      void session?.request(Method.SCENE_SET_TRANSFORM, {
        id: msg.id,
        px: p[0], py: p[1], pz: p[2],
        rw: r[0], rx: r[1], ry: r[2], rz: r[3],
        sx: s[0], sy: s[1], sz: s[2],
      });
      break;
    }
    case "edit":
      void handleEdit(msg);
      break;
    case "undo":
      void session?.request(Method.EDIT_UNDO, {});
      break;
    case "redo":
      void session?.request(Method.EDIT_REDO, {});
      break;
  }
}

async function handleEdit(msg: Record<string, unknown>): Promise<void> {
  if (!session) return;
  const b = (msg.brush ?? {}) as { shape?: string; mode?: string; radius?: number; material?: number };
  const res = (await session.request(Method.TERRAIN_RAYCAST, {
    ox: msg.ox, oy: msg.oy, oz: msg.oz,
    dx: msg.dx, dy: msg.dy, dz: msg.dz,
    max_distance: 250,
  })) as { hit: { point: [number, number, number] } | null };
  if (!res.hit) return;
  const [x, y, z] = res.hit.point;
  await session.request(Method.TERRAIN_BRUSH, {
    shape: b.shape ?? "sphere", x, y, z,
    mode: b.mode ?? "remove", radius: b.radius ?? 2,
    material: b.material ?? (b.mode === "add" ? 1 : 0),
  });
}

window.__fireEditorHost = { post: (m) => handleViewportMessage(m as Record<string, unknown>) };

// --- daemon → viewport ---
function handleNotification(method: string, params: unknown): void {
  events.push({ method, params });
  const p = (params ?? {}) as Record<string, unknown>;
  switch (method) {
    case Notification.CHUNK_UNLOAD:
      toViewport({ type: "unload", coord: [p.cx, p.cy, p.cz] });
      break;
    case Notification.SCENE_CHANGED:
      toViewport({ type: "objects", objects: p.objects });
      break;
    case Notification.EDIT_STATE:
      toViewport({ type: "editState", state: p });
      break;
    case Notification.STREAM_DONE: {
      const done = { sent: Number(p.sent) || 0, removed: Number(p.removed) || 0 };
      while (streamWaiters.length) streamWaiters.shift()!(done);
      break;
    }
    case Notification.LOG:
      log("daemon:", p.message ?? p);
      break;
  }
}

function handleBinary(frame: { schemaId: number; payloadId: number; payload: Uint8Array }): void {
  if (frame.schemaId === SchemaId.MESH) toViewport({ type: "mesh", payload: frame.payload });
  else if (frame.schemaId === SchemaId.TEXTURE) toViewport({ type: "groundLut", payload: frame.payload });
}

// --- boot ---
async function boot(): Promise<void> {
  if (!session) return;
  log(`connected ws://${HOST}:${PORT}`);
  await session.request(Method.HELLO, { protocol_version: PROTOCOL_VERSION, client: "browser-harness" });
  const open = SAVE
    ? await session.request(Method.WORLD_OPEN, { save_path: SAVE })
    : await session.request(Method.WORLD_OPEN, { seed: SEED ? Number(SEED) : 1337 });
  const config = (open as { config?: unknown }).config;
  log("world open", open);
  if (config) toViewport({ type: "config", config });
  const tree = (await session.request(Method.SCENE_TREE, {})) as { objects?: unknown };
  toViewport({ type: "objects", objects: tree.objects ?? [] });
  await session.request(Method.WORLD_GROUND_LUT, {}); // broadcasts the TEXTURE frame
  toViewport({ type: "cameraPose", x: CAM[0], y: CAM[1], z: CAM[2], tx: 0, ty: 0, tz: 8 });
  await session.request(Method.CHUNKS_SET_CENTER, { x: CAM[0], y: CAM[1], z: CAM[2], resend: true });
  log("boot complete", window.__fireSceneDebug?.snapshot());
}

// --- public API for agents / Chrome MCP ---
window.fireHarness = {
  rpc: (method, params = {}) => session!.request(method, params),
  open: (opts) =>
    session!.request(Method.WORLD_OPEN, opts.save ? { save_path: opts.save } : { seed: opts.seed ?? 1337 }),
  setCamera: (x, y, z, target) =>
    toViewport({ type: "cameraPose", x, y, z, tx: target?.[0], ty: target?.[1], tz: target?.[2] }),
  select: (id) => toViewport({ type: "select", id }),
  waitForStreamDone: (timeoutMs = 30000) =>
    new Promise((resolve, reject) => {
      const t = setTimeout(() => reject(new Error("stream.done timeout")), timeoutMs);
      streamWaiters.push((d) => { clearTimeout(t); resolve(d); });
    }),
  state: () => window.__fireSceneDebug?.snapshot(),
  events,
};

// --- connect ---
const ws = new WebSocket(`ws://${HOST}:${PORT}`);
ws.binaryType = "arraybuffer";
const transport: RpcTransport = { send: (t) => ws.send(t) };
session = new RpcSession(transport);
session.onNotification = handleNotification;
session.onBinary = handleBinary;
ws.onmessage = (ev) => {
  if (typeof ev.data === "string") session!.handleText(ev.data);
  else session!.handleBinary(new Uint8Array(ev.data as ArrayBuffer));
};
ws.onopen = () => void boot().catch((e) => log("boot failed", e));
ws.onerror = (e) => log("ws error", e);
ws.onclose = () => { session?.failAllPending(new Error("connection closed")); log("ws closed"); };
