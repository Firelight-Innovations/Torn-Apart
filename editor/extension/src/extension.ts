// Fire Editor extension entry point (EDITOR_PRD Phases E0–E1).
// Owns the daemon lifecycle, the WebSocket client, the status bar + output
// channel, and the Scene View webview. Relays MESH binary frames from the daemon
// down to the three.js viewport and camera moves back up to chunks.set_center.
import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";

import { DaemonController, DaemonState } from "./daemon";
import { HierarchyProvider, SceneObjectDTO } from "./hierarchyView";
import { InspectorViewProvider } from "./inspectorViewProvider";
import { FireEditorClient, RpcRemoteError } from "./protocol/client";
import { Method, Notification, SchemaId } from "./protocol/generated";
import { SceneViewPanel } from "./sceneViewPanel";

let output: vscode.OutputChannel;
let status: vscode.StatusBarItem;
let daemon: DaemonController | undefined;
let client: FireEditorClient | undefined;
let sceneView: SceneViewPanel | undefined;
let currentConfig: Record<string, unknown> | undefined;
let worldOpen = false;
let extensionUri: vscode.Uri;
let repoRootPath: string | undefined;
// The scene file Ctrl+S writes to: set by Open World from Save / first Save As.
let currentSavePath: string | undefined;

let hierarchy: HierarchyProvider;
let hierarchyView: vscode.TreeView<number>;
let inspector: InspectorViewProvider;
// Where newly-created objects spawn: the point the Scene View camera is looking
// at (reported via the webview's "focus" message). Defaults to the origin.
let lastFocus = { x: 0, y: 0, z: 0 };
// Guards selection echo: tree -> viewport -> tree would otherwise loop.
let syncingSelection = false;
// The one selection shared by tree, viewport and inspector (the selection hub).
let selectedObjectId: number | null = null;

export function activate(context: vscode.ExtensionContext): void {
  extensionUri = context.extensionUri;
  output = vscode.window.createOutputChannel("Fire Editor");
  status = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  status.command = "fireEditor.showStatus";
  context.subscriptions.push(output, status);
  setStatus("stopped");
  status.show();

  context.subscriptions.push(
    vscode.commands.registerCommand("fireEditor.restartDaemon", () => {
      output.appendLine("[extension] restart requested");
      daemon?.restart();
    }),
    vscode.commands.registerCommand("fireEditor.showLog", () => output.show(true)),
    vscode.commands.registerCommand("fireEditor.showStatus", () =>
      vscode.window.showInformationMessage(
        `Fire Editor: ${daemon?.state ?? "stopped"}${daemon?.port ? ` (port ${daemon.port})` : ""}`
      )
    ),
    vscode.commands.registerCommand("fireEditor.openSceneView", () => openSceneView()),
    vscode.commands.registerCommand("fireEditor.openWorldSeed", () => openWorldBySeed()),
    vscode.commands.registerCommand("fireEditor.openWorldSave", () => openWorldBySave()),
    vscode.commands.registerCommand("fireEditor.saveScene", () => saveScene()),
    vscode.commands.registerCommand("fireEditor.saveSceneAs", () => saveScene(true))
  );

  // --- Inspector (properties panel, below the Hierarchy) ---
  inspector = new InspectorViewProvider(extensionUri, (msg) => {
    switch (msg.type) {
      case "rename":
        void sceneRequest(Method.SCENE_RENAME, { id: msg.id, name: msg.name });
        break;
      case "setTransform": {
        const p = msg.position as number[];
        const r = msg.rotation as number[]; // (w, x, y, z)
        const s = msg.scale as number[];
        void sceneRequest(Method.SCENE_SET_TRANSFORM, {
          id: msg.id,
          px: p[0], py: p[1], pz: p[2],
          rw: r[0], rx: r[1], ry: r[2], rz: r[3],
          sx: s[0], sy: s[1], sz: s[2],
        });
        break;
      }
    }
  });
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(InspectorViewProvider.viewId, inspector)
  );

  // --- Scene hierarchy (Phase E2) ---
  hierarchy = new HierarchyProvider();
  hierarchy.onReparent = (id, parent) =>
    sceneRequest(Method.SCENE_REPARENT, parent === null ? { id } : { id, parent });
  hierarchyView = vscode.window.createTreeView("fireEditor.hierarchy", {
    treeDataProvider: hierarchy,
    dragAndDropController: hierarchy,
    showCollapseAll: true,
  });
  context.subscriptions.push(
    hierarchy,
    hierarchyView,
    hierarchyView.onDidChangeSelection((e) => {
      if (syncingSelection) return;
      setSelection(e.selection.length ? e.selection[0] : null, "tree");
    }),
    vscode.commands.registerCommand("fireEditor.refreshHierarchy", () => refreshHierarchy()),
    vscode.commands.registerCommand("fireEditor.createEmpty", () => createObject("empty")),
    vscode.commands.registerCommand("fireEditor.createCube", () => createObject("cube")),
    vscode.commands.registerCommand("fireEditor.createSphere", () => createObject("sphere")),
    vscode.commands.registerCommand("fireEditor.createLight", () => createObject("light")),
    vscode.commands.registerCommand("fireEditor.createSpawn", () => createObject("spawn")),
    vscode.commands.registerCommand("fireEditor.renameObject", (id?: number) => renameObject(id)),
    vscode.commands.registerCommand("fireEditor.deleteObject", (id?: number) => deleteObject(id)),
    vscode.commands.registerCommand("fireEditor.frameObject", (id?: number) => frameObject(id))
  );

  const repoRoot = findRepoRoot(context.extensionPath);
  if (!repoRoot) {
    output.appendLine(
      "[extension] no Torn Apart repo detected (need fire_engine/ + editor/fire_editor). Idle."
    );
    return;
  }
  repoRootPath = repoRoot;
  output.appendLine(`[extension] repo root: ${repoRoot}`);
  if (vscode.workspace.getConfiguration("fireEditor").get<boolean>("autoStart", true)) {
    startDaemon(context, repoRoot);
  }
}

function startDaemon(context: vscode.ExtensionContext, repoRoot: string): void {
  const cfg = vscode.workspace.getConfiguration("fireEditor");
  daemon = new DaemonController({
    repoRoot,
    pythonPath: cfg.get<string>("pythonPath", ""),
    logLevel: cfg.get<string>("logLevel", "info"),
  });
  context.subscriptions.push({ dispose: () => daemon?.dispose() });

  daemon.on("log", (line: string) => output.appendLine(line));
  daemon.on("state", (s: DaemonState) => setStatus(s));
  daemon.on("error", (msg: string) => {
    output.appendLine(`[extension] ERROR: ${msg}`);
    vscode.window.showErrorMessage(`Fire Editor: ${msg}`);
  });
  daemon.on("listening", (port: number) => connectClient(port));
  daemon.start();
}

async function connectClient(port: number): Promise<void> {
  client?.dispose();
  worldOpen = false;
  client = new FireEditorClient();
  client.onClose = () => output.appendLine("[extension] client disconnected");
  client.onBinary = (frame) => {
    if (frame.schemaId === SchemaId.MESH && sceneView) sceneView.postMesh(frame.payload);
  };
  client.onNotification = (method, params) => {
    const p = (params ?? {}) as Record<string, unknown>;
    if (method === "log") {
      output.appendLine(`[daemon:${p.level ?? "info"}] ${p.message ?? ""}`);
    } else if (method === "chunk.unload" && sceneView) {
      sceneView.postUnload([Number(p.cx), Number(p.cy), Number(p.cz)]);
    } else if (method === "stream.done") {
      output.appendLine(`[extension] stream done: sent ${p.sent}, removed ${p.removed}`);
    } else if (method === "edit.state" && sceneView) {
      sceneView.postEditState(p);
    } else if (method === Notification.SCENE_CHANGED) {
      applyObjects((p.objects as SceneObjectDTO[]) ?? []);
    }
  };

  try {
    await client.connect(port);
    const hello = await client.hello("vscode-fire-editor");
    output.appendLine(
      `[extension] handshake ok — engine ${hello.engine_version}, ` +
        `daemon ${hello.daemon_version}, protocol ${hello.protocol_version}`
    );
    setStatus("listening", true);
    // First-run UX: pressing F5 should land the user straight in the 3D editor,
    // not a hidden status-bar item. Auto-open the Scene View once connected
    // (opt out with fireEditor.autoOpenSceneView = false).
    if (
      vscode.workspace
        .getConfiguration("fireEditor")
        .get<boolean>("autoOpenSceneView", true)
    ) {
      openSceneView();
    }
  } catch (e) {
    if (e instanceof RpcRemoteError) {
      vscode.window.showErrorMessage(
        `Fire Editor handshake failed: ${e.rpc.message}. Rebuild the daemon/extension.`
      );
    } else {
      output.appendLine(`[extension] connect failed: ${(e as Error).message}`);
    }
  }
}

function openSceneView(): void {
  if (!client) {
    vscode.window.showWarningMessage("Fire Editor: daemon not connected yet.");
    return;
  }
  const setCenter = (x: number, y: number, z: number) => {
    client?.request(Method.CHUNKS_SET_CENTER, { x, y, z }).catch((e) =>
      output.appendLine(`[extension] set_center failed: ${(e as Error).message}`)
    );
  };
  const onMessage = (msg: Record<string, unknown>) => {
    switch (msg.type) {
      case "camera":
        setCenter(Number(msg.x), Number(msg.y), Number(msg.z));
        break;
      case "focus":
        lastFocus = { x: Number(msg.x), y: Number(msg.y), z: Number(msg.z) };
        break;
      case "selectObject": {
        // Viewport click -> reveal + select the node in the tree + inspector.
        const id = msg.id === null || msg.id === undefined ? null : Number(msg.id);
        setSelection(id, "viewport");
        break;
      }
      case "transform": {
        // Gizmo drag (throttled) -> authoritative transform on the daemon.
        const p = msg.position as number[];
        const r = msg.rotation as number[]; // (w, x, y, z)
        const s = msg.scale as number[];
        void sceneRequest(Method.SCENE_SET_TRANSFORM, {
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
        client?.request(Method.EDIT_UNDO, {}).catch(() => undefined);
        break;
      case "redo":
        client?.request(Method.EDIT_REDO, {}).catch(() => undefined);
        break;
    }
  };
  const onReady = async () => {
    if (!worldOpen) await openWorldBySeed(1337);
    if (currentConfig) sceneView?.postConfig(currentConfig);
    await refreshHierarchy(); // pushes current objects into the fresh viewport
    setCenter(20, -20, 24); // initial camera spot above the flat ground
  };
  sceneView = SceneViewPanel.createOrShow(extensionUri, onMessage, onReady);
}

async function handleEdit(msg: Record<string, unknown>): Promise<void> {
  if (!client) return;
  const b = (msg.brush ?? {}) as { shape?: string; mode?: string; radius?: number; material?: number };
  try {
    const res = (await client.request(Method.TERRAIN_RAYCAST, {
      ox: msg.ox, oy: msg.oy, oz: msg.oz,
      dx: msg.dx, dy: msg.dy, dz: msg.dz,
      max_distance: 250,
    })) as { hit: { point: [number, number, number] } | null };
    if (!res.hit) return;
    const [x, y, z] = res.hit.point;
    await client.request(Method.TERRAIN_BRUSH, {
      shape: b.shape ?? "sphere",
      x, y, z,
      mode: b.mode ?? "remove",
      radius: b.radius ?? 2,
      material: b.material ?? (b.mode === "add" ? 1 : 0),
    });
  } catch (e) {
    output.appendLine(`[extension] edit failed: ${(e as Error).message}`);
  }
}

async function openWorldBySeed(defaultSeed?: number): Promise<void> {
  if (!client) return;
  let seed = defaultSeed;
  if (seed === undefined) {
    const input = await vscode.window.showInputBox({
      prompt: "World seed",
      value: "1337",
      validateInput: (v) => (/^-?\d+$/.test(v) ? undefined : "enter an integer seed"),
    });
    if (input === undefined) return;
    seed = parseInt(input, 10);
  }
  await doOpenWorld({ seed });
}

async function openWorldBySave(): Promise<void> {
  if (!client) return;
  const uri = await vscode.window.showOpenDialog({
    canSelectMany: false,
    filters: { "Torn Apart saves": ["ta"] },
    title: "Open Torn Apart save",
  });
  if (!uri || uri.length === 0) return;
  await doOpenWorld({ save_path: uri[0].fsPath });
  currentSavePath = uri[0].fsPath; // Ctrl+S round-trips the opened file
}

async function doOpenWorld(params: Record<string, unknown>): Promise<void> {
  try {
    const res = (await client!.request(Method.WORLD_OPEN, params)) as {
      ok: boolean;
      seed: number;
      config: Record<string, unknown>;
      edited_chunks: number;
    };
    worldOpen = res.ok;
    currentConfig = res.config;
    sceneView?.reset();
    sceneView?.postConfig(currentConfig);
    await refreshHierarchy(); // a save may carry placed objects
    output.appendLine(
      `[extension] world open — seed ${res.seed}, edited chunks ${res.edited_chunks}`
    );
  } catch (e) {
    const msg = e instanceof RpcRemoteError ? e.rpc.message : (e as Error).message;
    vscode.window.showErrorMessage(`Fire Editor: world.open failed — ${msg}`);
  }
}

// --- Scene hierarchy helpers ---

/** Push a fresh object list to the tree, the viewport gizmos and the inspector. */
function applyObjects(objects: SceneObjectDTO[]): void {
  hierarchy.setObjects(objects);
  sceneView?.postObjects(objects);
  // Keep the inspector tracking daemon-side changes (gizmo drags, renames).
  if (selectedObjectId !== null && !hierarchy.has(selectedObjectId)) {
    selectedObjectId = null;
  }
  inspector.postObject(selectedObjectId !== null ? hierarchy.get(selectedObjectId) ?? null : null);
}

/**
 * The selection hub: tree click, viewport click and programmatic selection all
 * funnel here, then fan out to the other two surfaces + the inspector.
 */
function setSelection(id: number | null, source: "tree" | "viewport" | "code"): void {
  selectedObjectId = id !== null && hierarchy.has(id) ? id : null;
  if (source !== "viewport") sceneView?.postSelect(selectedObjectId);
  if (source !== "tree") revealInTree(selectedObjectId);
  inspector.postObject(selectedObjectId !== null ? hierarchy.get(selectedObjectId) ?? null : null);
}

// --- Save Scene (Ctrl+S in the Scene View / Hierarchy) ---

async function saveScene(forceDialog = false): Promise<void> {
  if (!client || !worldOpen) {
    vscode.window.showWarningMessage("Fire Editor: no world open to save.");
    return;
  }
  let target = currentSavePath;
  if (forceDialog || !target) {
    // Authored scenes belong in scenes/ (committed content), not saves/
    // (gitignored player state) — DECISIONS.md 2026-06-12.
    const defaultDir = repoRootPath ? path.join(repoRootPath, "scenes") : undefined;
    const uri = await vscode.window.showSaveDialog({
      title: "Save Fire Editor scene",
      filters: { "Torn Apart saves": ["ta"] },
      defaultUri: defaultDir
        ? vscode.Uri.file(path.join(defaultDir, "scene.ta"))
        : undefined,
    });
    if (!uri) return;
    target = uri.fsPath;
  }
  try {
    const res = (await client.request(Method.WORLD_SAVE, { path: target })) as {
      ok: boolean;
      edited_chunks: number;
    };
    currentSavePath = target;
    vscode.window.setStatusBarMessage(
      `Fire Editor: saved ${path.basename(target)} (${res.edited_chunks} edited chunk${res.edited_chunks === 1 ? "" : "s"})`,
      4000
    );
  } catch (e) {
    const msg = e instanceof RpcRemoteError ? e.rpc.message : (e as Error).message;
    vscode.window.showErrorMessage(`Fire Editor: save failed — ${msg}`);
  }
}

/** Re-fetch the whole tree from the daemon (after open / on demand). */
async function refreshHierarchy(): Promise<void> {
  if (!client || !worldOpen) {
    applyObjects([]);
    return;
  }
  try {
    const res = (await client.request(Method.SCENE_TREE, {})) as { objects: SceneObjectDTO[] };
    applyObjects(res.objects ?? []);
  } catch (e) {
    output.appendLine(`[extension] scene.tree failed: ${(e as Error).message}`);
  }
}

/** Fire a scene mutation; scene.changed will refresh the tree + viewport. */
async function sceneRequest(method: string, params: Record<string, unknown>): Promise<unknown> {
  if (!client) {
    vscode.window.showWarningMessage("Fire Editor: daemon not connected yet.");
    return undefined;
  }
  try {
    return await client.request(method, params);
  } catch (e) {
    const msg = e instanceof RpcRemoteError ? e.rpc.message : (e as Error).message;
    vscode.window.showErrorMessage(`Fire Editor: ${method} failed — ${msg}`);
    return undefined;
  }
}

/** The currently-selected tree node id, if any. */
function selectedId(): number | undefined {
  return hierarchyView?.selection.length ? hierarchyView.selection[0] : undefined;
}

async function createObject(kind: string): Promise<void> {
  // Parent the new object under the current selection (Unity-style), and spawn
  // it where the camera is looking so it lands in view.
  const parent = selectedId();
  const res = (await sceneRequest(Method.SCENE_CREATE, {
    kind,
    ...(parent !== undefined ? { parent } : {}),
    x: lastFocus.x,
    y: lastFocus.y,
    z: lastFocus.z,
  })) as { object?: SceneObjectDTO } | undefined;
  if (res?.object) {
    // scene.changed has already refreshed the tree; reveal the newcomer.
    await refreshHierarchy();
    revealInTree(res.object.id);
  }
}

async function renameObject(id?: number): Promise<void> {
  const target = id ?? selectedId();
  if (target === undefined) return;
  const current = hierarchy.get(target);
  const name = await vscode.window.showInputBox({
    prompt: "Rename object",
    value: current?.name ?? "",
  });
  if (name === undefined) return;
  await sceneRequest(Method.SCENE_RENAME, { id: target, name });
}

async function deleteObject(id?: number): Promise<void> {
  const target = id ?? selectedId();
  if (target === undefined) return;
  await sceneRequest(Method.SCENE_DELETE, { id: target });
}

function frameObject(id?: number): void {
  const target = id ?? selectedId();
  if (target !== undefined) sceneView?.postFrame(target);
}

/** Select a node in the tree + viewport without bouncing the echo back. */
function revealInTree(id: number | null): void {
  if (id === null || !hierarchy.has(id)) {
    sceneView?.postSelect(null);
    return;
  }
  syncingSelection = true;
  hierarchyView
    .reveal(id, { select: true, focus: false })
    .then(undefined, () => undefined)
    .then(() => {
      syncingSelection = false;
      sceneView?.postSelect(id);
    });
}

function setStatus(state: DaemonState, connected = false): void {
  const map: Record<DaemonState, string> = {
    stopped: "$(circle-slash) Fire Editor",
    starting: "$(sync~spin) Fire Editor: starting",
    listening: connected
      ? "$(flame) Fire Editor: connected"
      : "$(flame) Fire Editor: listening",
    restarting: "$(sync~spin) Fire Editor: restarting",
    crashed: "$(error) Fire Editor: crashed",
  };
  status.text = map[state];
  status.tooltip = "Fire Editor daemon — click for status";
}

function isRepoRoot(root: string): boolean {
  return (
    fs.existsSync(path.join(root, "fire_engine")) &&
    fs.existsSync(path.join(root, "editor", "fire_editor"))
  );
}

/**
 * Locate the Torn Apart repo root. Prefer an open workspace folder, but fall
 * back to the extension's own install location (<repo>/editor/extension) so the
 * daemon still starts when the Extension Development Host launches with no
 * folder open — which is exactly what happens when ${workspaceFolder} fails to
 * resolve (e.g. a space in the repo path).
 */
function findRepoRoot(extensionPath?: string): string | undefined {
  for (const f of vscode.workspace.workspaceFolders ?? []) {
    if (isRepoRoot(f.uri.fsPath)) return f.uri.fsPath;
  }
  if (extensionPath) {
    // <repo>/editor/extension -> up two -> <repo>
    const derived = path.resolve(extensionPath, "..", "..");
    if (isRepoRoot(derived)) return derived;
  }
  return undefined;
}

export function deactivate(): void {
  client?.dispose();
  daemon?.dispose();
}
