// Fire Editor extension entry point (EDITOR_PRD Phases E0–E1).
// Owns the daemon lifecycle, the WebSocket client, the status bar + output
// channel, and the Scene View webview. Relays MESH binary frames from the daemon
// down to the three.js viewport and camera moves back up to chunks.set_center.
import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";

import { DaemonController, DaemonState } from "./daemon";
import { FireEditorClient, RpcRemoteError } from "./protocol/client";
import { Method, SchemaId } from "./protocol/generated";
import { SceneViewPanel } from "./sceneViewPanel";

let output: vscode.OutputChannel;
let status: vscode.StatusBarItem;
let daemon: DaemonController | undefined;
let client: FireEditorClient | undefined;
let sceneView: SceneViewPanel | undefined;
let currentConfig: Record<string, unknown> | undefined;
let worldOpen = false;
let extensionUri: vscode.Uri;

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
    vscode.commands.registerCommand("fireEditor.openWorldSave", () => openWorldBySave())
  );

  const repoRoot = findRepoRoot();
  if (!repoRoot) {
    output.appendLine(
      "[extension] no Torn Apart workspace detected (need torn_apart/ + editor/). Idle."
    );
    return;
  }
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
  const onCamera = (x: number, y: number, z: number) => {
    client?.request(Method.CHUNKS_SET_CENTER, { x, y, z }).catch((e) =>
      output.appendLine(`[extension] set_center failed: ${(e as Error).message}`)
    );
  };
  const onReady = async () => {
    if (!worldOpen) await openWorldBySeed(1337);
    if (currentConfig) sceneView?.postConfig(currentConfig);
    onCamera(20, -20, 24); // initial camera spot above the flat ground
  };
  sceneView = SceneViewPanel.createOrShow(extensionUri, onCamera, onReady);
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
    output.appendLine(
      `[extension] world open — seed ${res.seed}, edited chunks ${res.edited_chunks}`
    );
  } catch (e) {
    const msg = e instanceof RpcRemoteError ? e.rpc.message : (e as Error).message;
    vscode.window.showErrorMessage(`Fire Editor: world.open failed — ${msg}`);
  }
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

function findRepoRoot(): string | undefined {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders) return undefined;
  for (const f of folders) {
    const root = f.uri.fsPath;
    if (
      fs.existsSync(path.join(root, "torn_apart")) &&
      fs.existsSync(path.join(root, "editor", "fire_editor"))
    ) {
      return root;
    }
  }
  return undefined;
}

export function deactivate(): void {
  client?.dispose();
  daemon?.dispose();
}
