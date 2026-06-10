// Fire Editor extension entry point (EDITOR_PRD Phase E0).
// Owns the daemon lifecycle, the WebSocket client connection, the status bar
// indicator, and the output channel. Later phases add the Scene View, Hierarchy,
// Inspector, Texture Lab, and Model Workspace panels onto the same client.
import * as fs from "fs";
import * as path from "path";
import * as vscode from "vscode";

import { DaemonController, DaemonState } from "./daemon";
import { FireEditorClient, RpcRemoteError } from "./protocol/client";

let output: vscode.OutputChannel;
let status: vscode.StatusBarItem;
let daemon: DaemonController | undefined;
let client: FireEditorClient | undefined;

export function activate(context: vscode.ExtensionContext): void {
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
    vscode.commands.registerCommand("fireEditor.showStatus", () => {
      vscode.window.showInformationMessage(
        `Fire Editor: ${daemon?.state ?? "stopped"}${
          daemon?.port ? ` (port ${daemon.port})` : ""
        }`
      );
    })
  );

  const repoRoot = findRepoRoot();
  if (!repoRoot) {
    output.appendLine(
      "[extension] no Torn Apart workspace detected (need torn_apart/ + editor/). Idle."
    );
    return;
  }

  const cfg = vscode.workspace.getConfiguration("fireEditor");
  if (cfg.get<boolean>("autoStart", true)) {
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
  client = new FireEditorClient();
  client.onClose = () => {
    output.appendLine("[extension] client disconnected");
  };
  client.onNotification = (method, params) => {
    if (method === "log" && params && typeof params === "object") {
      const p = params as { level?: string; message?: string };
      output.appendLine(`[daemon:${p.level ?? "info"}] ${p.message ?? ""}`);
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
