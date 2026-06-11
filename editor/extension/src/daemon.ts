// Daemon process lifecycle (EDITOR_PRD Phase E0): spawn `python -m fire_editor`
// from the repo .venv, parse the announced port, stream logs, auto-respawn on
// crash. UI-free (callbacks only) so extension.ts owns the vscode surface and
// this stays unit-friendly.
import { ChildProcessWithoutNullStreams, spawn } from "child_process";
import { EventEmitter } from "events";
import * as fs from "fs";
import * as path from "path";

export type DaemonState =
  | "stopped"
  | "starting"
  | "listening"
  | "crashed"
  | "restarting";

export interface DaemonOptions {
  /** Repo root (cwd for the daemon; must contain fire_engine/ and editor/). */
  repoRoot: string;
  /** Explicit python path, or "" to autodetect the repo .venv. */
  pythonPath: string;
  logLevel: string;
  /** Max automatic respawns before giving up. */
  maxRestarts?: number;
}

/** Resolve the venv python for the current platform, or fall back to PATH. */
export function resolvePython(repoRoot: string, override: string): string {
  if (override) return override;
  const candidates =
    process.platform === "win32"
      ? [path.join(repoRoot, ".venv", "Scripts", "python.exe")]
      : [path.join(repoRoot, ".venv", "bin", "python")];
  for (const c of candidates) if (fs.existsSync(c)) return c;
  return process.platform === "win32" ? "python" : "python3";
}

export class DaemonController extends EventEmitter {
  private proc?: ChildProcessWithoutNullStreams;
  private restarts = 0;
  private disposing = false;
  private stdoutBuffer = "";
  state: DaemonState = "stopped";
  port?: number;

  constructor(private readonly opts: DaemonOptions) {
    super();
  }

  /** Events: "state"(DaemonState), "listening"(port), "log"(line), "error"(msg). */
  start(): void {
    this.disposing = false;
    this.spawnOnce();
  }

  private setState(s: DaemonState): void {
    this.state = s;
    this.emit("state", s);
  }

  private spawnOnce(): void {
    const python = resolvePython(this.opts.repoRoot, this.opts.pythonPath);
    const editorDir = path.join(this.opts.repoRoot, "editor");
    // PYTHONPATH: repo root (fire_engine) + editor/ (fire_editor).
    const sep = path.delimiter;
    const env = {
      ...process.env,
      PYTHONPATH: [this.opts.repoRoot, editorDir, process.env.PYTHONPATH]
        .filter(Boolean)
        .join(sep),
      PYTHONUNBUFFERED: "1",
    };

    this.setState("starting");
    this.port = undefined;
    this.stdoutBuffer = "";
    this.emit("log", `[extension] spawning: ${python} -m fire_editor --port 0`);

    const proc = spawn(
      python,
      ["-m", "fire_editor", "--port", "0", "--log-level", this.opts.logLevel],
      { cwd: this.opts.repoRoot, env }
    );
    this.proc = proc;

    proc.stdout.on("data", (chunk: Buffer) => this.onStdout(chunk.toString()));
    proc.stderr.on("data", (chunk: Buffer) =>
      chunk
        .toString()
        .split(/\r?\n/)
        .filter((l) => l.length)
        .forEach((l) => this.emit("log", l))
    );
    proc.on("error", (err) => {
      this.emit("error", `failed to spawn daemon: ${err.message}`);
      this.setState("crashed");
    });
    proc.on("exit", (code, signal) => this.onExit(code, signal));
  }

  private onStdout(text: string): void {
    this.stdoutBuffer += text;
    let idx: number;
    while ((idx = this.stdoutBuffer.indexOf("\n")) >= 0) {
      const line = this.stdoutBuffer.slice(0, idx).trim();
      this.stdoutBuffer = this.stdoutBuffer.slice(idx + 1);
      if (!line) continue;
      // Readiness line: {"event":"listening","port":N}
      try {
        const obj = JSON.parse(line);
        if (obj.event === "listening" && typeof obj.port === "number") {
          this.port = obj.port;
          this.restarts = 0;
          this.setState("listening");
          this.emit("listening", obj.port);
          continue;
        }
      } catch {
        /* not JSON — treat as a log line */
      }
      this.emit("log", line);
    }
  }

  private onExit(code: number | null, signal: NodeJS.Signals | null): void {
    this.proc = undefined;
    if (this.disposing) {
      this.setState("stopped");
      return;
    }
    this.emit("log", `[extension] daemon exited (code=${code}, signal=${signal})`);
    const max = this.opts.maxRestarts ?? 5;
    if (this.restarts < max) {
      this.restarts++;
      this.setState("restarting");
      const delayMs = Math.min(500 * this.restarts, 4000);
      this.emit(
        "log",
        `[extension] respawning in ${delayMs}ms (attempt ${this.restarts}/${max})`
      );
      setTimeout(() => {
        if (!this.disposing) this.spawnOnce();
      }, delayMs);
    } else {
      this.setState("crashed");
      this.emit(
        "error",
        `daemon crashed ${max} times; giving up. See the Fire Editor log.`
      );
    }
  }

  restart(): void {
    this.restarts = 0;
    if (this.proc) {
      this.disposing = false;
      this.proc.kill();
      // onExit will respawn because disposing is false.
    } else {
      this.start();
    }
  }

  dispose(): void {
    this.disposing = true;
    this.proc?.kill();
    this.proc = undefined;
    this.setState("stopped");
  }
}
