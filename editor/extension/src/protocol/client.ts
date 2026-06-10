// WebSocket JSON-RPC client for the Fire Editor daemon (EDITOR_PRD §4).
// Owns the single connection to the daemon. Text frames are JSON-RPC; binary
// frames are protocol payloads delivered to `onBinary`. No vscode import here so
// the extension host owns UI separately and this stays portable.
import WebSocket from "ws";
import { decodeFrame, DecodedFrame } from "./binary";
import {
  PROTOCOL_VERSION,
  Method,
  HelloResult,
} from "./generated";

type Pending = {
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
};

export interface JsonRpcError {
  code: number;
  message: string;
  data?: unknown;
}

export class RpcRemoteError extends Error {
  constructor(public readonly rpc: JsonRpcError) {
    super(`[${rpc.code}] ${rpc.message}`);
  }
}

export class FireEditorClient {
  private ws?: WebSocket;
  private nextId = 1;
  private pending = new Map<number, Pending>();

  /** Notification handler: (method, params). */
  onNotification: (method: string, params: unknown) => void = () => {};
  /** Binary frame handler (mesh/texture payloads). */
  onBinary: (frame: DecodedFrame) => void = () => {};
  /** Called when the socket closes for any reason. */
  onClose: (code: number) => void = () => {};

  connect(port: number, host = "127.0.0.1"): Promise<void> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(`ws://${host}:${port}`);
      ws.binaryType = "arraybuffer";
      this.ws = ws;

      ws.on("open", () => resolve());
      ws.on("error", (err) => reject(err));
      ws.on("close", (code) => {
        this.failAllPending(new Error("connection closed"));
        this.onClose(code);
      });
      ws.on("message", (data, isBinary) => this.onMessage(data, isBinary));
    });
  }

  /** Handshake: announce our protocol version; daemon rejects on mismatch. */
  async hello(clientName: string): Promise<HelloResult> {
    return (await this.request(Method.HELLO, {
      protocol_version: PROTOCOL_VERSION,
      client: clientName,
    })) as HelloResult;
  }

  request<T = unknown>(method: string, params: Record<string, unknown> = {}): Promise<T> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error("daemon not connected"));
    }
    const id = this.nextId++;
    const payload = JSON.stringify({ jsonrpc: "2.0", id, method, params });
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
      this.ws!.send(payload);
    });
  }

  notify(method: string, params: Record<string, unknown> = {}): void {
    this.ws?.send(JSON.stringify({ jsonrpc: "2.0", method, params }));
  }

  dispose(): void {
    this.failAllPending(new Error("client disposed"));
    this.ws?.close();
    this.ws = undefined;
  }

  private onMessage(data: WebSocket.RawData, isBinary: boolean): void {
    if (isBinary) {
      const bytes =
        data instanceof ArrayBuffer
          ? new Uint8Array(data)
          : new Uint8Array(data as Buffer);
      try {
        this.onBinary(decodeFrame(bytes));
      } catch (e) {
        console.error("Fire Editor: bad binary frame", e);
      }
      return;
    }
    let msg: any;
    try {
      msg = JSON.parse(data.toString());
    } catch {
      return;
    }
    if (msg.id !== undefined && msg.id !== null) {
      const p = this.pending.get(msg.id);
      if (!p) return;
      this.pending.delete(msg.id);
      if (msg.error) p.reject(new RpcRemoteError(msg.error));
      else p.resolve(msg.result);
    } else if (msg.method) {
      this.onNotification(msg.method, msg.params);
    }
  }

  private failAllPending(err: Error): void {
    for (const p of this.pending.values()) p.reject(err);
    this.pending.clear();
  }
}
