// Transport-agnostic JSON-RPC 2.0 core for the Fire Editor protocol.
//
// The extension host's `client.ts` wraps a Node `ws` socket and feeds frames in
// here. Keeping the request/response correlation, notification fan-out and
// binary decode transport-agnostic means the socket layer stays a thin shim.
//
// No `ws` / `vscode` / DOM imports here: the caller owns the socket and pumps
// `handleText` / `handleBinary` / `handleClose`; this module only sends via the
// injected `RpcTransport`.
import { decodeFrame, DecodedFrame } from "./binary";

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

/** Whatever can put text on the wire (a `ws` socket, a browser `WebSocket`). */
export interface RpcTransport {
  send(text: string): void;
}

type Pending = {
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
};

export class RpcSession {
  private nextId = 1;
  private pending = new Map<number, Pending>();

  /** Notification handler: (method, params). */
  onNotification: (method: string, params: unknown) => void = () => {};
  /** Binary frame handler (mesh/texture payloads). */
  onBinary: (frame: DecodedFrame) => void = () => {};

  constructor(private readonly transport: RpcTransport) {}

  request<T = unknown>(method: string, params: Record<string, unknown> = {}): Promise<T> {
    const id = this.nextId++;
    const payload = JSON.stringify({ jsonrpc: "2.0", id, method, params });
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
      this.transport.send(payload);
    });
  }

  notify(method: string, params: Record<string, unknown> = {}): void {
    this.transport.send(JSON.stringify({ jsonrpc: "2.0", method, params }));
  }

  /** Feed one inbound text frame (a JSON-RPC response or notification). */
  handleText(raw: string): void {
    let msg: { id?: number | null; error?: JsonRpcError; result?: unknown; method?: string; params?: unknown };
    try {
      msg = JSON.parse(raw);
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

  /** Feed one inbound binary frame (decoded and handed to `onBinary`). */
  handleBinary(bytes: Uint8Array): void {
    try {
      this.onBinary(decodeFrame(bytes));
    } catch (e) {
      console.error("Fire Editor: bad binary frame", e);
    }
  }

  /** Reject every in-flight request (call on socket close / dispose). */
  failAllPending(err: Error): void {
    for (const p of this.pending.values()) p.reject(err);
    this.pending.clear();
  }
}
