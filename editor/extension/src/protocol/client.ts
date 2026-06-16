// WebSocket JSON-RPC client for the Fire Editor daemon (EDITOR_PRD §4).
// Owns the single Node `ws` connection used by the extension host. The actual
// JSON-RPC correlation / notification / binary-decode logic lives in the
// transport-agnostic `RpcSession`; this class is just the `ws` glue. No vscode
// import here so the extension host
// owns UI separately and this stays portable.
import WebSocket from "ws";
import { DecodedFrame } from "./binary";
import { RpcSession, RpcTransport, RpcRemoteError, JsonRpcError } from "./rpcSession";
import {
  PROTOCOL_VERSION,
  Method,
  HelloResult,
} from "./generated";

export { RpcRemoteError };
export type { JsonRpcError };

export class FireEditorClient {
  private ws?: WebSocket;
  private session?: RpcSession;

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

      const transport: RpcTransport = { send: (text) => ws.send(text) };
      const session = new RpcSession(transport);
      session.onNotification = (m, p) => this.onNotification(m, p);
      session.onBinary = (f) => this.onBinary(f);
      this.session = session;

      ws.on("open", () => resolve());
      ws.on("error", (err) => reject(err));
      ws.on("close", (code) => {
        session.failAllPending(new Error("connection closed"));
        this.onClose(code);
      });
      ws.on("message", (data, isBinary) => {
        if (isBinary) {
          const bytes =
            data instanceof ArrayBuffer
              ? new Uint8Array(data)
              : new Uint8Array(data as Buffer);
          session.handleBinary(bytes);
        } else {
          session.handleText(data.toString());
        }
      });
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
    if (!this.session || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error("daemon not connected"));
    }
    return this.session.request<T>(method, params);
  }

  notify(method: string, params: Record<string, unknown> = {}): void {
    if (this.session && this.ws?.readyState === WebSocket.OPEN) {
      this.session.notify(method, params);
    }
  }

  dispose(): void {
    this.session?.failAllPending(new Error("client disposed"));
    this.ws?.close();
    this.ws = undefined;
    this.session = undefined;
  }
}
