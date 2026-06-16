// Unit tests for the transport-agnostic JSON-RPC core used by the extension
// host client. A fake transport captures outbound text; we feed inbound frames
// via handleText/handleBinary.
import { test } from "node:test";
import assert from "node:assert/strict";

import { RpcSession, RpcRemoteError } from "../protocol/rpcSession";
import { encodeFrame } from "../protocol/binary";
import { SchemaId } from "../protocol/generated";

function makeSession() {
  const sent: string[] = [];
  const session = new RpcSession({ send: (t) => sent.push(t) });
  return { session, sent };
}

test("request sends a well-formed JSON-RPC envelope and resolves on result", async () => {
  const { session, sent } = makeSession();
  const p = session.request("world.open", { seed: 1337 });
  assert.equal(sent.length, 1);
  const msg = JSON.parse(sent[0]);
  assert.equal(msg.jsonrpc, "2.0");
  assert.equal(msg.method, "world.open");
  assert.deepEqual(msg.params, { seed: 1337 });
  assert.equal(typeof msg.id, "number");

  session.handleText(JSON.stringify({ jsonrpc: "2.0", id: msg.id, result: { ok: true } }));
  assert.deepEqual(await p, { ok: true });
});

test("error responses reject with RpcRemoteError carrying code/message", async () => {
  const { session, sent } = makeSession();
  const p = session.request("scene.tree");
  const id = JSON.parse(sent[0]).id;
  session.handleText(
    JSON.stringify({ jsonrpc: "2.0", id, error: { code: -32001, message: "no world open" } })
  );
  await assert.rejects(p, (e: unknown) => {
    assert.ok(e instanceof RpcRemoteError);
    assert.equal(e.rpc.code, -32001);
    assert.match(e.rpc.message, /no world open/);
    return true;
  });
});

test("ids are unique and responses correlate to the right request", async () => {
  const { session, sent } = makeSession();
  const a = session.request("a");
  const b = session.request("b");
  const idA = JSON.parse(sent[0]).id;
  const idB = JSON.parse(sent[1]).id;
  assert.notEqual(idA, idB);
  // Resolve out of order.
  session.handleText(JSON.stringify({ jsonrpc: "2.0", id: idB, result: "B" }));
  session.handleText(JSON.stringify({ jsonrpc: "2.0", id: idA, result: "A" }));
  assert.equal(await a, "A");
  assert.equal(await b, "B");
});

test("notifications fan out to onNotification (no id)", () => {
  const { session } = makeSession();
  const got: Array<[string, unknown]> = [];
  session.onNotification = (m, p) => got.push([m, p]);
  session.handleText(JSON.stringify({ jsonrpc: "2.0", method: "stream.done", params: { sent: 5 } }));
  assert.deepEqual(got, [["stream.done", { sent: 5 }]]);
});

test("binary frames decode and fan out to onBinary", () => {
  const { session } = makeSession();
  const frames: Array<{ schemaId: number; payloadId: number; len: number }> = [];
  session.onBinary = (f) => frames.push({ schemaId: f.schemaId, payloadId: f.payloadId, len: f.payload.length });
  session.handleBinary(encodeFrame(SchemaId.TEXTURE, 7, new Uint8Array([1, 2, 3, 4])));
  assert.deepEqual(frames, [{ schemaId: SchemaId.TEXTURE, payloadId: 7, len: 4 }]);
});

test("failAllPending rejects every in-flight request", async () => {
  const { session } = makeSession();
  const p = session.request("hang");
  session.failAllPending(new Error("closed"));
  await assert.rejects(p, /closed/);
});
