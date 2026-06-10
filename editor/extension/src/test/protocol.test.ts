// Pure-node unit tests for the protocol bindings (run via `npm test`).
// No vscode/ws dependency, so they execute under `node --test` without
// downloading the VS Code test harness. Mirrors tests/editor/test_protocol.py.
import { test } from "node:test";
import assert from "node:assert/strict";

import { encodeFrame, decodeFrame, BinaryFrameError } from "../protocol/binary";
import { SchemaId, PROTOCOL_VERSION, BINARY_MAGIC } from "../protocol/generated";

test("binary frame round-trips", () => {
  const payload = new Uint8Array([1, 2, 3, 250, 255, 0]);
  const frame = encodeFrame(SchemaId.MESH, 42, payload);
  const out = decodeFrame(frame);
  assert.equal(out.schemaId, SchemaId.MESH);
  assert.equal(out.payloadId, 42);
  assert.deepEqual(Array.from(out.payload), Array.from(payload));
});

test("empty payload round-trips", () => {
  const frame = encodeFrame(SchemaId.TEXTURE, 0, new Uint8Array(0));
  const out = decodeFrame(frame);
  assert.equal(out.schemaId, SchemaId.TEXTURE);
  assert.equal(out.payloadId, 0);
  assert.equal(out.payload.length, 0);
});

test("bad magic is rejected", () => {
  const frame = encodeFrame(SchemaId.MESH, 1, new Uint8Array([9]));
  frame[0] ^= 0xff;
  assert.throws(() => decodeFrame(frame), BinaryFrameError);
});

test("truncated header is rejected", () => {
  assert.throws(() => decodeFrame(new Uint8Array([0, 1])), BinaryFrameError);
});

test("generated constants are sane", () => {
  assert.equal(PROTOCOL_VERSION, 1);
  assert.equal(BINARY_MAGIC, 0x46495245);
});
