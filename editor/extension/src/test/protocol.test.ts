// Pure-node unit tests for the protocol bindings (run via `npm test`).
// No vscode/ws dependency, so they execute under `node --test` without
// downloading the VS Code test harness. Mirrors tests/editor/test_protocol.py.
import { test } from "node:test";
import assert from "node:assert/strict";

import { encodeFrame, decodeFrame, BinaryFrameError } from "../protocol/binary";
import { SchemaId, PROTOCOL_VERSION, BINARY_MAGIC } from "../protocol/generated";
import { decodeMeshPayload } from "../protocol/meshPayload";

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

test("mesh payload decodes per the documented layout", () => {
  // Build a payload matching editor/fire_editor/meshcodec.py:
  // i32 cx,cy,cz, u32 N, u32 M, f32[N*3] pos, f32[N*3] norm, f32[N*4] col, f32[N*2] uv, u32[M] idx
  const n = 1;
  const m = 3;
  const buf = new ArrayBuffer(20 + (n * 3 + n * 3 + n * 4 + n * 2) * 4 + m * 4);
  const dv = new DataView(buf);
  dv.setInt32(0, 5, true);
  dv.setInt32(4, -7, true);
  dv.setInt32(8, 2, true);
  dv.setUint32(12, n, true);
  dv.setUint32(16, m, true);
  let o = 20;
  for (const v of [1.5, 2.5, 3.5]) { dv.setFloat32(o, v, true); o += 4; } // pos
  for (const v of [0, 0, 1]) { dv.setFloat32(o, v, true); o += 4; }       // norm
  for (const v of [0.2, 0.4, 0.6, 1.0]) { dv.setFloat32(o, v, true); o += 4; } // col
  for (const v of [0.0, 1.0]) { dv.setFloat32(o, v, true); o += 4; }      // uv
  for (const v of [0, 1, 2]) { dv.setUint32(o, v, true); o += 4; }        // idx

  const mesh = decodeMeshPayload(new Uint8Array(buf));
  assert.deepEqual(mesh.coord, [5, -7, 2]);
  assert.equal(mesh.vertexCount, 1);
  assert.equal(mesh.indexCount, 3);
  assert.deepEqual(Array.from(mesh.positions), [1.5, 2.5, 3.5]);
  assert.deepEqual(Array.from(mesh.indices), [0, 1, 2]);
});

test("generated constants are sane", () => {
  assert.ok(Number.isInteger(PROTOCOL_VERSION) && PROTOCOL_VERSION >= 1);
  assert.equal(BINARY_MAGIC, 0x46495245);
});
