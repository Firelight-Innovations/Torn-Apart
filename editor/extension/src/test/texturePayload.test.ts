// Unit tests for the TEXTURE payload decoder — mirror of the Python
// tests/editor/test_texture_payload.py codec round-trip, in the webview's
// decode direction.
import { test } from "node:test";
import assert from "node:assert/strict";

import { decodeTexturePayload, TEXTURE_SUBHEADER_SIZE } from "../protocol/texturePayload";

/** Build a payload the way fire_editor/texturecodec.py encode_texture_payload does. */
function encode(width: number, height: number, rgba: Uint8Array): Uint8Array {
  const out = new Uint8Array(TEXTURE_SUBHEADER_SIZE + rgba.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, width, true);
  view.setUint32(4, height, true);
  out.set(rgba, TEXTURE_SUBHEADER_SIZE);
  return out;
}

test("decodes width/height and copies the rgba bytes", () => {
  const rgba = new Uint8Array(2 * 4 * 4); // 4x2 RGBA
  for (let i = 0; i < rgba.length; i++) rgba[i] = i & 0xff;
  const out = decodeTexturePayload(encode(4, 2, rgba));
  assert.equal(out.width, 4);
  assert.equal(out.height, 2);
  assert.deepEqual(Array.from(out.data), Array.from(rgba));
});

test("returns a copy, not a view onto the frame buffer", () => {
  const rgba = new Uint8Array([10, 20, 30, 40]);
  const payload = encode(1, 1, rgba);
  const out = decodeTexturePayload(payload);
  // Mutating the source frame must not change the decoded texture.
  payload[TEXTURE_SUBHEADER_SIZE] = 99;
  assert.equal(out.data[0], 10);
});

test("throws on a truncated payload", () => {
  const full = encode(4, 2, new Uint8Array(4 * 2 * 4));
  const truncated = full.slice(0, full.length - 5);
  assert.throws(() => decodeTexturePayload(truncated), /truncated/);
});

test("decodes a payload offset inside a larger ArrayBuffer", () => {
  // Simulate a Uint8Array that is a subarray (non-zero byteOffset).
  const rgba = new Uint8Array([1, 2, 3, 4]);
  const inner = encode(1, 1, rgba);
  const backing = new Uint8Array(inner.length + 8);
  backing.set(inner, 8);
  const view = backing.subarray(8); // byteOffset = 8
  const out = decodeTexturePayload(view);
  assert.equal(out.width, 1);
  assert.deepEqual(Array.from(out.data), [1, 2, 3, 4]);
});
