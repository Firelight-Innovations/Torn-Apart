// Binary frame codec — mirror of editor/fire_editor/binary.py (EDITOR_PRD §4).
// Frame layout, little-endian: [u32 magic][u32 schema_id][u32 payload_id][payload].
// Pure module (no vscode/ws imports) so it is unit-testable under `node --test`.
import { BINARY_HEADER_SIZE, BINARY_MAGIC } from "./generated";

export interface DecodedFrame {
  schemaId: number;
  payloadId: number;
  payload: Uint8Array;
}

export class BinaryFrameError extends Error {}

export function encodeFrame(
  schemaId: number,
  payloadId: number,
  payload: Uint8Array
): Uint8Array {
  const out = new Uint8Array(BINARY_HEADER_SIZE + payload.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, BINARY_MAGIC, true);
  view.setUint32(4, schemaId, true);
  view.setUint32(8, payloadId, true);
  out.set(payload, BINARY_HEADER_SIZE);
  return out;
}

export function decodeFrame(data: Uint8Array): DecodedFrame {
  if (data.length < BINARY_HEADER_SIZE) {
    throw new BinaryFrameError(
      `frame too short: ${data.length} < header ${BINARY_HEADER_SIZE}`
    );
  }
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const magic = view.getUint32(0, true);
  if (magic !== BINARY_MAGIC) {
    throw new BinaryFrameError(
      `bad magic 0x${magic.toString(16)}, expected 0x${BINARY_MAGIC.toString(16)}`
    );
  }
  const schemaId = view.getUint32(4, true);
  const payloadId = view.getUint32(8, true);
  // Copy so the caller owns a tight buffer (typed arrays into three.js later).
  const payload = data.slice(BINARY_HEADER_SIZE);
  return { schemaId, payloadId, payload };
}
