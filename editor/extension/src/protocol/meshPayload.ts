// MESH payload decoder — mirror of editor/fire_editor/meshcodec.py.
// Decodes the payload that follows the 12-byte protocol frame header (already
// stripped by client.decodeFrame). Pure module: no three/vscode imports, so it
// is shared by the host and the webview bundle.
//
// Layout (little-endian):
//   i32 cx, i32 cy, i32 cz, u32 N, u32 M,
//   f32[N*3] positions, f32[N*3] normals, f32[N*4] colors, f32[N*2] uvs, u32[M] indices

export interface DecodedMesh {
  coord: [number, number, number];
  vertexCount: number;
  indexCount: number;
  positions: Float32Array;
  normals: Float32Array;
  colors: Float32Array;
  uvs: Float32Array;
  indices: Uint32Array;
}

const SUBHEADER = 20;

export function decodeMeshPayload(payload: Uint8Array): DecodedMesh {
  // Ensure a 4-byte-aligned backing buffer for typed-array views.
  let bytes = payload;
  if (payload.byteOffset % 4 !== 0) {
    bytes = payload.slice();
  }
  const base = bytes.byteOffset;
  const buf = bytes.buffer;
  const view = new DataView(buf, base, bytes.byteLength);
  const cx = view.getInt32(0, true);
  const cy = view.getInt32(4, true);
  const cz = view.getInt32(8, true);
  const n = view.getUint32(12, true);
  const m = view.getUint32(16, true);

  let off = base + SUBHEADER;
  const f = (count: number): Float32Array => {
    const a = new Float32Array(buf, off, count);
    off += count * 4;
    return a;
  };
  const positions = f(n * 3);
  const normals = f(n * 3);
  const colors = f(n * 4);
  const uvs = f(n * 2);
  const indices = new Uint32Array(buf, off, m);

  return {
    coord: [cx, cy, cz],
    vertexCount: n,
    indexCount: m,
    positions,
    normals,
    colors,
    uvs,
    indices,
  };
}

/** Stable string key for a chunk coordinate. */
export function chunkKey(c: [number, number, number]): string {
  return `${c[0]},${c[1]},${c[2]}`;
}
