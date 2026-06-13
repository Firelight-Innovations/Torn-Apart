// TEXTURE payload decoder — mirror of editor/fire_editor/texturecodec.py.
// Layout (after the 12-byte frame header), little-endian:
//   u32 width
//   u32 height
//   u8[height*width*4] rgba    (row-major, row 0 first)

export interface TexturePayload {
  width: number;
  height: number;
  /** RGBA8, row-major, length = width * height * 4. */
  data: Uint8Array;
}

export const TEXTURE_SUBHEADER_SIZE = 8;

export function decodeTexturePayload(payload: Uint8Array): TexturePayload {
  const view = new DataView(
    payload.buffer as ArrayBuffer,
    payload.byteOffset,
    payload.byteLength
  );
  const width = view.getUint32(0, true);
  const height = view.getUint32(4, true);
  const expected = TEXTURE_SUBHEADER_SIZE + width * height * 4;
  if (payload.byteLength < expected) {
    throw new Error(
      `TEXTURE payload truncated: ${payload.byteLength} bytes, expected ${expected}`
    );
  }
  // Copy so the texture owns its bytes (the frame buffer may be reused).
  const data = payload.slice(
    TEXTURE_SUBHEADER_SIZE,
    TEXTURE_SUBHEADER_SIZE + width * height * 4
  );
  return { width, height, data };
}
