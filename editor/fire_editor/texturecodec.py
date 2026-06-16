"""Texture payload codec — serialise an RGBA image to a TEXTURE binary frame.

Used to ship the procedural-ground palette LUT (and any future small textures)
to the viewport as raw bytes (EDITOR_PRD hard rule 5 — never base64 binary
data). The webview maps the payload straight into a three.js ``DataTexture``.

TEXTURE payload layout (little-endian; this is the *payload* that follows the
12-byte protocol frame header from :mod:`fire_editor.binary`)::

    u32 width
    u32 height
    u8[height*width*4] rgba       # row-major, row 0 first

Row order is the numpy array's natural order (row 0 = texture V=0 on the
client; the ground LUT is addressed by explicit row index, so orientation
conventions never matter for it).
"""

from __future__ import annotations

import struct

import numpy as np

# u32 width, u32 height
_SUBHEADER = struct.Struct("<II")
TEXTURE_SUBHEADER_SIZE = _SUBHEADER.size  # 8


def encode_texture_payload(rgba: np.ndarray) -> bytes:
    """Pack an ``(H, W, 4) uint8`` RGBA image into a TEXTURE frame payload.

    Args:
        rgba: Image array, shape ``(height, width, 4)``, dtype ``uint8``.

    Returns:
        Payload bytes (concatenate after :func:`fire_editor.binary.encode_frame`).

    Raises:
        ValueError: On wrong dtype or shape.
    """
    arr = np.asarray(rgba)
    if arr.dtype != np.uint8:
        raise ValueError(f"texture payload must be uint8, got {arr.dtype}")
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"texture payload must be (H, W, 4), got {arr.shape}")
    arr = np.ascontiguousarray(arr)
    h, w = int(arr.shape[0]), int(arr.shape[1])
    return _SUBHEADER.pack(w, h) + arr.tobytes()


def decode_texture_payload(payload: bytes) -> dict:
    """Unpack a TEXTURE payload (used by tests and tooling).

    Returns a dict with ``width``, ``height`` and the ``(H, W, 4) uint8``
    numpy array ``rgba``.
    """
    w, h = _SUBHEADER.unpack_from(payload, 0)
    rgba = np.frombuffer(
        payload, dtype=np.uint8, count=h * w * 4, offset=TEXTURE_SUBHEADER_SIZE
    ).reshape(h, w, 4)
    return {"width": w, "height": h, "rgba": rgba}
