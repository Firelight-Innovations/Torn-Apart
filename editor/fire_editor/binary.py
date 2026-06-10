"""Binary frame codec for the Fire Editor protocol (EDITOR_PRD §4).

Bulk data (mesh buffers, texture payloads) crosses the wire as length-implicit
binary WebSocket frames, never base64 through JSON (EDITOR_PRD hard rule 5).

Frame layout, little-endian::

    [u32 magic][u32 schema_id][u32 payload_id][payload bytes...]

``schema_id`` selects the payload interpretation (``SchemaId`` in the generated
bindings: MESH, TEXTURE). ``payload_id`` is referenced from a JSON-RPC message
so the client can correlate a binary frame with the control-channel message that
announced it.

Example::

    frame = encode_frame(SchemaId.TEXTURE, 7, rgba_bytes)
    schema_id, payload_id, payload = decode_frame(frame)
    assert schema_id == SchemaId.TEXTURE and payload_id == 7
"""
from __future__ import annotations

import struct

from ._generated import BINARY_HEADER_SIZE, BINARY_MAGIC

# u32 magic, u32 schema_id, u32 payload_id — little-endian, packed (no padding).
_HEADER = struct.Struct("<III")
assert _HEADER.size == BINARY_HEADER_SIZE, "header struct must match generated size"


class BinaryFrameError(ValueError):
    """Raised when a binary frame is malformed (bad magic or truncated header)."""


def encode_frame(schema_id: int, payload_id: int, payload: bytes) -> bytes:
    """Pack a payload into a protocol binary frame.

    Args:
        schema_id: Payload schema selector (``SchemaId.*``).
        payload_id: Correlation id referenced by the announcing JSON-RPC message.
        payload: Raw payload bytes (e.g. concatenated mesh buffers, RGBA8).

    Returns:
        ``bytes`` ready to send as a single binary WebSocket frame.
    """
    return _HEADER.pack(BINARY_MAGIC, schema_id, payload_id) + payload


def decode_frame(data: bytes) -> tuple[int, int, bytes]:
    """Unpack a protocol binary frame.

    Args:
        data: Bytes of a received binary WebSocket frame.

    Returns:
        ``(schema_id, payload_id, payload)``.

    Raises:
        BinaryFrameError: If shorter than the header or the magic is wrong.
    """
    if len(data) < BINARY_HEADER_SIZE:
        raise BinaryFrameError(
            f"frame too short: {len(data)} < header {BINARY_HEADER_SIZE}"
        )
    magic, schema_id, payload_id = _HEADER.unpack_from(data, 0)
    if magic != BINARY_MAGIC:
        raise BinaryFrameError(f"bad magic 0x{magic:08X}, expected 0x{BINARY_MAGIC:08X}")
    return schema_id, payload_id, data[BINARY_HEADER_SIZE:]
