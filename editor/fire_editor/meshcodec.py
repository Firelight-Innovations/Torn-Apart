"""Mesh payload codec — serialise engine ``MeshArrays`` to a MESH binary frame.

The daemon meshes chunks with the engine's culled-face mesher and ships the raw
numpy buffers to the viewport as a single binary frame (EDITOR_PRD hard rule 5 —
never base64 mesh data). The webview maps these typed arrays straight into a
three.js ``BufferGeometry``.

MESH payload layout (little-endian; this is the *payload* that follows the
12-byte protocol frame header from :mod:`fire_editor.binary`)::

    i32 cx, i32 cy, i32 cz        # chunk coord (self-describing routing)
    u32 vertex_count N
    u32 index_count  M
    f32[N*3] positions            # world meters
    f32[N*3] normals              # flat per-face
    f32[N*4] colors               # RGBA, greyscale x sunlight
    f32[N*2] uvs                  # planar, tile @ 1 m
    u32[M]   indices

Positions are absolute world meters (DECISIONS: "Mesher emits world-space
vertices"), so the viewport attaches every chunk at the origin with no offset.
"""
from __future__ import annotations

import struct
from typing import Any

import numpy as np

# i32 cx, i32 cy, i32 cz, u32 N, u32 M
_SUBHEADER = struct.Struct("<iiiII")
MESH_SUBHEADER_SIZE = _SUBHEADER.size  # 20


def encode_mesh_payload(coord: tuple[int, int, int], mesh: Any) -> bytes:
    """Pack a ``MeshArrays`` into a MESH frame payload.

    Args:
        coord: ``(cx, cy, cz)`` chunk coordinate.
        mesh: An engine ``terrain.MeshArrays`` (positions/normals/colors/uvs f32,
            indices u32). Empty meshes (``face_count == 0``) encode to a valid
            header with ``N = M = 0``.

    Returns:
        Payload bytes (concatenate after :func:`fire_editor.binary.encode_frame`).
    """
    positions = np.ascontiguousarray(mesh.positions, dtype=np.float32)
    normals = np.ascontiguousarray(mesh.normals, dtype=np.float32)
    colors = np.ascontiguousarray(mesh.colors, dtype=np.float32)
    uvs = np.ascontiguousarray(mesh.uvs, dtype=np.float32)
    indices = np.ascontiguousarray(mesh.indices, dtype=np.uint32)

    n = int(positions.shape[0])
    m = int(indices.shape[0])
    header = _SUBHEADER.pack(int(coord[0]), int(coord[1]), int(coord[2]), n, m)
    return b"".join(
        (
            header,
            positions.tobytes(),
            normals.tobytes(),
            colors.tobytes(),
            uvs.tobytes(),
            indices.tobytes(),
        )
    )


def decode_mesh_payload(payload: bytes) -> dict:
    """Unpack a MESH payload back into arrays (used by tests and tooling).

    Returns a dict with ``coord``, ``vertex_count``, ``index_count`` and the
    five numpy arrays ``positions``, ``normals``, ``colors``, ``uvs``, ``indices``.
    """
    cx, cy, cz, n, m = _SUBHEADER.unpack_from(payload, 0)
    off = MESH_SUBHEADER_SIZE
    f32 = np.dtype("<f4")
    u32 = np.dtype("<u4")

    def take(count: int, dtype) -> np.ndarray:
        nonlocal off
        nbytes = count * dtype.itemsize
        arr = np.frombuffer(payload, dtype=dtype, count=count, offset=off)
        off += nbytes
        return arr

    positions = take(n * 3, f32).reshape(n, 3)
    normals = take(n * 3, f32).reshape(n, 3)
    colors = take(n * 4, f32).reshape(n, 4)
    uvs = take(n * 2, f32).reshape(n, 2)
    indices = take(m, u32)
    return {
        "coord": (cx, cy, cz),
        "vertex_count": n,
        "index_count": m,
        "positions": positions,
        "normals": normals,
        "colors": colors,
        "uvs": uvs,
        "indices": indices,
    }
