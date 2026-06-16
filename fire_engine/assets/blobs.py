"""Base64 numpy codec for ``.asset`` binary payloads.

A numpy array serialises to a JSON-friendly dict
``{"dtype": "<f4", "shape": [N, 3], "base64": "..."}`` — contiguous,
little-endian bytes, Base64-encoded — so an .asset stays a single portable
text file. Buildings carry no blobs (a floorplan is all primitives); blobs
exist for genuinely binary content (imported glTF meshes, baked bitmaps).

Docs: docs/systems/assets.md
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np

from fire_engine.assets.types import AssetError


def encode_array(arr: np.ndarray) -> dict[str, Any]:
    """Encode a numpy array as a Base64 blob dict (little-endian, contiguous).

    The output is byte-stable for a given array (same dtype/shape/values always
    produce the same dict), so re-saving an unchanged asset is a no-op git diff.

    Args:
        arr: any numeric numpy array.

    Returns:
        ``{"dtype": str, "shape": list[int], "base64": str}`` where ``dtype`` is
        the little-endian dtype string (e.g. ``"<f4"``) and ``base64`` is the
        ASCII Base64 of the array's contiguous little-endian bytes.

    Example::

        d = encode_array(np.arange(6, dtype="<f4").reshape(2, 3))
        np.array_equal(decode_array(d), np.arange(6).reshape(2, 3))  # True

    Docs: docs/systems/assets.md
    """
    le = np.dtype(arr.dtype).newbyteorder("<")
    a = np.ascontiguousarray(arr, dtype=le)
    return {
        "dtype": le.str,
        "shape": [int(n) for n in a.shape],
        "base64": base64.b64encode(a.tobytes()).decode("ascii"),
    }


def decode_array(d: dict[str, Any]) -> np.ndarray:
    """Inverse of :func:`encode_array` — returns a fresh writable array.

    Args:
        d: a blob dict ``{"dtype", "shape", "base64"}``.

    Returns:
        A new (writable, owns-its-data) ``np.ndarray`` of the stored dtype/shape.

    Raises:
        AssetError: if the dict is missing keys or the bytes don't fit the shape.

    Docs: docs/systems/assets.md
    """
    try:
        raw = base64.b64decode(d["base64"])
        arr = np.frombuffer(raw, dtype=np.dtype(d["dtype"]))
        return arr.reshape(tuple(int(n) for n in d["shape"])).copy()
    except (KeyError, ValueError, TypeError) as e:
        raise AssetError(f"malformed blob: {e}") from e
