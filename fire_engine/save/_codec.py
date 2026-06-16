"""
save/_codec.py — msgpack + zlib codec and config digest for SaveManager.

Private implementation module.  Encodes/decodes system delta dicts to/from
msgpack bytes, handling numpy arrays and tuple dict-keys.  Also computes the
blake2b config digest used to detect incompatible config changes.

Docs: docs/systems/save.md
"""

from __future__ import annotations

import hashlib
from typing import Any

import msgpack  # type: ignore[import-untyped]  # msgpack has no py.typed or stubs
import numpy as np

from fire_engine.core.config import Config

# ---------------------------------------------------------------------------
# Config digest
# ---------------------------------------------------------------------------

# blake2b digest size in bytes for the config digest (16 bytes = 32 hex chars).
_CONFIG_DIGEST_SIZE: int = 16


def compute_config_digest(config: Config) -> str:
    """
    Compute a stable blake2b hex digest of the config fields that affect
    world generation / save compatibility.

    Fields included: ``world_seed``, ``voxel_size``, ``chunk_size``,
    ``light_grid_scale``.  Debug flags and view_distance_chunks are excluded
    (changing them does not invalidate a save file).

    Parameters
    ----------
    config : Config
        The current engine config.

    Returns
    -------
    str
        32-character lowercase hex string.

    Docs: docs/systems/save.md
    """
    canonical = (
        f"{config.world_seed}:{config.voxel_size}:{config.chunk_size}:{config.light_grid_scale}"
    )
    return hashlib.blake2b(
        canonical.encode("ascii"),
        digest_size=_CONFIG_DIGEST_SIZE,
    ).hexdigest()


# ---------------------------------------------------------------------------
# Numpy + tuple-key msgpack encoding
# ---------------------------------------------------------------------------

_NDARRAY_TAG = "__ndarray__"
_DELTA_KV_TAG = "__delta_type__"


def encode_value(obj: Any) -> Any:
    """
    Recursively encode a value so it is msgpack-serialisable.

    Transforms:
    - ``numpy.ndarray`` → ``[_NDARRAY_TAG, dtype_str, shape_list, raw_bytes]``
    - ``dict`` with non-string keys → ``{_DELTA_KV_TAG: "kv_pairs", "pairs": [...]}``
    - ``dict`` with string keys → encode values recursively
    - ``list`` / ``tuple`` → encode elements recursively (tuples become lists)
    - primitives (int, float, str, bool, None) → pass through

    Parameters
    ----------
    obj : Any
        The object to encode.

    Returns
    -------
    Any
        A msgpack-serialisable representation.

    Docs: docs/systems/save.md
    """
    if isinstance(obj, np.ndarray):
        return [_NDARRAY_TAG, str(obj.dtype), list(obj.shape), obj.tobytes()]
    if isinstance(obj, dict):
        # Check if any key is non-string (e.g. tuple key for chunk coords)
        if obj and not all(isinstance(k, str) for k in obj):
            pairs = [[encode_value(k), encode_value(v)] for k, v in obj.items()]
            return {_DELTA_KV_TAG: "kv_pairs", "pairs": pairs}
        return {k: encode_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode_value(x) for x in obj]
    # Primitives: int, float, str, bool, None, bytes
    return obj


def decode_value(obj: Any) -> Any:
    """
    Recursively decode a value produced by :func:`encode_value`.

    Inverse transforms:
    - ``[_NDARRAY_TAG, ...]`` → ``numpy.ndarray``
    - ``{_DELTA_KV_TAG: "kv_pairs", ...}`` → dict with tuple keys
    - dicts with string keys → decode values recursively
    - lists → decode elements recursively

    Parameters
    ----------
    obj : Any
        Object as decoded from msgpack.

    Returns
    -------
    Any
        The original Python / numpy value.

    Docs: docs/systems/save.md
    """
    if isinstance(obj, list):
        # Check for numpy array tag
        if (
            len(obj) == 4
            and isinstance(obj[0], (str, bytes))
            and (obj[0] == _NDARRAY_TAG or obj[0] == _NDARRAY_TAG.encode())
        ):
            _tag, dtype_str, shape, raw = obj
            if isinstance(dtype_str, bytes):
                dtype_str = dtype_str.decode()
            return np.frombuffer(raw, dtype=np.dtype(dtype_str)).reshape(shape)
        return [decode_value(x) for x in obj]
    if isinstance(obj, dict):
        # Decode bytes keys (msgpack may return bytes for string keys)
        decoded_dict: dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(k, bytes):
                k = k.decode()
            decoded_dict[k] = v
        obj = decoded_dict

        delta_type = obj.get(_DELTA_KV_TAG)
        if delta_type == "kv_pairs":
            result: dict[Any, Any] = {}
            for pair in obj["pairs"]:
                key = decode_value(pair[0])
                val = decode_value(pair[1])
                # If key decoded to a list (was a tuple), convert to tuple
                if isinstance(key, list):
                    key = tuple(int(x) for x in key)
                result[key] = val
            return result
        return {k: decode_value(v) for k, v in obj.items()}
    return obj


def encode_delta(delta: dict[str, Any]) -> bytes:
    """
    Encode a system delta dict to msgpack bytes.

    Parameters
    ----------
    delta : dict
        As returned by ``Saveable.get_delta()``.

    Returns
    -------
    bytes
        Raw msgpack-encoded bytes (not compressed).

    Docs: docs/systems/save.md
    """
    encoded = encode_value(delta)
    return bytes(msgpack.packb(encoded, use_bin_type=True))


def decode_delta(data: bytes) -> dict[str, Any]:
    """
    Decode a system delta from raw msgpack bytes.

    Parameters
    ----------
    data : bytes
        As produced by :func:`encode_delta`.

    Returns
    -------
    dict
        The original delta as passed to :func:`encode_delta`.

    Docs: docs/systems/save.md
    """
    raw = msgpack.unpackb(data, raw=False)
    result: dict[str, Any] = decode_value(raw)
    return result
