"""Tests for fire_engine.assets.blobs — the Base64 numpy codec."""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.assets.blobs import decode_array, encode_array
from fire_engine.assets.types import AssetError


def test_round_trip_preserves_values_shape_dtype() -> None:
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = decode_array(encode_array(arr))
    assert np.array_equal(out, arr)
    assert out.dtype == arr.dtype
    assert out.shape == arr.shape


@pytest.mark.parametrize(
    "arr",
    [
        np.arange(12, dtype=np.int32).reshape(3, 4),
        np.array([1, 2, 3], dtype=np.uint8),
        np.linspace(0.0, 1.0, 9, dtype=np.float64).reshape(3, 3),
        np.zeros((0, 3), dtype=np.float32),  # empty
    ],
)
def test_round_trip_various_dtypes(arr: np.ndarray) -> None:
    out = decode_array(encode_array(arr))
    assert np.array_equal(out, arr)
    assert out.dtype == arr.dtype
    assert list(out.shape) == list(arr.shape)


def test_encoding_is_deterministic() -> None:
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert encode_array(arr) == encode_array(arr.copy())


def test_dtype_string_is_little_endian() -> None:
    arr = np.arange(4, dtype=np.float32)
    assert encode_array(arr)["dtype"] == "<f4"


def test_big_endian_input_normalised_to_little_endian() -> None:
    be = np.arange(4, dtype=">f4")
    d = encode_array(be)
    assert d["dtype"] == "<f4"
    assert np.array_equal(decode_array(d), be)


def test_decoded_array_is_writable() -> None:
    out = decode_array(encode_array(np.arange(3, dtype=np.float32)))
    out[0] = 9.0  # must not raise (frombuffer views are read-only; we copy)
    assert out[0] == 9.0


def test_malformed_blob_raises_asset_error() -> None:
    with pytest.raises(AssetError):
        decode_array({"dtype": "<f4", "shape": [99], "base64": "AAAA"})
    with pytest.raises(AssetError):
        decode_array({"shape": [1]})  # missing keys
