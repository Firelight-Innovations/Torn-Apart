"""
tests/save/test__codec.py — Tests for fire_engine/save/_codec.py.

Covers: compute_config_digest (determinism, sensitivity, format),
encode_value / decode_value round-trips (numpy arrays, tuple keys, primitives),
encode_delta / decode_delta round-trips (terrain-like, edge cases).
Headless: no panda3d.

Mirror of fire_engine/save/_codec.py.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from fire_engine.core.config import load_config
from fire_engine.save._codec import (
    compute_config_digest,
    decode_delta,
    decode_value,
    encode_delta,
    encode_value,
)

# ---------------------------------------------------------------------------
# compute_config_digest — determinism and sensitivity
# ---------------------------------------------------------------------------


class TestComputeConfigDigest:
    """Determinism and field-sensitivity tests for compute_config_digest."""

    def test_same_config_same_digest(self):
        """Same Config always produces the same digest (determinism)."""
        cfg = load_config()
        d1 = compute_config_digest(cfg)
        d2 = compute_config_digest(cfg)
        assert d1 == d2

    def test_digest_is_32_hex_chars(self):
        """Digest is exactly 32 lowercase hex characters (blake2b digest_size=16)."""
        cfg = load_config()
        digest = compute_config_digest(cfg)
        assert len(digest) == 32
        assert all(c in "0123456789abcdef" for c in digest)

    def test_different_world_seed_changes_digest(self):
        """Changing world_seed changes the digest."""
        cfg = load_config()
        cfg2 = dataclasses.replace(cfg, world_seed=cfg.world_seed + 1)
        assert compute_config_digest(cfg) != compute_config_digest(cfg2)

    def test_different_chunk_size_changes_digest(self):
        """Changing chunk_size changes the digest."""
        cfg = load_config()
        cfg2 = dataclasses.replace(cfg, chunk_size=cfg.chunk_size + 1)
        assert compute_config_digest(cfg) != compute_config_digest(cfg2)

    def test_different_voxel_size_changes_digest(self):
        """Changing voxel_size changes the digest."""
        cfg = load_config()
        cfg2 = dataclasses.replace(cfg, voxel_size=cfg.voxel_size * 2.0)
        assert compute_config_digest(cfg) != compute_config_digest(cfg2)

    def test_different_light_grid_scale_changes_digest(self):
        """Changing light_grid_scale changes the digest."""
        cfg = load_config()
        cfg2 = dataclasses.replace(cfg, light_grid_scale=cfg.light_grid_scale + 1)
        assert compute_config_digest(cfg) != compute_config_digest(cfg2)

    def test_show_fps_does_not_change_digest(self):
        """Changing show_fps (debug flag) does NOT change the digest."""
        cfg = load_config()
        cfg2 = dataclasses.replace(cfg, show_fps=not cfg.show_fps)
        assert compute_config_digest(cfg) == compute_config_digest(cfg2)

    def test_digest_is_lowercase(self):
        """Digest string is all lowercase."""
        cfg = load_config()
        digest = compute_config_digest(cfg)
        assert digest == digest.lower()


# ---------------------------------------------------------------------------
# encode_value / decode_value — primitive pass-through
# ---------------------------------------------------------------------------


class TestEncodeDecodePrimitives:
    """Primitive types pass through encode/decode unchanged."""

    def test_int_passthrough(self):
        """Integer values encode and decode to the same value."""
        assert decode_value(encode_value(42)) == 42

    def test_float_passthrough(self):
        """Float values encode and decode to the same value."""
        assert decode_value(encode_value(3.14)) == pytest.approx(3.14)

    def test_none_passthrough(self):
        """None encodes and decodes to None."""
        assert decode_value(encode_value(None)) is None

    def test_bool_passthrough(self):
        """Boolean values encode and decode to the same value."""
        assert decode_value(encode_value(True)) is True
        assert decode_value(encode_value(False)) is False

    def test_string_passthrough(self):
        """String values encode and decode to the same value."""
        assert decode_value(encode_value("hello")) == "hello"

    def test_list_of_ints_passthrough(self):
        """A list of ints encodes and decodes to the same list."""
        original = [1, 2, 3, 4]
        assert decode_value(encode_value(original)) == original


# ---------------------------------------------------------------------------
# encode_value / decode_value — numpy arrays
# ---------------------------------------------------------------------------


class TestEncodeDecodeNumpyArrays:
    """Numpy array encoding/decoding correctness."""

    def test_uint8_3d_array_round_trip(self):
        """uint8 [32,32,32] terrain array survives encode→decode intact."""
        arr = np.arange(32 * 32 * 32, dtype=np.uint8).reshape(32, 32, 32)
        decoded = decode_value(encode_value(arr))
        assert isinstance(decoded, np.ndarray)
        assert decoded.dtype == np.uint8
        assert decoded.shape == (32, 32, 32)
        assert np.array_equal(arr, decoded)

    def test_float32_array_round_trip(self):
        """float32 array round-trips with correct dtype and shape."""
        arr = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(10, 10)
        decoded = decode_value(encode_value(arr))
        assert decoded.dtype == np.float32
        assert decoded.shape == (10, 10)
        assert np.array_equal(arr, decoded)

    def test_float64_array_round_trip(self):
        """float64 array round-trips with correct dtype."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        decoded = decode_value(encode_value(arr))
        assert decoded.dtype == np.float64
        assert np.array_equal(arr, decoded)

    def test_int32_array_round_trip(self):
        """int32 array round-trips with correct dtype and values."""
        arr = np.array([-100, 0, 100, 32767], dtype=np.int32)
        decoded = decode_value(encode_value(arr))
        assert decoded.dtype == np.int32
        assert np.array_equal(arr, decoded)

    def test_nan_float32_round_trips_bit_identical(self):
        """float32 array with NaN round-trips bit-identically."""
        arr = np.array([1.0, float("nan"), float("inf"), -float("inf")], dtype=np.float32)
        decoded = decode_value(encode_value(arr))
        assert isinstance(decoded, np.ndarray)
        assert decoded.dtype == np.float32
        assert arr.tobytes() == decoded.tobytes()

    def test_empty_array_round_trips(self):
        """Zero-element uint8 array encodes and decodes with correct dtype."""
        arr = np.array([], dtype=np.uint8)
        decoded = decode_value(encode_value(arr))
        assert isinstance(decoded, np.ndarray)
        assert decoded.dtype == np.uint8
        assert decoded.shape == (0,)


# ---------------------------------------------------------------------------
# encode_value / decode_value — dict key types
# ---------------------------------------------------------------------------


class TestEncodeDecodeDictKeys:
    """Dict encoding for string keys vs tuple keys."""

    def test_string_key_dict_no_wrapper(self):
        """Dict with only string keys encodes without __delta_type__ wrapper."""
        data = {"health": 42, "name": "hero"}
        encoded = encode_value(data)
        assert isinstance(encoded, dict)
        assert "__delta_type__" not in encoded
        decoded = decode_value(encoded)
        assert decoded == data

    def test_tuple_key_dict_round_trip(self):
        """Dict with tuple keys encodes with kv_pairs wrapper and decodes correctly."""
        arr = np.zeros((4,), dtype=np.uint8)
        original = {
            (0, 0, 0): arr.copy(),
            (1, -2, 3): arr + 1,
        }
        encoded = encode_value(original)
        # kv_pairs wrapper must be present
        assert isinstance(encoded, dict)
        assert encoded.get("__delta_type__") == "kv_pairs"

        decoded = decode_value(encoded)
        assert isinstance(decoded, dict)
        assert set(decoded.keys()) == {(0, 0, 0), (1, -2, 3)}
        for k in original:
            assert np.array_equal(original[k], decoded[k])

    def test_mixed_chunk_coord_delta_round_trip(self):
        """Terrain-style {(cx,cy,cz): ndarray} dict round-trips correctly."""
        arr1 = np.ones((32, 32, 32), dtype=np.uint8) * 7
        arr2 = np.zeros((32, 32, 32), dtype=np.uint8)
        original = {(0, 0, 1): arr1, (2, -1, 3): arr2}

        decoded = decode_value(encode_value(original))
        assert set(decoded.keys()) == {(0, 0, 1), (2, -1, 3)}
        assert np.array_equal(decoded[(0, 0, 1)], arr1)
        assert np.array_equal(decoded[(2, -1, 3)], arr2)

    def test_empty_dict_round_trip(self):
        """Empty dict encodes and decodes to empty dict."""
        decoded = decode_value(encode_value({}))
        assert decoded == {}

    def test_nested_string_key_dicts_round_trip(self):
        """Nested dicts with string keys encode/decode recursively."""
        original = {"outer": {"inner": 99, "list": [1, 2, 3]}}
        decoded = decode_value(encode_value(original))
        assert decoded["outer"]["inner"] == 99
        assert decoded["outer"]["list"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# encode_delta / decode_delta — full pipeline (msgpack bytes)
# ---------------------------------------------------------------------------


class TestEncodeDelta:
    """encode_delta produces bytes; decode_delta inverts it."""

    def test_empty_delta_round_trip(self):
        """Empty delta encodes and decodes to empty dict."""
        raw = encode_delta({})
        recovered = decode_delta(raw)
        assert recovered == {}

    def test_string_key_delta_round_trip(self):
        """String-keyed delta (e.g. clock state) round-trips."""
        delta = {"tick": 42, "day": 3, "name": "test"}
        raw = encode_delta(delta)
        recovered = decode_delta(raw)
        assert recovered == delta

    def test_tuple_key_terrain_delta_round_trip(self):
        """Tuple-keyed terrain delta round-trips correctly."""
        arr1 = np.ones((32, 32, 32), dtype=np.uint8) * 5
        arr2 = np.zeros((32, 32, 32), dtype=np.uint8)
        delta = {(0, 0, 1): arr1, (2, 3, -1): arr2}

        raw = encode_delta(delta)
        assert isinstance(raw, bytes)
        recovered = decode_delta(raw)

        assert set(recovered.keys()) == {(0, 0, 1), (2, 3, -1)}
        assert np.array_equal(recovered[(0, 0, 1)], arr1)
        assert np.array_equal(recovered[(2, 3, -1)], arr2)

    def test_encode_delta_returns_bytes(self):
        """encode_delta always returns bytes (not str or dict)."""
        raw = encode_delta({"x": 1})
        assert isinstance(raw, bytes)
        assert len(raw) > 0

    def test_2d_uint8_array_in_delta(self):
        """2D uint8 array inside a delta dict round-trips correctly."""
        arr = np.arange(256, dtype=np.uint8).reshape(16, 16)
        raw = encode_delta({"arr": arr})
        recovered = decode_delta(raw)
        assert np.array_equal(recovered["arr"], arr)
        assert recovered["arr"].dtype == np.uint8

    def test_determinism_same_delta_same_bytes(self):
        """
        Encoding the same deterministic delta twice produces the same bytes.
        (Relies on msgpack dict key ordering being stable in Python 3.7+.)
        """
        arr = np.ones((4, 4), dtype=np.float32) * 3.14
        delta = {"tick": 1, "arr": arr}
        raw1 = encode_delta(delta)
        raw2 = encode_delta(delta)
        assert raw1 == raw2
