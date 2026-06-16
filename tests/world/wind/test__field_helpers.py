"""
tests/world/wind/test__field_helpers.py — Tests for fire_engine/world/wind/_field_helpers.py.

Covers:
  vertical_profile — boundary-layer wind-speed multiplier (floor, cap, monotone)
  pack_wind_field  — fp16 BGRA GPU-texture packing (byte length, channel order)
  _snapshot_materials — chunk → materials dict extraction (object or bare array)

Categories:
  CORRECTNESS — known input → known output
  DETERMINISM — pure functions; same input → same bytes (no seed needed)
  ROUND-TRIP  — pack → decode → compare to source arrays

Headless only. No panda3d. No per-element Python loops.
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.core.config import Config
from fire_engine.world.wind._field_helpers import (
    _snapshot_materials,
    pack_wind_field,
    vertical_profile,
)
from fire_engine.world.wind.types import WindSnapshot

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snap(cells: int = 8, vx_val: float = 3.0, vy_val: float = -1.5) -> WindSnapshot:
    """Create a WindSnapshot with uniform vx/vy/turb values for easy assertions."""
    field = np.zeros((cells, cells, 4), dtype=np.float32)
    field[..., 0] = vx_val
    field[..., 1] = vy_val
    field[..., 2] = 0.5  # turb
    # channel 3 is reserved (unused in pack); leave at 0
    return WindSnapshot(
        field=field,
        origin_m=(0.0, 0.0),
        cell_m=4.0,
        cells=cells,
        wind_time=0.0,
    )


# ---------------------------------------------------------------------------
# vertical_profile
# ---------------------------------------------------------------------------


class TestVerticalProfile:
    def test_floor_at_ground_and_below(self):
        """At z == z_ground and below, profile == floor."""
        cfg = Config()
        z = np.array([cfg.ground_height_m - 10.0, cfg.ground_height_m], dtype=np.float32)
        m = vertical_profile(z, cfg.ground_height_m, cfg)
        for val in m:
            assert val == pytest.approx(cfg.wind_profile_floor, abs=1e-6)

    def test_monotone_nondecreasing(self):
        """Profile is monotonically non-decreasing with height."""
        cfg = Config()
        zs = np.linspace(cfg.ground_height_m, cfg.ground_height_m + 200.0, 100)
        m = vertical_profile(zs, cfg.ground_height_m, cfg)
        diffs = np.diff(m)
        assert np.all(diffs >= -1e-6), f"profile decreased at diffs={diffs[diffs < -1e-6]}"

    def test_cap_at_high_altitude(self):
        """At extreme altitude the profile saturates at cap."""
        cfg = Config()
        z_high = np.array([cfg.ground_height_m + 10_000.0], dtype=np.float32)
        m = vertical_profile(z_high, cfg.ground_height_m, cfg)
        assert m[0] == pytest.approx(cfg.wind_profile_cap, abs=1e-5)

    def test_value_at_z_ref_is_one(self):
        """At z_ground + z_ref the profile should equal exactly 1.0 (0^shear = 0 => floor;
        but z_ref gives (z_ref/z_ref)^shear = 1.0). Clamp keeps it in [floor, cap]."""
        cfg = Config()
        z_ref_abs = cfg.ground_height_m + cfg.wind_profile_z_ref
        m = vertical_profile(np.array([z_ref_abs], dtype=np.float32), cfg.ground_height_m, cfg)
        # (1.0)^shear == 1.0; within [floor, cap], so should equal 1.0.
        assert m[0] == pytest.approx(1.0, abs=1e-5)

    def test_output_dtype_is_float32(self):
        cfg = Config()
        z = np.linspace(0.0, 100.0, 10, dtype=np.float32)
        m = vertical_profile(z, cfg.ground_height_m, cfg)
        assert m.dtype == np.float32

    def test_output_shape_matches_input(self):
        cfg = Config()
        z = np.zeros((3, 4), dtype=np.float32)
        m = vertical_profile(z, cfg.ground_height_m, cfg)
        assert m.shape == (3, 4)

    def test_no_nan_or_inf(self):
        """No NaN/Inf anywhere in a wide z range including below-ground."""
        cfg = Config()
        z = np.linspace(cfg.ground_height_m - 50.0, cfg.ground_height_m + 1000.0, 200)
        m = vertical_profile(z, cfg.ground_height_m, cfg)
        assert np.isfinite(m).all()

    def test_deterministic_pure_function(self):
        """Same inputs → bit-identical outputs on two calls."""
        cfg = Config()
        z = np.linspace(0.0, 200.0, 50, dtype=np.float32)
        a = vertical_profile(z, cfg.ground_height_m, cfg)
        b = vertical_profile(z, cfg.ground_height_m, cfg)
        np.testing.assert_array_equal(a, b)

    def test_all_values_in_floor_cap_range(self):
        """Every output value is in [floor, cap]."""
        cfg = Config()
        z = np.linspace(cfg.ground_height_m - 20.0, cfg.ground_height_m + 500.0, 200)
        m = vertical_profile(z, cfg.ground_height_m, cfg)
        assert m.min() >= cfg.wind_profile_floor - 1e-6
        assert m.max() <= cfg.wind_profile_cap + 1e-6


# ---------------------------------------------------------------------------
# pack_wind_field
# ---------------------------------------------------------------------------


class TestPackWindField:
    def test_byte_length(self):
        """Packed buffer has cells*cells*4*2 bytes (fp16 RGBA)."""
        cells = 8
        snap = _snap(cells=cells)
        data = pack_wind_field(snap)
        assert len(data) == cells * cells * 4 * 2

    def test_byte_length_64_cells(self):
        """Same formula for 64-cell grid."""
        cells = 64
        snap = _snap(cells=cells)
        data = pack_wind_field(snap)
        assert len(data) == cells * cells * 4 * 2

    def test_channel_order_bgra_at_cell(self):
        """
        Decode the buffer and assert BGRA channel order for a known cell.

        pack layout: float16 row-major (y, x) BGRA:
          B = turb, G = vy, R = vx, A = hypot(vx, vy)
        """
        cells = 8
        vx_val, vy_val, turb_val = 3.0, -1.5, 0.5
        snap = _snap(cells=cells, vx_val=vx_val, vy_val=vy_val)
        data = pack_wind_field(snap)
        buf = np.frombuffer(data, dtype=np.float16).reshape(cells, cells, 4).astype(np.float32)

        # Check a few cells; all cells have same values in this uniform snapshot.
        for i, j in [(0, 0), (3, 5), (7, 7)]:
            # Row-major: texel for field[x=i, y=j] is buf[j, i]
            b, g, r, a = buf[j, i]
            np.testing.assert_allclose(b, turb_val, atol=1e-2, err_msg="B != turb")
            np.testing.assert_allclose(g, vy_val, atol=1e-2, err_msg="G != vy")
            np.testing.assert_allclose(r, vx_val, atol=1e-2, err_msg="R != vx")
            expected_speed = float(np.hypot(vx_val, vy_val))
            np.testing.assert_allclose(a, expected_speed, atol=1e-2, err_msg="A != speed")

    def test_row_major_transposition(self):
        """
        The buffer is transposed (y outer, x inner) relative to field[x, y].

        Build a gradient field: field[i, j, 0] = float(i) so vx varies in x.
        After transposition, row 0 (y=0) of the buffer should equal
        field[:, 0, 0] (varying x at fixed y=0).
        """
        cells = 8
        field = np.zeros((cells, cells, 4), dtype=np.float32)
        for i in range(cells):
            field[i, :, 0] = float(i)  # vx = i (varies with x-index)
        snap = WindSnapshot(
            field=field, origin_m=(0.0, 0.0), cell_m=4.0, cells=cells, wind_time=0.0
        )
        data = pack_wind_field(snap)
        buf = np.frombuffer(data, dtype=np.float16).reshape(cells, cells, 4).astype(np.float32)

        # In buf[y=0, x=0..7], the R channel should be 0.0, 1.0, 2.0, ...
        r_row0 = buf[0, :, 2]  # BGRA: channel 2 = R = vx
        expected = np.arange(cells, dtype=np.float32)
        np.testing.assert_allclose(r_row0, expected, atol=0.01)

    def test_pure_function_deterministic(self):
        """Same snapshot → bit-identical bytes on two calls."""
        snap = _snap(cells=16)
        a = pack_wind_field(snap)
        b = pack_wind_field(snap)
        assert a == b

    def test_no_nan_in_packed_bytes(self):
        """The packed fp16 buffer must not contain NaN."""
        snap = _snap(cells=32)
        data = pack_wind_field(snap)
        buf = np.frombuffer(data, dtype=np.float16)
        assert not np.isnan(buf.astype(np.float32)).any()

    def test_speed_channel_is_hypot(self):
        """The A channel (index 3) equals hypot(vx, vy) for every texel."""
        cells = 8
        vx_val, vy_val = 3.0, 4.0  # speed == 5.0 exactly
        snap = _snap(cells=cells, vx_val=vx_val, vy_val=vy_val)
        data = pack_wind_field(snap)
        buf = np.frombuffer(data, dtype=np.float16).reshape(cells, cells, 4).astype(np.float32)
        speed_channel = buf[:, :, 3]  # A channel
        expected_speed = float(np.hypot(vx_val, vy_val))  # 5.0
        np.testing.assert_allclose(speed_channel, expected_speed, atol=1e-2)


# ---------------------------------------------------------------------------
# _snapshot_materials
# ---------------------------------------------------------------------------


class TestSnapshotMaterials:
    def test_bare_array_passthrough(self):
        """A dict of coord -> ndarray is returned as-is (arrays are the value)."""
        arr = np.zeros((32, 32, 32), dtype=np.uint8)
        chunks = {(0, 0, 0): arr}
        result = _snapshot_materials(chunks)
        assert (0, 0, 0) in result
        assert result[(0, 0, 0)] is arr  # same object — reference, not copy

    def test_chunk_object_with_materials_attr(self):
        """A chunk object with a .materials attribute has that array extracted."""

        class FakeChunk:
            def __init__(self):
                self.materials = np.ones((32, 32, 32), dtype=np.uint8)

        chunk = FakeChunk()
        chunks = {(1, 2, 3): chunk}
        result = _snapshot_materials(chunks)
        assert (1, 2, 3) in result
        assert result[(1, 2, 3)] is chunk.materials

    def test_mixed_types_in_same_dict(self):
        """A dict with both bare arrays and chunk objects is handled correctly."""

        class FakeChunk:
            def __init__(self, arr):
                self.materials = arr

        arr_bare = np.zeros((32, 32, 32), dtype=np.uint8)
        arr_obj = np.ones((32, 32, 32), dtype=np.uint8)
        chunk_obj = FakeChunk(arr_obj)

        chunks = {(0, 0, 0): arr_bare, (1, 0, 0): chunk_obj}
        result = _snapshot_materials(chunks)
        assert result[(0, 0, 0)] is arr_bare
        assert result[(1, 0, 0)] is arr_obj

    def test_empty_dict_returns_empty(self):
        """An empty input dict returns an empty output dict."""
        result = _snapshot_materials({})
        assert result == {}

    def test_output_count_equals_input_count(self):
        """One entry per input chunk — no extra or missing entries."""
        chunks = {(i, 0, 0): np.zeros((32, 32, 32), dtype=np.uint8) for i in range(5)}
        result = _snapshot_materials(chunks)
        assert len(result) == 5

    def test_multiple_chunks_all_preserved(self):
        """All coord keys from the input appear in the output."""
        keys = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        chunks = {k: np.zeros((32, 32, 32), dtype=np.uint8) for k in keys}
        result = _snapshot_materials(chunks)
        assert set(result.keys()) == set(keys)
