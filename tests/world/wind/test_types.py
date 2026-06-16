"""
tests/world/wind/test_types.py — Tests for fire_engine/world/wind/types.py.

Covers the three frozen dataclasses: WindSnapshot, VenturiJob, VenturiResult.

Categories:
  CORRECTNESS — field values round-trip through construction
  DETERMINISM — frozen dataclasses are immutable; no seed needed
  ROUND-TRIP  — construct → access fields → assert values unchanged

Headless only. No panda3d. No per-element Python loops.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from fire_engine.world.wind.types import VenturiJob, VenturiResult, WindSnapshot

# ---------------------------------------------------------------------------
# WindSnapshot
# ---------------------------------------------------------------------------


class TestWindSnapshot:
    def _make_snapshot(
        self,
        cells: int = 8,
        origin_m: tuple[float, float] = (0.0, 0.0),
        cell_m: float = 4.0,
        wind_time: float = 0.0,
    ) -> WindSnapshot:
        field = np.zeros((cells, cells, 4), dtype=np.float32)
        return WindSnapshot(
            field=field,
            origin_m=origin_m,
            cell_m=cell_m,
            cells=cells,
            wind_time=wind_time,
        )

    def test_field_shape_round_trip(self):
        """field array is stored and retrieved unchanged."""
        cells = 16
        snap = self._make_snapshot(cells=cells)
        assert snap.field.shape == (cells, cells, 4)
        assert snap.field.dtype == np.float32

    def test_origin_m_round_trip(self):
        """origin_m tuple is stored and retrieved unchanged."""
        origin = (-128.0, 64.5)
        snap = self._make_snapshot(origin_m=origin)
        assert snap.origin_m == origin

    def test_cell_m_round_trip(self):
        """cell_m float is stored correctly."""
        snap = self._make_snapshot(cell_m=8.0)
        assert snap.cell_m == 8.0

    def test_cells_round_trip(self):
        """cells int is stored correctly."""
        snap = self._make_snapshot(cells=32)
        assert snap.cells == 32

    def test_wind_time_round_trip(self):
        """wind_time float is stored correctly."""
        snap = self._make_snapshot(wind_time=42.5)
        assert snap.wind_time == 42.5

    def test_is_frozen(self):
        """WindSnapshot is frozen — attempting mutation raises FrozenInstanceError."""
        snap = self._make_snapshot()
        with pytest.raises(dataclasses.FrozenInstanceError):
            snap.wind_time = 99.9  # type: ignore[misc]

    def test_field_values_preserved(self):
        """Field array values set before construction are still there after."""
        cells = 4
        field = np.arange(cells * cells * 4, dtype=np.float32).reshape(cells, cells, 4)
        snap = WindSnapshot(
            field=field, origin_m=(0.0, 0.0), cell_m=4.0, cells=cells, wind_time=0.0
        )
        np.testing.assert_array_equal(snap.field, field)

    def test_equality_by_value(self):
        """Two snapshots with identical fields and metadata compare equal."""
        cells = 4
        field = np.zeros((cells, cells, 4), dtype=np.float32)
        a = WindSnapshot(field=field, origin_m=(0.0, 0.0), cell_m=4.0, cells=cells, wind_time=1.0)
        b = WindSnapshot(field=field, origin_m=(0.0, 0.0), cell_m=4.0, cells=cells, wind_time=1.0)
        assert a == b

    def test_inequality_on_wind_time(self):
        """Two snapshots with different wind_time are not equal."""
        cells = 4
        field = np.zeros((cells, cells, 4), dtype=np.float32)
        a = WindSnapshot(field=field, origin_m=(0.0, 0.0), cell_m=4.0, cells=cells, wind_time=1.0)
        b = WindSnapshot(field=field, origin_m=(0.0, 0.0), cell_m=4.0, cells=cells, wind_time=2.0)
        assert a != b


# ---------------------------------------------------------------------------
# VenturiJob
# ---------------------------------------------------------------------------


class TestVenturiJob:
    def _make_job(
        self,
        origin_cell: tuple[int, int] = (0, 0),
        cells: int = 16,
        cell_m: float = 4.0,
        seq: int = 1,
    ) -> VenturiJob:
        return VenturiJob(
            origin_cell=origin_cell,
            cells=cells,
            cell_m=cell_m,
            chunk_size=32,
            voxel_size=0.5,
            ground_band=(8.0, 24.0),
            materials={},
            venturi_iters=8,
            venturi_max=2.2,
            deflect_gain=0.15,
            seq=seq,
        )

    def test_origin_cell_round_trip(self):
        job = self._make_job(origin_cell=(-8, 16))
        assert job.origin_cell == (-8, 16)

    def test_cells_round_trip(self):
        job = self._make_job(cells=64)
        assert job.cells == 64

    def test_cell_m_round_trip(self):
        job = self._make_job(cell_m=8.0)
        assert job.cell_m == 8.0

    def test_chunk_size_round_trip(self):
        job = self._make_job()
        assert job.chunk_size == 32

    def test_voxel_size_round_trip(self):
        job = self._make_job()
        assert job.voxel_size == 0.5

    def test_ground_band_round_trip(self):
        job = self._make_job()
        assert job.ground_band == (8.0, 24.0)

    def test_materials_empty_round_trip(self):
        job = self._make_job()
        assert job.materials == {}

    def test_materials_with_array_round_trip(self):
        """A job carrying a material array stores it as-is."""
        arr = np.ones((32, 32, 32), dtype=np.uint8)
        job = VenturiJob(
            origin_cell=(0, 0),
            cells=8,
            cell_m=4.0,
            chunk_size=32,
            voxel_size=0.5,
            ground_band=(8.0, 24.0),
            materials={(0, 0, 0): arr},
            venturi_iters=8,
            venturi_max=2.2,
            deflect_gain=0.15,
            seq=7,
        )
        assert (0, 0, 0) in job.materials
        np.testing.assert_array_equal(job.materials[(0, 0, 0)], arr)

    def test_venturi_iters_round_trip(self):
        job = self._make_job()
        assert job.venturi_iters == 8

    def test_venturi_max_round_trip(self):
        job = self._make_job()
        assert job.venturi_max == pytest.approx(2.2)

    def test_deflect_gain_round_trip(self):
        job = self._make_job()
        assert job.deflect_gain == pytest.approx(0.15)

    def test_seq_round_trip(self):
        job = self._make_job(seq=99)
        assert job.seq == 99

    def test_is_frozen(self):
        """VenturiJob is frozen — mutation raises."""
        job = self._make_job()
        with pytest.raises(dataclasses.FrozenInstanceError):
            job.seq = 0  # type: ignore[misc]

    def test_equality_same_fields(self):
        """Two identical jobs are equal."""
        a = self._make_job(seq=3)
        b = self._make_job(seq=3)
        assert a == b

    def test_inequality_on_seq(self):
        """Jobs with different seq are not equal."""
        a = self._make_job(seq=1)
        b = self._make_job(seq=2)
        assert a != b


# ---------------------------------------------------------------------------
# VenturiResult
# ---------------------------------------------------------------------------


class TestVenturiResult:
    def _make_result(
        self,
        origin_cell: tuple[int, int] = (0, 0),
        cells: int = 16,
        seq: int = 1,
    ) -> VenturiResult:
        speedup = np.ones((cells, cells), dtype=np.float32)
        deflect = np.zeros((cells, cells, 2), dtype=np.float32)
        return VenturiResult(
            origin_cell=origin_cell,
            speedup=speedup,
            deflect=deflect,
            seq=seq,
        )

    def test_origin_cell_round_trip(self):
        res = self._make_result(origin_cell=(-4, 8))
        assert res.origin_cell == (-4, 8)

    def test_speedup_shape_round_trip(self):
        res = self._make_result(cells=32)
        assert res.speedup.shape == (32, 32)
        assert res.speedup.dtype == np.float32

    def test_deflect_shape_round_trip(self):
        res = self._make_result(cells=32)
        assert res.deflect.shape == (32, 32, 2)
        assert res.deflect.dtype == np.float32

    def test_seq_round_trip(self):
        res = self._make_result(seq=77)
        assert res.seq == 77

    def test_speedup_values_preserved(self):
        """Non-trivial speedup values survive round-trip through the dataclass."""
        cells = 8
        speedup = np.arange(cells * cells, dtype=np.float32).reshape(cells, cells) + 1.0
        deflect = np.zeros((cells, cells, 2), dtype=np.float32)
        res = VenturiResult(origin_cell=(0, 0), speedup=speedup, deflect=deflect, seq=1)
        np.testing.assert_array_equal(res.speedup, speedup)

    def test_deflect_values_preserved(self):
        """Non-zero deflect values survive round-trip."""
        cells = 8
        speedup = np.ones((cells, cells), dtype=np.float32)
        deflect = np.arange(cells * cells * 2, dtype=np.float32).reshape(cells, cells, 2) * 0.01
        res = VenturiResult(origin_cell=(0, 0), speedup=speedup, deflect=deflect, seq=1)
        np.testing.assert_array_equal(res.deflect, deflect)

    def test_is_frozen(self):
        """VenturiResult is frozen — mutation raises."""
        res = self._make_result()
        with pytest.raises(dataclasses.FrozenInstanceError):
            res.seq = 0  # type: ignore[misc]

    def test_identity_result_is_identity(self):
        """A result representing 'no terrain effect' has speedup==1 and deflect==0."""
        cells = 16
        res = self._make_result(cells=cells)
        np.testing.assert_array_equal(res.speedup, np.ones((cells, cells), dtype=np.float32))
        np.testing.assert_array_equal(res.deflect, np.zeros((cells, cells, 2), dtype=np.float32))

    def test_equality_on_arrays(self):
        """Two results with identical arrays compare equal."""
        cells = 4
        speedup = np.ones((cells, cells), dtype=np.float32)
        deflect = np.zeros((cells, cells, 2), dtype=np.float32)
        a = VenturiResult(origin_cell=(0, 0), speedup=speedup, deflect=deflect, seq=5)
        b = VenturiResult(origin_cell=(0, 0), speedup=speedup, deflect=deflect, seq=5)
        assert a == b

    def test_inequality_on_seq(self):
        """Results with different seq are not equal."""
        cells = 4
        speedup = np.ones((cells, cells), dtype=np.float32)
        deflect = np.zeros((cells, cells, 2), dtype=np.float32)
        a = VenturiResult(origin_cell=(0, 0), speedup=speedup, deflect=deflect, seq=1)
        b = VenturiResult(origin_cell=(0, 0), speedup=speedup, deflect=deflect, seq=2)
        assert a != b
