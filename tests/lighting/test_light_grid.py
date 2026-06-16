"""
tests/lighting/test_light_grid.py — Headless tests for fire_engine.lighting.light_grid.

Covers:
- LIGHT_FULL / LIGHT_AMBIENT constants.
- occupancy_from_materials: known downsampling cases.
- LightGrid store: set/get/has_valid/invalidate/remove/loaded_coords.

No panda3d imports.
"""

from __future__ import annotations

import numpy as np

from fire_engine.lighting.light_grid import (
    LIGHT_AMBIENT,
    LIGHT_FULL,
    LightGrid,
    occupancy_from_materials,
)


class TestConstants:
    def test_light_full_is_255(self):
        assert LIGHT_FULL == 255

    def test_light_ambient_is_40(self):
        assert LIGHT_AMBIENT == 40

    def test_ambient_less_than_full(self):
        assert LIGHT_AMBIENT < LIGHT_FULL


class TestOccupancyFromMaterials:
    def test_all_air_returns_all_false(self):
        mat = np.zeros((32, 32, 32), dtype=np.uint8)
        occ = occupancy_from_materials(mat)
        assert occ.shape == (16, 16, 16)
        assert occ.dtype == bool
        assert not occ.any()

    def test_all_solid_returns_all_true(self):
        mat = np.ones((32, 32, 32), dtype=np.uint8)
        occ = occupancy_from_materials(mat)
        assert occ.all()

    def test_single_voxel_occupies_correct_cell(self):
        mat = np.zeros((32, 32, 32), dtype=np.uint8)
        mat[0, 0, 0] = 1
        occ = occupancy_from_materials(mat)
        assert occ[0, 0, 0]
        other = occ.copy()
        other[0, 0, 0] = False
        assert not other.any()

    def test_voxel_maps_to_correct_cell(self):
        mat = np.zeros((32, 32, 32), dtype=np.uint8)
        # Voxel (2, 4, 6) lives in light cell (1, 2, 3).
        mat[2, 4, 6] = 1
        occ = occupancy_from_materials(mat)
        assert occ[1, 2, 3]
        assert not occ[0, 2, 3]

    def test_two_voxels_in_same_cell(self):
        mat = np.zeros((32, 32, 32), dtype=np.uint8)
        mat[0, 0, 0] = 1
        mat[1, 1, 1] = 1
        occ = occupancy_from_materials(mat)
        assert occ[0, 0, 0]
        other = occ.copy()
        other[0, 0, 0] = False
        assert not other.any()

    def test_voxel_at_boundary_correct_cell(self):
        mat = np.zeros((32, 32, 32), dtype=np.uint8)
        mat[30, 30, 30] = 1  # should land in cell (15, 15, 15)
        occ = occupancy_from_materials(mat)
        assert occ[15, 15, 15]
        other = occ.copy()
        other[15, 15, 15] = False
        assert not other.any()


class TestLightGrid:
    def test_get_returns_none_before_set(self):
        lg = LightGrid()
        assert lg.get((0, 0, 0)) is None

    def test_has_valid_false_before_set(self):
        lg = LightGrid()
        assert not lg.has_valid((0, 0, 0))

    def test_set_and_get(self):
        lg = LightGrid()
        arr = np.full((16, 16, 16), 200, dtype=np.uint8)
        lg.set((1, 2, 3), arr)
        got = lg.get((1, 2, 3))
        assert got is arr  # same object stored
        assert lg.has_valid((1, 2, 3))

    def test_invalidate(self):
        lg = LightGrid()
        arr = np.full((16, 16, 16), 100, dtype=np.uint8)
        lg.set((0, 0, 0), arr)
        lg.invalidate((0, 0, 0))
        assert not lg.has_valid((0, 0, 0))
        assert lg.get((0, 0, 0)) is arr  # array is still accessible

    def test_remove(self):
        lg = LightGrid()
        arr = np.full((16, 16, 16), 50, dtype=np.uint8)
        lg.set((0, 0, 0), arr)
        lg.remove((0, 0, 0))
        assert lg.get((0, 0, 0)) is None
        assert not lg.has_valid((0, 0, 0))

    def test_loaded_coords(self):
        lg = LightGrid()
        lg.set((0, 0, 0), np.zeros((16, 16, 16), dtype=np.uint8))
        lg.set((1, 0, 0), np.zeros((16, 16, 16), dtype=np.uint8))
        coords = lg.loaded_coords()
        assert set(coords) == {(0, 0, 0), (1, 0, 0)}

    def test_invalidate_nonexistent_no_error(self):
        lg = LightGrid()
        lg.invalidate((99, 99, 99))  # must not raise

    def test_remove_nonexistent_no_error(self):
        lg = LightGrid()
        lg.remove((99, 99, 99))  # must not raise

    def test_multiple_chunks_independent(self):
        lg = LightGrid()
        a = np.full((16, 16, 16), 255, dtype=np.uint8)
        b = np.full((16, 16, 16), 40, dtype=np.uint8)
        lg.set((0, 0, 0), a)
        lg.set((1, 0, 0), b)
        assert lg.get((0, 0, 0)) is a
        assert lg.get((1, 0, 0)) is b
        lg.remove((0, 0, 0))
        assert lg.get((0, 0, 0)) is None
        assert lg.get((1, 0, 0)) is b
