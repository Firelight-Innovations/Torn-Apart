"""
tests/world/terrain/test_chunk.py — Chunk construction, properties, and solidity mask.
Headless: no panda3d imports.
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.world.terrain.chunk import Chunk


class TestChunkConstruction:
    def test_default_all_air(self):
        c = Chunk((0, 0, 0))
        assert c.materials.shape == (32, 32, 32)
        assert c.materials.dtype == np.uint8
        assert np.all(c.materials == 0)

    def test_coord_stored_as_ints(self):
        c = Chunk((1, -2, 3))
        assert c.coord == (1, -2, 3)
        assert all(isinstance(v, int) for v in c.coord)

    def test_dirty_true_on_init(self):
        c = Chunk((0, 0, 0))
        assert c.dirty is True

    def test_edited_false_on_init(self):
        c = Chunk((0, 0, 0))
        assert c.edited is False

    def test_custom_materials_taken_by_ref(self):
        arr = np.ones((32, 32, 32), dtype=np.uint8)
        c = Chunk((0, 0, 0), materials=arr)
        assert np.array_equal(c.materials, arr)

    def test_wrong_shape_raises(self):
        bad = np.zeros((16, 16, 16), dtype=np.uint8)
        with pytest.raises(ValueError):
            Chunk((0, 0, 0), materials=bad)

    def test_custom_chunk_size(self):
        c = Chunk((0, 0, 0), chunk_size=16)
        assert c.materials.shape == (16, 16, 16)

    def test_chunk_meters_default(self):
        c = Chunk((0, 0, 0))
        # 32 voxels * 0.5 m/voxel = 16 m
        assert c.chunk_meters == pytest.approx(16.0)

    def test_chunk_meters_custom_voxel_size(self):
        c = Chunk((0, 0, 0), chunk_size=32, voxel_size=1.0)
        assert c.chunk_meters == pytest.approx(32.0)


class TestChunkWorldOrigin:
    def test_origin_at_zero(self):
        c = Chunk((0, 0, 0))
        o = c.world_origin
        assert o.x == pytest.approx(0.0)
        assert o.y == pytest.approx(0.0)
        assert o.z == pytest.approx(0.0)

    def test_origin_positive_coord(self):
        c = Chunk((1, 2, 3))
        o = c.world_origin
        assert o.x == pytest.approx(16.0)
        assert o.y == pytest.approx(32.0)
        assert o.z == pytest.approx(48.0)

    def test_origin_negative_coord(self):
        c = Chunk((-1, 0, -2))
        o = c.world_origin
        assert o.x == pytest.approx(-16.0)
        assert o.y == pytest.approx(0.0)
        assert o.z == pytest.approx(-32.0)


class TestChunkSolidityMask:
    def test_all_air_mask_false(self):
        c = Chunk((0, 0, 0))
        mask = c.is_solid_mask()
        assert mask.dtype == bool
        assert mask.shape == (32, 32, 32)
        assert not mask.any()

    def test_set_voxel_solid(self):
        c = Chunk((0, 0, 0))
        c.materials[5, 5, 5] = 1
        mask = c.is_solid_mask()
        assert mask[5, 5, 5] is np.True_
        # All other voxels still air.
        mask[5, 5, 5] = False
        assert not mask.any()

    def test_material_id_gt_1_still_solid(self):
        c = Chunk((0, 0, 0))
        c.materials[0, 0, 0] = 2  # MATERIAL_GRASS
        assert c.is_solid_mask()[0, 0, 0] is np.True_

    def test_fully_solid_chunk(self):
        c = Chunk((0, 0, 0))
        c.materials[:] = 1
        assert c.is_solid_mask().all()

    def test_solid_mask_counts_match_materials(self):
        c = Chunk((0, 0, 0))
        c.materials[0:5, 0:5, 0:5] = 1
        mask = c.is_solid_mask()
        assert mask.sum() == 5 * 5 * 5
