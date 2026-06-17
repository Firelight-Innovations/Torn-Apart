"""
tests/world/terrain/lod/test_downsample.py — downsample_block reduce policy.

Headless, no panda3d.  Verifies the ANY solidity (thin 1-voxel wall survives 2×
and 4×), max-id material (grass 2 beats dirt 1), air/solid extremes, exact
output shapes for k=2/4/8, determinism, the majority mode, and that the reduce
is a whole-array op (runs fast on a 256³ input — no Python voxel loop).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from fire_engine.world.terrain.lod.downsample import downsample_block


def _tile(rank: int) -> np.ndarray:
    """All-air tiled block of shape (32k, 32k, 32k)."""
    k = 1 << rank
    return np.zeros((32 * k, 32 * k, 32 * k), dtype=np.uint8)


class TestShapes:
    @pytest.mark.parametrize("rank", [1, 2, 3])
    def test_output_is_32_cube(self, rank: int) -> None:
        out = downsample_block(_tile(rank), rank)
        assert out.shape == (32, 32, 32)
        assert out.dtype == np.uint8

    def test_non_cube_z_span(self) -> None:
        # A thin-slab Z span: (32k, 32k, 7k) -> (32, 32, 7).
        k = 2
        tile = np.zeros((32 * k, 32 * k, 7 * k), dtype=np.uint8)
        out = downsample_block(tile, rank=1)
        assert out.shape == (32, 32, 7)

    def test_bad_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            downsample_block(np.zeros((63, 64, 64), dtype=np.uint8), rank=1)


class TestSolidityAny:
    @pytest.mark.parametrize("rank", [1, 2])
    def test_thin_wall_survives(self, rank: int) -> None:
        # A single solid x-plane (1-voxel-thick wall). ANY must keep it solid.
        tile = _tile(rank)
        tile[10, :, :] = 1
        out = downsample_block(tile, rank)
        k = 1 << rank
        coarse_x = 10 // k
        # The whole coarse x-plane is solid; everything else is air.
        assert out[coarse_x, :, :].all()
        mask = np.ones(32, dtype=bool)
        mask[coarse_x] = False
        assert (out[mask, :, :] == 0).all()

    def test_single_voxel_survives(self) -> None:
        tile = _tile(2)  # k=4
        tile[5, 6, 7] = 1
        out = downsample_block(tile, rank=2)
        assert out[1, 1, 1] == 1  # 5//4, 6//4, 7//4
        assert int((out > 0).sum()) == 1


class TestMaterialMaxId:
    def test_grass_beats_dirt(self) -> None:
        # A coarse cell with one grass (2) voxel among dirt (1) voxels -> 2.
        tile = _tile(1)  # k=2: cell (0,0,0) spans [0:2,0:2,0:2]
        tile[0:2, 0:2, 0:2] = 1  # fill with dirt
        tile[1, 1, 1] = 2  # one grass voxel
        out = downsample_block(tile, rank=1, mode="any")
        assert out[0, 0, 0] == 2

    def test_air_block_all_zero(self) -> None:
        out = downsample_block(_tile(2), rank=2)
        assert int(out.sum()) == 0

    def test_fully_solid_block_all_nonzero(self) -> None:
        tile = _tile(1)
        tile[...] = 1
        out = downsample_block(tile, rank=1)
        assert (out > 0).all()
        assert (out == 1).all()


class TestMajorityMode:
    def test_majority_picks_dominant_id(self) -> None:
        # Cell (0,0,0) at k=2 has 8 voxels: 7 dirt(1) + 1 grass(2).
        tile = _tile(1)
        tile[0:2, 0:2, 0:2] = 1
        tile[1, 1, 1] = 2
        any_mode = downsample_block(tile, rank=1, mode="any")
        maj_mode = downsample_block(tile, rank=1, mode="majority")
        assert any_mode[0, 0, 0] == 2  # max-id -> grass
        assert maj_mode[0, 0, 0] == 1  # majority -> dirt (differs from any)

    def test_majority_solidity_still_any(self) -> None:
        # One solid voxel in an otherwise-air cell: majority must still mark it
        # solid (solidity is ANY regardless of mode).
        tile = _tile(1)
        tile[1, 1, 1] = 3
        out = downsample_block(tile, rank=1, mode="majority")
        assert out[0, 0, 0] == 3

    def test_majority_air_block_zero(self) -> None:
        out = downsample_block(_tile(1), rank=1, mode="majority")
        assert int(out.sum()) == 0


class TestDeterminism:
    def test_byte_identical_twice(self) -> None:
        rng = np.random.default_rng(0)  # local generator; pure-layer test fixture only
        tile = rng.integers(0, 4, size=(64, 64, 64), dtype=np.uint8)
        a = downsample_block(tile, rank=1)
        b = downsample_block(tile.copy(), rank=1)
        assert np.array_equal(a, b)


class TestVectorizedPerformance:
    def test_256_cube_is_fast(self) -> None:
        # A 256³ input (k=8, rank 3) must reduce well under a generous budget —
        # a Python per-voxel loop over 16.7M voxels would blow this away.
        tile = np.zeros((256, 256, 256), dtype=np.uint8)
        tile[::8, ::8, ::8] = 1
        t0 = time.perf_counter()
        out = downsample_block(tile, rank=3)
        elapsed = time.perf_counter() - t0
        assert out.shape == (32, 32, 32)
        assert elapsed < 2.0, f"downsample too slow ({elapsed:.3f}s) — likely a voxel loop"
