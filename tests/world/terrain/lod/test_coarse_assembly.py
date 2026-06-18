"""
tests/world/terrain/lod/test_coarse_assembly.py — assemble_coarse_materials.

Headless, no panda3d.  Verifies the coarse-node gather+downsample: output shape
is always 32³, the ``k³`` covered chunks are read from the provider at the right
chunk coords, a flat grass floor survives the reduce, max-id vs majority modes
are forwarded, the result equals a direct tile + ``downsample_block`` (so the
tiling order matches), it is deterministic, and it never calls the provider more
than ``k³`` times (a chunk-level gather, NOT a per-voxel loop — Hard Rule 4).
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.world.terrain.lod.coarse_assembly import assemble_coarse_materials
from fire_engine.world.terrain.lod.downsample import downsample_block
from fire_engine.world.terrain.lod.node import LodNode

_N = 32


def _flat_floor(_coord: tuple[int, int, int]) -> np.ndarray:
    """Every L0 chunk: bottom layer grass (id 2), air above."""
    mats = np.zeros((_N, _N, _N), dtype=np.uint8)
    mats[:, :, 0] = 2
    return mats


def _ground_node_floor_provider(node: LodNode):
    """Provider where only the node's BOTTOM chunk-layer (dz=0) has a floor.

    Mirrors real terrain: the surface lives in the lowest chunk layer of a
    column, the chunk layers above are air.  After the downsample the floor then
    occupies exactly the single bottom coarse z-layer.
    """
    oz = node.chunk_origin()[2]

    def provider(coord: tuple[int, int, int]) -> np.ndarray:
        mats = np.zeros((_N, _N, _N), dtype=np.uint8)
        if coord[2] == oz:  # bottom chunk layer of the node only
            mats[:, :, 0] = 2
        return mats

    return provider


class TestShape:
    @pytest.mark.parametrize("rank", [1, 2, 3])
    def test_output_is_32_cube(self, rank: int) -> None:
        node = LodNode(rank, 0, 0, 0)
        out = assemble_coarse_materials(node, _flat_floor)
        assert out.shape == (_N, _N, _N)
        assert out.dtype == np.uint8


class TestProviderCoords:
    def test_reads_exactly_the_covered_chunks(self) -> None:
        # The provider must be queried for exactly node.covered_chunks(), and
        # never more than k**3 times (chunk-level gather, not a voxel loop).
        node = LodNode(2, 1, -1, 0)  # k=4
        requested: list[tuple[int, int, int]] = []

        def provider(coord: tuple[int, int, int]) -> np.ndarray:
            requested.append(coord)
            return np.zeros((_N, _N, _N), dtype=np.uint8)

        assemble_coarse_materials(node, provider)
        assert len(requested) == (1 << node.rank) ** 3  # 64
        assert set(requested) == set(node.covered_chunks())


class TestReducePreservesFloor:
    @pytest.mark.parametrize("rank", [1, 2, 3])
    def test_ground_floor_survives_in_bottom_layer(self, rank: int) -> None:
        # Only the node's bottom chunk-layer has a floor (real-terrain shape):
        # after the downsample exactly the single bottom coarse z-layer is solid
        # grass everywhere, and everything above is air.
        node = LodNode(rank, 0, 0, 0)
        out = assemble_coarse_materials(node, _ground_node_floor_provider(node))
        assert (out[:, :, 0] == 2).all()
        assert (out[:, :, 1:] == 0).all()

    @pytest.mark.parametrize("rank", [1, 2, 3])
    def test_floor_in_every_chunk_yields_one_coarse_layer_per_chunk(self, rank: int) -> None:
        # A floor in EVERY covered chunk (incl. stacked ones) survives as a
        # solid coarse layer at the bottom of each chunk's tile slot, i.e. at
        # coarse z = 0, 32/k, 64/k, ...  (thin floors never erode — ANY solidity).
        node = LodNode(rank, 0, 0, 0)
        out = assemble_coarse_materials(node, _flat_floor)
        k = node.factor
        step = _N // k
        for layer in range(k):
            assert (out[:, :, layer * step] == 2).all(), layer


class TestMatchesDirectTile:
    """assemble == manual tile + downsample (pins the tiling axis order)."""

    def test_equals_manual_tile_downsample(self) -> None:
        node = LodNode(1, 0, 0, 0)  # k=2

        # Distinct content per covered chunk so a wrong tile slot would show up.
        def provider(coord: tuple[int, int, int]) -> np.ndarray:
            cx, _cy, _cz = coord
            mats = np.zeros((_N, _N, _N), dtype=np.uint8)
            mats[:, :, 0] = np.uint8(1 + (cx & 1))  # 1 or 2 by x-parity
            return mats

        out = assemble_coarse_materials(node, provider, mode="any")

        # Manual tile in the same C-order the assembler uses.
        k = node.factor
        ox, oy, oz = node.chunk_origin()
        tile = np.zeros((_N * k, _N * k, _N * k), dtype=np.uint8)
        for dx in range(k):
            for dy in range(k):
                for dz in range(k):
                    m = provider((ox + dx, oy + dy, oz + dz))
                    tile[
                        dx * _N : (dx + 1) * _N,
                        dy * _N : (dy + 1) * _N,
                        dz * _N : (dz + 1) * _N,
                    ] = m
        expected = downsample_block(tile, node.rank, "any")
        assert np.array_equal(out, expected)


class TestModeForwarded:
    def test_any_vs_majority_differ(self) -> None:
        # A coarse cell whose 8 merged voxels are 7 dirt(1) + 1 grass(2):
        # max-id -> 2, majority -> 1.  Provide it via one chunk at rank 1.
        node = LodNode(1, 0, 0, 0)  # k=2, cell (0,0,0) spans chunk (0,0,0)[0:2]^3

        def provider(coord: tuple[int, int, int]) -> np.ndarray:
            mats = np.zeros((_N, _N, _N), dtype=np.uint8)
            if coord == (0, 0, 0):
                mats[0:2, 0:2, 0:2] = 1  # dirt
                mats[1, 1, 1] = 2  # one grass voxel
            return mats

        any_out = assemble_coarse_materials(node, provider, mode="any")
        maj_out = assemble_coarse_materials(node, provider, mode="majority")
        assert any_out[0, 0, 0] == 2  # max-id
        assert maj_out[0, 0, 0] == 1  # majority


class TestDeterminism:
    def test_byte_identical_twice(self) -> None:
        def provider(coord: tuple[int, int, int]) -> np.ndarray:
            # Deterministic per-coord content (local generator seeded off the
            # coord — pure-layer test fixture only, not engine randomness).
            cx, cy, cz = coord
            seed = (cx * 73856093) ^ (cy * 19349663) ^ (cz * 83492791)
            local = np.random.default_rng(seed % (2**32))
            return local.integers(0, 4, size=(_N, _N, _N), dtype=np.uint8)

        node = LodNode(2, 3, -1, 0)
        a = assemble_coarse_materials(node, provider)
        b = assemble_coarse_materials(node, provider)
        assert np.array_equal(a, b)
