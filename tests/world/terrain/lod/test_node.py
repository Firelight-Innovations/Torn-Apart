"""
tests/world/terrain/lod/test_node.py — LodNode addressing + snap rule.

Headless, no panda3d.  Covers the ``chunk >> L`` snap (including negatives), the
covered-chunk-block enumeration (``k³`` unique chunks), and the world-origin /
voxel-size scaling for ranks 0..3.
"""

from __future__ import annotations

import pytest

from fire_engine.world.terrain.lod.node import LodNode, rank_factor


class TestRankFactor:
    @pytest.mark.parametrize(
        ("rank", "expected"),
        [(0, 1), (1, 2), (2, 4), (3, 8), (4, 16)],
    )
    def test_factor(self, rank: int, expected: int) -> None:
        assert rank_factor(rank) == expected
        assert LodNode(rank, 0, 0, 0).factor == expected


class TestSnapRule:
    def test_positive_snap(self) -> None:
        n = LodNode.for_chunk((5, 6, 7), rank=1)
        assert (n.rank, n.nx, n.ny, n.nz) == (1, 2, 3, 3)  # 5>>1, 6>>1, 7>>1

    def test_negative_snap_floors(self) -> None:
        # >> floors toward -inf: -3 >> 1 == -2, -1 >> 1 == -1.
        n = LodNode.for_chunk((-3, -1, -8), rank=1)
        assert (n.nx, n.ny, n.nz) == (-2, -1, -4)

    def test_rank3_snap(self) -> None:
        n = LodNode.for_chunk((17, 0, 31), rank=3)  # 17>>3=2, 31>>3=3
        assert (n.nx, n.ny, n.nz) == (2, 0, 3)

    def test_key_tuple(self) -> None:
        n = LodNode(2, 1, -1, 0)
        assert n.key == (2, 1, -1, 0)

    def test_all_chunks_in_node_snap_back_to_it(self) -> None:
        node = LodNode(2, 1, -2, 0)  # k=4
        for c in node.covered_chunks():
            assert LodNode.for_chunk(c, rank=2) == node


class TestCoveredChunks:
    @pytest.mark.parametrize("rank", [0, 1, 2, 3])
    def test_block_size_is_k_cubed_unique(self, rank: int) -> None:
        node = LodNode(rank, 2, -1, 1)
        chunks = node.covered_chunks()
        k = 1 << rank
        assert len(chunks) == k**3
        assert len(set(chunks)) == k**3  # all unique

    def test_block_origin_and_extent(self) -> None:
        node = LodNode(1, 2, -2, 0)  # k=2
        assert node.chunk_origin() == (4, -4, 0)
        chunks = set(node.covered_chunks())
        assert chunks == {
            (4, -4, 0),
            (4, -4, 1),
            (4, -3, 0),
            (4, -3, 1),
            (5, -4, 0),
            (5, -4, 1),
            (5, -3, 0),
            (5, -3, 1),
        }


class TestWorldOriginAndVoxel:
    @pytest.mark.parametrize(
        ("rank", "expected_vs"),
        [(0, 0.5), (1, 1.0), (2, 2.0), (3, 4.0)],
    )
    def test_voxel_size_scales(self, rank: int, expected_vs: float) -> None:
        assert LodNode(rank, 0, 0, 0).voxel_size(base_voxel_size=0.5) == expected_vs

    def test_world_origin_origin_node(self) -> None:
        n = LodNode(3, 0, 0, 0)
        assert n.world_origin(base_chunk_meters=16.0).to_numpy().tolist() == [0.0, 0.0, 0.0]

    @pytest.mark.parametrize("rank", [1, 2, 3])
    def test_world_origin_lands_on_chunk_block_min_corner(self, rank: int) -> None:
        # Node origin must equal the world min-corner of its covered L0 block:
        # (nx*k) * base_chunk_meters along each axis.
        node = LodNode(rank, 3, -2, 1)
        k = 1 << rank
        base_chunk_m = 16.0
        wo = node.world_origin(base_chunk_meters=base_chunk_m).to_numpy()
        expected = [
            (3 * k) * base_chunk_m,
            (-2 * k) * base_chunk_m,
            (1 * k) * base_chunk_m,
        ]
        assert wo.tolist() == expected
