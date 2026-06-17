"""
tests/world/terrain/lod/test_coarse_chunk.py — _CoarseChunk mesher shim.

Headless, no panda3d.  Verifies the shim exposes the exact four members the
meshers read, runs ``build_mesh_faceted`` / ``build_mesh`` UNCHANGED and
byte-identically to a real ``Chunk`` at the scaled voxel size, that a flat node
yields the expected top-face count, a fully-solid node (solid neighbours) is
empty, and the world origin lands at the correct world meters for L1/L2/L3.
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.world.terrain.chunk import Chunk
from fire_engine.world.terrain.lod.coarse_chunk import _CoarseChunk
from fire_engine.world.terrain.lod.node import LodNode
from fire_engine.world.terrain.meshing import build_mesh
from fire_engine.world.terrain.surface_nets import build_mesh_faceted

_BASE_VS = 0.5
_BASE_CHUNK_M = 16.0


def _flat_materials() -> np.ndarray:
    """A flat node: bottom 4 layers solid grass (id 2), air above."""
    mats = np.zeros((32, 32, 32), dtype=np.uint8)
    mats[:, :, 0:4] = 2
    return mats


class TestShimAttributes:
    def test_exposes_four_members(self) -> None:
        node = LodNode(1, 2, -2, 0)
        mats = _flat_materials()
        cc = _CoarseChunk(node, mats, base_voxel_size=_BASE_VS)
        assert cc.materials.shape == (32, 32, 32)
        assert cc.materials.dtype == np.uint8
        assert cc._voxel_size == _BASE_VS * 2  # 0.5 * 2**1
        assert np.array_equal(cc.is_solid_mask(), mats > 0)
        wo = cc.world_origin.to_numpy()
        assert wo.tolist() == [64.0, -64.0, 0.0]  # (2,-2,0)*(16*2)

    @pytest.mark.parametrize(
        ("rank", "expected_vs"),
        [(1, 1.0), (2, 2.0), (3, 4.0)],
    )
    def test_world_origin_scales_per_rank(self, rank: int, expected_vs: float) -> None:
        node = LodNode(rank, 3, -1, 1)
        cc = _CoarseChunk(node, _flat_materials(), base_voxel_size=_BASE_VS)
        assert cc._voxel_size == expected_vs
        k = 1 << rank
        wo = cc.world_origin.to_numpy()
        assert wo.tolist() == [
            (3 * k) * _BASE_CHUNK_M,
            (-1 * k) * _BASE_CHUNK_M,
            (1 * k) * _BASE_CHUNK_M,
        ]


class TestMeshesLikeRealChunk:
    """A real Chunk built at the node coord + scaled voxel size meshes the same."""

    def _ref_chunk(self, node: LodNode, mats: np.ndarray) -> Chunk:
        return Chunk(
            (node.nx, node.ny, node.nz),
            mats.copy(),
            chunk_size=32,
            voxel_size=node.voxel_size(_BASE_VS),
        )

    @pytest.mark.parametrize("rank", [1, 2, 3])
    def test_faceted_byte_identical(self, rank: int) -> None:
        node = LodNode(rank, 1, 2, 0)
        mats = _flat_materials()
        cc = _CoarseChunk(node, mats, base_voxel_size=_BASE_VS)
        ref = self._ref_chunk(node, mats)
        shim_mesh = build_mesh_faceted(cc)
        ref_mesh = build_mesh_faceted(ref)
        assert np.array_equal(shim_mesh.positions, ref_mesh.positions)
        assert np.array_equal(shim_mesh.normals, ref_mesh.normals)
        assert np.array_equal(shim_mesh.indices, ref_mesh.indices)
        assert np.array_equal(shim_mesh.face_materials, ref_mesh.face_materials)

    def test_blocky_runs_on_shim(self) -> None:
        node = LodNode(2, 0, 0, 0)
        mats = _flat_materials()
        cc = _CoarseChunk(node, mats, base_voxel_size=_BASE_VS)
        ref = self._ref_chunk(node, mats)
        shim_mesh = build_mesh(cc)
        ref_mesh = build_mesh(ref)
        assert np.array_equal(shim_mesh.positions, ref_mesh.positions)
        assert shim_mesh.face_materials is None  # blocky single-texture


def _top_face_count(mesh, thresh: float = 0.5) -> int:
    """Number of faces whose flat normal points +Z (the top skin)."""
    nrm = mesh.normals.reshape(-1, mesh.verts_per_face, 3)[:, 0, :]
    return int((nrm[:, 2] > thresh).sum())


class TestFlatTopFaceCount:
    def test_flat_node_top_faces_equal_single_flat_chunk(self) -> None:
        # A flat coarse node has the SAME top-face count as a single flat L0
        # chunk with identical materials (the top grass layer survives the
        # downsample; voxel scale changes positions, not face topology).
        mats = _flat_materials()
        node = LodNode(1, 0, 0, 0)
        coarse_mesh = build_mesh_faceted(_CoarseChunk(node, mats, base_voxel_size=_BASE_VS))
        l0_chunk = Chunk((0, 0, 0), mats.copy(), chunk_size=32, voxel_size=_BASE_VS)
        l0_mesh = build_mesh_faceted(l0_chunk)
        # Byte-identical topology: same total faces and same top-skin count.
        assert coarse_mesh.face_count == l0_mesh.face_count
        assert _top_face_count(coarse_mesh) == _top_face_count(l0_mesh)
        # Full 32x32 top surface (every column has an upward-facing top face).
        assert _top_face_count(coarse_mesh) == 32 * 32


class TestFullySolidEmpty:
    def test_fully_solid_with_solid_neighbours_is_empty(self) -> None:
        # A fully-solid node whose 26 neighbours are also solid has no exposed
        # interior faces -> empty mesh.
        from fire_engine.world.terrain.surface_nets import NEIGHBOR_OFFSETS_26

        node = LodNode(1, 0, 0, 0)
        mats = np.ones((32, 32, 32), dtype=np.uint8)
        cc = _CoarseChunk(node, mats, base_voxel_size=_BASE_VS)
        solid_nb = np.ones((32, 32, 32), dtype=np.uint8)
        neighbors = {off: solid_nb for off in NEIGHBOR_OFFSETS_26}
        mesh = build_mesh_faceted(cc, neighbors)
        assert mesh.is_empty
