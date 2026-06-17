"""
tests/world/terrain/lod/test_job.py — LodJob/LodResult + build_lod_mesh.

Headless: no panda3d imports.  The key test asserts build_lod_mesh reproduces
ChunkManager.mesh_chunk byte-for-byte for both mesh styles, proving the threaded
path is equivalent to the synchronous one (Hard Rule 12 determinism).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from fire_engine.core import EventBus, load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.world.terrain.chunk_manager import ChunkManager
from fire_engine.world.terrain.lod.job import LodJob, LodResult, build_lod_mesh


def _copy_neighbors(
    raw: dict[tuple[int, int, int], np.ndarray | str],
) -> dict[tuple[int, int, int], np.ndarray | str]:
    """Snapshot a neighbour dict the way the caller would (copy arrays)."""
    out: dict[tuple[int, int, int], np.ndarray | str] = {}
    for k, v in raw.items():
        out[k] = v if isinstance(v, str) else np.asarray(v).copy()
    return out


def _make_manager(config):
    set_world_seed(1337)
    return ChunkManager(config, EventBus())


def _assert_mesh_equal(a, b) -> None:
    assert np.array_equal(a.positions, b.positions)
    assert np.array_equal(a.normals, b.normals)
    assert np.array_equal(a.uvs, b.uvs)
    assert np.array_equal(a.colors, b.colors)
    assert np.array_equal(a.indices, b.indices)
    # face_materials may be None (blocky) or an array (faceted).
    if a.face_materials is None or b.face_materials is None:
        assert a.face_materials is None and b.face_materials is None
    else:
        assert np.array_equal(a.face_materials, b.face_materials)
    assert a.verts_per_face == b.verts_per_face


class TestEquivalenceFaceted:
    def test_matches_mesh_chunk_faceted(self):
        config = load_config()  # mesh_style defaults to "faceted"
        cm = _make_manager(config)
        coord = (0, 0, 0)
        cm.get_or_create(coord)
        # Load a couple of neighbours so the dict isn't all-generated baseline.
        cm.get_or_create((1, 0, 0))
        cm.get_or_create((0, 0, -1))

        chunk = cm.chunks[coord]
        job = LodJob(
            coord=coord,
            materials=chunk.materials.copy(),
            neighbors=_copy_neighbors(cm._neighbor_materials(coord)),
            chunk_size=int(config.chunk_size),
            voxel_size=float(config.voxel_size),
            shade_strength=float(config.facet_shade_strength),
            mesh_style="faceted",
            seq=7,
        )
        result = build_lod_mesh(job)
        expected = cm.mesh_chunk(coord)
        _assert_mesh_equal(result.mesh, expected)
        assert result.coord == coord
        assert result.seq == 7

    def test_matches_after_brush_poke(self):
        config = load_config()
        cm = _make_manager(config)
        coord = (0, 0, 1)
        chunk = cm.get_or_create(coord)
        # Poke a voxel directly (simulate an edit) and mark it edited/dirty.
        chunk.materials[10, 12, 4] = 0
        chunk.materials[3, 3, 3] = 1
        chunk.edited = True

        job = LodJob(
            coord=coord,
            materials=chunk.materials.copy(),
            neighbors=_copy_neighbors(cm._neighbor_materials(coord)),
            chunk_size=int(config.chunk_size),
            voxel_size=float(config.voxel_size),
            shade_strength=float(config.facet_shade_strength),
            mesh_style="faceted",
            seq=1,
        )
        result = build_lod_mesh(job)
        expected = cm.mesh_chunk(coord)
        _assert_mesh_equal(result.mesh, expected)


class TestEquivalenceBlocky:
    def test_matches_mesh_chunk_blocky(self):
        config = dataclasses.replace(load_config(), mesh_style="blocky")
        cm = _make_manager(config)
        coord = (0, 0, 0)
        cm.get_or_create(coord)
        cm.get_or_create((1, 0, 0))
        cm.get_or_create((0, -1, 0))

        chunk = cm.chunks[coord]
        job = LodJob(
            coord=coord,
            materials=chunk.materials.copy(),
            neighbors=_copy_neighbors(cm._neighbor_solids(coord)),
            chunk_size=int(config.chunk_size),
            voxel_size=float(config.voxel_size),
            shade_strength=float(config.facet_shade_strength),
            mesh_style="blocky",
            seq=42,
        )
        result = build_lod_mesh(job)
        expected = cm.mesh_chunk(coord)
        _assert_mesh_equal(result.mesh, expected)
        assert result.mesh.face_materials is None  # blocky → single-texture
        assert result.seq == 42


class TestCoarseRank:
    """build_lod_mesh(rank > 0) meshes a _CoarseChunk; rank 0 stays native L0."""

    def test_rank1_meshes_coarse_node(self):
        import numpy as np

        from fire_engine.world.terrain.lod.coarse_chunk import _CoarseChunk
        from fire_engine.world.terrain.lod.node import LodNode
        from fire_engine.world.terrain.surface_nets import build_mesh_faceted

        config = load_config()
        base_vs = float(config.voxel_size)
        rank = 1
        node = LodNode(rank, 2, -1, 0)

        # A downsampled coarse block (already 32³): flat grass floor.
        coarse_mats = np.zeros((32, 32, 32), dtype=np.uint8)
        coarse_mats[:, :, 0:3] = 2

        job = LodJob(
            coord=(node.nx, node.ny, node.nz),
            materials=coarse_mats.copy(),
            neighbors={},
            chunk_size=int(config.chunk_size),
            voxel_size=base_vs * (1 << rank),  # scaled coarse voxel edge
            shade_strength=float(config.facet_shade_strength),
            mesh_style="faceted",
            seq=4,
            rank=rank,
        )
        result = build_lod_mesh(job)
        assert result.rank == rank
        assert result.coord == (node.nx, node.ny, node.nz)
        assert not result.mesh.is_empty

        # Equals meshing a _CoarseChunk at the node coord + scaled voxel size.
        shim = _CoarseChunk(node, coarse_mats.copy(), base_voxel_size=base_vs)
        ref = build_mesh_faceted(shim, {}, None, shade_strength=float(config.facet_shade_strength))
        _assert_mesh_equal(result.mesh, ref)

        # The coarse voxel edge is reflected in world positions (2× a native L0).
        l0_job = dataclasses.replace(job, voxel_size=base_vs, rank=0, coord=(0, 0, 0))
        l0 = build_lod_mesh(l0_job).mesh
        # Same topology, but the coarse mesh's positions span 2× the metres.
        assert l0.face_count == result.mesh.face_count
        coarse_extent = result.mesh.positions.max(axis=0) - result.mesh.positions.min(axis=0)
        l0_extent = l0.positions.max(axis=0) - l0.positions.min(axis=0)
        assert np.allclose(coarse_extent[:2], l0_extent[:2] * 2)

    def test_rank0_still_native(self):
        config = load_config()
        cm = _make_manager(config)
        coord = (0, 0, 0)
        cm.get_or_create(coord)
        job = LodJob(
            coord=coord,
            materials=cm.chunks[coord].materials.copy(),
            neighbors=_copy_neighbors(cm._neighbor_materials(coord)),
            chunk_size=int(config.chunk_size),
            voxel_size=float(config.voxel_size),
            shade_strength=float(config.facet_shade_strength),
            mesh_style="faceted",
            seq=1,
            rank=0,
        )
        result = build_lod_mesh(job)
        assert result.rank == 0
        _assert_mesh_equal(result.mesh, cm.mesh_chunk(coord))


class TestDeterminism:
    def test_twice_same_job_byte_identical(self):
        config = load_config()
        cm = _make_manager(config)
        coord = (2, -1, 0)
        chunk = cm.get_or_create(coord)
        job = LodJob(
            coord=coord,
            materials=chunk.materials.copy(),
            neighbors=_copy_neighbors(cm._neighbor_materials(coord)),
            chunk_size=int(config.chunk_size),
            voxel_size=float(config.voxel_size),
            shade_strength=float(config.facet_shade_strength),
            mesh_style="faceted",
            seq=0,
        )
        a = build_lod_mesh(job)
        b = build_lod_mesh(job)
        _assert_mesh_equal(a.mesh, b.mesh)
        assert isinstance(a, LodResult)
