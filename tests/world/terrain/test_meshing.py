"""
tests/world/terrain/test_meshing.py — Culled-face mesher fixture tests.
Headless: no panda3d imports.
"""

from __future__ import annotations

import numpy as np

from fire_engine.world.terrain.chunk import Chunk
from fire_engine.world.terrain.meshing import WORLD_FLOOR_SOLID, MeshArrays, build_mesh


class TestMesherFixtures:
    def test_single_voxel_six_faces(self):
        c = Chunk((0, 0, 0))
        c.materials[5, 5, 5] = 1
        mesh = build_mesh(c, neighbor_solids=None)
        assert mesh.face_count == 6
        assert mesh.tri_count == 12
        assert mesh.vertex_count == 24
        assert mesh.indices.shape[0] == 36
        assert mesh.positions.dtype == np.float32
        assert mesh.indices.dtype == np.uint32

    def test_two_adjacent_ten_faces(self):
        c = Chunk((0, 0, 0))
        c.materials[5, 5, 5] = 1
        c.materials[6, 5, 5] = 1  # adjacent along +X
        mesh = build_mesh(c, neighbor_solids=None)
        # 12 total faces minus 2 shared (one each side) = 10
        assert mesh.face_count == 10
        assert mesh.tri_count == 20

    def test_fully_buried_zero_faces(self):
        c = Chunk((0, 0, 0))
        # 3x3x3 block; the centre voxel (5,5,5) is fully buried.
        c.materials[4:7, 4:7, 4:7] = 1
        # Mesh only the centre's exposure by checking a single interior voxel:
        # Easier: a fully solid chunk's interior produces only boundary faces.
        solid = Chunk((0, 0, 0))
        solid.materials[:] = 1
        mesh = build_mesh(solid, neighbor_solids=None)
        # All-solid 32^3 with air on all 6 open sides → only the 6 outer 32x32
        # faces are exposed: 6 * 32 * 32 = 6144 faces.
        assert mesh.face_count == 6 * 32 * 32

    def test_interior_voxel_buried(self):
        """A single voxel surrounded on all 6 sides emits 0 faces."""
        c = Chunk((0, 0, 0))
        c.materials[5, 5, 5] = 1
        c.materials[4, 5, 5] = 1
        c.materials[6, 5, 5] = 1
        c.materials[5, 4, 5] = 1
        c.materials[5, 6, 5] = 1
        c.materials[5, 5, 4] = 1
        c.materials[5, 5, 6] = 1
        # Count faces of just the centre by diffing: build full, then remove centre.
        full = build_mesh(c, neighbor_solids=None).face_count
        c.materials[5, 5, 5] = 0
        without = build_mesh(c, neighbor_solids=None).face_count
        # Removing the buried centre exposes 6 new inner faces on its neighbours;
        # the centre itself contributed 0 faces. So faces increase by 6.
        assert without - full == 6

    def test_chunk_boundary_face_culled_with_neighbor(self):
        """A solid voxel on +X face of A with solid neighbor in B → shared
        interior face culled when neighbor_solids supplies B (no leak)."""
        a = Chunk((0, 0, 0))
        a.materials[31, 5, 5] = 1  # on the +X boundary of A
        # Neighbor B (chunk +X) has a solid voxel at its x=0 facing A's x=31.
        b_solid = np.zeros((32, 32, 32), dtype=bool)
        b_solid[0, 5, 5] = True
        # Without neighbor: the +X face is exposed → 6 faces.
        mesh_open = build_mesh(a, neighbor_solids=None)
        assert mesh_open.face_count == 6
        # With neighbor B solid across the +X face: that face is culled → 5.
        mesh_culled = build_mesh(a, neighbor_solids={(1, 0, 0): b_solid})
        assert mesh_culled.face_count == 5

    def test_world_floor_pads_solid(self):
        """A voxel on the -Z boundary: with WORLD_FLOOR_SOLID the bottom face
        is culled (no see-through floor)."""
        a = Chunk((0, 0, -2))
        a.materials[5, 5, 0] = 1  # on the -Z boundary
        open_mesh = build_mesh(a, neighbor_solids=None)
        assert open_mesh.face_count == 6
        floored = build_mesh(a, neighbor_solids={(0, 0, -1): WORLD_FLOOR_SOLID})
        assert floored.face_count == 5

    def test_empty_chunk_empty_mesh(self):
        c = Chunk((0, 0, 0))
        mesh = build_mesh(c, neighbor_solids=None)
        assert mesh.is_empty
        assert mesh.face_count == 0
        assert mesh.positions.shape == (0, 3)
        assert mesh.indices.shape == (0,)

    def test_light_sampler_full_bright_default(self):
        c = Chunk((0, 0, 0))
        c.materials[5, 5, 5] = 1
        mesh = build_mesh(c, neighbor_solids=None)
        assert np.allclose(mesh.colors[:, :3], 1.0)
        assert np.allclose(mesh.colors[:, 3], 1.0)

    def test_light_sampler_hook(self):
        c = Chunk((0, 0, 0))
        c.materials[5, 5, 5] = 1

        captured = {}

        def sampler(face_centers: np.ndarray) -> np.ndarray:
            captured["shape"] = face_centers.shape
            # darken everything to 0.25
            return np.full((face_centers.shape[0],), 0.25, dtype=np.float32)

        mesh = build_mesh(c, neighbor_solids=None, light_sampler=sampler)
        assert captured["shape"] == (6, 3)  # 6 faces, world xyz
        assert np.allclose(mesh.colors[:, :3], 0.25)
        assert np.allclose(mesh.colors[:, 3], 1.0)

    def test_mesh_arrays_properties(self):
        """MeshArrays dataclass property accessors are correct."""
        positions = np.zeros((8, 3), np.float32)
        normals = np.zeros((8, 3), np.float32)
        uvs = np.zeros((8, 2), np.float32)
        colors = np.ones((8, 4), np.float32)
        indices = np.arange(12, dtype=np.uint32)
        m = MeshArrays(
            positions=positions,
            normals=normals,
            uvs=uvs,
            colors=colors,
            indices=indices,
        )
        # verts_per_face default is 4 → 8 verts / 4 = 2 faces
        assert m.face_count == 2
        assert m.vertex_count == 8
        assert m.tri_count == 4
        assert not m.is_empty

    def test_world_floor_solid_sentinel_value(self):
        """WORLD_FLOOR_SOLID is a string sentinel (not None, not ndarray)."""
        assert isinstance(WORLD_FLOOR_SOLID, str)
        assert WORLD_FLOOR_SOLID  # truthy
