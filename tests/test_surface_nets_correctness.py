"""
tests/test_surface_nets_correctness.py — Golden-master / characterisation tests
for surface_nets.py and related internals.

Scope: pins CURRENT behaviour only.  Do NOT fix bugs here — mark suspicions in
comments so a human can investigate.  Headless; no panda3d imports.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from fire_engine.core import load_config
from fire_engine.terrain.chunk import Chunk
from fire_engine.terrain.generation import (
    MATERIAL_DIRT,
    MATERIAL_GRASS,
    generate_chunk,
)
from fire_engine.terrain.meshing import MeshArrays, WORLD_FLOOR_SOLID
from fire_engine.terrain.surface_nets import (
    NEIGHBOR_OFFSETS_26,
    _build_padded_materials,
    _cell_vertices,
    build_mesh_faceted,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    return load_config()


def _simple_sampler(face_centers: np.ndarray) -> np.ndarray:
    """Headless full-bright light sampler (mirrors existing test pattern)."""
    return np.ones(face_centers.shape[0], dtype=np.float32)


def _full_neighbor_materials(coord, cfg) -> dict:
    """All 26 neighbours from the deterministic generated baseline."""
    return {
        off: generate_chunk(
            (coord[0] + off[0], coord[1] + off[1], coord[2] + off[2]), cfg
        )
        for off in NEIGHBOR_OFFSETS_26
    }


# ===========================================================================
# 1. NEIGHBOR_OFFSETS_26 structural invariants
# ===========================================================================

class TestNeighborOffsets26:

    def test_exactly_26_entries(self):
        """All 3^3 - 1 = 26 offsets must be present."""
        assert len(NEIGHBOR_OFFSETS_26) == 26

    def test_all_unique(self):
        """No duplicate offsets."""
        assert len(set(NEIGHBOR_OFFSETS_26)) == 26

    def test_no_zero_offset(self):
        """(0, 0, 0) must not appear — it is the chunk itself, not a neighbour."""
        assert (0, 0, 0) not in NEIGHBOR_OFFSETS_26

    def test_all_components_in_minus1_0_plus1(self):
        """Every component must be in {-1, 0, 1}."""
        for off in NEIGHBOR_OFFSETS_26:
            for c in off:
                assert c in (-1, 0, 1), f"Bad component {c} in offset {off}"

    def test_covers_all_face_edge_corner_neighbours(self):
        """
        The 26 offsets must be exactly the set
        {-1,0,1}^3 \\ {(0,0,0)}.
        """
        expected = set(
            o for o in itertools.product((-1, 0, 1), repeat=3) if o != (0, 0, 0)
        )
        assert set(NEIGHBOR_OFFSETS_26) == expected

    def test_all_entries_are_3_tuples(self):
        """Each offset must be a 3-element sequence."""
        for off in NEIGHBOR_OFFSETS_26:
            assert len(off) == 3


# ===========================================================================
# 2. Empty chunk → empty mesh
# ===========================================================================

class TestEmptyChunk:

    def test_all_air_gives_zero_faces(self):
        c = Chunk((0, 0, 0))          # default: all zeros = air
        mesh = build_mesh_faceted(c)
        assert mesh.face_count == 0

    def test_all_air_mesh_is_empty(self):
        c = Chunk((3, -1, 2))
        mesh = build_mesh_faceted(c)
        assert mesh.is_empty

    def test_all_air_positions_are_zero_length(self):
        c = Chunk((0, 0, 0))
        mesh = build_mesh_faceted(c)
        assert mesh.positions.shape == (0, 3)

    def test_all_air_face_materials_is_empty_uint8(self):
        c = Chunk((0, 0, 0))
        mesh = build_mesh_faceted(c)
        assert mesh.face_materials is not None
        assert mesh.face_materials.shape == (0,)
        assert mesh.face_materials.dtype == np.uint8

    def test_all_air_indices_are_zero_length(self):
        c = Chunk((0, 0, 0))
        mesh = build_mesh_faceted(c)
        assert mesh.indices.shape == (0,)

    def test_verts_per_face_is_6_even_on_empty(self):
        """verts_per_face is set on the returned MeshArrays even when empty."""
        c = Chunk((0, 0, 0))
        mesh = build_mesh_faceted(c)
        assert mesh.verts_per_face == 6


# ===========================================================================
# 3. Solid slab — basic face count and internal consistency
# ===========================================================================

class TestSolidSlab:

    def _make_slab(self) -> Chunk:
        """A 32×32×1 slab at z=0 (material id 1) in an otherwise-air chunk."""
        c = Chunk((0, 0, 0))
        c.materials[:, :, 0] = 1
        return c

    def test_slab_has_faces(self):
        mesh = build_mesh_faceted(self._make_slab())
        assert mesh.face_count > 0

    def test_verts_per_face_is_module_constant(self):
        """verts_per_face from the returned mesh must equal 6 (two flat triangles)."""
        mesh = build_mesh_faceted(self._make_slab())
        # Read the constant from the mesh, not hard-coded here.
        vpf = mesh.verts_per_face
        assert vpf == 6

    def test_positions_length_equals_faces_times_verts_per_face(self):
        mesh = build_mesh_faceted(self._make_slab())
        vpf = mesh.verts_per_face
        assert mesh.positions.shape[0] == mesh.face_count * vpf

    def test_normals_length_matches_positions(self):
        mesh = build_mesh_faceted(self._make_slab())
        assert mesh.normals.shape[0] == mesh.positions.shape[0]

    def test_indices_length_equals_tris_times_3(self):
        """indices is a flat triangle list: tri_count * 3."""
        mesh = build_mesh_faceted(self._make_slab())
        assert mesh.indices.shape[0] == mesh.tri_count * 3

    def test_tri_count_is_twice_face_count(self):
        """Two triangles per face is the surface-nets contract."""
        mesh = build_mesh_faceted(self._make_slab())
        assert mesh.tri_count == mesh.face_count * 2

    def test_face_materials_length_equals_face_count(self):
        mesh = build_mesh_faceted(self._make_slab())
        assert mesh.face_materials is not None
        assert mesh.face_materials.shape[0] == mesh.face_count

    def test_colors_shape_matches_positions(self):
        """colors is (N, 4) with same N as positions."""
        mesh = build_mesh_faceted(self._make_slab())
        assert mesh.colors.shape == (mesh.positions.shape[0], 4)


# ===========================================================================
# 4. face_materials — grass/dirt assignment rule
# ===========================================================================

class TestFaceMaterials:

    def test_top_face_of_grass_column_is_grass(self, cfg):
        """
        A column that is grass on top (the generated baseline):
        the +Z-facing (up) exposed face of the grass voxel must carry
        MATERIAL_GRASS, not MATERIAL_DIRT.
        """
        coord = (0, 0, 0)
        c = Chunk(coord, generate_chunk(coord, cfg))
        mesh = build_mesh_faceted(c, _full_neighbor_materials(coord, cfg))
        # All exposed faces on the flat surface are +Z and must be grass.
        # The existing test already pinned that all face_materials == MATERIAL_GRASS
        # for the full ground plane.  Here we pin the rule for a single column.
        assert MATERIAL_GRASS in np.unique(mesh.face_materials)
        # Every face is on the grass-cap, so no face should carry material 0 (air).
        assert 0 not in np.unique(mesh.face_materials).tolist()

    def test_dirt_column_yields_dirt_face_materials(self):
        """
        A manually placed MATERIAL_DIRT voxel exposes MATERIAL_DIRT faces.
        face_material is the solid voxel's own material id, not the neighbour's.
        """
        c = Chunk((0, 0, 0))
        c.materials[10, 10, 10] = MATERIAL_DIRT   # id == 1
        mesh = build_mesh_faceted(c)
        assert mesh.face_materials is not None
        assert np.all(mesh.face_materials == MATERIAL_DIRT)

    def test_grass_voxel_all_faces_carry_grass_material(self):
        """
        A MATERIAL_GRASS voxel in empty space: ALL 6 faces (including side
        and bottom) carry MATERIAL_GRASS because face_material is the SOLID
        voxel's material, not based on direction.
        """
        c = Chunk((0, 0, 0))
        c.materials[5, 5, 5] = MATERIAL_GRASS
        mesh = build_mesh_faceted(c)
        # 6 faces, all carrying the same material.
        assert mesh.face_count == 6
        assert np.all(mesh.face_materials == MATERIAL_GRASS)

    def test_mixed_materials_in_same_chunk(self):
        """
        Two voxels of different materials in the same chunk: each face carries
        its own solid voxel's material (not a neighbour's).
        """
        c = Chunk((0, 0, 0))
        c.materials[2, 2, 2] = MATERIAL_DIRT    # id 1
        c.materials[20, 20, 20] = MATERIAL_GRASS  # id 2
        mesh = build_mesh_faceted(c)
        unique_mats = set(np.unique(mesh.face_materials).tolist())
        assert MATERIAL_DIRT in unique_mats
        assert MATERIAL_GRASS in unique_mats

    def test_face_materials_dtype_is_uint8(self):
        c = Chunk((0, 0, 0))
        c.materials[5, 5, 5] = 1
        mesh = build_mesh_faceted(c)
        assert mesh.face_materials.dtype == np.uint8

    def test_alpha_encodes_material_over_255(self):
        """
        Alpha channel of colors encodes face material: alpha[i] == mat_id / 255.
        Pin the exact 6-vertex-per-face expansion rule.
        """
        c = Chunk((0, 0, 0))
        c.materials[5, 5, 5] = MATERIAL_GRASS
        mesh = build_mesh_faceted(c, shade_strength=0.0)
        expected_alpha = (
            np.repeat(mesh.face_materials, 6).astype(np.float32) / 255.0
        )
        assert np.allclose(mesh.colors[:, 3], expected_alpha, atol=1e-6)


# ===========================================================================
# 5. Determinism
# ===========================================================================

class TestDeterminism:

    def test_isolated_voxel_same_arrays_twice(self):
        """Same isolated chunk → identical arrays on two calls."""
        c = Chunk((0, 0, 0))
        c.materials[10, 10, 10] = 1
        m1 = build_mesh_faceted(c)
        m2 = build_mesh_faceted(c)
        for field in ("positions", "normals", "uvs", "colors", "indices", "face_materials"):
            arr1, arr2 = getattr(m1, field), getattr(m2, field)
            assert np.array_equal(arr1, arr2), f"Field '{field}' not equal across calls"

    def test_with_sampler_deterministic(self, cfg):
        """With a deterministic sampler, output must be identical on two calls."""
        coord = (0, 0, 0)
        c = Chunk(coord, generate_chunk(coord, cfg))
        nm = _full_neighbor_materials(coord, cfg)
        m1 = build_mesh_faceted(c, nm, _simple_sampler)
        m2 = build_mesh_faceted(c, nm, _simple_sampler)
        for field in ("positions", "normals", "colors", "indices", "face_materials"):
            assert np.array_equal(getattr(m1, field), getattr(m2, field)), field

    def test_different_shade_strength_changes_colors(self):
        """shade_strength affects colors but NOT positions/indices/face_materials."""
        c = Chunk((0, 0, 0))
        c.materials[5, 5, 5] = 1
        m0 = build_mesh_faceted(c, shade_strength=0.0)
        m1 = build_mesh_faceted(c, shade_strength=1.0)
        # Colors differ (shade accent applied).
        assert not np.array_equal(m0.colors, m1.colors)
        # Geometry must be identical.
        assert np.array_equal(m0.positions, m1.positions)
        assert np.array_equal(m0.indices, m1.indices)
        assert np.array_equal(m0.face_materials, m1.face_materials)


# ===========================================================================
# 6. Border / neighbor_materials behaviour — pin current behaviour
# ===========================================================================

class TestBorderBehavior:

    def test_border_voxel_no_neighbors_has_exposed_faces(self):
        """
        A solid voxel at the chunk border with no neighbor_materials supplied
        (None → pad air) should expose ALL 6 faces, including the border face.
        """
        c = Chunk((0, 0, 0))
        c.materials[31, 31, 31] = 1   # corner voxel
        mesh_no_nb = build_mesh_faceted(c, neighbor_materials=None)
        assert mesh_no_nb.face_count == 6

    def test_border_voxel_with_solid_neighbor_reduces_faces(self):
        """
        If the neighbor chunk across a face has a solid voxel at the
        border, that face becomes hidden (fewer exposed faces).
        Pin current face count: 6 (no neighbour) vs 5 (solid neighbour).

        SUSPECTED ISSUE: check that the face count actually drops when a
        neighbour is provided.  If this fails, _build_padded_materials may
        not be using the neighbour's border slab correctly for edge/corner
        neighbours.
        """
        c = Chunk((0, 0, 0))
        c.materials[31, 31, 31] = 1   # solid at +X +Y +Z corner

        # Build a solid-filled neighbour in the +X direction.
        nb_mat = np.ones((32, 32, 32), dtype=np.uint8)
        mesh_with_nb = build_mesh_faceted(
            c,
            neighbor_materials={(1, 0, 0): nb_mat},
        )
        mesh_no_nb = build_mesh_faceted(c, neighbor_materials=None)

        # The +X face should now be hidden → one fewer exposed face.
        assert mesh_with_nb.face_count < mesh_no_nb.face_count

    def test_all_solid_neighbors_buries_border_voxel_partially(self):
        """
        A solid voxel at (0,0,0) with ALL 26 neighbour chunks solid:
        the 3 faces pointing INTO this chunk (+X, +Y, +Z) are exposed because
        those directions are the chunk interior (air by default, not filled by
        any neighbour pad).  Only the 3 outward faces (-X, -Y, -Z) are hidden
        by the neighbour shell.

        SUSPECTED BUG: A corner voxel at (0,0,0) surrounded by solid neighbour
        chunks should theoretically have 0 exposed faces, but the mesher sees
        3 exposed faces (the inward-pointing +X/+Y/+Z directions) because the
        neighbour padding only fills the 1-voxel shell, not the rest of the
        chunk interior.  This is expected IF the caller has only ONE solid voxel
        in the chunk; in practice callers mesh full chunks so the interior is
        also solid.  PIN current behaviour: 3 exposed faces.
        """
        c = Chunk((0, 0, 0))
        c.materials[0, 0, 0] = 1     # min-corner voxel, rest of chunk is air

        nb_mat = np.ones((32, 32, 32), dtype=np.uint8)
        nm = {off: nb_mat for off in NEIGHBOR_OFFSETS_26}

        mesh = build_mesh_faceted(c, neighbor_materials=nm)

        # The 3 outward faces (-X, -Y, -Z) are covered by the neighbour shell.
        # The 3 inward faces (+X, +Y, +Z) face into the air-filled chunk interior.
        # Current behaviour: 3 faces exposed.
        assert mesh.face_count == 3

    def test_world_floor_solid_sentinel_blocks_bottom_face(self):
        """
        Passing WORLD_FLOOR_SOLID as the (-1, 0, -1) neighbour value (world
        floor sentinel) should cause the padded shell region to be filled with
        solid dirt.  A voxel at z=0 whose -Z face was previously exposed should
        have its -Z face hidden after the floor sentinel is applied.

        Pins the _build_padded_materials sentinel path.
        """
        n = 32
        mat = np.zeros((n, n, n), dtype=np.uint8)
        mat[5, 5, 0] = 1    # solid at z=0, -Z face normally exposed

        # Without sentinel: pad is air below → -Z face is visible.
        pad_air = _build_padded_materials(mat, None, n)
        # With sentinel at the face-direction (-Z = (0,0,-1)):
        pad_floor = _build_padded_materials(mat, {(0, 0, -1): WORLD_FLOOR_SOLID}, n)

        # The z=0 shell slice of the padded array:
        # - air pad: pad_air[:, :, 0] should be 0 (air) at (5+1, 5+1)
        # - floor sentinel: pad_floor[:, :, 0] should be > 0 (dirt)
        assert pad_air[6, 6, 0] == 0
        assert pad_floor[6, 6, 0] > 0   # sentinel filled solid dirt

    def test_none_neighbor_pads_air(self):
        """
        An explicitly-None neighbour in the dict is treated as air (same as
        absent key), not WORLD_FLOOR_SOLID.
        """
        n = 32
        mat = np.zeros((n, n, n), dtype=np.uint8)
        mat[5, 5, 0] = 1

        pad_explicit_none = _build_padded_materials(mat, {(0, 0, -1): None}, n)
        pad_absent = _build_padded_materials(mat, {}, n)

        # Both should produce the same pad.
        assert np.array_equal(pad_explicit_none, pad_absent)


# ===========================================================================
# 7. _build_padded_materials internals
# ===========================================================================

class TestBuildPaddedMaterials:

    def test_interior_matches_chunk_materials(self):
        """Center region of the padded array must equal the chunk materials."""
        n = 32
        mat = np.arange(n ** 3, dtype=np.uint8).reshape(n, n, n)
        pad = _build_padded_materials(mat, None, n)
        assert pad.shape == (n + 2, n + 2, n + 2)
        assert np.array_equal(pad[1:n + 1, 1:n + 1, 1:n + 1], mat)

    def test_no_neighbors_pads_air(self):
        """Without neighbours the shell must be all zeros (air)."""
        n = 32
        mat = np.ones((n, n, n), dtype=np.uint8)
        pad = _build_padded_materials(mat, None, n)
        # All shell slabs must be air.
        assert np.all(pad[0, :, :] == 0)
        assert np.all(pad[n + 1, :, :] == 0)
        assert np.all(pad[:, 0, :] == 0)
        assert np.all(pad[:, n + 1, :] == 0)
        assert np.all(pad[:, :, 0] == 0)
        assert np.all(pad[:, :, n + 1] == 0)

    def test_face_neighbor_x_plus_fills_shell(self):
        """
        A +X face neighbour should fill the x = n+1 shell slab with that
        neighbour's x=0 slab.
        """
        n = 32
        mat = np.zeros((n, n, n), dtype=np.uint8)
        nb = np.full((n, n, n), fill_value=7, dtype=np.uint8)
        pad = _build_padded_materials(mat, {(1, 0, 0): nb}, n)
        # Shell at x = n+1 (index n+1) should be nb[0, :, :]
        assert np.array_equal(pad[n + 1, 1:n + 1, 1:n + 1], nb[0, :, :])

    def test_face_neighbor_x_minus_fills_shell(self):
        """
        A -X face neighbour should fill the x = 0 shell slab with that
        neighbour's x = n-1 slab.
        """
        n = 32
        mat = np.zeros((n, n, n), dtype=np.uint8)
        nb = np.full((n, n, n), fill_value=3, dtype=np.uint8)
        pad = _build_padded_materials(mat, {(-1, 0, 0): nb}, n)
        assert np.array_equal(pad[0, 1:n + 1, 1:n + 1], nb[n - 1, :, :])


# ===========================================================================
# 8. _cell_vertices shape and range
# ===========================================================================

class TestCellVertices:

    def test_output_shape(self):
        """_cell_vertices returns (n+1, n+1, n+1, 3)."""
        n = 32
        solid = np.zeros((n + 2, n + 2, n + 2), dtype=bool)
        vl = _cell_vertices(solid, n)
        assert vl.shape == (n + 1, n + 1, n + 1, 3)

    def test_all_air_gives_zero_positions(self):
        """With no solid voxels there are no sign changes; positions stay at 0."""
        n = 32
        solid = np.zeros((n + 2, n + 2, n + 2), dtype=bool)
        vl = _cell_vertices(solid, n)
        assert np.all(vl == 0.0)

    def test_positions_in_unit_cube(self):
        """
        Non-zero vertex positions must lie in [0, 1]^3 (the cell's unit cube).
        """
        n = 32
        mat = np.zeros((n, n, n), dtype=np.uint8)
        mat[10:12, 10:12, 10:12] = 1
        pad = _build_padded_materials(mat, None, n)
        vl = _cell_vertices(pad > 0, n)
        # Only check non-zero positions (inactive cells are 0,0,0 by convention).
        active = (vl != 0).any(axis=-1)
        if active.any():
            sub = vl[active]
            assert sub.min() >= 0.0 - 1e-6
            assert sub.max() <= 1.0 + 1e-6

    def test_dtype_is_float32(self):
        n = 32
        solid = np.zeros((n + 2,) * 3, dtype=bool)
        vl = _cell_vertices(solid, n)
        assert vl.dtype == np.float32
