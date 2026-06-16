"""
tests/test_meshing_seams.py — Golden-master / characterisation tests for
build_mesh (blocky culled-face mesher) focusing on seam correctness,
neighbor-pad behaviour, and the WORLD_FLOOR_SOLID sentinel.

Headless only: no panda3d imports anywhere.
Do NOT fix bugs found here — pin current behaviour and flag suspicions.
"""

from __future__ import annotations

import numpy as np

from fire_engine.world.terrain.chunk import Chunk
from fire_engine.world.terrain.meshing import (
    WORLD_FLOOR_SOLID,
    build_mesh,
)

# Read verts_per_face from the module: build an empty-chunk mesh and read the
# field, rather than hard-coding a guess.  The empty-chunk path still returns a
# MeshArrays with the correct verts_per_face default.
_VPF: int = build_mesh(Chunk((0, 0, 0))).verts_per_face


def _single_voxel_chunk(x: int = 5, y: int = 5, z: int = 5, mat: int = 1) -> Chunk:
    """Return a Chunk with exactly one solid voxel at (x, y, z)."""
    c = Chunk((0, 0, 0))
    c.materials[x, y, z] = mat
    return c


def _all_solid_chunk(coord=(0, 0, 0)) -> Chunk:
    """Return a fully-solid chunk."""
    c = Chunk(coord)
    c.materials[:] = 1
    return c


def _air_chunk(coord=(0, 0, 0)) -> Chunk:
    """Return an all-air chunk (default construction)."""
    return Chunk(coord)


def _solid_slab_z(z_start: int, z_end: int, mat: int = 1) -> Chunk:
    """Return a chunk with a horizontal slab of solid voxels [z_start:z_end]."""
    c = Chunk((0, 0, 0))
    c.materials[:, :, z_start:z_end] = mat
    return c


# ---------------------------------------------------------------------------
# Helper: expected counts for the blocky mesher
# ---------------------------------------------------------------------------


def _expected_counts(face_count: int):
    """Return (vertex_count, index_count, tri_count) for *face_count* quads."""
    return (
        face_count * _VPF,  # vertex_count
        face_count * 6,  # index_count  (2 tris × 3 indices each)
        face_count * 2,  # tri_count
    )


# ===========================================================================
# 1. Empty chunk → 0 faces / empty arrays
# ===========================================================================


class TestEmptyChunk:
    def test_all_air_zero_faces(self):
        mesh = build_mesh(_air_chunk())
        assert mesh.face_count == 0

    def test_all_air_is_empty(self):
        mesh = build_mesh(_air_chunk())
        assert mesh.is_empty

    def test_all_air_positions_shape(self):
        mesh = build_mesh(_air_chunk())
        assert mesh.positions.shape == (0, 3)

    def test_all_air_indices_shape(self):
        mesh = build_mesh(_air_chunk())
        assert mesh.indices.shape == (0,)

    def test_all_air_normals_shape(self):
        mesh = build_mesh(_air_chunk())
        assert mesh.normals.shape == (0, 3)


# ===========================================================================
# 2. Single solid voxel surrounded by air → exactly 6 faces (cube)
# ===========================================================================


class TestSingleVoxel:
    def test_six_faces(self):
        mesh = build_mesh(_single_voxel_chunk(), neighbor_solids=None)
        assert mesh.face_count == 6

    def test_vertex_count(self):
        mesh = build_mesh(_single_voxel_chunk(), neighbor_solids=None)
        expected_verts, _, _ = _expected_counts(6)
        assert mesh.vertex_count == expected_verts

    def test_tri_count(self):
        mesh = build_mesh(_single_voxel_chunk(), neighbor_solids=None)
        assert mesh.tri_count == 12

    def test_indices_length(self):
        mesh = build_mesh(_single_voxel_chunk(), neighbor_solids=None)
        _, expected_idx, _ = _expected_counts(6)
        assert mesh.indices.shape[0] == expected_idx

    def test_positions_dtype(self):
        mesh = build_mesh(_single_voxel_chunk(), neighbor_solids=None)
        assert mesh.positions.dtype == np.float32

    def test_indices_dtype(self):
        mesh = build_mesh(_single_voxel_chunk(), neighbor_solids=None)
        assert mesh.indices.dtype == np.uint32

    def test_verts_per_face_is_four(self):
        """Blocky mesher always emits 4 vertices per face."""
        mesh = build_mesh(_single_voxel_chunk(), neighbor_solids=None)
        assert mesh.verts_per_face == 4


# ===========================================================================
# 3. Fully-solid chunk with all neighbors solid → 0 exterior faces
# ===========================================================================


class TestFullySolidWithSolidNeighbors:
    def _all_solid_neighbor_solids(self) -> dict:
        full = np.ones((32, 32, 32), dtype=bool)
        return {
            (1, 0, 0): full,
            (-1, 0, 0): full,
            (0, 1, 0): full,
            (0, -1, 0): full,
            (0, 0, 1): full,
            (0, 0, -1): full,
        }

    def test_zero_faces_when_all_neighbors_solid(self):
        chunk = _all_solid_chunk()
        mesh = build_mesh(chunk, neighbor_solids=self._all_solid_neighbor_solids())
        assert mesh.face_count == 0

    def test_is_empty_when_all_neighbors_solid(self):
        chunk = _all_solid_chunk()
        mesh = build_mesh(chunk, neighbor_solids=self._all_solid_neighbor_solids())
        assert mesh.is_empty

    def test_all_solid_no_neighbors_has_surface_faces(self):
        """Without neighbors, a solid chunk exposes all 6 outer 32x32 faces."""
        chunk = _all_solid_chunk()
        mesh = build_mesh(chunk, neighbor_solids=None)
        # 6 directions × 32 × 32 = 6144 faces
        assert mesh.face_count == 6 * 32 * 32

    def test_all_solid_no_neighbors_face_count_6144(self):
        chunk = _all_solid_chunk()
        mesh = build_mesh(chunk, neighbor_solids=None)
        assert mesh.face_count == 6144


# ===========================================================================
# 4. Neighbor influence: border voxel face present (absent) or culled (solid)
# ===========================================================================


class TestNeighborCulling:
    """A solid voxel on the +X border of a chunk (x=31)."""

    def _border_chunk_px(self):
        c = Chunk((0, 0, 0))
        c.materials[31, 5, 5] = 1
        return c

    def test_plus_x_face_visible_when_no_neighbor(self):
        """Without a +X neighbor the border face is open → 6 faces."""
        mesh = build_mesh(self._border_chunk_px(), neighbor_solids=None)
        assert mesh.face_count == 6

    def test_plus_x_face_culled_when_neighbor_solid_at_x0(self):
        """Solid voxel at (0, 5, 5) of the +X chunk culls the shared face."""
        nb = np.zeros((32, 32, 32), dtype=bool)
        nb[0, 5, 5] = True
        mesh = build_mesh(self._border_chunk_px(), neighbor_solids={(1, 0, 0): nb})
        assert mesh.face_count == 5

    def test_plus_x_face_still_visible_when_neighbor_solid_elsewhere(self):
        """A solid neighbor voxel NOT at (0,5,5) doesn't cull the face."""
        nb = np.zeros((32, 32, 32), dtype=bool)
        nb[0, 10, 10] = True  # different YZ position — not the matching face
        mesh = build_mesh(self._border_chunk_px(), neighbor_solids={(1, 0, 0): nb})
        assert mesh.face_count == 6

    def test_minus_x_border_culled_when_neg_x_neighbor_solid(self):
        """Voxel at x=0; the -X neighbor at x=31 culls the shared face."""
        c = Chunk((0, 0, 0))
        c.materials[0, 5, 5] = 1
        nb = np.zeros((32, 32, 32), dtype=bool)
        nb[31, 5, 5] = True
        mesh = build_mesh(c, neighbor_solids={(-1, 0, 0): nb})
        assert mesh.face_count == 5

    def test_plus_z_face_culled_by_solid_neighbor_above(self):
        """Voxel at z=31; solid +Z neighbor at z=0 culls the top face."""
        c = Chunk((0, 0, 0))
        c.materials[5, 5, 31] = 1
        nb = np.zeros((32, 32, 32), dtype=bool)
        nb[5, 5, 0] = True
        mesh = build_mesh(c, neighbor_solids={(0, 0, 1): nb})
        assert mesh.face_count == 5

    def test_absent_neighbor_pads_air_all_dirs_except_neg_z(self):
        """For any absent lateral neighbor the pad is AIR → face remains open."""
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)):
            # Place voxel on the border in that direction.
            idx = [5, 5, 5]
            axis = next(i for i, v in enumerate(d) if v != 0)
            idx[axis] = 31 if d[axis] > 0 else 0
            c = Chunk((0, 0, 0))
            c.materials[idx[0], idx[1], idx[2]] = 1
            # All other directions absent (air), only specify this one explicitly absent.
            mesh = build_mesh(c, neighbor_solids={})
            # The border face for this direction should remain open (6 faces total).
            assert mesh.face_count == 6, (
                f"Direction {d}: expected 6 faces with absent neighbor, got {mesh.face_count}"
            )


# ===========================================================================
# 5. WORLD_FLOOR_SOLID: pins -Z sentinel behaviour
# ===========================================================================


class TestWorldFloorSolid:
    def _bottom_z_chunk(self):
        """Chunk with a single solid voxel on the -Z border (z=0)."""
        c = Chunk((0, 0, -2))
        c.materials[5, 5, 0] = 1
        return c

    def test_absent_neg_z_neighbor_gives_six_faces(self):
        """Without WORLD_FLOOR_SOLID, -Z is padded AIR → 6 faces open."""
        mesh = build_mesh(self._bottom_z_chunk(), neighbor_solids=None)
        assert mesh.face_count == 6

    def test_world_floor_solid_culls_neg_z_face(self):
        """WORLD_FLOOR_SOLID on -Z pads it SOLID → bottom face culled → 5 faces."""
        mesh = build_mesh(
            self._bottom_z_chunk(),
            neighbor_solids={(0, 0, -1): WORLD_FLOOR_SOLID},
        )
        assert mesh.face_count == 5

    def test_world_floor_solid_only_culls_neg_z_not_other_faces(self):
        """Only the actual -Z face is culled; the other 5 remain open."""
        mesh = build_mesh(
            self._bottom_z_chunk(),
            neighbor_solids={(0, 0, -1): WORLD_FLOOR_SOLID},
        )
        # 6 faces - 1 (-Z culled) = 5
        assert mesh.face_count == 5

    def test_world_floor_solid_on_non_border_voxel_no_effect(self):
        """WORLD_FLOOR_SOLID on -Z with voxel NOT at z=0 border → still 6 faces
        (the sentinel pads the slab below the chunk, not a voxel inside it)."""
        c = Chunk((0, 0, -2))
        c.materials[5, 5, 5] = 1  # z=5, well above the -Z border
        mesh = build_mesh(c, neighbor_solids={(0, 0, -1): WORLD_FLOOR_SOLID})
        assert mesh.face_count == 6

    def test_world_floor_solid_accepted_for_any_direction(self):
        """
        CURRENT BEHAVIOUR PIN (suspected unintended):
        build_mesh accepts WORLD_FLOOR_SOLID for any direction, not just -Z.
        Passing it for +X pads that slab SOLID too — culling the +X border face.
        Pin this as observed current behaviour; may be a bug.
        """
        c = Chunk((0, 0, 0))
        c.materials[31, 5, 5] = 1  # voxel on the +X border
        # Absent +X → open, 6 faces
        mesh_open = build_mesh(c, neighbor_solids=None)
        assert mesh_open.face_count == 6
        # WORLD_FLOOR_SOLID on +X → the +X slab pads SOLID → 5 faces
        mesh_sentinel = build_mesh(c, neighbor_solids={(1, 0, 0): WORLD_FLOOR_SOLID})
        assert mesh_sentinel.face_count == 5  # current behaviour: sentinel culls any dir


# ===========================================================================
# 6. Normals: axis-aligned unit vectors pointing outward
# ===========================================================================


class TestNormals:
    def test_single_voxel_normals_are_axis_aligned(self):
        """All normals for a single voxel are unit-length axis-aligned vectors."""
        mesh = build_mesh(_single_voxel_chunk(), neighbor_solids=None)
        # Each normal row should be one of the 6 unit axis directions.
        norms = mesh.normals  # (N, 3)
        # All components are 0 or ±1.
        assert np.all((norms == 0) | (norms == 1) | (norms == -1))
        # Each row has exactly one non-zero component (axis-aligned).
        nonzero_per_row = np.count_nonzero(norms, axis=1)
        assert np.all(nonzero_per_row == 1)

    def test_normals_unit_length(self):
        mesh = build_mesh(_single_voxel_chunk(), neighbor_solids=None)
        lengths = np.linalg.norm(mesh.normals, axis=1)
        assert np.allclose(lengths, 1.0)

    def test_normals_point_outward(self):
        """For a single voxel, each face normal should point away from the solid.
        The centroid is at the voxel centre; face normals should have positive dot
        product with (face_centre - voxel_centre)."""
        c = _single_voxel_chunk(x=5, y=5, z=5)
        vs = 0.5  # default voxel_size
        voxel_centre = np.array([(5 + 0.5) * vs, (5 + 0.5) * vs, (5 + 0.5) * vs])
        mesh = build_mesh(c, neighbor_solids=None)

        # Each group of verts_per_face vertices shares the same normal.
        vpf = mesh.verts_per_face
        n_faces = mesh.face_count
        for fi in range(n_faces):
            verts = mesh.positions[fi * vpf : (fi + 1) * vpf]  # (vpf, 3)
            face_centre = verts.mean(axis=0)
            normal = mesh.normals[fi * vpf]  # flat — all same in face
            outward = face_centre - voxel_centre
            dot = float(np.dot(normal, outward))
            assert dot > 0, (
                f"Face {fi}: normal {normal} does not point away from voxel centre "
                f"(dot={dot:.4f}, face_centre={face_centre}, voxel_centre={voxel_centre})"
            )

    def test_top_face_normal_is_plus_z(self):
        """A solid slab exposed only on top should have +Z normals."""
        chunk = _solid_slab_z(0, 16)  # solid voxels z=0..15, air above
        mesh = build_mesh(chunk, neighbor_solids=None)
        # Only the top face layer should be exposed (bottom is open as AIR, sides too).
        # Filter normals that are +Z.
        nrm = mesh.normals
        top_mask = nrm[:, 2] > 0.5
        assert top_mask.any(), "Expected some +Z normals for solid slab top"
        assert np.allclose(nrm[top_mask], [0, 0, 1])

    def test_normals_full_solid_all_six_directions(self):
        """A fully solid chunk without neighbors exposes faces in all 6 directions."""
        mesh = build_mesh(_all_solid_chunk(), neighbor_solids=None)
        norms = mesh.normals  # (N, 3)
        found_dirs = set(map(tuple, np.unique(norms, axis=0).tolist()))
        expected_dirs = {(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)}
        assert found_dirs == expected_dirs


# ===========================================================================
# 7. Determinism: same inputs → identical arrays
# ===========================================================================


class TestDeterminism:
    def test_single_voxel_deterministic(self):
        c = _single_voxel_chunk()
        m1 = build_mesh(c, neighbor_solids=None)
        m2 = build_mesh(c, neighbor_solids=None)
        assert np.array_equal(m1.positions, m2.positions)
        assert np.array_equal(m1.normals, m2.normals)
        assert np.array_equal(m1.indices, m2.indices)

    def test_full_arrays_deterministic(self):
        """All MeshArrays fields are byte-identical across two calls."""
        c = _single_voxel_chunk()
        m1 = build_mesh(c, neighbor_solids=None)
        m2 = build_mesh(c, neighbor_solids=None)
        for field in ("positions", "normals", "uvs", "colors", "indices"):
            assert np.array_equal(getattr(m1, field), getattr(m2, field)), (
                f"Field '{field}' differs between two calls"
            )

    def test_deterministic_with_neighbors(self):
        """Same chunk + same neighbor_solids → identical output."""
        c = _single_voxel_chunk(x=31, y=5, z=5)
        nb = np.zeros((32, 32, 32), dtype=bool)
        nb[0, 5, 5] = True
        ns = {(1, 0, 0): nb}
        m1 = build_mesh(c, neighbor_solids=ns)
        m2 = build_mesh(c, neighbor_solids=ns)
        assert np.array_equal(m1.positions, m2.positions)
        assert np.array_equal(m1.indices, m2.indices)


# ===========================================================================
# 8. face_materials: blocky mesher should return None (no material split)
# ===========================================================================


class TestFaceMaterials:
    def test_blocky_face_materials_is_none(self):
        """build_mesh (blocky) does not set face_materials — it is None."""
        mesh = build_mesh(_single_voxel_chunk(), neighbor_solids=None)
        assert mesh.face_materials is None

    def test_empty_chunk_face_materials_is_none(self):
        mesh = build_mesh(_air_chunk())
        assert mesh.face_materials is None


# ===========================================================================
# 9. Colors: full-bright default, alpha = 1.0, light_sampler hook
# ===========================================================================


class TestColors:
    def test_default_colors_full_bright_rgb(self):
        """Without a light_sampler, RGB channels are 1.0 (full-bright)."""
        mesh = build_mesh(_single_voxel_chunk(), neighbor_solids=None)
        assert np.allclose(mesh.colors[:, :3], 1.0)

    def test_default_colors_alpha_one(self):
        """Blocky mesher sets alpha to 1.0 (no material id packed in alpha)."""
        mesh = build_mesh(_single_voxel_chunk(), neighbor_solids=None)
        assert np.allclose(mesh.colors[:, 3], 1.0)

    def test_light_sampler_scales_rgb(self):
        """light_sampler value 0.25 multiplies into RGB channels."""
        c = _single_voxel_chunk()

        def sampler(face_centers: np.ndarray) -> np.ndarray:
            return np.full(face_centers.shape[0], 0.25, dtype=np.float32)

        mesh = build_mesh(c, neighbor_solids=None, light_sampler=sampler)
        assert np.allclose(mesh.colors[:, :3], 0.25)

    def test_light_sampler_receives_f_by_3_array(self):
        """light_sampler is called with float32 (F, 3) world-space centres."""
        c = _single_voxel_chunk()
        captured: dict = {}

        def sampler(fc: np.ndarray) -> np.ndarray:
            captured["shape"] = fc.shape
            captured["dtype"] = fc.dtype
            return np.ones(fc.shape[0], dtype=np.float32)

        build_mesh(c, neighbor_solids=None, light_sampler=sampler)
        assert captured["shape"] == (6, 3)
        assert captured["dtype"] == np.float32


# ===========================================================================
# 10. Position / world-space consistency
# ===========================================================================


class TestPositions:
    def test_single_voxel_positions_within_voxel_bounds(self):
        """Blocky mesher: all vertex positions sit on the faces of the unit voxel
        cube [5*0.5, 6*0.5]^3."""
        vs = 0.5
        lo, hi = 5 * vs, 6 * vs
        mesh = build_mesh(_single_voxel_chunk(x=5, y=5, z=5), neighbor_solids=None)
        pos = mesh.positions
        # Allow a tiny float tolerance.
        assert pos[:, 0].min() >= lo - 1e-5
        assert pos[:, 0].max() <= hi + 1e-5
        assert pos[:, 1].min() >= lo - 1e-5
        assert pos[:, 1].max() <= hi + 1e-5
        assert pos[:, 2].min() >= lo - 1e-5
        assert pos[:, 2].max() <= hi + 1e-5

    def test_world_origin_offset_in_positions(self):
        """Chunk at coord (2, 3, 1) offsets all positions by (32, 48, 16) m."""
        c = Chunk((2, 3, 1))
        c.materials[0, 0, 0] = 1
        mesh = build_mesh(c, neighbor_solids=None)
        vs = 0.5
        chunk_m = 32 * vs  # 16.0 m
        origin = np.array([2 * chunk_m, 3 * chunk_m, 1 * chunk_m])
        # All positions should be within one voxel of the chunk origin corner.
        assert np.all(mesh.positions[:, 0] >= origin[0] - 1e-5)
        assert np.all(mesh.positions[:, 1] >= origin[1] - 1e-5)
        assert np.all(mesh.positions[:, 2] >= origin[2] - 1e-5)
