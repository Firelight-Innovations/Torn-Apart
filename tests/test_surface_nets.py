"""
tests/test_surface_nets.py — Faceted (surface-nets) mesher, grass/dirt material
generation, brush border-dirty propagation, and ChunkManager mesh-style
dispatch.  Headless: no panda3d imports anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.core import load_config, EventBus
from fire_engine.core.config import Config
from fire_engine.core.math3d import Vec3
from fire_engine.world.terrain.chunk import Chunk
from fire_engine.world.terrain.chunk_manager import ChunkManager
from fire_engine.world.terrain.generation import (
    MATERIAL_DIRT,
    MATERIAL_GRASS,
    generate_chunk,
)
from fire_engine.world.terrain.brush import SphereBrush, BrushMode, apply_brush
from fire_engine.world.terrain.meshing import build_mesh
from fire_engine.world.terrain.surface_nets import (
    NEIGHBOR_OFFSETS_26,
    _build_padded_materials,
    _cell_vertices,
    build_mesh_faceted,
)


@pytest.fixture
def cfg():
    return load_config()


def _full_neighbor_materials(coord, cfg) -> dict:
    """All 26 neighbours from the deterministic generated baseline."""
    return {
        off: generate_chunk((coord[0] + off[0], coord[1] + off[1], coord[2] + off[2]), cfg)
        for off in NEIGHBOR_OFFSETS_26
    }


# ===========================================================================
# Generation: grass skin + dirt bulk
# ===========================================================================


class TestGrassDirtGeneration:
    def test_top_layer_is_grass_rest_is_dirt(self, cfg):
        m = generate_chunk((0, 0, 0), cfg)  # world z ∈ [0, 16), ground at 8
        vs = cfg.voxel_size
        zc = (np.arange(32) + 0.5) * vs  # voxel-centre world Z
        below = zc < cfg.ground_height_m
        top = below & (zc + vs >= cfg.ground_height_m)
        # Exactly one grass layer, everything solid beneath it is dirt.
        assert top.sum() == 1
        for z in range(32):
            if top[z]:
                assert np.all(m[:, :, z] == MATERIAL_GRASS)
            elif below[z]:
                assert np.all(m[:, :, z] == MATERIAL_DIRT)
            else:
                assert np.all(m[:, :, z] == 0)

    def test_buried_chunk_has_no_grass(self, cfg):
        """A chunk fully below another solid chunk is all dirt (no grass skin)."""
        m = generate_chunk((0, 0, -1), cfg)  # world z ∈ [-16, 0)
        assert np.all(m == MATERIAL_DIRT)

    def test_grass_layer_is_chunk_border_consistent(self, cfg):
        """Grass assignment is a pure function of world Z (no seam mismatch)."""
        a = generate_chunk((0, 0, 0), cfg)
        b = generate_chunk((1, 0, 0), cfg)
        assert np.array_equal(a[31, :, :] > 0, b[0, :, :] > 0)
        assert np.array_equal(a[31, :, :] == MATERIAL_GRASS, b[0, :, :] == MATERIAL_GRASS)


# ===========================================================================
# Faceted mesher: counts, geometry, determinism
# ===========================================================================


class TestFacetedMesher:
    def test_single_voxel_counts(self):
        c = Chunk((0, 0, 0))
        c.materials[5, 5, 5] = 1
        mesh = build_mesh_faceted(c)
        assert mesh.verts_per_face == 6
        assert mesh.face_count == 6
        assert mesh.tri_count == 12
        assert mesh.vertex_count == 36
        assert mesh.face_materials is not None
        assert np.all(mesh.face_materials == 1)

    def test_single_voxel_verts_inside_voxel_cube(self):
        """Surface-nets pulls vertices onto the surface: an isolated voxel
        becomes an octahedron-ish solid entirely inside the voxel's cube."""
        c = Chunk((0, 0, 0))
        c.materials[5, 5, 5] = 1
        mesh = build_mesh_faceted(c)
        lo, hi = 5 * 0.5, 6 * 0.5  # voxel spans [2.5, 3.0] m
        assert mesh.positions.min() >= lo - 1e-5
        assert mesh.positions.max() <= hi + 1e-5

    def test_winding_faces_outward(self):
        """Triangles wind CCW seen from outside: normal · face direction > 0."""
        c = Chunk((0, 0, 0))
        c.materials[5, 5, 5] = 1
        mesh = build_mesh_faceted(c)
        # Triangle order follows face order; faces are grouped by direction
        # (+X, -X, +Y, -Y, +Z, -Z) — one face each for a single voxel.
        dirs = np.array(
            [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
            dtype=np.float32,
        )
        tri_normals = mesh.normals.reshape(-1, 3, 3)[:, 0, :]  # (12, 3)
        per_face = tri_normals.reshape(6, 2, 3)
        dots = np.einsum("fij,fj->fi", per_face, dirs)
        assert np.all(dots > 0.0)

    def test_face_count_matches_blocky_mesher(self, cfg):
        """The exposure mask is shared with the cube mesher — face counts equal."""
        c = Chunk((0, 0, 0), generate_chunk((0, 0, 0), cfg))
        # Carve an arbitrary hole for non-trivial topology.
        c.materials[10:14, 10:14, 13:16] = 0
        blocky = build_mesh(c, neighbor_solids=None)
        faceted = build_mesh_faceted(c, neighbor_materials=None)
        assert faceted.face_count == blocky.face_count
        assert faceted.tri_count == blocky.tri_count

    def test_flat_ground_top_is_exactly_planar(self, cfg):
        """Flat baseline stays perfectly flat: every vertex of every top face
        sits exactly on the ground plane and every normal is +Z."""
        coord = (0, 0, 0)
        c = Chunk(coord, generate_chunk(coord, cfg))
        mesh = build_mesh_faceted(c, _full_neighbor_materials(coord, cfg))
        assert mesh.face_count > 0
        assert np.allclose(mesh.positions[:, 2], cfg.ground_height_m, atol=1e-5)
        assert np.allclose(mesh.normals[:, 2], 1.0, atol=1e-5)
        # The flat surface skin is all grass.
        assert np.all(mesh.face_materials == MATERIAL_GRASS)

    def test_carving_exposes_dirt(self, cfg):
        coord = (0, 0, 0)
        c = Chunk(coord, generate_chunk(coord, cfg))
        c.materials[8:12, 8:12, 0:16] = 0  # carve a shaft to the floor
        mesh = build_mesh_faceted(c, _full_neighbor_materials(coord, cfg))
        mats = set(np.unique(mesh.face_materials).tolist())
        assert MATERIAL_DIRT in mats  # shaft walls are dirt
        assert MATERIAL_GRASS in mats  # surrounding skin still grass

    def test_determinism(self, cfg):
        coord = (0, 0, 0)
        c = Chunk(coord, generate_chunk(coord, cfg))
        c.materials[4:9, 4:9, 12:16] = 0
        nm = _full_neighbor_materials(coord, cfg)
        m1 = build_mesh_faceted(c, nm)
        m2 = build_mesh_faceted(c, nm)
        for field in ("positions", "normals", "uvs", "colors", "indices", "face_materials"):
            assert np.array_equal(getattr(m1, field), getattr(m2, field))

    def test_empty_chunk_empty_mesh(self):
        c = Chunk((0, 0, 5))
        mesh = build_mesh_faceted(c)
        assert mesh.is_empty
        assert mesh.face_count == 0
        assert mesh.face_materials.shape == (0,)

    def test_light_sampler_contract(self, cfg):
        """Sampler gets (F, 3) world-meter centres; light lands in colors."""
        coord = (0, 0, 0)
        c = Chunk(coord, generate_chunk(coord, cfg))
        seen = {}

        def sampler(face_centers):
            seen["shape"] = face_centers.shape
            return np.full((face_centers.shape[0],), 0.5, dtype=np.float32)

        mesh = build_mesh_faceted(
            c, _full_neighbor_materials(coord, cfg), sampler, shade_strength=0.0
        )
        assert seen["shape"] == (mesh.face_count, 3)
        # shade_strength=0 → grey is exactly the sampled light.
        assert np.allclose(mesh.colors[:, 0], 0.5, atol=1e-6)
        # Alpha encodes the face material id (id / 255) for the GPU ground
        # shader's per-material palette selection, not a constant 1.0.
        expected_alpha = np.repeat(mesh.face_materials, 6).astype(np.float32) / 255.0
        assert np.allclose(mesh.colors[:, 3], expected_alpha)
        assert np.all(mesh.face_materials > 0)  # all faces are solid voxels

    def test_facet_shade_darkens_side_faces(self):
        """With shade_strength > 0, side facets are darker than top facets."""
        c = Chunk((0, 0, 0))
        c.materials[5, 5, 5] = 1
        mesh = build_mesh_faceted(c, shade_strength=0.5)
        grey = mesh.colors[:, 0].reshape(6, 6)  # (face, vert) grey
        # Faces are direction-grouped: +Z is index 4, -Z is index 5.
        top = grey[4].mean()
        bottom = grey[5].mean()
        assert top > bottom


# ===========================================================================
# Cross-chunk seams
# ===========================================================================


class TestSeams:
    def test_border_cell_vertices_match_across_chunks(self, cfg):
        """Both chunks compute byte-identical world positions for shared
        border dual cells (the no-cracks guarantee)."""
        a_coord, b_coord = (0, 0, 0), (1, 0, 0)
        a = Chunk(a_coord, generate_chunk(a_coord, cfg))
        b = Chunk(b_coord, generate_chunk(b_coord, cfg))
        # Carve a crater spanning the shared X border at the surface.
        store = {a_coord: a, b_coord: b}

        def provider(coord):
            return store.setdefault(coord, Chunk(coord, generate_chunk(coord, cfg)))

        apply_brush(
            SphereBrush(2.5),
            Vec3(16.0, 8.0, 8.0),
            BrushMode.REMOVE,
            chunk_provider=provider,
        )

        def neighbor_mats(coord):
            return {
                off: provider((coord[0] + off[0], coord[1] + off[1], coord[2] + off[2])).materials
                for off in NEIGHBOR_OFFSETS_26
            }

        n, vs = 32, cfg.voxel_size

        def cell_world(chunk, nm):
            pad = _build_padded_materials(chunk.materials, nm, n)
            vl = _cell_vertices(pad > 0, n)  # (33,33,33,3)
            m = n + 1
            grid = np.stack(
                np.meshgrid(np.arange(m), np.arange(m), np.arange(m), indexing="ij"),
                axis=-1,
            ).astype(np.float32)
            origin = chunk.world_origin.to_numpy().astype(np.float32)
            return origin + (grid - 0.5 + vl) * vs

        wa = cell_world(a, neighbor_mats(a_coord))
        wb = cell_world(b, neighbor_mats(b_coord))
        # A's max-X cell plane is B's min-X cell plane (the shared lattice).
        assert np.allclose(wa[32, :, :, :], wb[0, :, :, :], atol=1e-5)

    def test_brush_at_border_dirties_neighbor(self, cfg):
        """An edit whose changed voxels touch a chunk border flags the
        adjacent chunk dirty (remesh) but not edited (no save delta)."""
        store: dict = {}

        def provider(coord):
            return store.setdefault(coord, Chunk(coord, generate_chunk(coord, cfg)))

        # Small sphere fully inside chunk (0,0,0) but touching its +X border
        # (border plane at x = 16 m): centre 1 m from the plane, radius 1 m.
        touched = apply_brush(
            SphereBrush(1.0),
            Vec3(15.0, 8.0, 7.5),
            BrushMode.REMOVE,
            chunk_provider=provider,
        )
        assert touched == {(0, 0, 0)}
        nb = store[(1, 0, 0)]
        assert nb.dirty is True
        assert nb.edited is False


# ===========================================================================
# ChunkManager dispatch + streaming integration
# ===========================================================================


class TestMeshStyleDispatch:
    def test_default_config_is_faceted(self, cfg):
        assert cfg.mesh_style == "faceted"
        cm = ChunkManager(cfg, EventBus())
        mesh = cm.mesh_chunk((0, 0, 0))
        assert mesh.verts_per_face == 6
        assert mesh.face_materials is not None

    def test_blocky_style_uses_cube_mesher(self):
        cfg = Config(mesh_style="blocky")
        cm = ChunkManager(cfg, EventBus())
        mesh = cm.mesh_chunk((0, 0, 0))
        assert mesh.verts_per_face == 4
        assert mesh.face_materials is None

    def test_neighbor_materials_has_no_side_effects(self, cfg):
        """Building neighbour materials must not load chunks into the store
        (otherwise desired-set streaming would skip meshing them later)."""
        cm = ChunkManager(cfg, EventBus())
        cm.get_or_create((0, 0, 0))
        before = set(cm.chunks.keys())
        cm._neighbor_materials((0, 0, 0))
        assert set(cm.chunks.keys()) == before

    def test_stream_frame_produces_faceted_meshes(self, cfg):
        cm = ChunkManager(cfg, EventBus())
        cm.stream_frame(Vec3(0.0, 0.0, 8.0))
        assert len(cm.pending_meshes) > 0
        for mesh in cm.pending_meshes.values():
            assert mesh.verts_per_face == 6
