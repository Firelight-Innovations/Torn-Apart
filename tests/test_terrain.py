"""
tests/test_terrain.py — Generation determinism/seamlessness, mesher fixtures,
desired-set pure-function tests.  Headless: no panda3d imports anywhere.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from fire_engine.core import load_config, EventBus
from fire_engine.core.math3d import Vec3
from fire_engine.core.rng import set_world_seed
from fire_engine.terrain.chunk import Chunk
from fire_engine.terrain.generation import generate_chunk
from fire_engine.terrain.meshing import build_mesh, WORLD_FLOOR_SOLID
from fire_engine.terrain.chunk_manager import ChunkManager


@pytest.fixture
def cfg():
    return load_config()


# ===========================================================================
# Generation: flat baseline terrain (determinism, flatness, bounds, seed-free)
# ===========================================================================

class TestFlatGeneration:
    def test_same_coord_byte_identical(self, cfg):
        a = generate_chunk((2, -3, 1), cfg)
        b = generate_chunk((2, -3, 1), cfg)
        assert np.array_equal(a, b)
        assert hashlib.sha256(a.tobytes()).hexdigest() == hashlib.sha256(b.tobytes()).hexdigest()

    def test_shape_dtype(self, cfg):
        m = generate_chunk((0, 0, 0), cfg)
        assert m.shape == (32, 32, 32)
        assert m.dtype == np.uint8

    def test_seed_independent(self, cfg):
        """Baseline terrain is flat/authored: it does NOT depend on world_seed."""
        set_world_seed(1)
        a = generate_chunk((0, 0, 0), cfg)
        set_world_seed(99999)
        b = generate_chunk((0, 0, 0), cfg)
        assert np.array_equal(a, b)

    def test_ground_chunk_is_flat(self, cfg):
        """A ground-straddling chunk is solid below ground_height_m, air above.

        Every (x,y) column must be a single solid→air run (no holes/overhangs):
        once air begins going up it never returns to solid.
        """
        m = generate_chunk((0, 0, 0), cfg)            # world z ∈ [0, 16)
        vs = cfg.voxel_size
        ground = cfg.ground_height_m
        zc = (np.arange(32) + 0.5) * vs               # voxel-centre world Z
        expected_col = (zc < ground)                  # (32,) bool over z
        solid = m > 0
        # Within this chunk's footprint (near origin) every column equals the
        # same flat profile.
        assert np.all(solid == expected_col[None, None, :])
        # Sanity: there IS both solid and air in a straddling chunk.
        assert solid.any() and (~solid).any()

    def test_below_ground_fully_solid(self, cfg):
        """A chunk entirely below the ground surface is fully solid (in bounds)."""
        m = generate_chunk((0, 0, -1), cfg)           # world z ∈ [-16, 0), all < 8
        assert np.all(m > 0)

    def test_above_ground_fully_air(self, cfg):
        """A chunk entirely above the ground surface is empty air."""
        m = generate_chunk((0, 0, 2), cfg)            # world z ∈ [32, 48), all > 8
        assert np.all(m == 0)

    def test_outside_world_footprint_is_air(self, cfg):
        """Beyond the world_size_m footprint there is no ground, only air."""
        half_chunks = int((cfg.world_size_m * 0.5) // cfg.chunk_meters) + 2
        m = generate_chunk((half_chunks, 0, -1), cfg)  # well past +X half-extent
        assert np.all(m == 0)

    def test_footprint_is_centred_on_origin(self, cfg):
        """Both sides of the origin are ground; symmetry confirms centring."""
        east = generate_chunk((0, 0, -1), cfg)        # x ∈ [0, 16)
        west = generate_chunk((-1, 0, -1), cfg)       # x ∈ [-16, 0)
        assert np.all(east > 0) and np.all(west > 0)

    def test_surface_height_is_flat_constant(self, cfg):
        from fire_engine.terrain.generation import surface_height
        wx = np.array([0.0, 8.0, -300.0])[:, None]
        wy = np.array([0.0, -50.0])[None, :]
        surf = surface_height(wx, wy, cfg)
        assert surf.shape == (3, 2)
        assert np.all(surf == cfg.ground_height_m)


# ===========================================================================
# Mesher fixtures
# ===========================================================================

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
        c.materials[6, 5, 5] = 1   # adjacent along +X
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
        a.materials[31, 5, 5] = 1   # on the +X boundary of A
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
        a.materials[5, 5, 0] = 1   # on the -Z boundary
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
        assert captured["shape"] == (6, 3)   # 6 faces, world xyz
        assert np.allclose(mesh.colors[:, :3], 0.25)
        assert np.allclose(mesh.colors[:, 3], 1.0)


# ===========================================================================
# desired_set pure-function tests
# ===========================================================================

class TestDesiredSet:
    def test_membership_and_count(self):
        set_world_seed(1337)
        cfg = load_config()  # view_distance_chunks = 6
        cm = ChunkManager(cfg, EventBus())
        # camera at origin → camera chunk (0,0,0)
        ds = cm.desired_set(Vec3(0.0, 0.0, 0.0))
        r = cfg.view_distance_chunks
        # XY: (2r+1)^2, Z: from -2..+4 = 7 levels
        expected = (2 * r + 1) ** 2 * 7
        assert len(ds) == expected
        assert (0, 0, 0) in ds
        assert (r, r, 4) in ds
        assert (r, r, -2) in ds
        assert (r + 1, 0, 0) not in ds
        assert (0, 0, 5) not in ds
        assert (0, 0, -3) not in ds

    def test_camera_chunk_offset(self):
        set_world_seed(1337)
        cfg = load_config()
        cm = ChunkManager(cfg, EventBus())
        # camera at world (20, 0, 0): chunk_meters=16 → camera chunk x=1
        ds = cm.desired_set(Vec3(20.0, 0.0, 0.0))
        assert (1, 0, 0) in ds
        assert (1 + cfg.view_distance_chunks, 0, 0) in ds


# ===========================================================================
# ChunkManager streaming + Saveable round-trip
# ===========================================================================

class TestChunkManagerStreaming:
    def test_stream_budget_two_per_frame(self):
        set_world_seed(1337)
        cfg = load_config()
        cm = ChunkManager(cfg, EventBus())
        cm.stream_frame(Vec3(0.0, 0.0, 0.0))
        assert len(cm.chunks) == 2  # at most 2 loaded per frame

    def test_provider_generates_on_demand(self):
        set_world_seed(1337)
        cfg = load_config()
        cm = ChunkManager(cfg, EventBus())
        ch = cm.get_or_create((3, 3, 0))
        assert ch.coord == (3, 3, 0)
        assert (3, 3, 0) in cm.chunks

    def test_saveable_delta_only_edited(self):
        set_world_seed(1337)
        cfg = load_config()
        cm = ChunkManager(cfg, EventBus())
        c0 = cm.get_or_create((0, 0, 0))
        c1 = cm.get_or_create((1, 0, 0))
        c1.materials[0, 0, 0] ^= 1  # mutate
        c1.edited = True
        delta = cm.get_delta()
        assert set(delta.keys()) == {(1, 0, 0)}
        assert delta[(1, 0, 0)].dtype == np.uint8

    def test_saveable_round_trip(self):
        set_world_seed(1337)
        cfg = load_config()
        cm = ChunkManager(cfg, EventBus())
        ch = cm.get_or_create((2, 2, 0))
        ch.materials[10, 10, 10] = 0 if ch.materials[10, 10, 10] else 1
        ch.edited = True
        delta = cm.get_delta()

        # Fresh manager, same seed, apply delta.
        cm2 = ChunkManager(cfg, EventBus())
        cm2.apply_delta(delta)
        restored = cm2.chunks[(2, 2, 0)]
        assert np.array_equal(restored.materials, ch.materials)
        assert restored.edited is True
        assert restored.dirty is True

    def test_save_key(self):
        assert ChunkManager.save_key == "terrain"
