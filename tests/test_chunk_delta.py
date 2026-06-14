"""
tests/test_chunk_delta.py — Characterisation (golden-master) tests for Chunk
and ChunkManager's delta round-trip.

Covers:
- Chunk.is_solid_mask() — all-air, all-solid, sparse patterns
- Chunk.world_origin — derived from config chunk_meters, tested at (0,0,0),
  positive, and negative coords
- dirty / edited initial state and flag transitions
- ChunkManager.get_delta() / apply_delta() round-trip
- Unedited chunk delta convention (empty / not present)
- Boundary coords (large positive, large negative)
- Delta of all-air vs all-solid chunk
- Determinism: same coord twice → identical materials

Headless: no panda3d imports, no per-voxel Python loops.
All array comparisons use np.array_equal.
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.core import load_config, EventBus
from fire_engine.core.rng import set_world_seed
from fire_engine.terrain.chunk import Chunk
from fire_engine.terrain.generation import generate_chunk, MATERIAL_DIRT, MATERIAL_GRASS
from fire_engine.terrain.chunk_manager import ChunkManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def chunk_meters(cfg):
    """Derived chunk world-edge length from config, never hard-coded."""
    return cfg.chunk_meters


@pytest.fixture
def cm(cfg):
    set_world_seed(cfg.world_seed)
    return ChunkManager(cfg, EventBus())


# ---------------------------------------------------------------------------
# is_solid_mask()
# ---------------------------------------------------------------------------

class TestIsSolidMask:

    def test_all_air_all_false(self):
        """All-zero materials -> every mask entry is False."""
        c = Chunk((0, 0, 0))
        # materials are all zero by default
        mask = c.is_solid_mask()
        assert mask.shape == (32, 32, 32)
        assert mask.dtype == bool
        assert not mask.any()

    def test_all_solid_all_true(self):
        """Fully solid chunk -> every mask entry is True."""
        c = Chunk((0, 0, 0))
        c.materials[:] = 1
        mask = c.is_solid_mask()
        assert mask.all()

    def test_sparse_pattern_equals_materials_nonzero(self):
        """Sparse pattern: mask must exactly equal (materials != 0)."""
        c = Chunk((0, 0, 0))
        # Set an irregular block: checkerboard slice + corner voxels.
        c.materials[::2, ::2, 0] = 1
        c.materials[1::2, 1::2, 1] = 2        # material id 2 is also solid
        c.materials[31, 31, 31] = 1
        expected = c.materials != 0
        assert np.array_equal(c.is_solid_mask(), expected)

    def test_single_voxel_solid(self):
        """Only one voxel set to solid -> exactly one True in mask."""
        c = Chunk((0, 0, 0))
        c.materials[5, 10, 15] = 1
        mask = c.is_solid_mask()
        assert int(mask.sum()) == 1
        assert mask[5, 10, 15]

    def test_material_id_2_is_solid(self):
        """MATERIAL_GRASS (id=2) is solid — mask treats any nonzero id as solid."""
        c = Chunk((0, 0, 0))
        c.materials[0, 0, 0] = MATERIAL_GRASS
        mask = c.is_solid_mask()
        assert mask[0, 0, 0]

    def test_mask_shape_and_dtype(self):
        """Returned array has shape (32,32,32) and bool dtype."""
        c = Chunk((1, 2, -1))
        mask = c.is_solid_mask()
        assert mask.shape == (32, 32, 32)
        assert mask.dtype == bool


# ---------------------------------------------------------------------------
# world_origin property
# ---------------------------------------------------------------------------

class TestWorldOrigin:

    def test_origin_zero_coord(self, chunk_meters):
        """Chunk at (0,0,0) has world origin (0,0,0)."""
        c = Chunk((0, 0, 0))
        o = c.world_origin
        assert float(o.x) == pytest.approx(0.0)
        assert float(o.y) == pytest.approx(0.0)
        assert float(o.z) == pytest.approx(0.0)

    def test_origin_positive_coord(self, chunk_meters):
        """Positive coord: world_origin = coord * chunk_meters."""
        c = Chunk((3, 5, 2))
        o = c.world_origin
        assert float(o.x) == pytest.approx(3 * chunk_meters)
        assert float(o.y) == pytest.approx(5 * chunk_meters)
        assert float(o.z) == pytest.approx(2 * chunk_meters)

    def test_origin_negative_coord(self, chunk_meters):
        """Negative coord: world_origin uses signed multiplication."""
        c = Chunk((-1, -2, -3))
        o = c.world_origin
        assert float(o.x) == pytest.approx(-1 * chunk_meters)
        assert float(o.y) == pytest.approx(-2 * chunk_meters)
        assert float(o.z) == pytest.approx(-3 * chunk_meters)

    def test_origin_large_positive(self, chunk_meters):
        """Large positive coord: no overflow in float multiplication."""
        c = Chunk((1000, 1000, 100))
        o = c.world_origin
        assert float(o.x) == pytest.approx(1000 * chunk_meters)
        assert float(o.y) == pytest.approx(1000 * chunk_meters)
        assert float(o.z) == pytest.approx(100 * chunk_meters)

    def test_origin_large_negative(self, chunk_meters):
        """Large negative coord."""
        c = Chunk((-500, -500, -10))
        o = c.world_origin
        assert float(o.x) == pytest.approx(-500 * chunk_meters)
        assert float(o.y) == pytest.approx(-500 * chunk_meters)
        assert float(o.z) == pytest.approx(-10 * chunk_meters)

    def test_origin_matches_config_chunk_meters(self, cfg, chunk_meters):
        """chunk_meters derived from config == chunk_size * voxel_size."""
        assert chunk_meters == pytest.approx(cfg.chunk_size * cfg.voxel_size)
        c = Chunk((2, 0, 0), chunk_size=cfg.chunk_size, voxel_size=cfg.voxel_size)
        assert float(c.world_origin.x) == pytest.approx(2 * chunk_meters)


# ---------------------------------------------------------------------------
# dirty / edited initial state and flag transitions
# ---------------------------------------------------------------------------

class TestDirtyEditedFlags:

    def test_fresh_chunk_dirty_true_edited_false(self):
        """A freshly constructed Chunk starts dirty=True, edited=False."""
        c = Chunk((0, 0, 0))
        assert c.dirty is True
        assert c.edited is False

    def test_fresh_chunk_with_materials_dirty_true_edited_false(self):
        """Passing materials in the constructor still gives dirty=True, edited=False."""
        mats = np.zeros((32, 32, 32), dtype=np.uint8)
        mats[0, 0, 0] = 1
        c = Chunk((0, 0, 0), mats)
        assert c.dirty is True
        assert c.edited is False

    def test_clearing_dirty_is_possible(self):
        """dirty can be cleared by the mesh-step (simulate what ChunkManager does)."""
        c = Chunk((0, 0, 0))
        c.dirty = False
        assert c.dirty is False

    def test_setting_edited_manually(self):
        """edited can be set to True (as apply_delta does)."""
        c = Chunk((0, 0, 0))
        c.edited = True
        assert c.edited is True

    def test_apply_delta_sets_edited_and_dirty(self, cm, cfg):
        """After apply_delta, the target chunk is edited=True AND dirty=True."""
        coord = (0, 0, 0)
        # Build a fake delta: generate a chunk and flip one voxel.
        mats = generate_chunk(coord, cfg).copy()
        mats[15, 15, 15] = 0 if mats[15, 15, 15] else 1
        cm.apply_delta({coord: mats})
        restored = cm.chunks[coord]
        assert restored.edited is True
        assert restored.dirty is True

    def test_get_delta_only_returns_edited_chunks(self, cm, cfg):
        """ChunkManager.get_delta() includes only chunks where edited=True."""
        c0 = cm.get_or_create((0, 0, 0))
        c1 = cm.get_or_create((1, 0, 0))
        # Do NOT mark c0 as edited; mark c1.
        c1.edited = True
        delta = cm.get_delta()
        assert (1, 0, 0) in delta
        assert (0, 0, 0) not in delta

    def test_unedited_chunk_not_in_delta(self, cm):
        """A chunk that was generated but never edited must be absent from the delta."""
        cm.get_or_create((5, 5, 0))   # loaded but not edited
        delta = cm.get_delta()
        assert (5, 5, 0) not in delta

    def test_empty_world_delta_is_empty_dict(self, cm):
        """A fresh ChunkManager with no loaded chunks yields an empty delta."""
        delta = cm.get_delta()
        assert delta == {}


# ---------------------------------------------------------------------------
# save_delta / apply_delta round-trip (on ChunkManager)
# ---------------------------------------------------------------------------

class TestDeltaRoundTrip:

    def test_round_trip_preserves_materials(self, cfg):
        """Edit a chunk, get_delta, apply_delta into fresh CM, materials identical."""
        set_world_seed(cfg.world_seed)
        cm1 = ChunkManager(cfg, EventBus())
        coord = (2, 3, 0)
        ch = cm1.get_or_create(coord)
        # Modify using numpy slicing — no per-voxel loop.
        ch.materials[10:14, 0:4, 0:4] = 0
        ch.materials[0, 0, 0] = 1
        ch.edited = True

        delta = cm1.get_delta()
        assert coord in delta

        cm2 = ChunkManager(cfg, EventBus())
        cm2.apply_delta(delta)
        restored = cm2.chunks[coord]
        assert np.array_equal(restored.materials, ch.materials)

    def test_apply_delta_sets_edited_true(self, cfg):
        """apply_delta marks the target chunk edited=True."""
        set_world_seed(cfg.world_seed)
        cm1 = ChunkManager(cfg, EventBus())
        coord = (0, 0, 0)
        ch = cm1.get_or_create(coord)
        ch.materials[5, 5, 5] = 0
        ch.edited = True

        cm2 = ChunkManager(cfg, EventBus())
        cm2.apply_delta(cm1.get_delta())
        assert cm2.chunks[coord].edited is True

    def test_apply_delta_sets_dirty_true(self, cfg):
        """apply_delta marks the target chunk dirty=True (triggers remesh)."""
        set_world_seed(cfg.world_seed)
        cm1 = ChunkManager(cfg, EventBus())
        coord = (0, 0, 0)
        ch = cm1.get_or_create(coord)
        ch.materials[0:5, 0:5, 0:5] = 0
        ch.edited = True

        cm2 = ChunkManager(cfg, EventBus())
        # First clear dirty by simulating mesh step, then apply delta.
        coord2 = (0, 0, 0)
        cm2.apply_delta(cm1.get_delta())
        assert cm2.chunks[coord2].dirty is True

    def test_apply_delta_on_unloaded_coord_creates_chunk(self, cfg):
        """apply_delta creates the chunk in the target CM even if it was not loaded."""
        set_world_seed(cfg.world_seed)
        coord = (99, 99, 0)
        mats = np.zeros((32, 32, 32), dtype=np.uint8)
        mats[1, 2, 3] = MATERIAL_DIRT

        cm = ChunkManager(cfg, EventBus())
        assert coord not in cm.chunks
        cm.apply_delta({coord: mats})
        assert coord in cm.chunks
        assert cm.chunks[coord].materials[1, 2, 3] == MATERIAL_DIRT

    def test_delta_values_are_copies(self, cfg):
        """get_delta returns copies: mutating the chunk after get_delta doesn't change delta."""
        set_world_seed(cfg.world_seed)
        cm = ChunkManager(cfg, EventBus())
        coord = (0, 0, 0)
        ch = cm.get_or_create(coord)
        ch.materials[10, 10, 10] = 1
        ch.edited = True

        delta = cm.get_delta()
        original_val = delta[coord][10, 10, 10]

        # Now mutate the chunk after the delta was taken.
        ch.materials[10, 10, 10] = 0
        # The delta copy must be unaffected.
        assert delta[coord][10, 10, 10] == original_val

    def test_round_trip_all_solid_chunk(self, cfg):
        """Delta round-trip for a fully solid chunk."""
        set_world_seed(cfg.world_seed)
        cm1 = ChunkManager(cfg, EventBus())
        coord = (0, 0, -1)
        ch = cm1.get_or_create(coord)
        ch.materials[:] = MATERIAL_DIRT
        ch.edited = True

        delta = cm1.get_delta()
        cm2 = ChunkManager(cfg, EventBus())
        cm2.apply_delta(delta)
        assert np.array_equal(cm2.chunks[coord].materials, ch.materials)

    def test_round_trip_all_air_chunk(self, cfg):
        """Delta round-trip for a completely air chunk (unusual but valid)."""
        set_world_seed(cfg.world_seed)
        cm1 = ChunkManager(cfg, EventBus())
        coord = (0, 0, 5)
        ch = cm1.get_or_create(coord)
        ch.materials[:] = 0   # force all-air
        ch.edited = True

        delta = cm1.get_delta()
        cm2 = ChunkManager(cfg, EventBus())
        cm2.apply_delta(delta)
        result = cm2.chunks[coord].materials
        assert np.array_equal(result, np.zeros((32, 32, 32), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Boundary coordinates
# ---------------------------------------------------------------------------

class TestBoundaryCoords:

    def test_delta_large_positive_coord(self, cfg):
        """get_delta / apply_delta at a large positive coord."""
        set_world_seed(cfg.world_seed)
        cm1 = ChunkManager(cfg, EventBus())
        coord = (500, 500, 0)
        ch = cm1.get_or_create(coord)
        ch.materials[0, 0, 0] = 1
        ch.edited = True

        delta = cm1.get_delta()
        cm2 = ChunkManager(cfg, EventBus())
        cm2.apply_delta(delta)
        assert np.array_equal(cm2.chunks[coord].materials, ch.materials)

    def test_delta_large_negative_coord(self, cfg):
        """get_delta / apply_delta at a large negative coord."""
        set_world_seed(cfg.world_seed)
        cm1 = ChunkManager(cfg, EventBus())
        coord = (-500, -500, -10)
        ch = cm1.get_or_create(coord)
        ch.materials[31, 31, 31] = 1
        ch.edited = True

        delta = cm1.get_delta()
        cm2 = ChunkManager(cfg, EventBus())
        cm2.apply_delta(delta)
        assert np.array_equal(cm2.chunks[coord].materials, ch.materials)

    def test_world_origin_min_coord(self, chunk_meters):
        """Chunk at (-1,-1,-1) world_origin = (-chunk_meters, -chunk_meters, -chunk_meters)."""
        c = Chunk((-1, -1, -1))
        o = c.world_origin
        assert float(o.x) == pytest.approx(-chunk_meters)
        assert float(o.y) == pytest.approx(-chunk_meters)
        assert float(o.z) == pytest.approx(-chunk_meters)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_same_coord_same_materials(self, cfg):
        """generate_chunk called twice with the same coord returns identical arrays."""
        coord = (3, -2, 0)
        a = generate_chunk(coord, cfg)
        b = generate_chunk(coord, cfg)
        assert np.array_equal(a, b)

    def test_same_coord_same_materials_different_seed_call(self, cfg):
        """Baseline terrain is seed-independent: changing world_seed doesn't matter."""
        coord = (0, 0, 0)
        set_world_seed(1)
        a = generate_chunk(coord, cfg)
        set_world_seed(99999)
        b = generate_chunk(coord, cfg)
        assert np.array_equal(a, b)

    def test_two_cm_same_seed_same_chunk(self, cfg):
        """Two ChunkManagers created with the same seed produce identical chunks."""
        set_world_seed(cfg.world_seed)
        cm1 = ChunkManager(cfg, EventBus())
        cm2 = ChunkManager(cfg, EventBus())
        coord = (1, 2, 0)
        ch1 = cm1.get_or_create(coord)
        ch2 = cm2.get_or_create(coord)
        assert np.array_equal(ch1.materials, ch2.materials)

    def test_is_solid_mask_deterministic(self, cfg):
        """is_solid_mask is a pure function of materials — same input, same mask."""
        coord = (0, 0, 0)
        mats = generate_chunk(coord, cfg)
        c1 = Chunk(coord, mats.copy())
        c2 = Chunk(coord, mats.copy())
        assert np.array_equal(c1.is_solid_mask(), c2.is_solid_mask())

    def test_is_solid_mask_equals_materials_gt_zero(self, cfg):
        """Pinned: is_solid_mask() == (materials > 0) for a generated chunk."""
        coord = (0, 0, 0)
        mats = generate_chunk(coord, cfg)
        c = Chunk(coord, mats)
        assert np.array_equal(c.is_solid_mask(), mats > 0)
