"""
tests/world/terrain/test_chunk_manager.py — ChunkManager streaming, desired_set,
provider, Saveable round-trip, and remesh_edited.
Headless: no panda3d imports.
"""

from __future__ import annotations

import numpy as np

from fire_engine.core import EventBus, load_config
from fire_engine.core.math3d import Vec3
from fire_engine.core.rng import set_world_seed
from fire_engine.world.terrain.chunk_manager import ChunkManager


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
        cm.get_or_create((0, 0, 0))
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


class TestRemeshEdited:
    """remesh_edited: same-frame remesh of brush edits, bypassing the budget."""

    def _crater(self, cm):
        """Carve a corner-spanning crater; return apply_brush's touched set."""
        from fire_engine.world.terrain.brush import BrushMode, SphereBrush, apply_brush

        return apply_brush(
            SphereBrush(2.5),
            Vec3(16.0, 16.0, 8.0),
            BrushMode.REMOVE,
            chunk_provider=cm.get_or_create,
        )

    def test_remeshes_all_touched_and_dirty_neighbors_now(self):
        set_world_seed(1337)
        cfg = load_config()
        cm = ChunkManager(cfg, EventBus())
        touched = self._crater(cm)
        assert len(touched) >= 2  # corner hit spans chunks

        n = cm.remesh_edited(touched)

        # Everything the brush dirtied (touched + border neighbours) is
        # remeshed immediately — no dirty chunk left in the neighbourhood.
        assert n >= len(touched)
        assert not any(ch.dirty for ch in cm.chunks.values())
        for c in touched:
            assert c in cm.pending_meshes

    def test_unrelated_dirty_chunks_stay_budgeted(self):
        set_world_seed(1337)
        cfg = load_config()
        cm = ChunkManager(cfg, EventBus())
        touched = self._crater(cm)
        far = cm.get_or_create((40, 40, 0))  # far outside the crater
        far.dirty = True

        cm.remesh_edited(touched)

        assert far.dirty is True  # left to stream_frame's budget
        assert (40, 40, 0) not in cm.pending_meshes

    def test_idempotent_on_clean_chunks(self):
        set_world_seed(1337)
        cfg = load_config()
        cm = ChunkManager(cfg, EventBus())
        touched = self._crater(cm)
        cm.remesh_edited(touched)
        assert cm.remesh_edited(touched) == 0  # nothing dirty → no work
