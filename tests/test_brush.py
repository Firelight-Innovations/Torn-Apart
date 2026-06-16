"""
tests/test_brush.py — Brush rasterisation correctness, round-trips, multi-chunk
coverage, and event publishing.  Headless: no panda3d imports.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fire_engine.core import EventBus, TerrainEditedEvent, load_config
from fire_engine.core.math3d import Vec3
from fire_engine.core.rng import set_world_seed
from fire_engine.world.terrain.brush import (
    BoxBrush,
    BrushMode,
    CylinderBrush,
    SphereBrush,
    apply_brush,
)
from fire_engine.world.terrain.chunk import Chunk


@pytest.fixture
def cfg():
    return load_config()


def make_provider(cfg, store=None):
    """Dict-backed chunk_provider that generates empty chunks on miss."""
    store = {} if store is None else store

    def provider(coord):
        ch = store.get(coord)
        if ch is None:
            ch = Chunk(coord, chunk_size=cfg.chunk_size, voxel_size=cfg.voxel_size)
            store[coord] = ch
        return ch

    return provider, store


class TestSphereCount:
    def test_sphere_voxel_count_analytic(self, cfg):
        set_world_seed(1)
        provider, store = make_provider(cfg)
        r = 3.0
        center = Vec3(8.0, 8.0, 8.0)  # interior of chunk (0,0,0)
        apply_brush(SphereBrush(r), center, BrushMode.ADD, chunk_provider=provider)
        total_solid = sum(int((ch.materials > 0).sum()) for ch in store.values())
        vs = cfg.voxel_size
        analytic = (4.0 / 3.0) * math.pi * r**3 / (vs**3)
        # Discretisation tolerance: ±12%
        assert abs(total_solid - analytic) / analytic < 0.12


class TestRoundTrip:
    def test_add_then_remove_round_trips(self, cfg):
        set_world_seed(1)
        provider, _store = make_provider(cfg)
        center = Vec3(8.0, 8.0, 8.0)
        # baseline copy
        ch = provider((0, 0, 0))
        before = ch.materials.copy()
        apply_brush(SphereBrush(2.5), center, BrushMode.ADD, chunk_provider=provider)
        assert not np.array_equal(ch.materials, before)
        apply_brush(SphereBrush(2.5), center, BrushMode.REMOVE, chunk_provider=provider)
        assert np.array_equal(ch.materials, before)


class TestMultiChunk:
    def test_brush_touches_correct_chunk_set(self, cfg):
        set_world_seed(1)
        provider, _store = make_provider(cfg)
        # Center on the boundary between chunks along X at world x=16 (border of
        # chunk 0 and 1), radius spanning into both. chunk_meters = 16.
        center = Vec3(16.0, 8.0, 8.0)
        touched = apply_brush(SphereBrush(3.0), center, BrushMode.ADD, chunk_provider=provider)
        # Sphere at x=16±3 spans chunk 0 (x∈[0,16)) and chunk 1 (x∈[16,32)).
        assert (0, 0, 0) in touched
        assert (1, 0, 0) in touched
        # Should NOT reach chunk 2 or negative-x chunks.
        assert (2, 0, 0) not in touched
        assert (-1, 0, 0) not in touched

    def test_corner_brush_eight_chunks(self, cfg):
        set_world_seed(1)
        provider, _store = make_provider(cfg)
        # Centre exactly on the 8-chunk corner at (16,16,16).
        center = Vec3(16.0, 16.0, 16.0)
        touched = apply_brush(SphereBrush(2.0), center, BrushMode.ADD, chunk_provider=provider)
        expected = {(cx, cy, cz) for cx in (0, 1) for cy in (0, 1) for cz in (0, 1)}
        assert expected.issubset(touched)


class TestEvents:
    def test_remove_sets_flags_and_publishes(self, cfg):
        set_world_seed(1)
        provider, _store = make_provider(cfg)
        # Pre-fill chunk so REMOVE actually changes voxels.
        ch = provider((0, 0, 0))
        ch.materials[:] = 1
        ch.edited = False
        ch.dirty = False

        bus = EventBus()
        events = []
        bus.subscribe(TerrainEditedEvent, lambda e: events.append(e))

        touched = apply_brush(
            SphereBrush(2.0),
            Vec3(8, 8, 8),
            BrushMode.REMOVE,
            chunk_provider=provider,
            bus=bus,
        )
        assert (0, 0, 0) in touched
        assert ch.edited is True
        assert ch.dirty is True
        assert len(events) == len(touched)
        assert all(isinstance(e, TerrainEditedEvent) for e in events)
        assert events[0].chunk_coords == (0, 0, 0)

    def test_no_event_when_no_change(self, cfg):
        """REMOVE on an empty chunk changes nothing → no flags, no event."""
        set_world_seed(1)
        provider, _store = make_provider(cfg)
        ch = provider((0, 0, 0))  # empty
        bus = EventBus()
        events = []
        bus.subscribe(TerrainEditedEvent, lambda e: events.append(e))
        touched = apply_brush(
            SphereBrush(2.0),
            Vec3(8, 8, 8),
            BrushMode.REMOVE,
            chunk_provider=provider,
            bus=bus,
        )
        assert touched == set()
        assert events == []
        assert ch.edited is False


class TestOtherShapes:
    def test_box_brush_extent(self, cfg):
        set_world_seed(1)
        provider, _store = make_provider(cfg)
        apply_brush(
            BoxBrush(Vec3(2.0, 2.0, 2.0)),
            Vec3(8, 8, 8),
            BrushMode.ADD,
            chunk_provider=provider,
        )
        ch = provider((0, 0, 0))
        count = int((ch.materials > 0).sum())
        # Box 4x4x4 m at vs=0.5 → ~8x8x8 = 512 voxels (centres within ±2.0 m).
        # voxel centres span ±2.0 inclusive → 9 along each axis (-2..+2 step .5)
        # Actually centres at 6.0..10.0 inclusive within 8±2 → 9^3 = 729.
        assert 500 <= count <= 800

    def test_cylinder_brush(self, cfg):
        set_world_seed(1)
        provider, _store = make_provider(cfg)
        touched = apply_brush(
            CylinderBrush(radius_m=2.0, height_m=4.0),
            Vec3(8, 8, 8),
            BrushMode.ADD,
            chunk_provider=provider,
        )
        ch = provider((0, 0, 0))
        count = int((ch.materials > 0).sum())
        # Cylinder volume π r² h / vs³ ≈ π*4*4 / 0.125 ≈ 402 voxels.
        analytic = math.pi * 2.0**2 * 4.0 / (cfg.voxel_size**3)
        assert abs(count - analytic) / analytic < 0.2
        assert (0, 0, 0) in touched
