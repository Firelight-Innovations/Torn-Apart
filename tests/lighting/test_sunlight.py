"""
tests/lighting/test_sunlight.py — Headless tests for fire_engine.lighting.sunlight.

Covers:
- make_light_sampler: empty grid fallback, fully lit chunk, shadowed region.
- SunlightComputer column pass: empty column → LIGHT_FULL, solid layer → shadow.
- SunlightComputer events: TerrainEditedEvent / ChunkLoadedEvent trigger recompute.
- Determinism: same materials → byte-identical light arrays.

No panda3d imports.
"""

from __future__ import annotations

import numpy as np

from fire_engine.core import ChunkLoadedEvent, EventBus, TerrainEditedEvent, load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.lighting.light_grid import LIGHT_AMBIENT, LIGHT_FULL, LightGrid
from fire_engine.lighting.sunlight import SunlightComputer, make_light_sampler
from fire_engine.world.terrain.chunk import Chunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config():
    return load_config()


class _FakeChunkProvider:
    def __init__(self, chunks_dict=None):
        self.chunks = chunks_dict or {}


# ---------------------------------------------------------------------------
# make_light_sampler
# ---------------------------------------------------------------------------


class TestMakeLightSampler:
    def test_no_light_data_returns_full_bright(self):
        cfg = _config()
        lg = LightGrid()
        sampler = make_light_sampler(lg, cfg)
        positions = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
        result = sampler(positions)
        assert result.shape == (1,)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, [1.0])

    def test_empty_positions_returns_empty(self):
        cfg = _config()
        lg = LightGrid()
        sampler = make_light_sampler(lg, cfg)
        positions = np.empty((0, 3), dtype=np.float32)
        result = sampler(positions)
        assert result.shape == (0,)
        assert result.dtype == np.float32

    def test_manual_grid_lookup_correctness(self):
        """Known light array values must be returned at the right world positions."""
        cfg = _config()
        lg = LightGrid()
        arr = np.zeros((16, 16, 16), dtype=np.uint8)
        arr[:, :, 0] = 40
        arr[:, :, 8] = 128
        arr[:, :, 15] = 255
        lg.set((0, 0, 0), arr)
        sampler = make_light_sampler(lg, cfg)
        positions = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 8.5], [0.5, 0.5, 15.5]], dtype=np.float32)
        result = sampler(positions)
        np.testing.assert_allclose(result[0], 40.0 / 255.0, atol=1e-5)
        np.testing.assert_allclose(result[1], 128.0 / 255.0, atol=1e-5)
        np.testing.assert_allclose(result[2], 255.0 / 255.0, atol=1e-5)

    def test_positions_in_different_chunks(self):
        cfg = _config()
        lg = LightGrid()
        arr0 = np.full((16, 16, 16), LIGHT_AMBIENT, dtype=np.uint8)
        arr1 = np.full((16, 16, 16), LIGHT_FULL, dtype=np.uint8)
        lg.set((0, 0, 0), arr0)
        lg.set((1, 0, 0), arr1)
        sampler = make_light_sampler(lg, cfg)
        positions = np.array([[8.0, 8.0, 8.0], [24.0, 8.0, 8.0]], dtype=np.float32)
        result = sampler(positions)
        np.testing.assert_allclose(result[0], LIGHT_AMBIENT / 255.0, atol=1e-5)
        np.testing.assert_allclose(result[1], LIGHT_FULL / 255.0, atol=1e-5)


# ---------------------------------------------------------------------------
# SunlightComputer: column pass
# ---------------------------------------------------------------------------


class TestColumnPass:
    def _make_computer(self, chunks_dict):
        cfg = _config()
        bus = EventBus()
        lg = LightGrid()
        provider = _FakeChunkProvider(chunks_dict)
        sc = SunlightComputer(cfg, provider, lg, bus)
        return sc, lg, bus

    def test_empty_column_all_full(self):
        chunk = Chunk((0, 0, 0))
        sc, lg, _ = self._make_computer({(0, 0, 0): chunk})
        sc.recompute_column(0, 0)
        arr = lg.get((0, 0, 0))
        assert arr is not None
        assert arr.shape == (16, 16, 16)
        assert (arr == LIGHT_FULL).all()

    def test_solid_top_half_shadows_bottom(self):
        mat = np.zeros((32, 32, 32), dtype=np.uint8)
        mat[:, :, 16:] = 1  # solid top 16 voxels = cells z=8..15
        chunk = Chunk((0, 0, 0), mat)
        sc, lg, _ = self._make_computer({(0, 0, 0): chunk})
        sc.recompute_column(0, 0)
        arr = lg.get((0, 0, 0))
        # Bottommost cells (z=0,1) are deep in shadow.
        assert arr[:, :, 0].max() == LIGHT_AMBIENT
        # Topmost cells just below solid at z=15 also shadowed.
        assert arr[:, :, 15].max() == LIGHT_AMBIENT

    def test_full_air_column_stays_fully_lit(self):
        chunk = Chunk((0, 0, 0))  # all air
        sc, lg, _ = self._make_computer({(0, 0, 0): chunk})
        sc.recompute_column(0, 0)
        arr = lg.get((0, 0, 0))
        assert (arr == LIGHT_FULL).all()


# ---------------------------------------------------------------------------
# SunlightComputer event handlers
# ---------------------------------------------------------------------------


class TestEventHandlers:
    def test_terrain_edited_event_triggers_recompute(self):
        cfg = _config()
        bus = EventBus()
        lg = LightGrid()
        chunk = Chunk((0, 0, 0))
        provider = _FakeChunkProvider({(0, 0, 0): chunk})
        SunlightComputer(cfg, provider, lg, bus)
        chunk.dirty = False

        class _FakeBrush:
            pass

        bus.publish(TerrainEditedEvent(chunk_coords=(0, 0, 0), brush=_FakeBrush()))
        assert lg.get((0, 0, 0)) is not None
        assert chunk.dirty

    def test_chunk_loaded_event_triggers_recompute(self):
        cfg = _config()
        bus = EventBus()
        lg = LightGrid()
        chunk = Chunk((0, 0, 2))
        provider = _FakeChunkProvider({(0, 0, 2): chunk})
        SunlightComputer(cfg, provider, lg, bus)
        bus.publish(ChunkLoadedEvent(coord=(0, 0, 2)))
        assert lg.get((0, 0, 2)) is not None


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_materials_same_light(self):
        set_world_seed(42)
        mat = np.zeros((32, 32, 32), dtype=np.uint8)
        mat[:, :, 12:20] = 1

        def _compute():
            cfg = _config()
            bus = EventBus()
            lg = LightGrid()
            chunk = Chunk((0, 0, 0), mat.copy())
            provider = _FakeChunkProvider({(0, 0, 0): chunk})
            sc = SunlightComputer(cfg, provider, lg, bus)
            sc.recompute_column(0, 0)
            return lg.get((0, 0, 0)).copy()

        arr1 = _compute()
        arr2 = _compute()
        np.testing.assert_array_equal(arr1, arr2, err_msg="Determinism violated")
