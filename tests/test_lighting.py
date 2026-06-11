"""
tests/test_lighting.py — Headless tests for the lighting package (Phase 4).

No panda3d imports anywhere in this file.  All tests operate on numpy arrays,
Chunk objects, and the lighting API directly.

Test coverage
-------------
- occupancy_from_materials: known downsampling cases.
- LightGrid store: set/get/has_valid/invalidate/remove.
- SunlightComputer column pass: empty column → all 255.
- SunlightComputer column pass: occupied layer → correct shadow boundary.
- Box-blur: range conservation, shape preservation, penumbra (gradient) appears.
- Event invalidation: TerrainEditedEvent triggers recompute + dirty flags.
- Determinism: same materials → byte-identical light arrays.
- make_light_sampler: face centres in lit / shadowed regions return correct values.
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.core import EventBus, TerrainEditedEvent, ChunkLoadedEvent, load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.terrain.chunk import Chunk
from fire_engine.lighting import (
    LightGrid,
    occupancy_from_materials,
    SunlightComputer,
    make_light_sampler,
    LIGHT_FULL,
    LIGHT_AMBIENT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(coord, solid_z_range=None):
    """Return a Chunk with optional solid fill in a Z range (local voxel indices)."""
    c = Chunk(coord)
    if solid_z_range is not None:
        z_lo, z_hi = solid_z_range
        c.materials[:, :, z_lo:z_hi] = 1
    return c


def _config():
    return load_config()


class _FakeChunkProvider:
    """Minimal chunk provider: a dict of Chunk objects."""

    def __init__(self, chunks_dict=None):
        self.chunks = chunks_dict or {}


# ---------------------------------------------------------------------------
# occupancy_from_materials
# ---------------------------------------------------------------------------

class TestOccupancyFromMaterials:
    def test_all_air_returns_all_false(self):
        mat = np.zeros((32, 32, 32), dtype=np.uint8)
        occ = occupancy_from_materials(mat)
        assert occ.shape == (16, 16, 16)
        assert occ.dtype == bool
        assert not occ.any()

    def test_all_solid_returns_all_true(self):
        mat = np.ones((32, 32, 32), dtype=np.uint8)
        occ = occupancy_from_materials(mat)
        assert occ.all()

    def test_single_voxel_occupies_correct_cell(self):
        mat = np.zeros((32, 32, 32), dtype=np.uint8)
        # Voxels [0,0,0] and [1,1,1] share light cell (0,0,0).
        mat[0, 0, 0] = 1
        occ = occupancy_from_materials(mat)
        assert occ[0, 0, 0] == True
        # All other cells must be False.
        other = occ.copy()
        other[0, 0, 0] = False
        assert not other.any()

    def test_voxel_maps_to_correct_cell(self):
        mat = np.zeros((32, 32, 32), dtype=np.uint8)
        # Voxel (2, 4, 6) lives in light cell (1, 2, 3).
        mat[2, 4, 6] = 1
        occ = occupancy_from_materials(mat)
        assert occ[1, 2, 3] == True
        # Neighboring cell should be False.
        assert occ[0, 2, 3] == False

    def test_two_voxels_in_same_cell(self):
        mat = np.zeros((32, 32, 32), dtype=np.uint8)
        mat[0, 0, 0] = 1
        mat[1, 1, 1] = 1
        occ = occupancy_from_materials(mat)
        assert occ[0, 0, 0] == True
        other = occ.copy()
        other[0, 0, 0] = False
        assert not other.any()

    def test_voxel_at_boundary_correct_cell(self):
        mat = np.zeros((32, 32, 32), dtype=np.uint8)
        mat[30, 30, 30] = 1  # should land in cell (15, 15, 15)
        occ = occupancy_from_materials(mat)
        assert occ[15, 15, 15] == True
        other = occ.copy()
        other[15, 15, 15] = False
        assert not other.any()


# ---------------------------------------------------------------------------
# LightGrid store
# ---------------------------------------------------------------------------

class TestLightGrid:
    def test_get_returns_none_before_set(self):
        lg = LightGrid()
        assert lg.get((0, 0, 0)) is None

    def test_has_valid_false_before_set(self):
        lg = LightGrid()
        assert not lg.has_valid((0, 0, 0))

    def test_set_and_get(self):
        lg = LightGrid()
        arr = np.full((16, 16, 16), 200, dtype=np.uint8)
        lg.set((1, 2, 3), arr)
        got = lg.get((1, 2, 3))
        assert got is arr   # same object stored
        assert lg.has_valid((1, 2, 3))

    def test_invalidate(self):
        lg = LightGrid()
        arr = np.full((16, 16, 16), 100, dtype=np.uint8)
        lg.set((0, 0, 0), arr)
        lg.invalidate((0, 0, 0))
        assert not lg.has_valid((0, 0, 0))
        assert lg.get((0, 0, 0)) is arr  # array is still accessible

    def test_remove(self):
        lg = LightGrid()
        arr = np.full((16, 16, 16), 50, dtype=np.uint8)
        lg.set((0, 0, 0), arr)
        lg.remove((0, 0, 0))
        assert lg.get((0, 0, 0)) is None
        assert not lg.has_valid((0, 0, 0))

    def test_loaded_coords(self):
        lg = LightGrid()
        lg.set((0, 0, 0), np.zeros((16, 16, 16), dtype=np.uint8))
        lg.set((1, 0, 0), np.zeros((16, 16, 16), dtype=np.uint8))
        coords = lg.loaded_coords()
        assert set(coords) == {(0, 0, 0), (1, 0, 0)}


# ---------------------------------------------------------------------------
# SunlightComputer — column pass
# ---------------------------------------------------------------------------

class TestColumnPass:

    def _make_computer(self, chunks_dict):
        cfg = _config()
        bus = EventBus()
        lg = LightGrid()
        provider = _FakeChunkProvider(chunks_dict)
        sc = SunlightComputer(cfg, provider, lg, bus)
        return sc, lg, bus, provider

    def test_empty_column_all_full(self):
        """A chunk with all-air materials → every light cell should be LIGHT_FULL."""
        chunk = Chunk((0, 0, 0))
        # All materials are 0 (air) by default.
        sc, lg, _, _ = self._make_computer({(0, 0, 0): chunk})
        sc.recompute_column(0, 0)
        arr = lg.get((0, 0, 0))
        assert arr is not None
        assert arr.shape == (16, 16, 16)
        assert (arr == LIGHT_FULL).all(), f"Expected all {LIGHT_FULL}, got min={arr.min()}"

    def test_solid_bottom_half_lit_top_half(self):
        """
        A chunk filled solid in the bottom half (z<16 voxels → z<8 light cells)
        and air above → top half should be LIGHT_FULL, bottom shadowed.
        """
        mat = np.zeros((32, 32, 32), dtype=np.uint8)
        mat[:, :, :16] = 1  # solid bottom 16 voxels = 8 light cells
        chunk = Chunk((0, 0, 0), mat)
        sc, lg, _, _ = self._make_computer({(0, 0, 0): chunk})
        sc.recompute_column(0, 0)
        arr = lg.get((0, 0, 0))
        assert arr is not None
        # Without blur the boundary is sharp: cells 0..7 are occupied (shadowed),
        # cells 8..15 are air above (but occupancy of lower half casts shadow up).
        # Actually: the solid is in cells 0..7 (z=0..7 light coords).
        # The column-pass casts shadow DOWNWARD from the first occupied cell.
        # Cells 8..15 are ABOVE the solid (no occupancy above them) → LIGHT_FULL.
        # Cells 0..7 have occupancy AT that level → LIGHT_AMBIENT.
        # After blur, the boundary may become a gradient — so just check:
        # - cells 15 (topmost) must be LIGHT_FULL (no solid above, ambient floor preserved).
        # - cells 0 (bottommost) must be LIGHT_AMBIENT (solid present at same level).
        assert arr[:, :, 15].min() == LIGHT_FULL  # topmost light cells → full sun
        # Cell 0 is entirely shadowed: must be exactly LIGHT_AMBIENT (no one above to blur in more).
        assert arr[:, :, 0].max() == LIGHT_AMBIENT  # bottommost → only ambient

    def test_single_occupied_layer_boundary(self):
        """
        Single solid layer at z light cells = 8 (mid-chunk).
        Cells 9..15 (above) → LIGHT_FULL. Cells 0..7 (below) → LIGHT_AMBIENT.
        Cell 8 itself → LIGHT_AMBIENT (occupied → shadowed).
        After blur the boundary gradient should contain intermediate values.
        """
        mat = np.zeros((32, 32, 32), dtype=np.uint8)
        # Light cell 8 = voxels 16..17 (z).
        mat[:, :, 16:18] = 1   # fill voxels z=16,17 → light cell z=8 occupied
        chunk = Chunk((0, 0, 0), mat)
        sc, lg, _, _ = self._make_computer({(0, 0, 0): chunk})
        sc.recompute_column(0, 0)
        arr = lg.get((0, 0, 0))

        # Topmost cells (14, 15) have no occupancy at or above → LIGHT_FULL after blur.
        # The blur only blends ±1 cell, so cells far from the boundary remain pure.
        top_vals = arr[:, :, 14:16]
        assert top_vals.min() == LIGHT_FULL, (
            f"Expected LIGHT_FULL at top, got {top_vals.min()}"
        )

        # Bottommost cells (0, 1) are deep in shadow → LIGHT_AMBIENT.
        bot_vals = arr[:, :, 0:2]
        assert bot_vals.max() == LIGHT_AMBIENT, (
            f"Expected LIGHT_AMBIENT at bottom, got {bot_vals.max()}"
        )

    def test_stacked_chunks_shadow_propagates_down(self):
        """
        Two vertically stacked chunks: top chunk has a solid mid-layer.
        Bottom chunk cells should be shadowed (ambient).
        """
        # Top chunk (0,0,1): solid at light cells z=0..1 (voxels z=0..3 of this chunk).
        mat_top = np.zeros((32, 32, 32), dtype=np.uint8)
        mat_top[:, :, 0:4] = 1  # voxels 0-3 = light cells 0-1 of the top chunk
        top_chunk = Chunk((0, 0, 1), mat_top)

        # Bottom chunk (0,0,0): all air.
        bot_chunk = Chunk((0, 0, 0))

        chunks = {(0, 0, 0): bot_chunk, (0, 0, 1): top_chunk}
        sc, lg, _, _ = self._make_computer(chunks)
        sc.recompute_column(0, 0)

        arr_bot = lg.get((0, 0, 0))
        assert arr_bot is not None
        # Entire bottom chunk is below the solid layer → LIGHT_AMBIENT everywhere.
        assert arr_bot.max() == LIGHT_AMBIENT, (
            f"Bottom chunk should be all ambient, max={arr_bot.max()}"
        )


# ---------------------------------------------------------------------------
# Box blur
# ---------------------------------------------------------------------------

class TestBoxBlur:

    def _compute_blurred(self, mat):
        cfg = _config()
        bus = EventBus()
        lg = LightGrid()
        chunk = Chunk((0, 0, 0), mat)
        provider = _FakeChunkProvider({(0, 0, 0): chunk})
        sc = SunlightComputer(cfg, provider, lg, bus)
        sc.recompute_column(0, 0)
        return lg.get((0, 0, 0))

    def test_range_conservation(self):
        """After blur, values stay within [LIGHT_AMBIENT, LIGHT_FULL]."""
        mat = np.zeros((32, 32, 32), dtype=np.uint8)
        mat[:, :, 16:18] = 1  # sharp shadow boundary
        arr = self._compute_blurred(mat)
        assert arr.min() >= LIGHT_AMBIENT
        assert arr.max() <= LIGHT_FULL

    def test_shape_preserved(self):
        """Output shape is (16, 16, 16)."""
        mat = np.zeros((32, 32, 32), dtype=np.uint8)
        mat[:, :, 10:20] = 1
        arr = self._compute_blurred(mat)
        assert arr.shape == (16, 16, 16)

    def test_blur_produces_gradient_at_boundary(self):
        """
        A sharp light/shadow boundary should show intermediate values in the
        cells adjacent to the boundary (penumbra).
        """
        mat = np.zeros((32, 32, 32), dtype=np.uint8)
        # Solid at light cells z=7,8 (voxels 14-17) — sharp boundary between 8 and 9.
        mat[:, :, 14:18] = 1
        arr = self._compute_blurred(mat)

        # Look for an intermediate value: strictly between AMBIENT and FULL.
        has_intermediate = (
            (arr > LIGHT_AMBIENT) & (arr < LIGHT_FULL)
        ).any()
        assert has_intermediate, (
            f"Expected a gradient (intermediate value between {LIGHT_AMBIENT} "
            f"and {LIGHT_FULL}) but got unique values: {np.unique(arr)}"
        )

    def test_fully_lit_region_stays_full(self):
        """Top cells with no solid above stay at LIGHT_FULL after blur."""
        mat = np.zeros((32, 32, 32), dtype=np.uint8)
        mat[:, :, 0:4] = 1  # solid only at bottom
        arr = self._compute_blurred(mat)
        # Topmost cells, far from boundary, should still be LIGHT_FULL.
        assert arr[:, :, 15].min() == LIGHT_FULL


# ---------------------------------------------------------------------------
# Event invalidation
# ---------------------------------------------------------------------------

class TestEventInvalidation:

    def test_terrain_edited_event_triggers_recompute_and_dirty(self):
        """
        Publishing a TerrainEditedEvent for a chunk should recompute its column
        light and mark the chunk dirty.
        """
        cfg = _config()
        bus = EventBus()
        lg = LightGrid()
        chunk = Chunk((2, 3, 0))  # all air initially
        provider = _FakeChunkProvider({(2, 3, 0): chunk})
        sc = SunlightComputer(cfg, provider, lg, bus)

        # Manually clear dirty flag to test that it gets re-set.
        chunk.dirty = False

        # Simulate a terrain edit by publishing a TerrainEditedEvent.
        class _FakeBrush:
            pass
        bus.publish(TerrainEditedEvent(chunk_coords=(2, 3, 0), brush=_FakeBrush()))

        # Light should now be computed for this chunk.
        arr = lg.get((2, 3, 0))
        assert arr is not None, "Light array should be computed after terrain edit"

        # Chunk should be marked dirty for remeshing.
        assert chunk.dirty, "Chunk should be marked dirty after terrain edit recompute"

    def test_chunk_loaded_event_triggers_recompute(self):
        """
        Publishing a ChunkLoadedEvent should cause the computer to compute
        light for that chunk's column.
        """
        cfg = _config()
        bus = EventBus()
        lg = LightGrid()
        chunk = Chunk((0, 0, 2))
        provider = _FakeChunkProvider({(0, 0, 2): chunk})
        sc = SunlightComputer(cfg, provider, lg, bus)

        bus.publish(ChunkLoadedEvent(coord=(0, 0, 2)))

        arr = lg.get((0, 0, 2))
        assert arr is not None, "Light should be computed after ChunkLoadedEvent"

    def test_terrain_edit_marks_all_column_chunks_dirty(self):
        """
        Editing a chunk should mark ALL chunks in the same (cx,cy) column dirty.
        """
        cfg = _config()
        bus = EventBus()
        lg = LightGrid()
        chunk0 = Chunk((1, 1, 0))
        chunk1 = Chunk((1, 1, 1))
        chunk2 = Chunk((1, 1, 2))
        provider = _FakeChunkProvider({
            (1, 1, 0): chunk0,
            (1, 1, 1): chunk1,
            (1, 1, 2): chunk2,
        })
        sc = SunlightComputer(cfg, provider, lg, bus)

        # Clear dirty flags.
        chunk0.dirty = False
        chunk1.dirty = False
        chunk2.dirty = False

        class _FakeBrush:
            pass
        # Edit chunk1 — should dirty all three in the column.
        bus.publish(TerrainEditedEvent(chunk_coords=(1, 1, 1), brush=_FakeBrush()))

        assert chunk0.dirty, "chunk0 should be dirty (same column)"
        assert chunk1.dirty, "chunk1 should be dirty (edited + same column)"
        assert chunk2.dirty, "chunk2 should be dirty (same column)"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_same_materials_same_light(self):
        """Same materials array → byte-identical light output."""
        set_world_seed(42)
        mat = np.zeros((32, 32, 32), dtype=np.uint8)
        mat[:, :, 12:20] = 1
        mat[5:10, 5:10, 8:12] = 1

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

    def test_different_materials_different_light(self):
        """Different materials → different light arrays (with high probability)."""
        cfg = _config()
        bus1, bus2 = EventBus(), EventBus()
        lg1, lg2 = LightGrid(), LightGrid()

        mat1 = np.zeros((32, 32, 32), dtype=np.uint8)
        mat1[:, :, 16:20] = 1

        mat2 = np.zeros((32, 32, 32), dtype=np.uint8)
        mat2[:, :, 0:4] = 1

        prov1 = _FakeChunkProvider({(0, 0, 0): Chunk((0, 0, 0), mat1)})
        prov2 = _FakeChunkProvider({(0, 0, 0): Chunk((0, 0, 0), mat2)})

        SunlightComputer(cfg, prov1, lg1, bus1).recompute_column(0, 0)
        SunlightComputer(cfg, prov2, lg2, bus2).recompute_column(0, 0)

        arr1 = lg1.get((0, 0, 0))
        arr2 = lg2.get((0, 0, 0))
        assert not (arr1 == arr2).all(), "Different materials should yield different light"


# ---------------------------------------------------------------------------
# make_light_sampler
# ---------------------------------------------------------------------------

class TestMakeLightSampler:

    def _fully_lit_setup(self):
        """Return a sampler backed by a chunk that is all-air (full sun)."""
        cfg = _config()
        bus = EventBus()
        lg = LightGrid()
        chunk = Chunk((0, 0, 0))  # all air
        provider = _FakeChunkProvider({(0, 0, 0): chunk})
        sc = SunlightComputer(cfg, provider, lg, bus)
        sc.recompute_column(0, 0)
        sampler = make_light_sampler(lg, cfg)
        return sampler, cfg

    def test_no_light_data_returns_full_bright(self):
        """Positions in a chunk with no computed light → default full bright (1.0)."""
        cfg = _config()
        lg = LightGrid()  # empty — no data
        sampler = make_light_sampler(lg, cfg)
        positions = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
        result = sampler(positions)
        assert result.shape == (1,)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, [1.0])

    def test_fully_lit_chunk_returns_near_one(self):
        """Face centres in a fully lit (all air) chunk → light ≈ 1.0."""
        sampler, cfg = self._fully_lit_setup()
        # Face centres scattered across chunk (0,0,0): world coords [0, 16) m.
        positions = np.array([
            [0.5, 0.5, 15.5],   # near top of chunk
            [8.0, 8.0, 8.0],    # mid chunk
            [15.5, 15.5, 0.5],  # near bottom of chunk
        ], dtype=np.float32)
        result = sampler(positions)
        assert result.shape == (3,)
        np.testing.assert_allclose(result, [1.0, 1.0, 1.0], atol=1e-6)

    def test_shadowed_region_returns_near_ambient(self):
        """Face centres deep in shadow (no blur influence) → light ≈ LIGHT_AMBIENT/255."""
        cfg = _config()
        bus = EventBus()
        lg = LightGrid()
        # Solid entire top of chunk to cast a deep shadow on lower cells.
        mat = np.zeros((32, 32, 32), dtype=np.uint8)
        mat[:, :, 28:32] = 1   # solid top 4 voxels = top 2 light cells
        chunk = Chunk((0, 0, 0), mat)
        provider = _FakeChunkProvider({(0, 0, 0): chunk})
        sc = SunlightComputer(cfg, provider, lg, bus)
        sc.recompute_column(0, 0)
        sampler = make_light_sampler(lg, cfg)

        # Deep shadow at the bottom: light cells z=0,1 (far from the blur boundary).
        positions = np.array([
            [0.5, 0.5, 0.5],   # world Z 0.5 m → light cell z=0 of chunk (0,0,0)
        ], dtype=np.float32)
        result = sampler(positions)
        expected = float(LIGHT_AMBIENT) / 255.0
        np.testing.assert_allclose(result, [expected], atol=0.05)

    def test_empty_positions(self):
        """Empty input array → empty output array."""
        cfg = _config()
        lg = LightGrid()
        sampler = make_light_sampler(lg, cfg)
        positions = np.empty((0, 3), dtype=np.float32)
        result = sampler(positions)
        assert result.shape == (0,)
        assert result.dtype == np.float32

    def test_positions_map_to_correct_cells(self):
        """
        Verify the cell lookup formula for a known configuration:
        place a manual light array and check specific positions sample the right cell.
        """
        cfg = _config()
        lg = LightGrid()
        # Set a custom array for chunk (0,0,0): gradient across Z light cells.
        arr = np.zeros((16, 16, 16), dtype=np.uint8)
        arr[:, :, 0] = 40    # bottom cell
        arr[:, :, 8] = 128   # mid cell
        arr[:, :, 15] = 255  # top cell
        lg.set((0, 0, 0), arr)

        sampler = make_light_sampler(lg, cfg)

        # chunk (0,0,0) world origin = (0, 0, 0).
        # light_cell_meters = 1.0 m, so cell z = floor(world_z / 1.0).
        # Cell z=0: world Z in [0, 1). Test at Z=0.5.
        # Cell z=8: world Z in [8, 9). Test at Z=8.5.
        # Cell z=15: world Z in [15, 16). Test at Z=15.5.
        positions = np.array([
            [0.5, 0.5, 0.5],    # cell (0,0,0) → value 40
            [0.5, 0.5, 8.5],    # cell (0,0,8) → value 128
            [0.5, 0.5, 15.5],   # cell (0,0,15) → value 255
        ], dtype=np.float32)
        result = sampler(positions)

        np.testing.assert_allclose(result[0], 40.0 / 255.0, atol=1e-5)
        np.testing.assert_allclose(result[1], 128.0 / 255.0, atol=1e-5)
        np.testing.assert_allclose(result[2], 255.0 / 255.0, atol=1e-5)

    def test_positions_in_different_chunks(self):
        """Positions spanning two chunks are dispatched to the correct arrays."""
        cfg = _config()
        lg = LightGrid()

        arr0 = np.full((16, 16, 16), LIGHT_AMBIENT, dtype=np.uint8)
        arr1 = np.full((16, 16, 16), LIGHT_FULL, dtype=np.uint8)
        lg.set((0, 0, 0), arr0)
        lg.set((1, 0, 0), arr1)

        sampler = make_light_sampler(lg, cfg)

        # Chunk (0,0,0) world X in [0,16). Centre → X=8.
        # Chunk (1,0,0) world X in [16,32). Centre → X=24.
        positions = np.array([
            [8.0, 8.0, 8.0],   # chunk (0,0,0) → LIGHT_AMBIENT
            [24.0, 8.0, 8.0],  # chunk (1,0,0) → LIGHT_FULL
        ], dtype=np.float32)
        result = sampler(positions)

        np.testing.assert_allclose(result[0], LIGHT_AMBIENT / 255.0, atol=1e-5)
        np.testing.assert_allclose(result[1], LIGHT_FULL / 255.0, atol=1e-5)

    def test_output_dtype_and_range(self):
        """Output is float32 and values are in [0, 1]."""
        sampler, cfg = self._fully_lit_setup()
        positions = np.random.default_rng(0).uniform(0, 16, (100, 3)).astype(np.float32)
        result = sampler(positions)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0
