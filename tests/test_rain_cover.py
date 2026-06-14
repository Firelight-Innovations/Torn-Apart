"""
tests/test_rain_cover.py — Headless rain-cover heightmap (terrain/rain_cover.py).

Validates the top-down highest-solid-voxel reduction that the M6 volumetric
rain renderer samples to discard rain under roofs:

* a roof slab over part of the grid raises those columns' cover height to the
  roof top Z while open columns stay at the ground (or the open-sky sentinel);
* removing the roof (a TerrainEditedEvent-style re-fold) lowers the height;
* recentering shifts the window correctly;
* the vectorised column reduction matches a brute-force reference;
* determinism (same chunks → byte-identical heightmap).

Headless: no panda3d imports.
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.core import load_config
from fire_engine.world.terrain.chunk import Chunk
from fire_engine.world.terrain.rain_cover import RainCoverField, OPEN_SKY_Z


@pytest.fixture
def cfg():
    # Small window so fixtures stay cheap and assertions are easy to reason about.
    base = load_config()
    from dataclasses import replace
    return replace(base, rain_cover_cells=64, rain_cover_cell_m=1.0)


VS = 0.5          # voxel size (m)
N = 32            # chunk edge (voxels)
CHUNK_M = N * VS  # 16 m


def _voxel_top_z(cz: int, z_idx: int) -> float:
    """World Z of the TOP face of voxel z_idx in chunk-Z layer cz."""
    return cz * CHUNK_M + (z_idx + 1) * VS


# ---------------------------------------------------------------------------
# Brute-force reference for the per-chunk top-solid reduction
# ---------------------------------------------------------------------------

def _brute_top_z(chunk: Chunk) -> np.ndarray:
    """Reference (Python-loop) highest-solid top-face Z per [x, y] column."""
    solid = chunk.materials > 0
    out = np.full((N, N), OPEN_SKY_Z, dtype=np.float32)
    origin_z = chunk.coord[2] * CHUNK_M
    for x in range(N):
        for y in range(N):
            zs = np.where(solid[x, y, :])[0]
            if zs.size:
                out[x, y] = origin_z + (int(zs.max()) + 1) * VS
    return out


# ===========================================================================
# Per-chunk reduction correctness
# ===========================================================================

class TestColumnReduction:
    def test_matches_brute_force(self, cfg):
        rng = np.random.default_rng(7)
        mats = (rng.random((N, N, N)) < 0.3).astype(np.uint8)
        chunk = Chunk((0, 0, 0), mats)
        field = RainCoverField(cfg)
        got = field._chunk_top_z(chunk)
        assert got is not None
        ref = _brute_top_z(chunk)
        assert np.array_equal(got, ref)

    def test_all_air_returns_none(self, cfg):
        chunk = Chunk((0, 0, 0), np.zeros((N, N, N), dtype=np.uint8))
        assert RainCoverField(cfg)._chunk_top_z(chunk) is None


# ===========================================================================
# Roof over the grid: covered columns = roof top, open columns = ground
# ===========================================================================

class TestRoofCover:
    def _floor_chunk(self) -> Chunk:
        # Solid bottom 4 voxels (ground), spanning the whole chunk footprint.
        mats = np.zeros((N, N, N), dtype=np.uint8)
        mats[:, :, 0:4] = 1
        return Chunk((0, 0, 0), mats)

    def _roof_chunk(self) -> Chunk:
        # A roof slab one voxel thick at z=20 over HALF the footprint (x < 16).
        mats = np.zeros((N, N, N), dtype=np.uint8)
        mats[0:16, :, 20] = 1
        return Chunk((0, 0, 1), mats)   # chunk-Z layer 1 → world Z 16..32 m

    def test_roof_raises_covered_columns(self, cfg):
        field = RainCoverField(cfg)
        field.recenter((CHUNK_M * 0.5, CHUNK_M * 0.5))   # window over chunk (0,0)
        chunks = {(0, 0, 0): self._floor_chunk(), (0, 0, 1): self._roof_chunk()}
        field.rebuild_all(chunks)

        ground_z = _voxel_top_z(0, 3)            # top of the floor slab (2.0 m)
        roof_z = _voxel_top_z(1, 20)             # top of the roof slab

        # Map a world XY to a texel.
        def texel(wx, wy):
            ox, oy = field.origin_m
            col = int(np.floor((wx - ox) / field.cell_m))
            row = int(np.floor((wy - oy) / field.cell_m))
            return row, col

        # A covered column (x world ~ 4 m, under the roof half x<16 voxels → x<8 m).
        r, c = texel(4.0, 8.0)
        assert field.height[r, c] == pytest.approx(roof_z)
        # An open column (x world ~ 12 m → voxel x≈24, no roof) sees the ground.
        r2, c2 = texel(12.0, 8.0)
        assert field.height[r2, c2] == pytest.approx(ground_z)

    def test_removing_roof_lowers_height(self, cfg):
        field = RainCoverField(cfg)
        field.recenter((CHUNK_M * 0.5, CHUNK_M * 0.5))
        floor = self._floor_chunk()
        roof = self._roof_chunk()
        chunks = {(0, 0, 0): floor, (0, 0, 1): roof}
        field.rebuild_all(chunks)

        def texel(wx, wy):
            ox, oy = field.origin_m
            return (int(np.floor((wy - oy) / field.cell_m)),
                    int(np.floor((wx - ox) / field.cell_m)))

        r, c = texel(4.0, 8.0)
        assert field.height[r, c] == pytest.approx(_voxel_top_z(1, 20))

        # "Edit" removes the roof: clear that chunk's voxels and re-fold the column.
        roof.materials[...] = 0
        field.rebuild_columns(chunks, [(0, 0)])
        assert field.height[r, c] == pytest.approx(_voxel_top_z(0, 3))   # back to ground


# ===========================================================================
# Recenter shifts the window
# ===========================================================================

class TestRecenter:
    def test_recenter_shifts_origin_and_data(self, cfg):
        # A floor slab spanning a 3x3 chunk area so both windows have ground.
        chunks = {}
        for cx in range(-1, 2):
            for cy in range(-1, 2):
                mats = np.zeros((N, N, N), dtype=np.uint8)
                mats[:, :, 0:4] = 1
                chunks[(cx, cy, 0)] = Chunk((cx, cy, 0), mats)

        field = RainCoverField(cfg)
        field.recenter((0.0, 0.0))
        o0 = field.origin_m
        field.rebuild_all(chunks)
        ground_z = _voxel_top_z(0, 3)
        # Center texel is over the floor → ground height.
        mid = cfg.rain_cover_cells // 2
        assert field.height[mid, mid] == pytest.approx(ground_z)

        # Recenter elsewhere → origin moves by the requested delta (snapped).
        field.recenter((20.0, -10.0))
        o1 = field.origin_m
        assert o1[0] > o0[0]
        assert o1[1] < o0[1]
        # Origin stays snapped to the cell grid.
        assert o1[0] % field.cell_m == pytest.approx(0.0)
        assert o1[1] % field.cell_m == pytest.approx(0.0)
        field.rebuild_all(chunks)
        assert field.height[mid, mid] == pytest.approx(ground_z)


# ===========================================================================
# Determinism
# ===========================================================================

class TestDeterminism:
    def test_same_chunks_identical_heightmap(self, cfg):
        rng = np.random.default_rng(99)
        chunks = {}
        for cz in (0, 1):
            mats = (rng.random((N, N, N)) < 0.25).astype(np.uint8)
            chunks[(0, 0, cz)] = Chunk((0, 0, cz), mats)

        a = RainCoverField(cfg)
        a.recenter((CHUNK_M * 0.5, CHUNK_M * 0.5))
        a.rebuild_all(chunks)

        b = RainCoverField(cfg)
        b.recenter((CHUNK_M * 0.5, CHUNK_M * 0.5))
        b.rebuild_all(chunks)

        assert np.array_equal(a.height, b.height)
        assert a.height.dtype == np.float32

    def test_higher_chunk_layer_wins(self, cfg):
        # Two solid layers in the same column: the higher one must win.
        low = np.zeros((N, N, N), dtype=np.uint8); low[:, :, 0:4] = 1
        high = np.zeros((N, N, N), dtype=np.uint8); high[:, :, 10] = 1
        chunks = {(0, 0, 0): Chunk((0, 0, 0), low),
                  (0, 0, 1): Chunk((0, 0, 1), high)}
        field = RainCoverField(cfg)
        field.recenter((CHUNK_M * 0.5, CHUNK_M * 0.5))
        field.rebuild_all(chunks)
        mid_col = field.cells // 2
        # Find a texel firmly inside chunk (0,0)'s footprint.
        ox, oy = field.origin_m
        c = int(np.floor((4.0 - ox) / field.cell_m))
        r = int(np.floor((4.0 - oy) / field.cell_m))
        assert field.height[r, c] == pytest.approx(_voxel_top_z(1, 10))
