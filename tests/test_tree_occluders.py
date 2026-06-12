"""
tests/test_tree_occluders.py — static tree occupancy splats for the cascades.

Headless (no panda3d): ``lighting/occluders.py`` + its hook in
``lighting/volume.py::assemble_geometry`` and
``lighting/assembly_worker.py::assemble_packed``.  Covers:

1. Determinism: identical inputs → byte-identical splats.
2. The None/empty path leaves ``assemble_geometry`` byte-identical.
3. Trunk column + canopy ellipsoid land at the expected cells with the
   expected occupancy bytes and splat albedo at cascade-0 resolution.
4. Terrain solids win the max-combine (occupancy never lowered, terrain
   albedo never repainted).
5. Out-of-window instances are skipped without error.
6. Coarse cells scale the contribution down (a tree is a wisp in an 8 m
   cell), and sub-cell canopies still register fractionally.
7. ``assemble_packed`` threads the occluders through the job.
"""

from __future__ import annotations

import numpy as np

from fire_engine.lighting import (
    TreeOccluderSet,
    VolumeWindow,
    assemble_geometry,
    splat_tree_occluders,
)
from fire_engine.lighting.assembly_worker import AssemblyJob, assemble_packed
from fire_engine.lighting.occluders import (
    CANOPY_CENTER_FRAC,
    CANOPY_HALF_HEIGHT_FRAC,
    TRUNK_SIDE_M,
    TRUNK_TOP_FRAC,
)
from fire_engine.lighting.palette import MaterialPalette

TRUNK_OCC = 0.85
CANOPY_OCC = 0.30

# Reference tree: base at (8, 8, 4), 6 m tall, 2.5 m canopy radius.
_TREE = dict(x=8.0, y=8.0, z=4.0, height_m=6.0, canopy_r_m=2.5,
             bark_rgb=(0.2, 0.1, 0.05), leaf_rgb=(0.05, 0.2, 0.05))


def _vol(n: int = 32) -> np.ndarray:
    return np.zeros((n, n, n, 4), dtype=np.uint8)


def _splat(vol, occ_set, origin=(0, 0, 0), cell_m=0.5,
           trunk=TRUNK_OCC, canopy=CANOPY_OCC):
    splat_tree_occluders(vol, origin, cell_m, occ_set, trunk, canopy)


class TestDeterminism:
    def test_same_inputs_identical_output(self):
        occ = TreeOccluderSet.single(**_TREE)
        a, b = _vol(), _vol()
        _splat(a, occ)
        _splat(b, occ)
        assert np.array_equal(a, b)

    def test_merge_preserves_instances(self):
        s1 = TreeOccluderSet.single(**_TREE)
        s2 = TreeOccluderSet.single(**{**_TREE, "x": 4.0})
        merged = TreeOccluderSet.merge([s1, s2])
        assert merged.count == 2
        separate = _vol()
        _splat(separate, s1)
        _splat(separate, s2)
        together = _vol()
        _splat(together, merged)
        assert np.array_equal(separate, together)


class TestSplatShape:
    def test_trunk_column_occupied(self):
        vol = _vol()
        _splat(vol, TreeOccluderSet.single(**_TREE))
        # Trunk: cell (16, 16) in XY, Z cells 8 .. <(4 + 0.45*6)/0.5>.
        expect = np.uint8(round(255 * TRUNK_OCC))   # full cell cross-section
        z_top = int(np.floor((4.0 + TRUNK_TOP_FRAC * 6.0) / 0.5))
        for zc in range(8, z_top + 1):
            assert vol[16, 16, zc, 3] >= np.uint8(round(255 * CANOPY_OCC)), zc
        assert vol[16, 16, 9, 3] == expect
        # Bark albedo written where the trunk raised occupancy.
        assert tuple(vol[16, 16, 9, :3]) == tuple(
            np.clip(np.float32(_TREE["bark_rgb"]) * 255, 0, 255)
            .astype(np.uint8))

    def test_canopy_cell_occupancy_and_albedo(self):
        vol = _vol()
        _splat(vol, TreeOccluderSet.single(**_TREE))
        cz = 4.0 + CANOPY_CENTER_FRAC * 6.0          # 7.9 m
        czi = int(np.floor(cz / 0.5))
        expect = np.uint8(round(255 * CANOPY_OCC))   # canopy >> cell volume
        assert vol[16, 16, czi, 3] == expect
        assert tuple(vol[16, 16, czi, :3]) == tuple(
            np.clip(np.float32(_TREE["leaf_rgb"]) * 255, 0, 255)
            .astype(np.uint8))
        # Outside the canopy radius stays air.
        assert vol[16 + 8, 16, czi, 3] == 0          # 4 m off-axis > 2.5 m

    def test_terrain_solid_wins(self):
        vol = _vol()
        vol[..., 3] = 255                            # everything solid rock
        vol[..., :3] = 90
        _splat(vol, TreeOccluderSet.single(**_TREE))
        assert (vol[..., 3] == 255).all()            # never lowered
        assert (vol[..., :3] == 90).all()            # never repainted

    def test_out_of_window_skipped(self):
        vol = _vol()
        far = TreeOccluderSet.single(**{**_TREE, "x": 500.0, "y": -500.0})
        _splat(vol, far)
        assert vol.sum() == 0

    def test_zero_opacity_is_noop(self):
        vol = _vol()
        _splat(vol, TreeOccluderSet.single(**_TREE), trunk=0.0, canopy=0.0)
        assert vol.sum() == 0


class TestCoarseCells:
    def test_coarse_cell_scales_contribution_down(self):
        # 12 m oak in 8 m cells: canopy fills most of (but not over) a cell.
        big = TreeOccluderSet.single(x=8.0, y=8.0, z=0.0, height_m=12.0,
                                     canopy_r_m=5.0)
        vol = _vol(16)
        _splat(vol, big, origin=(-8, -8, -8), cell_m=8.0)
        cz = CANOPY_CENTER_FRAC * 12.0
        cv = CANOPY_HALF_HEIGHT_FRAC * 12.0
        ratio = min(1.0, (4.0 / 3.0) * np.pi * 5.0 * 5.0 * cv / 8.0 ** 3)
        expect = np.uint8(round(255 * CANOPY_OCC * ratio))
        cxi = int(np.floor(8.0 / 8.0)) + 8
        czi = int(np.floor(cz / 8.0)) + 8
        assert 0 < ratio < 1.0
        assert vol[cxi, cxi, czi, 3] == expect
        assert expect < np.uint8(round(255 * CANOPY_OCC))

    def test_trunk_cross_section_scaling(self):
        tall = TreeOccluderSet.single(x=8.0, y=8.0, z=0.0, height_m=12.0,
                                      canopy_r_m=0.0)     # trunk only
        vol = _vol(16)
        _splat(vol, tall, origin=(-8, -8, -8), cell_m=8.0)
        eff = TRUNK_OCC * (TRUNK_SIDE_M / 8.0) ** 2
        expect = np.uint8(round(255 * eff))
        cxi = int(np.floor(8.0 / 8.0)) + 8
        assert vol[cxi, cxi, 8, 3] == expect

    def test_subcell_canopy_registers_fractionally(self):
        bush = TreeOccluderSet.single(x=8.0, y=8.0, z=0.0, height_m=12.0,
                                      canopy_r_m=2.0)
        vol = _vol(16)
        _splat(vol, bush, origin=(-8, -8, -8), cell_m=8.0, trunk=0.0)
        # Ellipsoid (r 2 m, vertical 4.2 m) ≈ 70 m³ in a 512 m³ cell: the
        # inside-test may miss every 8 m cell centre, but the centre cell
        # still picks up the volume-ratio fraction.
        assert vol[..., 3].max() > 0
        assert vol[..., 3].max() < np.uint8(round(255 * CANOPY_OCC))


class TestAssemblyIntegration:
    def _window(self):
        win = VolumeWindow(cells=32, cell_m=0.5)
        win.recenter((8.0, 8.0, 8.0))
        return win

    def test_assemble_geometry_none_is_byte_identical(self):
        win = self._window()
        pal = MaterialPalette()
        base = assemble_geometry(win, {}, pal, 32, 0.5)
        with_kw = assemble_geometry(win, {}, pal, 32, 0.5,
                                    occluders=None,
                                    trunk_occ=TRUNK_OCC,
                                    canopy_occ=CANOPY_OCC)
        empty = assemble_geometry(win, {}, pal, 32, 0.5,
                                  occluders=TreeOccluderSet.empty(),
                                  trunk_occ=TRUNK_OCC, canopy_occ=CANOPY_OCC)
        assert np.array_equal(base.albedo_occ, with_kw.albedo_occ)
        assert np.array_equal(base.albedo_occ, empty.albedo_occ)

    def test_assemble_geometry_splats(self):
        win = self._window()
        pal = MaterialPalette()
        vol = assemble_geometry(win, {}, pal, 32, 0.5,
                                occluders=TreeOccluderSet.single(**_TREE),
                                trunk_occ=TRUNK_OCC, canopy_occ=CANOPY_OCC)
        assert vol.albedo_occ[16, 16, 9, 3] == np.uint8(round(255 * TRUNK_OCC))

    def test_assemble_packed_threads_occluders(self):
        common = dict(cascade_index=0, origin_cell=(0, 0, 0), cells=32,
                      cell_m=0.5, chunk_size=32, voxel_size=0.5,
                      materials={}, palette=MaterialPalette(), seq=1)
        plain = assemble_packed(AssemblyJob(**common))
        treed = assemble_packed(AssemblyJob(
            **common, occluders=TreeOccluderSet.single(**_TREE),
            trunk_occ=TRUNK_OCC, canopy_occ=CANOPY_OCC))
        assert plain.albedo_bytes != treed.albedo_bytes
        assert plain.emis_bytes == treed.emis_bytes   # emission untouched
