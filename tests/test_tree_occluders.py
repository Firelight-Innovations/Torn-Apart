"""
tests/test_tree_occluders.py — static tree occupancy splats for the cascades.

Headless (no panda3d): ``lighting/occluders.py`` + its hook in
``lighting/volume.py::assemble_geometry`` and
``lighting/assembly_worker.py::assemble_packed``.  Covers:

1. Determinism: identical inputs → byte-identical splats.
2. The None/empty path leaves ``assemble_geometry`` byte-identical.
3. Trunk column + canopy cells land where expected, with Beer–Lambert
   per-meter extinction semantics (occ = 1 − exp(−sigma·falloff·cell_m)).
4. THE key property: total transmittance through a canopy is the same
   marched at 0.5 m cells and at 2 m cells — attenuation compounds with
   METERS crossed, not cell count (the old flat per-cell opacity went
   near-black at cascade 0: 0.7¹⁰ vs 0.7³).
5. Rim falloff: the canopy medium thins toward the edge (soft gradient).
6. Terrain solids win the max-combine; out-of-window instances skipped.
7. Coarse/sub-cell scaling; ``assemble_packed`` threads the occluders.
8. ``mesh_leaf_area_m2`` — the leaf-thickness measure feeding sigma.
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
from fire_engine.procedural.flora.mesher import TreeMesh, mesh_leaf_area_m2

TRUNK_OCC = 0.85
CANOPY_GAIN = 1.0

# Reference tree: base at (8, 8, 4), 6 m tall, 2.5 m canopy radius,
# sigma 0.25/m (transmittance ~0.88 per meter of crown centre).
_TREE = dict(x=8.0, y=8.0, z=4.0, height_m=6.0, canopy_r_m=2.5,
             canopy_sigma=0.25,
             bark_rgb=(0.2, 0.1, 0.05), leaf_rgb=(0.05, 0.2, 0.05))


def _vol(n: int = 32) -> np.ndarray:
    return np.zeros((n, n, n, 4), dtype=np.uint8)


def _splat(vol, occ_set, origin=(0, 0, 0), cell_m=0.5,
           trunk=TRUNK_OCC, canopy=CANOPY_GAIN):
    splat_tree_occluders(vol, origin, cell_m, occ_set, trunk, canopy)


def _cell_occ_expected(d2: float, sigma: float, cell_m: float) -> np.uint8:
    """The splat's per-cell canopy byte for a cell at normalised dist² d2."""
    falloff = np.sqrt(max(0.0, 1.0 - d2))
    return np.uint8(np.rint(255.0 * (1.0 - np.exp(-sigma * falloff * cell_m))))


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
        expect = np.uint8(round(255 * TRUNK_OCC))   # full cell cross-section
        assert vol[16, 16, 9, 3] == expect
        # Bark albedo written where the trunk raised occupancy.
        assert tuple(vol[16, 16, 9, :3]) == tuple(
            np.clip(np.float32(_TREE["bark_rgb"]) * 255, 0, 255)
            .astype(np.uint8))

    def test_canopy_cell_beer_lambert(self):
        vol = _vol()
        _splat(vol, TreeOccluderSet.single(**_TREE))
        # Cell containing the canopy centre (7.9 m): centre (8.25, 8.25, 7.75).
        cz = _TREE["z"] + CANOPY_CENTER_FRAC * _TREE["height_m"]
        cv = CANOPY_HALF_HEIGHT_FRAC * _TREE["height_m"]
        czi = int(np.floor(cz / 0.5))
        d2 = (0.25 / 2.5) ** 2 + (0.25 / 2.5) ** 2 \
            + ((czi * 0.5 + 0.25 - cz) / cv) ** 2
        expect = _cell_occ_expected(d2, _TREE["canopy_sigma"], 0.5)
        assert vol[16, 16, czi, 3] == expect
        # A single 0.5 m cell of leaf medium is translucent, far from solid.
        assert 0 < expect < 64
        assert tuple(vol[16, 16, czi, :3]) == tuple(
            np.clip(np.float32(_TREE["leaf_rgb"]) * 255, 0, 255)
            .astype(np.uint8))
        # Outside the canopy radius stays air.
        assert vol[16 + 8, 16, czi, 3] == 0          # 4 m off-axis > 2.5 m

    def test_rim_falloff_thinner_than_core(self):
        vol = _vol()
        _splat(vol, TreeOccluderSet.single(**_TREE), trunk=0.0)
        cz = _TREE["z"] + CANOPY_CENTER_FRAC * _TREE["height_m"]
        czi = int(np.floor(cz / 0.5))
        core = int(vol[16, 16, czi, 3])
        rim = int(vol[16 + 4, 16, czi, 3])           # 2.25 m off-axis of 2.5
        assert core > rim > 0

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

    def test_zero_gain_is_noop(self):
        vol = _vol()
        _splat(vol, TreeOccluderSet.single(**_TREE), trunk=0.0, canopy=0.0)
        assert vol.sum() == 0


class TestCellSizeConsistency:
    def test_transmittance_matches_across_cell_sizes(self):
        """THE property the per-meter extinction exists for: marching the
        same canopy at 0.5 m and 2 m cells loses (about) the same light."""
        tree = TreeOccluderSet.single(x=16.0, y=16.0, z=2.0, height_m=10.0,
                                      canopy_r_m=4.0, canopy_sigma=0.3)

        def column_transmittance(cell_m: float, cells: int) -> float:
            vol = np.zeros((cells, cells, cells, 4), dtype=np.uint8)
            _splat(vol, tree, origin=(0, 0, 0), cell_m=cell_m, trunk=0.0)
            ci = int(np.floor(16.0 / cell_m))
            t = 1.0
            for occ in vol[ci, ci, :, 3]:
                t *= 1.0 - float(occ) / 255.0
            return t

        t_fine = column_transmittance(0.5, 64)       # 32 m box
        t_coarse = column_transmittance(2.0, 16)     # 32 m box
        # Both must transmit a meaningful fraction (the old flat per-cell
        # opacity gave ~0.7^16 ≈ 0.003 at fine cells = pitch black) …
        assert 0.05 < t_fine < 0.9
        # … and agree across cell sizes (cell-centre sampling differs a bit).
        assert abs(t_fine - t_coarse) < 0.15

    def test_fine_cells_not_black(self):
        """Regression for the 'completely black near trees' bug: a dense
        canopy at cascade-0 cells must still pass a visible share of light."""
        dense = TreeOccluderSet.single(x=16.0, y=16.0, z=0.0, height_m=12.0,
                                       canopy_r_m=5.0, canopy_sigma=0.35)
        vol = np.zeros((64, 64, 64, 4), dtype=np.uint8)
        _splat(vol, dense, origin=(0, 0, 0), cell_m=0.5, trunk=0.0)
        t = 1.0
        for occ in vol[32, 32, :, 3]:
            t *= 1.0 - float(occ) / 255.0
        # exp(-0.35 * ~8 m of weighted path) ≈ 0.1–0.2 — shaded, not black.
        assert t > 0.03


class TestCoarseCells:
    def test_trunk_cross_section_scaling(self):
        tall = TreeOccluderSet.single(x=8.0, y=8.0, z=0.0, height_m=12.0,
                                      canopy_r_m=0.0, canopy_sigma=0.0)
        vol = _vol(16)
        _splat(vol, tall, origin=(-8, -8, -8), cell_m=8.0)
        eff = TRUNK_OCC * (TRUNK_SIDE_M / 8.0) ** 2
        expect = np.uint8(round(255 * eff))
        cxi = int(np.floor(8.0 / 8.0)) + 8
        assert vol[cxi, cxi, 8, 3] == expect

    def test_subcell_canopy_registers_fractionally(self):
        bush = TreeOccluderSet.single(x=8.0, y=8.0, z=0.0, height_m=12.0,
                                      canopy_r_m=2.0, canopy_sigma=0.25)
        vol = _vol(16)
        _splat(vol, bush, origin=(-8, -8, -8), cell_m=8.0, trunk=0.0)
        cv = CANOPY_HALF_HEIGHT_FRAC * 12.0
        full = np.uint8(round(255 * (1.0 - np.exp(-0.25 * 2.0 * cv))))
        peak = vol[..., 3].max()
        # Present, but scaled well below the whole-crown opacity (the crown
        # fills a sliver of the 512 m³ cell).
        assert 0 < peak < full


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
                                    canopy_gain=CANOPY_GAIN)
        empty = assemble_geometry(win, {}, pal, 32, 0.5,
                                  occluders=TreeOccluderSet.empty(),
                                  trunk_occ=TRUNK_OCC,
                                  canopy_gain=CANOPY_GAIN)
        assert np.array_equal(base.albedo_occ, with_kw.albedo_occ)
        assert np.array_equal(base.albedo_occ, empty.albedo_occ)

    def test_assemble_geometry_splats(self):
        win = self._window()
        pal = MaterialPalette()
        vol = assemble_geometry(win, {}, pal, 32, 0.5,
                                occluders=TreeOccluderSet.single(**_TREE),
                                trunk_occ=TRUNK_OCC, canopy_gain=CANOPY_GAIN)
        assert vol.albedo_occ[16, 16, 9, 3] == np.uint8(round(255 * TRUNK_OCC))

    def test_assemble_packed_threads_occluders(self):
        common = dict(cascade_index=0, origin_cell=(0, 0, 0), cells=32,
                      cell_m=0.5, chunk_size=32, voxel_size=0.5,
                      materials={}, palette=MaterialPalette(), seq=1)
        plain = assemble_packed(AssemblyJob(**common))
        treed = assemble_packed(AssemblyJob(
            **common, occluders=TreeOccluderSet.single(**_TREE),
            trunk_occ=TRUNK_OCC, canopy_gain=CANOPY_GAIN))
        assert plain.albedo_bytes != treed.albedo_bytes
        assert plain.emis_bytes == treed.emis_bytes   # emission untouched


class TestLeafArea:
    @staticmethod
    def _mesh(uv_x: float) -> TreeMesh:
        """One unit quad (two triangles, area 1 m²) at atlas column uv_x."""
        pos = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
                       dtype=np.float32)
        return TreeMesh(
            positions=pos,
            normals=np.tile(np.float32([0, 1, 0]), (4, 1)),
            uvs=np.full((4, 2), uv_x, dtype=np.float32),
            colors=np.ones((4, 4), dtype=np.float32),
            indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32),
            height_m=1.0, radius_m=1.0)

    def test_leaf_quad_counts(self):
        assert mesh_leaf_area_m2(self._mesh(uv_x=0.75)) == 1.0

    def test_bark_quad_ignored(self):
        assert mesh_leaf_area_m2(self._mesh(uv_x=0.25)) == 0.0

    def test_empty_mesh(self):
        assert mesh_leaf_area_m2(TreeMesh.empty()) == 0.0

    def test_oak_leafier_than_dead_tree(self):
        # The real species: sigma's input must rank a leafy oak above a snag.
        from fire_engine.core.rng import set_world_seed, for_domain
        from fire_engine.procedural.flora.species import gnarled_oak, dead_tree
        set_world_seed(424242)
        oak = gnarled_oak.GnarledOakDef().generate(
            for_domain("procedural", "tree_gnarled_oak"))
        set_world_seed(424242)
        dead = dead_tree.DeadTreeDef().generate(
            for_domain("procedural", "tree_dead"))
        oak_area = np.mean([mesh_leaf_area_m2(m) for m in oak.meshes])
        dead_area = np.mean([mesh_leaf_area_m2(m) for m in dead.meshes])
        assert oak_area > dead_area
        assert oak_area > 0.0
