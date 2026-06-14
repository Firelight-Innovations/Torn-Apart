"""
tests/test_lighting_volume.py — Headless tests for the GPU-lighting core.

Covers the panda3d-free halves of the volumetric pipeline:
- ``VolumeWindow`` recenter/snap/hysteresis math,
- ``assemble_geometry`` occupancy/albedo/emission packing at both cascade
  scales (0.5 m direct and 2.0 m downsampled),
- ``MaterialPalette`` / ``build_default_palette`` determinism,
- ``LightSet`` packing layout, transient TTL fade/expiry, version bumps,
- ``derive_normal_map`` encoding.

The GPU half (lighting/gpu.py) imports panda3d and is intentionally NOT
imported here (headless suite rule).
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.core.rng import set_world_seed
from fire_engine.lighting.lights import (
    AreaLight,
    LIGHT_TYPE_AREA,
    LIGHT_TYPE_POINT,
    LightSet,
    PointLight,
)
from fire_engine.lighting.palette import MaterialPalette, build_default_palette
from fire_engine.lighting.volume import (
    EMISSION_SCALE,
    ChunkBlockCache,
    VolumeWindow,
    assemble_geometry,
)
from fire_engine.lighting.assembly_worker import (
    AssemblyJob,
    CascadeAssemblyWorker,
    assemble_packed,
)
from fire_engine.procedural.maps import (
    black_emission_map,
    derive_normal_map,
    flat_normal_map,
)

VOXEL = 0.5
CHUNK = 32


class _Chunk:
    """Minimal chunk stand-in: just a materials array."""

    def __init__(self) -> None:
        self.materials = np.zeros((CHUNK, CHUNK, CHUNK), dtype=np.uint8)


def _palette() -> MaterialPalette:
    """Hand-built palette with distinct albedos for material 1 and 2."""
    albedo = np.zeros((256, 3), dtype=np.float32)
    albedo[1] = (0.4, 0.3, 0.2)  # dirt-ish
    albedo[2] = (0.2, 0.5, 0.1)  # grass-ish
    return MaterialPalette(albedo=albedo, emission=np.zeros((256, 3), dtype=np.float32))


# ---------------------------------------------------------------------------
# VolumeWindow
# ---------------------------------------------------------------------------


class TestVolumeWindow:
    def test_first_recenter_places_window(self):
        win = VolumeWindow(cells=32, cell_m=0.5)
        assert win.recenter((8.0, 8.0, 8.0)) is True
        assert win.origin_cell is not None

    def test_origin_snaps_to_snap_cells(self):
        win = VolumeWindow(cells=96, cell_m=0.5, snap_cells=8)
        win.recenter((13.7, -4.2, 9.9))
        assert all(o % 8 == 0 for o in win.origin_cell)

    def test_window_centred_on_camera(self):
        win = VolumeWindow(cells=32, cell_m=0.5, snap_cells=8)
        win.recenter((8.0, 8.0, 8.0))
        ox, oy, oz = win.world_origin_m
        # Camera should be within half a box of the centre on every axis.
        centre = (ox + 8.0, oy + 8.0, oz + 8.0)  # 32 * 0.5 / 2 = 8 m
        assert all(abs(c - 8.0) <= 4.0 + 1e-6 for c in centre)

    def test_hysteresis_no_move_within_margin(self):
        win = VolumeWindow(cells=96, cell_m=0.5, margin_cells=8)
        win.recenter((0.0, 0.0, 0.0))
        assert win.recenter((1.0, 1.0, 1.0)) is False  # 1 m < 4 m margin

    def test_recenter_after_large_move(self):
        win = VolumeWindow(cells=96, cell_m=0.5, margin_cells=8)
        win.recenter((0.0, 0.0, 0.0))
        first = win.origin_cell
        assert win.recenter((20.0, 0.0, 0.0)) is True  # 20 m > margin
        assert win.origin_cell != first

    def test_world_origin_before_recenter_raises(self):
        win = VolumeWindow(cells=32, cell_m=0.5)
        with pytest.raises(ValueError):
            _ = win.world_origin_m

    def test_cells_must_divide_by_snap(self):
        with pytest.raises(ValueError):
            VolumeWindow(cells=30, cell_m=0.5, snap_cells=8)


# ---------------------------------------------------------------------------
# assemble_geometry — cascade 0 (cell == voxel)
# ---------------------------------------------------------------------------


class TestAssembleCascade0:
    def _window_at_origin(self, cells=32):
        win = VolumeWindow(cells=cells, cell_m=VOXEL, snap_cells=8)
        win.recenter((8.0, 8.0, 8.0))
        assert win.origin_cell == (0, 0, 0)  # snapped exactly to chunk 0
        return win

    def test_solid_voxel_lands_in_right_cell(self):
        win = self._window_at_origin()
        chunk = _Chunk()
        chunk.materials[5, 6, 7] = 1
        vol = assemble_geometry(
            win, {(0, 0, 0): chunk}, _palette(), chunk_size=CHUNK, voxel_size=VOXEL
        )
        assert vol.albedo_occ[5, 6, 7, 3] == 255  # occupied
        assert vol.albedo_occ[5, 6, 6, 3] == 0  # neighbour air
        # Albedo = palette[1] * 255.
        np.testing.assert_allclose(
            vol.albedo_occ[5, 6, 7, :3],
            np.clip(np.array([0.4, 0.3, 0.2]) * 255, 0, 255).astype(np.uint8),
        )

    def test_missing_chunks_are_air(self):
        win = self._window_at_origin()
        vol = assemble_geometry(win, {}, _palette(), chunk_size=CHUNK, voxel_size=VOXEL)
        assert int(vol.albedo_occ[..., 3].max()) == 0

    def test_emission_packing(self):
        win = self._window_at_origin()
        chunk = _Chunk()
        chunk.materials[1, 1, 1] = 3
        pal = _palette().with_emission(3, (2.0, 1.0, 0.5))
        vol = assemble_geometry(win, {(0, 0, 0): chunk}, pal, chunk_size=CHUNK, voxel_size=VOXEL)
        expected = np.clip(np.array([2.0, 1.0, 0.5]) * 255.0 / EMISSION_SCALE, 0, 255).astype(
            np.uint8
        )
        np.testing.assert_array_equal(vol.emission[1, 1, 1, :3], expected)

    def test_deterministic(self):
        chunk = _Chunk()
        chunk.materials[:, :, :8] = 1
        chunk.materials[:, :, 8] = 2
        outs = []
        for _ in range(2):
            win = self._window_at_origin()
            vol = assemble_geometry(
                win, {(0, 0, 0): chunk}, _palette(), chunk_size=CHUNK, voxel_size=VOXEL
            )
            outs.append(vol.albedo_occ.tobytes() + vol.emission.tobytes())
        assert outs[0] == outs[1]

    def test_spans_chunk_boundary(self):
        # Window snapped to (-8, -8, -8) cells covers chunks (-1..0)^3 corner.
        win = VolumeWindow(cells=32, cell_m=VOXEL, snap_cells=8)
        win.recenter((0.0, 0.0, 0.0))
        ox, oy, oz = win.origin_cell
        assert ox < 0 < ox + win.cells
        neg = _Chunk()
        neg.materials[CHUNK - 1, CHUNK - 1, CHUNK - 1] = 1  # voxel (-1,-1,-1)
        vol = assemble_geometry(
            win, {(-1, -1, -1): neg}, _palette(), chunk_size=CHUNK, voxel_size=VOXEL
        )
        idx = (-1 - ox, -1 - oy, -1 - oz)
        assert vol.albedo_occ[idx[0], idx[1], idx[2], 3] == 255


# ---------------------------------------------------------------------------
# Geometry-occupancy providers (buildings / future props)
# ---------------------------------------------------------------------------


class _RecordingProvider:
    """Records the args it was handed and stamps occupancy into one cell."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def rasterize_occupancy(self, origin_cell, cells, cell_m, albedo_occ, emission) -> None:
        self.calls.append((origin_cell, cells, cell_m, albedo_occ.shape, emission.shape))
        albedo_occ[2, 2, 2, 3] = 200  # mark a cell solid


class TestGeometryProviders:
    def _window(self):
        win = VolumeWindow(cells=32, cell_m=VOXEL, snap_cells=8)
        win.recenter((8.0, 8.0, 8.0))
        return win

    def test_empty_providers_is_byte_identical(self):
        chunk = _Chunk()
        chunk.materials[5, 6, 7] = 1
        a = assemble_geometry(
            self._window(), {(0, 0, 0): chunk}, _palette(), chunk_size=CHUNK, voxel_size=VOXEL
        )
        b = assemble_geometry(
            self._window(),
            {(0, 0, 0): chunk},
            _palette(),
            chunk_size=CHUNK,
            voxel_size=VOXEL,
            providers=(),
        )
        assert a.albedo_occ.tobytes() == b.albedo_occ.tobytes()
        assert a.emission.tobytes() == b.emission.tobytes()

    def test_provider_receives_window_params(self):
        prov = _RecordingProvider()
        win = self._window()
        vol = assemble_geometry(
            win, {}, _palette(), chunk_size=CHUNK, voxel_size=VOXEL, providers=(prov,)
        )
        assert len(prov.calls) == 1
        origin, cells, cell_m, ashape, eshape = prov.calls[0]
        assert origin == win.origin_cell
        assert cells == win.cells
        assert cell_m == win.cell_m
        assert ashape == (win.cells,) * 3 + (4,)
        assert eshape == (win.cells,) * 3 + (4,)
        # The provider's write landed in the returned volume.
        assert vol.albedo_occ[2, 2, 2, 3] == 200

    def test_building_rasterizer_is_protocol_and_noop(self):
        from fire_engine.buildings.occlusion import BuildingOccupancyRasterizer
        from fire_engine.lighting.volume import GeometryOccupancyProvider

        rasterizer = BuildingOccupancyRasterizer(manager=None)
        assert isinstance(rasterizer, GeometryOccupancyProvider)
        chunk = _Chunk()
        chunk.materials[5, 6, 7] = 1
        base = assemble_geometry(
            self._window(), {(0, 0, 0): chunk}, _palette(), chunk_size=CHUNK, voxel_size=VOXEL
        )
        withb = assemble_geometry(
            self._window(),
            {(0, 0, 0): chunk},
            _palette(),
            chunk_size=CHUNK,
            voxel_size=VOXEL,
            providers=(rasterizer,),
        )
        # v1 no-op: byte-identical to the no-provider assembly.
        assert base.albedo_occ.tobytes() == withb.albedo_occ.tobytes()
        assert base.emission.tobytes() == withb.emission.tobytes()


# ---------------------------------------------------------------------------
# assemble_geometry — cascade 1 (2.0 m cells, 4^3 voxels per cell)
# ---------------------------------------------------------------------------


class TestAssembleCascade1:
    def test_partial_occupancy_and_max_material_wins(self):
        # Coarse cells carry a FRACTIONAL occupancy alpha now (solid-voxel
        # fraction ×255), while albedo still picks the max material id.
        win = VolumeWindow(cells=16, cell_m=2.0, snap_cells=8)
        win.recenter((16.0, 16.0, 16.0))
        assert win.origin_cell == (0, 0, 0)
        chunk = _Chunk()
        chunk.materials[0, 0, 0] = 1  # one dirt voxel in cell (0,0,0)
        chunk.materials[1, 0, 0] = 2  # plus one grass voxel — max id wins
        vol = assemble_geometry(
            win, {(0, 0, 0): chunk}, _palette(), chunk_size=CHUNK, voxel_size=VOXEL
        )
        # 2 solid voxels of the cell's 4^3 = 64 sub-voxels → round(255*2/64).
        assert vol.albedo_occ[0, 0, 0, 3] == int(round(255.0 * 2 / 64))
        np.testing.assert_allclose(
            vol.albedo_occ[0, 0, 0, :3],
            np.clip(np.array([0.2, 0.5, 0.1]) * 255, 0, 255).astype(np.uint8),
        )
        assert vol.albedo_occ[1, 0, 0, 3] == 0

    def test_non_integer_cell_ratio_raises(self):
        win = VolumeWindow(cells=16, cell_m=0.75, snap_cells=8)
        win.recenter((0.0, 0.0, 0.0))
        with pytest.raises(ValueError):
            assemble_geometry(win, {}, _palette(), chunk_size=CHUNK, voxel_size=VOXEL)


# ---------------------------------------------------------------------------
# MaterialPalette
# ---------------------------------------------------------------------------


class TestPalette:
    def test_default_palette_air_is_zero_and_materials_differ(self):
        set_world_seed(1337)
        import fire_engine.procedural  # noqa: F401  (registration)

        pal = build_default_palette()
        assert (pal.albedo[0] == 0).all()
        assert not np.allclose(pal.albedo[1], pal.albedo[2])
        assert (pal.emission == 0).all()

    def test_default_palette_deterministic(self):
        set_world_seed(1337)
        import fire_engine.procedural  # noqa: F401

        a = build_default_palette().albedo.tobytes()
        b = build_default_palette().albedo.tobytes()
        assert a == b

    def test_with_emission_is_a_copy(self):
        base = _palette()
        glowing = base.with_emission(7, (1.0, 2.0, 3.0))
        assert (base.emission[7] == 0).all()
        np.testing.assert_allclose(glowing.emission[7], (1.0, 2.0, 3.0))


# ---------------------------------------------------------------------------
# LightSet
# ---------------------------------------------------------------------------


class TestLightSet:
    def test_pack_point_light_layout(self):
        ls = LightSet()
        ls.add(
            PointLight(position=(1.0, 2.0, 3.0), color=(1.0, 0.5, 0.25), intensity=2.0, radius=10.0)
        )
        arr, count = ls.pack(max_lights=4)
        assert count == 1 and arr.shape == (4, 12)
        np.testing.assert_allclose(arr[0, 0:3], (1.0, 2.0, 3.0))
        assert arr[0, 3] == 10.0
        np.testing.assert_allclose(arr[0, 4:7], (2.0, 1.0, 0.5))
        assert arr[0, 7] == LIGHT_TYPE_POINT
        assert (arr[1:] == 0).all()

    def test_pack_area_light_layout(self):
        ls = LightSet()
        ls.add(
            AreaLight(
                center=(0.0, 0.0, 5.0),
                half_extents=(2.0, 1.0, 0.5),
                color=(1.0, 1.0, 1.0),
                intensity=1.0,
                radius=8.0,
            )
        )
        arr, count = ls.pack(max_lights=2)
        assert count == 1
        assert arr[0, 7] == LIGHT_TYPE_AREA
        np.testing.assert_allclose(arr[0, 8:11], (2.0, 1.0, 0.5))

    def test_ttl_fade_and_expiry(self):
        ls = LightSet()
        ls.add(PointLight((0, 0, 0), (1, 1, 1), 10.0, 5.0, ttl_s=1.0))
        ls.update(0.5)
        arr, count = ls.pack(4)
        assert count == 1
        assert arr[0, 4] == pytest.approx(5.0)  # half-faded
        ls.update(0.6)
        assert ls.count == 0

    def test_version_bumps(self):
        ls = LightSet()
        v0 = ls.version
        lid = ls.add(PointLight((0, 0, 0), (1, 1, 1), 1.0, 5.0))
        assert ls.version > v0
        v1 = ls.version
        ls.remove(lid)
        assert ls.version > v1
        v2 = ls.version
        ls.remove(999)  # absent id: no bump
        assert ls.version == v2
        ls.update(0.1)  # no transient lights: no bump
        assert ls.version == v2

    def test_pack_respects_max_lights(self):
        ls = LightSet()
        for i in range(6):
            ls.add(PointLight((float(i), 0, 0), (1, 1, 1), 1.0, 5.0))
        arr, count = ls.pack(4)
        assert count == 4


# ---------------------------------------------------------------------------
# procedural/maps.py
# ---------------------------------------------------------------------------


class TestDerivedMaps:
    def test_flat_input_gives_flat_normal(self):
        rgba = np.full((8, 8, 4), 200, dtype=np.uint8)
        nm = derive_normal_map(rgba)
        assert nm.shape == (8, 8, 4) and nm.dtype == np.uint8
        assert (nm[..., 0] == 128).all()
        assert (nm[..., 1] == 128).all()
        assert (nm[..., 2] == 255).all()

    def test_gradient_tilts_normal(self):
        rgba = np.zeros((8, 8, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        rgba[:, 4:, :3] = 255  # bright right half → slope at the seam
        nm = derive_normal_map(rgba)
        assert (nm[:, 3:5, 0].astype(int) != 128).any()

    def test_deterministic(self):
        set_world_seed(1337)
        import fire_engine.procedural  # noqa: F401
        from fire_engine.procedural import get as get_procedural

        rgba = get_procedural("grass_ground")
        assert derive_normal_map(rgba).tobytes() == derive_normal_map(rgba).tobytes()

    def test_helper_maps(self):
        fn = flat_normal_map()
        assert tuple(fn[0, 0]) == (128, 128, 255, 255)
        em = black_emission_map()
        assert tuple(em[0, 0]) == (0, 0, 0, 255)


# ---------------------------------------------------------------------------
# assemble_geometry — fractional occupancy (coarse downsample)
# ---------------------------------------------------------------------------


def _hollow_box_chunk() -> _Chunk:
    """A chunk-sized hollow box: 1-voxel solid shell, air interior."""
    chunk = _Chunk()
    chunk.materials[:] = 1  # fill solid
    chunk.materials[1:-1, 1:-1, 1:-1] = 0  # hollow out the interior
    return chunk


class TestFractionalOccupancy:
    def test_fine_path_is_binary_and_byte_identical(self):
        # At cell_m == voxel_size (k == 1) alpha must be exactly 0/255 — the
        # previous any-solid behaviour — for arbitrary material content.
        win = VolumeWindow(cells=32, cell_m=VOXEL, snap_cells=8)
        win.recenter((8.0, 8.0, 8.0))
        chunk = _Chunk()
        chunk.materials[:, :, :8] = 1
        chunk.materials[:, :, 8] = 2
        chunk.materials[3, 4, 5] = 0
        vol = assemble_geometry(
            win, {(0, 0, 0): chunk}, _palette(), chunk_size=CHUNK, voxel_size=VOXEL
        )
        alpha = vol.albedo_occ[..., 3]
        solid = chunk.materials > 0
        expected = np.where(solid, 255, 0).astype(np.uint8)
        np.testing.assert_array_equal(alpha, expected)
        assert set(np.unique(alpha)).issubset({0, 255})

    def test_hollow_box_interior_air_walls_partial_8m(self):
        # 8 m cells over a 16 m chunk => 2x2x2 block; each cell is 16^3 voxels.
        win = VolumeWindow(cells=8, cell_m=8.0, snap_cells=8)
        win.recenter((40.0, 40.0, 40.0))  # centres an 8x8m window on origin 0
        assert win.origin_cell == (0, 0, 0)
        chunk = _hollow_box_chunk()
        vol = assemble_geometry(
            win, {(0, 0, 0): chunk}, _palette(), chunk_size=CHUNK, voxel_size=VOXEL
        )
        alpha = vol.albedo_occ[..., 3]
        # The 2x2x2 corner block is all wall cells (every 16^3 corner block of
        # the shell touches the shell) — partial, never fully solid, never air.
        corner = alpha[0:2, 0:2, 0:2]
        assert (corner > 0).all()
        assert (corner < 255).all()

    def test_hollow_box_interior_air_2m(self):
        # 2 m cells => 8x8x8 cells per chunk; each cell is 4^3 voxels.  Interior
        # cells fully inside the 1-voxel-shell hollow must read air (alpha 0).
        win = VolumeWindow(cells=8, cell_m=2.0, snap_cells=8)
        win.recenter((8.0, 8.0, 8.0))
        assert win.origin_cell == (0, 0, 0)
        chunk = _hollow_box_chunk()
        vol = assemble_geometry(
            win, {(0, 0, 0): chunk}, _palette(), chunk_size=CHUNK, voxel_size=VOXEL
        )
        alpha = vol.albedo_occ[..., 3]
        # Cell (2,2,2) spans voxels [8:12]^3 — fully in the hollow interior.
        assert alpha[2, 2, 2] == 0
        # Edge cell (0,0,0) spans [0:4]^3 — contains the 1-voxel shell faces.
        assert 0 < alpha[0, 0, 0] < 255

    def test_partial_fraction_value(self):
        # A cell with exactly one solid voxel out of 4^3 => round(255/64).
        win = VolumeWindow(cells=8, cell_m=2.0, snap_cells=8)
        win.recenter((8.0, 8.0, 8.0))
        chunk = _Chunk()
        chunk.materials[0, 0, 0] = 1  # single solid voxel in cell (0,0,0)
        vol = assemble_geometry(
            win, {(0, 0, 0): chunk}, _palette(), chunk_size=CHUNK, voxel_size=VOXEL
        )
        assert vol.albedo_occ[0, 0, 0, 3] == int(round(255.0 / 64.0))

    def test_albedo_still_max_material(self):
        # Albedo RGB selection unchanged: max material id wins within a cell.
        win = VolumeWindow(cells=8, cell_m=2.0, snap_cells=8)
        win.recenter((8.0, 8.0, 8.0))
        chunk = _Chunk()
        chunk.materials[0, 0, 0] = 1
        chunk.materials[1, 0, 0] = 2  # same 4^3 cell — id 2 wins
        vol = assemble_geometry(
            win, {(0, 0, 0): chunk}, _palette(), chunk_size=CHUNK, voxel_size=VOXEL
        )
        np.testing.assert_allclose(
            vol.albedo_occ[0, 0, 0, :3],
            np.clip(np.array([0.2, 0.5, 0.1]) * 255, 0, 255).astype(np.uint8),
        )


# ---------------------------------------------------------------------------
# ChunkBlockCache
# ---------------------------------------------------------------------------


def _coarse_window():
    win = VolumeWindow(cells=8, cell_m=2.0, snap_cells=8)
    win.recenter((8.0, 8.0, 8.0))
    return win


def _varied_chunk() -> _Chunk:
    chunk = _Chunk()
    chunk.materials[:, :, :8] = 1
    chunk.materials[2:5, 2:5, 2:5] = 2
    chunk.materials[0, 0, 0] = 0
    return chunk


class TestChunkBlockCache:
    def test_hit_equals_miss_bytes(self):
        chunk = _varied_chunk()
        chunks = {(0, 0, 0): chunk}
        pal = _palette()

        no_cache = assemble_geometry(
            _coarse_window(), chunks, pal, chunk_size=CHUNK, voxel_size=VOXEL
        )
        cache = ChunkBlockCache()
        miss = assemble_geometry(
            _coarse_window(), chunks, pal, chunk_size=CHUNK, voxel_size=VOXEL, cache=cache
        )
        assert len(cache) == 1  # populated on miss
        hit = assemble_geometry(
            _coarse_window(), chunks, pal, chunk_size=CHUNK, voxel_size=VOXEL, cache=cache
        )
        a = no_cache.albedo_occ.tobytes() + no_cache.emission.tobytes()
        b = miss.albedo_occ.tobytes() + miss.emission.tobytes()
        c = hit.albedo_occ.tobytes() + hit.emission.tobytes()
        assert a == b == c

    def test_invalidate_forces_recompute(self):
        chunk = _varied_chunk()
        chunks = {(0, 0, 0): chunk}
        pal = _palette()
        cache = ChunkBlockCache()
        assemble_geometry(
            _coarse_window(), chunks, pal, chunk_size=CHUNK, voxel_size=VOXEL, cache=cache
        )
        # Mutate the chunk; without invalidation the stale block is reused.
        chunk.materials[10, 10, 10] = 2
        stale = assemble_geometry(
            _coarse_window(), chunks, pal, chunk_size=CHUNK, voxel_size=VOXEL, cache=cache
        )
        cache.invalidate((0, 0, 0))
        assert len(cache) == 0
        fresh = assemble_geometry(
            _coarse_window(), chunks, pal, chunk_size=CHUNK, voxel_size=VOXEL, cache=cache
        )
        # Fresh must match a no-cache assembly of the mutated chunk.
        truth = assemble_geometry(_coarse_window(), chunks, pal, chunk_size=CHUNK, voxel_size=VOXEL)
        assert fresh.albedo_occ.tobytes() == truth.albedo_occ.tobytes()
        assert stale.albedo_occ.tobytes() != truth.albedo_occ.tobytes()

    def test_clear(self):
        chunk = _varied_chunk()
        cache = ChunkBlockCache()
        assemble_geometry(
            _coarse_window(),
            {(0, 0, 0): chunk},
            _palette(),
            chunk_size=CHUNK,
            voxel_size=VOXEL,
            cache=cache,
        )
        assert len(cache) == 1
        cache.clear()
        assert len(cache) == 0

    def test_lru_eviction_bounds_size(self):
        cache = ChunkBlockCache(max_entries=2)
        for i in range(5):
            cache.put(
                (i, 0, 0), 2.0, (np.zeros((2, 2, 2), np.uint8), np.zeros((2, 2, 2), np.uint16))
            )
        assert len(cache) == 2

    def test_fine_path_does_not_cache(self):
        # k == 1 (cascade 0) must not populate the cache (block aliases the
        # live chunk array — caching it would freeze the caller's materials).
        win = VolumeWindow(cells=32, cell_m=VOXEL, snap_cells=8)
        win.recenter((8.0, 8.0, 8.0))
        chunk = _Chunk()
        chunk.materials[5, 5, 5] = 1
        cache = ChunkBlockCache()
        assemble_geometry(
            win, {(0, 0, 0): chunk}, _palette(), chunk_size=CHUNK, voxel_size=VOXEL, cache=cache
        )
        assert len(cache) == 0
        # The chunk array stays writeable.
        chunk.materials[5, 5, 5] = 2  # must not raise


# ---------------------------------------------------------------------------
# CascadeAssemblyWorker + cache parity
# ---------------------------------------------------------------------------


def _coarse_job(materials: dict, seq: int = 0) -> AssemblyJob:
    win = _coarse_window()
    return AssemblyJob(
        cascade_index=2,
        origin_cell=win.origin_cell,
        cells=win.cells,
        cell_m=win.cell_m,
        chunk_size=CHUNK,
        voxel_size=VOXEL,
        materials={k: v.materials for k, v in materials.items()},
        palette=_palette(),
        seq=seq,
    )


class TestWorkerCacheParity:
    def test_assemble_packed_cache_matches_no_cache(self):
        materials = {(0, 0, 0): _varied_chunk()}
        job = _coarse_job(materials)
        plain = assemble_packed(job)
        cached = assemble_packed(job, cache=ChunkBlockCache())
        assert plain.albedo_bytes == cached.albedo_bytes
        assert plain.emis_bytes == cached.emis_bytes

    def test_worker_thread_output_matches_sync_no_cache(self):
        materials = {(0, 0, 0): _varied_chunk()}
        job = _coarse_job(materials)
        truth = assemble_packed(job)  # sync, no cache

        worker = CascadeAssemblyWorker()
        worker.start()
        try:
            worker.submit(job)
            results = []
            deadline = 5.0
            import time

            t0 = time.time()
            while not results and time.time() - t0 < deadline:
                results = worker.drain_results()
            assert results, "worker produced no result"
            res = results[0]
        finally:
            worker.stop()
        assert res.albedo_bytes == truth.albedo_bytes
        assert res.emis_bytes == truth.emis_bytes

    def test_worker_invalidate_recomputes(self):
        chunk = _varied_chunk()
        worker = CascadeAssemblyWorker()
        job1 = _coarse_job({(0, 0, 0): chunk}, seq=1)
        # Populate the worker's cache synchronously through its cache object.
        assemble_packed(job1, cache=worker.block_cache)
        assert len(worker.block_cache) == 1
        worker.invalidate_chunk((0, 0, 0))
        assert len(worker.block_cache) == 0
