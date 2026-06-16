"""
tests/lighting/test_volume.py — Headless tests for fire_engine.lighting.volume.

Covers:
- VolumeWindow: construction, recenter, hysteresis, snapping.
- assemble_geometry: cascade-0 (binary occupancy), cascade-1 (fractional).
- EMISSION_SCALE constant.
- GeometryVolume dataclass fields.

No panda3d imports.
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.lighting.palette import MaterialPalette
from fire_engine.lighting.volume import (
    EMISSION_SCALE,
    GeometryVolume,
    VolumeWindow,
    assemble_geometry,
)

VOXEL = 0.5
CHUNK = 32


class _Chunk:
    def __init__(self) -> None:
        self.materials = np.zeros((CHUNK, CHUNK, CHUNK), dtype=np.uint8)


def _palette() -> MaterialPalette:
    albedo = np.zeros((256, 3), dtype=np.float32)
    albedo[1] = (0.4, 0.3, 0.2)
    albedo[2] = (0.2, 0.5, 0.1)
    return MaterialPalette(albedo=albedo, emission=np.zeros((256, 3), dtype=np.float32))


# ---------------------------------------------------------------------------
# EMISSION_SCALE constant
# ---------------------------------------------------------------------------


class TestEmissionScale:
    def test_emission_scale_is_positive(self):
        assert EMISSION_SCALE > 0.0

    def test_emission_scale_value_is_8(self):
        assert EMISSION_SCALE == 8.0


# ---------------------------------------------------------------------------
# VolumeWindow
# ---------------------------------------------------------------------------


class TestVolumeWindow:
    def test_first_recenter_returns_true(self):
        win = VolumeWindow(cells=32, cell_m=VOXEL)
        assert win.recenter((8.0, 8.0, 8.0)) is True

    def test_origin_snaps_to_snap_cells(self):
        win = VolumeWindow(cells=96, cell_m=VOXEL, snap_cells=8)
        win.recenter((13.7, -4.2, 9.9))
        assert all(o % 8 == 0 for o in win.origin_cell)

    def test_hysteresis_no_move_within_margin(self):
        win = VolumeWindow(cells=96, cell_m=VOXEL, margin_cells=8)
        win.recenter((0.0, 0.0, 0.0))
        assert win.recenter((1.0, 1.0, 1.0)) is False

    def test_recenter_after_large_move_returns_true(self):
        win = VolumeWindow(cells=96, cell_m=VOXEL, margin_cells=8)
        win.recenter((0.0, 0.0, 0.0))
        assert win.recenter((20.0, 0.0, 0.0)) is True

    def test_world_origin_before_recenter_raises(self):
        win = VolumeWindow(cells=32, cell_m=VOXEL)
        with pytest.raises(ValueError):
            _ = win.world_origin_m

    def test_cells_must_divide_by_snap(self):
        with pytest.raises(ValueError):
            VolumeWindow(cells=30, cell_m=VOXEL, snap_cells=8)


# ---------------------------------------------------------------------------
# assemble_geometry — cascade 0
# ---------------------------------------------------------------------------


class TestAssembleCascade0:
    def _window_at_origin(self, cells=32):
        win = VolumeWindow(cells=cells, cell_m=VOXEL, snap_cells=8)
        win.recenter((8.0, 8.0, 8.0))
        assert win.origin_cell == (0, 0, 0)
        return win

    def test_solid_voxel_occupancy_255(self):
        win = self._window_at_origin()
        chunk = _Chunk()
        chunk.materials[5, 6, 7] = 1
        vol = assemble_geometry(win, {(0, 0, 0): chunk}, _palette(), CHUNK, VOXEL)
        assert vol.albedo_occ[5, 6, 7, 3] == 255

    def test_air_voxel_occupancy_0(self):
        win = self._window_at_origin()
        chunk = _Chunk()
        chunk.materials[5, 6, 7] = 1
        vol = assemble_geometry(win, {(0, 0, 0): chunk}, _palette(), CHUNK, VOXEL)
        assert vol.albedo_occ[5, 6, 6, 3] == 0

    def test_missing_chunks_are_air(self):
        win = self._window_at_origin()
        vol = assemble_geometry(win, {}, _palette(), CHUNK, VOXEL)
        assert int(vol.albedo_occ[..., 3].max()) == 0

    def test_emission_packing(self):
        win = self._window_at_origin()
        chunk = _Chunk()
        chunk.materials[1, 1, 1] = 3
        pal = _palette().with_emission(3, (2.0, 1.0, 0.5))
        vol = assemble_geometry(win, {(0, 0, 0): chunk}, pal, CHUNK, VOXEL)
        expected = np.clip(np.array([2.0, 1.0, 0.5]) * 255.0 / EMISSION_SCALE, 0, 255).astype(
            np.uint8
        )
        np.testing.assert_array_equal(vol.emission[1, 1, 1, :3], expected)

    def test_deterministic(self):
        chunk = _Chunk()
        chunk.materials[:, :, :8] = 1
        outs = []
        for _ in range(2):
            win = self._window_at_origin()
            vol = assemble_geometry(win, {(0, 0, 0): chunk}, _palette(), CHUNK, VOXEL)
            outs.append(vol.albedo_occ.tobytes() + vol.emission.tobytes())
        assert outs[0] == outs[1]

    def test_returns_geometry_volume(self):
        win = self._window_at_origin()
        vol = assemble_geometry(win, {}, _palette(), CHUNK, VOXEL)
        assert isinstance(vol, GeometryVolume)
        assert vol.albedo_occ.shape == (32, 32, 32, 4)
        assert vol.emission.shape == (32, 32, 32, 4)


# ---------------------------------------------------------------------------
# assemble_geometry — cascade 1 (fractional occupancy)
# ---------------------------------------------------------------------------


class TestAssembleCascade1:
    def test_partial_occupancy_value(self):
        win = VolumeWindow(cells=8, cell_m=2.0, snap_cells=8)
        win.recenter((8.0, 8.0, 8.0))
        chunk = _Chunk()
        chunk.materials[0, 0, 0] = 1  # one solid voxel in a 4^3 = 64 sub-voxel cell
        vol = assemble_geometry(win, {(0, 0, 0): chunk}, _palette(), CHUNK, VOXEL)
        assert vol.albedo_occ[0, 0, 0, 3] == round(255.0 / 64.0)

    def test_non_integer_cell_ratio_raises(self):
        win = VolumeWindow(cells=16, cell_m=0.75, snap_cells=8)
        win.recenter((0.0, 0.0, 0.0))
        with pytest.raises(ValueError):
            assemble_geometry(win, {}, _palette(), CHUNK, VOXEL)


# ---------------------------------------------------------------------------
# GeometryVolume dataclass
# ---------------------------------------------------------------------------


class TestGeometryVolume:
    def test_fields_accessible(self):
        ao = np.zeros((8, 8, 8, 4), dtype=np.uint8)
        em = np.zeros((8, 8, 8, 4), dtype=np.uint8)
        gv = GeometryVolume(albedo_occ=ao, emission=em, origin_cell=(0, 0, 0), cell_m=0.5)
        assert gv.albedo_occ is ao
        assert gv.emission is em
        assert gv.origin_cell == (0, 0, 0)
        assert gv.cell_m == 0.5
