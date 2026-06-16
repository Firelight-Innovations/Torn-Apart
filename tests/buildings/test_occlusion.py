"""
tests/buildings/test_occlusion.py — BuildingOccupancyRasterizer: no-op contract,
warn-once logging, array immutability, and construction.

Headless (no panda3d). The rasterizer is a v1 no-op, so these tests pin that
it does NOT modify the arrays it receives (regression guard for when the
algorithm is implemented).
"""

from __future__ import annotations

import numpy as np

from fire_engine.buildings.occlusion import BuildingOccupancyRasterizer


class _FakeManager:
    """Minimal stand-in so BuildingOccupancyRasterizer can be constructed."""

    def buildings(self):
        return []


def _rasterizer() -> BuildingOccupancyRasterizer:
    return BuildingOccupancyRasterizer(_FakeManager())


class TestConstruction:
    def test_instantiates_with_manager(self):
        r = _rasterizer()
        assert r is not None

    def test_stores_manager(self):
        mgr = _FakeManager()
        r = BuildingOccupancyRasterizer(mgr)
        assert r._manager is mgr

    def test_warn_flag_starts_false(self):
        r = _rasterizer()
        assert r._warned is False


class TestNoOpContract:
    """rasterize_occupancy must be a true no-op: arrays are unchanged."""

    def _arrays(self, cells: int = 4):
        albedo_occ = np.zeros((cells, cells, cells, 4), dtype=np.float32)
        albedo_occ[1, 1, 1, 3] = 0.7  # sentinel value
        emission = np.zeros((cells, cells, cells, 3), dtype=np.float32)
        emission[0, 0, 0, 0] = 1.0
        return albedo_occ, emission

    def test_albedo_occ_unchanged(self):
        r = _rasterizer()
        albedo_occ, emission = self._arrays()
        before = albedo_occ.copy()
        r.rasterize_occupancy((0, 0, 0), 4, 1.0, albedo_occ, emission)
        np.testing.assert_array_equal(albedo_occ, before)

    def test_emission_unchanged(self):
        r = _rasterizer()
        albedo_occ, emission = self._arrays()
        before = emission.copy()
        r.rasterize_occupancy((0, 0, 0), 4, 1.0, albedo_occ, emission)
        np.testing.assert_array_equal(emission, before)

    def test_returns_none(self):
        r = _rasterizer()
        albedo_occ, emission = self._arrays()
        result = r.rasterize_occupancy((0, 0, 0), 4, 1.0, albedo_occ, emission)
        assert result is None

    def test_warn_flag_set_after_first_call(self):
        r = _rasterizer()
        albedo_occ, emission = self._arrays()
        assert r._warned is False
        r.rasterize_occupancy((0, 0, 0), 4, 1.0, albedo_occ, emission)
        assert r._warned is True

    def test_warn_flag_stays_true_on_repeat_calls(self):
        r = _rasterizer()
        albedo_occ, emission = self._arrays()
        r.rasterize_occupancy((0, 0, 0), 4, 1.0, albedo_occ, emission)
        r.rasterize_occupancy((1, 0, 0), 4, 1.0, albedo_occ, emission)
        assert r._warned is True

    def test_different_origin_and_cell_size_still_noop(self):
        r = _rasterizer()
        albedo_occ = np.ones((8, 8, 8, 4), dtype=np.float32)
        emission = np.zeros((8, 8, 8, 3), dtype=np.float32)
        before_a = albedo_occ.copy()
        r.rasterize_occupancy((-16, -16, 0), 8, 0.5, albedo_occ, emission)
        np.testing.assert_array_equal(albedo_occ, before_a)

    def test_exported_in_all(self):
        import fire_engine.buildings.occlusion as m

        assert "BuildingOccupancyRasterizer" in m.__all__
