"""
tests/lighting/_impl/test_types.py — Headless tests for
fire_engine.lighting._impl.types.

Covers:
- PointLight, AreaLight, SpotLight dataclass field access and defaults.
- AssemblyJob / AssemblyResult frozen dataclass construction.
- GeometryVolume dataclass field access.
- Re-exports: symbols importable via parent modules.

No panda3d imports.
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.lighting._impl.types import (
    AreaLight,
    AssemblyJob,
    AssemblyResult,
    GeometryVolume,
    PointLight,
    SpotLight,
)
from fire_engine.lighting.palette import MaterialPalette

# ---------------------------------------------------------------------------
# PointLight
# ---------------------------------------------------------------------------


class TestPointLight:
    def test_fields_accessible(self):
        pl = PointLight(position=(1.0, 2.0, 3.0), color=(0.9, 0.8, 0.7), intensity=4.0, radius=12.0)
        assert pl.position == (1.0, 2.0, 3.0)
        assert pl.color == (0.9, 0.8, 0.7)
        assert pl.intensity == 4.0
        assert pl.radius == 12.0

    def test_ttl_default_is_none(self):
        pl = PointLight((0, 0, 0), (1, 1, 1), 1.0, 5.0)
        assert pl.ttl_s is None

    def test_ttl_can_be_set(self):
        pl = PointLight((0, 0, 0), (1, 1, 1), 1.0, 5.0, ttl_s=0.5)
        assert pl.ttl_s == pytest.approx(0.5)

    def test_position_is_mutable(self):
        """PointLight is a plain dataclass — position field can be mutated."""
        pl = PointLight((0.0, 0.0, 0.0), (1, 1, 1), 1.0, 5.0)
        pl.position = (3.0, 4.0, 5.0)
        assert pl.position == (3.0, 4.0, 5.0)


# ---------------------------------------------------------------------------
# AreaLight
# ---------------------------------------------------------------------------


class TestAreaLight:
    def test_fields_accessible(self):
        al = AreaLight(
            center=(0.0, 0.0, 5.0),
            half_extents=(2.0, 1.0, 0.5),
            color=(1.0, 1.0, 1.0),
            intensity=3.0,
            radius=10.0,
        )
        assert al.center == (0.0, 0.0, 5.0)
        assert al.half_extents == (2.0, 1.0, 0.5)
        assert al.intensity == pytest.approx(3.0)

    def test_ttl_default_is_none(self):
        al = AreaLight((0, 0, 0), (1, 1, 1), (1, 1, 1), 1.0, 5.0)
        assert al.ttl_s is None


# ---------------------------------------------------------------------------
# SpotLight
# ---------------------------------------------------------------------------


class TestSpotLight:
    def test_fields_accessible(self):
        sl = SpotLight(
            position=(0.0, 0.0, 10.0),
            direction=(0.0, 1.0, 0.0),
            color=(1.0, 0.95, 0.8),
            intensity=14.0,
            radius=30.0,
            cone_deg=42.0,
        )
        assert sl.position == (0.0, 0.0, 10.0)
        assert sl.direction == (0.0, 1.0, 0.0)
        assert sl.cone_deg == pytest.approx(42.0)

    def test_cone_deg_default_is_42(self):
        sl = SpotLight((0, 0, 0), (0, 1, 0), (1, 1, 1), 1.0, 5.0)
        assert sl.cone_deg == pytest.approx(42.0)

    def test_ttl_default_is_none(self):
        sl = SpotLight((0, 0, 0), (0, 1, 0), (1, 1, 1), 1.0, 5.0)
        assert sl.ttl_s is None


# ---------------------------------------------------------------------------
# AssemblyJob
# ---------------------------------------------------------------------------


class TestAssemblyJob:
    def _make_job(self, **overrides) -> AssemblyJob:
        defaults: dict = dict(
            cascade_index=0,
            origin_cell=(0, 0, 0),
            cells=32,
            cell_m=0.5,
            chunk_size=32,
            voxel_size=0.5,
            materials={},
            palette=MaterialPalette(),
            seq=0,
        )
        defaults.update(overrides)
        return AssemblyJob(**defaults)

    def test_fields_accessible(self):
        job = self._make_job(cascade_index=1, seq=5)
        assert job.cascade_index == 1
        assert job.seq == 5
        assert job.cells == 32
        assert job.cell_m == pytest.approx(0.5)

    def test_default_occluders_none(self):
        assert self._make_job().occluders is None

    def test_default_trunk_occ_zero(self):
        assert self._make_job().trunk_occ == pytest.approx(0.0)

    def test_default_canopy_gain_zero(self):
        assert self._make_job().canopy_gain == pytest.approx(0.0)

    def test_frozen(self):
        """AssemblyJob is frozen — mutation must raise."""
        job = self._make_job()
        with pytest.raises((TypeError, AttributeError)):
            job.seq = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AssemblyResult
# ---------------------------------------------------------------------------


class TestAssemblyResult:
    def test_fields_accessible(self):
        res = AssemblyResult(
            cascade_index=0,
            origin_cell=(1, 2, 3),
            albedo_bytes=b"abc",
            emis_bytes=b"def",
            seq=7,
        )
        assert res.cascade_index == 0
        assert res.origin_cell == (1, 2, 3)
        assert res.albedo_bytes == b"abc"
        assert res.seq == 7

    def test_frozen(self):
        res = AssemblyResult(0, (0, 0, 0), b"", b"", 0)
        with pytest.raises((TypeError, AttributeError)):
            res.seq = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GeometryVolume
# ---------------------------------------------------------------------------


class TestGeometryVolume:
    def test_fields_accessible(self):
        ao = np.zeros((4, 4, 4, 4), dtype=np.uint8)
        em = np.zeros((4, 4, 4, 4), dtype=np.uint8)
        gv = GeometryVolume(albedo_occ=ao, emission=em, origin_cell=(5, 6, 7), cell_m=1.0)
        assert gv.albedo_occ is ao
        assert gv.emission is em
        assert gv.origin_cell == (5, 6, 7)
        assert gv.cell_m == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Re-exports via parent modules
# ---------------------------------------------------------------------------


class TestReExports:
    def test_point_light_from_lights(self):
        from fire_engine.lighting.lights import PointLight as PL

        assert PL is PointLight

    def test_area_light_from_lights(self):
        from fire_engine.lighting.lights import AreaLight as AL

        assert AL is AreaLight

    def test_spot_light_from_lights(self):
        from fire_engine.lighting.lights import SpotLight as SL

        assert SL is SpotLight

    def test_geometry_volume_from_volume(self):
        from fire_engine.lighting.volume import GeometryVolume as GV

        assert GV is GeometryVolume
