"""
tests/buildings/test_types.py — correctness and round-trip tests for
buildings/types.py: BuildingDefaults, Opening, Wall, Room, StairsStub,
Foundation, RoofSlab.

Headless (no panda3d).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fire_engine.buildings.enums import OpeningKind, WallKind
from fire_engine.buildings.types import (
    BuildingDefaults,
    Foundation,
    Opening,
    RoofSlab,
    Room,
    StairsStub,
    Wall,
)
from fire_engine.core.config import Config

# ---------------------------------------------------------------------------
# BuildingDefaults
# ---------------------------------------------------------------------------


class TestBuildingDefaults:
    def test_from_config_field_values(self):
        cfg = Config()
        d = BuildingDefaults.from_config(cfg)
        assert d.storey_height_m == pytest.approx(cfg.building_default_storey_height_m)
        assert d.wall_thickness_m == pytest.approx(cfg.building_default_wall_thickness_m)
        assert d.slab_thickness_m == pytest.approx(cfg.building_slab_thickness_m)
        assert d.foundation_depth_m == pytest.approx(cfg.building_foundation_depth_m)

    def test_to_dict_keys(self):
        d = BuildingDefaults.from_config(Config()).to_dict()
        assert set(d.keys()) == {
            "storey_height_m",
            "wall_thickness_m",
            "slab_thickness_m",
            "foundation_depth_m",
        }

    def test_to_dict_values_are_plain_floats(self):
        d = BuildingDefaults.from_config(Config()).to_dict()
        for k, v in d.items():
            assert isinstance(v, float), f"key {k!r} has type {type(v)}"

    def test_round_trip(self):
        orig = BuildingDefaults.from_config(Config())
        restored = BuildingDefaults.from_dict(orig.to_dict())
        assert restored == orig

    def test_frozen_raises_on_assign(self):
        d = BuildingDefaults.from_config(Config())
        with pytest.raises((AttributeError, TypeError)):
            d.storey_height_m = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Opening
# ---------------------------------------------------------------------------


class TestOpening:
    def _opening(self) -> Opening:
        return Opening(
            id=7, kind=OpeningKind.WINDOW, offset_m=1.0, width_m=1.2, sill_m=0.9, head_m=2.2
        )

    def test_fields(self):
        o = self._opening()
        assert o.id == 7
        assert o.kind is OpeningKind.WINDOW
        assert o.offset_m == pytest.approx(1.0)
        assert o.width_m == pytest.approx(1.2)
        assert o.sill_m == pytest.approx(0.9)
        assert o.head_m == pytest.approx(2.2)

    def test_round_trip(self):
        o = self._opening()
        restored = Opening.from_dict(o.to_dict())
        assert restored.id == o.id
        assert restored.kind is o.kind
        assert restored.offset_m == pytest.approx(o.offset_m)
        assert restored.width_m == pytest.approx(o.width_m)
        assert restored.sill_m == pytest.approx(o.sill_m)
        assert restored.head_m == pytest.approx(o.head_m)

    def test_to_dict_kind_is_string(self):
        d = self._opening().to_dict()
        assert isinstance(d["kind"], str)
        assert d["kind"] == "window"

    def test_door_round_trip(self):
        door = Opening(
            id=3, kind=OpeningKind.DOOR, offset_m=2.0, width_m=0.9, sill_m=0.0, head_m=2.0
        )
        assert Opening.from_dict(door.to_dict()).kind is OpeningKind.DOOR


# ---------------------------------------------------------------------------
# Wall
# ---------------------------------------------------------------------------


class TestWall:
    def test_straight_kind(self):
        w = Wall(id=1, a=(0.0, 0.0), b=(4.0, 0.0))
        assert w.kind is WallKind.SEGMENT

    def test_arc_kind(self):
        w = Wall(id=1, a=(0.0, 0.0), b=(4.0, 0.0), bulge=0.5)
        assert w.kind is WallKind.ARC

    def test_straight_length(self):
        w = Wall(id=1, a=(0.0, 0.0), b=(3.0, 4.0))
        assert w.length_m() == pytest.approx(5.0)

    def test_chord_m(self):
        w = Wall(id=1, a=(0.0, 0.0), b=(3.0, 4.0))
        assert w.chord_m() == pytest.approx(5.0)

    def test_semicircle_length(self):
        # bulge=1 => semicircle; chord=2, radius=1 => arc len = pi
        w = Wall(id=1, a=(0.0, 0.0), b=(2.0, 0.0), bulge=1.0)
        assert w.length_m() == pytest.approx(math.pi)

    def test_arc_params_raises_on_straight(self):
        w = Wall(id=1, a=(0.0, 0.0), b=(2.0, 0.0))
        with pytest.raises(ValueError):
            w.arc_params()

    def test_tessellate_straight_returns_two_points(self):
        w = Wall(id=1, a=(0.0, 0.0), b=(4.0, 0.0))
        pts = w.tessellate(8)
        assert pts.shape == (2, 2)
        np.testing.assert_allclose(pts[0], [0.0, 0.0])
        np.testing.assert_allclose(pts[-1], [4.0, 0.0])

    def test_tessellate_arc_has_more_than_two_points(self):
        w = Wall(id=1, a=(0.0, 0.0), b=(2.0, 0.0), bulge=0.5)
        pts = w.tessellate(8)
        assert pts.shape[0] > 2

    def test_tessellate_endpoints_are_exact(self):
        w = Wall(id=1, a=(1.5, 2.3), b=(4.7, 0.1), bulge=0.3)
        pts = w.tessellate(8)
        assert tuple(pts[0]) == w.a
        assert tuple(pts[-1]) == w.b

    def test_round_trip_no_openings(self):
        w = Wall(id=5, a=(1.0, 2.0), b=(3.0, 4.0), bulge=0.25, thickness_m=0.4, height_m=1.5)
        d = w.to_dict()
        w2 = Wall.from_dict(d)
        assert w2.id == w.id
        assert w2.a == pytest.approx(w.a)
        assert w2.b == pytest.approx(w.b)
        assert w2.bulge == pytest.approx(w.bulge)
        assert w2.thickness_m == pytest.approx(w.thickness_m)
        assert w2.height_m == pytest.approx(w.height_m)
        assert w2.openings == []

    def test_round_trip_with_opening(self):
        o = Opening(
            id=9, kind=OpeningKind.WINDOW, offset_m=0.5, width_m=1.0, sill_m=0.8, head_m=2.0
        )
        w = Wall(id=1, a=(0.0, 0.0), b=(4.0, 0.0), openings=[o])
        w2 = Wall.from_dict(w.to_dict())
        assert len(w2.openings) == 1
        assert w2.openings[0].id == o.id

    def test_to_dict_height_m_none(self):
        w = Wall(id=1, a=(0.0, 0.0), b=(4.0, 0.0))
        assert w.to_dict()["height_m"] is None

    def test_bulge_zero_is_segment(self):
        w = Wall(id=1, a=(0.0, 0.0), b=(1.0, 0.0), bulge=0.0)
        assert w.kind is WallKind.SEGMENT


# ---------------------------------------------------------------------------
# Room
# ---------------------------------------------------------------------------


class TestRoom:
    def _square_room(self) -> Room:
        poly = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]], dtype=np.float64)
        return Room(id=11, polygon=poly, tag="living", meta={"style": "rustic"}, auto=False)

    def test_area_m2(self):
        r = self._square_room()
        assert r.area_m2() == pytest.approx(16.0)

    def test_centroid(self):
        r = self._square_room()
        cx, cy = r.centroid()
        assert cx == pytest.approx(2.0)
        assert cy == pytest.approx(2.0)

    def test_round_trip(self):
        r = self._square_room()
        d = r.to_dict()
        r2 = Room.from_dict(d)
        assert r2.id == r.id
        assert r2.tag == r.tag
        assert r2.auto == r.auto
        assert r2.meta == r.meta
        np.testing.assert_allclose(r2.polygon, r.polygon)

    def test_to_dict_polygon_is_list_of_lists(self):
        d = self._square_room().to_dict()
        assert isinstance(d["polygon"], list)
        assert all(isinstance(row, list) for row in d["polygon"])
        assert all(isinstance(v, float) for row in d["polygon"] for v in row)

    def test_auto_flag_persists(self):
        poly = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]], dtype=np.float64)
        r = Room(id=1, polygon=poly, auto=True)
        r2 = Room.from_dict(r.to_dict())
        assert r2.auto is True

    def test_area_positive_for_ccw(self):
        poly = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        r = Room(id=1, polygon=poly)
        assert r.area_m2() > 0.0


# ---------------------------------------------------------------------------
# StairsStub
# ---------------------------------------------------------------------------


class TestStairsStub:
    def _stub(self) -> StairsStub:
        return StairsStub(
            id=3,
            storey_from=0,
            storey_to=1,
            anchor=(2.0, 3.0),
            direction_rad=math.pi / 4,
            width_m=1.2,
        )

    def test_fields(self):
        s = self._stub()
        assert s.storey_from == 0
        assert s.storey_to == 1
        assert s.anchor == pytest.approx((2.0, 3.0))
        assert s.direction_rad == pytest.approx(math.pi / 4)
        assert s.width_m == pytest.approx(1.2)

    def test_round_trip(self):
        s = self._stub()
        s2 = StairsStub.from_dict(s.to_dict())
        assert s2.id == s.id
        assert s2.storey_from == s.storey_from
        assert s2.storey_to == s.storey_to
        assert s2.anchor == pytest.approx(s.anchor)
        assert s2.direction_rad == pytest.approx(s.direction_rad)
        assert s2.width_m == pytest.approx(s.width_m)

    def test_to_dict_anchor_is_list_of_floats(self):
        d = self._stub().to_dict()
        assert isinstance(d["anchor"], list)
        assert all(isinstance(v, float) for v in d["anchor"])


# ---------------------------------------------------------------------------
# Foundation
# ---------------------------------------------------------------------------


class TestFoundation:
    def _foundation(self) -> Foundation:
        poly = np.array([[0.0, 0.0], [6.0, 0.0], [6.0, 4.0], [0.0, 4.0]], dtype=np.float64)
        return Foundation(polygon=poly, depth_m=0.5)

    def test_fields(self):
        f = self._foundation()
        assert f.depth_m == pytest.approx(0.5)
        assert f.polygon.shape == (4, 2)

    def test_round_trip(self):
        f = self._foundation()
        f2 = Foundation.from_dict(f.to_dict())
        assert f2.depth_m == pytest.approx(f.depth_m)
        np.testing.assert_allclose(f2.polygon, f.polygon)

    def test_to_dict_polygon_is_plain(self):
        d = self._foundation().to_dict()
        assert isinstance(d["polygon"], list)
        assert isinstance(d["depth_m"], float)


# ---------------------------------------------------------------------------
# RoofSlab
# ---------------------------------------------------------------------------


class TestRoofSlab:
    def _roof(self) -> RoofSlab:
        poly = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 3.0], [0.0, 3.0]], dtype=np.float64)
        return RoofSlab(polygon=poly, thickness_m=0.2)

    def test_fields(self):
        r = self._roof()
        assert r.thickness_m == pytest.approx(0.2)
        assert r.polygon.shape == (4, 2)

    def test_round_trip(self):
        r = self._roof()
        r2 = RoofSlab.from_dict(r.to_dict())
        assert r2.thickness_m == pytest.approx(r.thickness_m)
        np.testing.assert_allclose(r2.polygon, r.polygon)

    def test_to_dict_polygon_is_plain(self):
        d = self._roof().to_dict()
        assert isinstance(d["polygon"], list)
        assert isinstance(d["thickness_m"], float)
