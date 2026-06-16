"""
tests/buildings/_impl/test_storey.py — Storey class
(buildings/_impl/storey.py): authoring API, element-id allocation,
wall/opening/room/stairs mutations, detect_rooms integration, and
serialisation round-trip.

Headless (no panda3d).
"""

from __future__ import annotations

import math

import pytest

from fire_engine.buildings._impl.storey import Storey
from fire_engine.buildings.enums import OpeningKind, WallKind
from fire_engine.buildings.model import Building, BuildingDefaults
from fire_engine.core.config import Config
from fire_engine.core.math3d import Quat, Vec3


def _building(name: str = "test") -> Building:
    d = BuildingDefaults.from_config(Config())
    return Building(name=name, position=Vec3(0.0, 0.0, 0.0), rotation=Quat.identity(), defaults=d)


def _storey() -> tuple[Building, Storey]:
    b = _building()
    s = b.add_storey()
    return b, s


# ---------------------------------------------------------------------------
# Storey construction
# ---------------------------------------------------------------------------


class TestStoreyAttributes:
    def test_index_starts_at_zero(self):
        _, s = _storey()
        assert s.index == 0

    def test_height_from_defaults(self):
        _, s = _storey()
        cfg = Config()
        assert s.height_m == pytest.approx(cfg.building_default_storey_height_m)

    def test_slab_from_defaults(self):
        _, s = _storey()
        cfg = Config()
        assert s.slab_m == pytest.approx(cfg.building_slab_thickness_m)

    def test_walls_empty(self):
        _, s = _storey()
        assert s.walls == []

    def test_rooms_empty(self):
        _, s = _storey()
        assert s.rooms == []

    def test_stairs_empty(self):
        _, s = _storey()
        assert s.stairs == []

    def test_id_is_positive(self):
        _, s = _storey()
        assert s.id > 0

    def test_second_storey_index_is_one(self):
        b = _building()
        s0 = b.add_storey()
        s1 = b.add_storey()
        assert s1.index == 1
        assert s1.index > s0.index


# ---------------------------------------------------------------------------
# add_wall
# ---------------------------------------------------------------------------


class TestAddWall:
    def test_returns_wall_appended(self):
        _, s = _storey()
        w = s.add_wall((0, 0), (4, 0))
        assert w in s.walls
        assert len(s.walls) == 1

    def test_default_thickness_from_building(self):
        _, s = _storey()
        w = s.add_wall((0, 0), (4, 0))
        cfg = Config()
        assert w.thickness_m == pytest.approx(cfg.building_default_wall_thickness_m)

    def test_explicit_thickness(self):
        _, s = _storey()
        w = s.add_wall((0, 0), (4, 0), thickness_m=0.4)
        assert w.thickness_m == pytest.approx(0.4)

    def test_bulge_zero_is_segment(self):
        _, s = _storey()
        w = s.add_wall((0, 0), (4, 0))
        assert w.kind is WallKind.SEGMENT

    def test_bulge_nonzero_is_arc(self):
        _, s = _storey()
        w = s.add_wall((0, 0), (4, 0), bulge=0.5)
        assert w.kind is WallKind.ARC

    def test_height_m_none_by_default(self):
        _, s = _storey()
        w = s.add_wall((0, 0), (4, 0))
        assert w.height_m is None

    def test_height_m_explicit(self):
        _, s = _storey()
        w = s.add_wall((0, 0), (4, 0), height_m=1.1)
        assert w.height_m == pytest.approx(1.1)

    def test_coincident_endpoints_raise(self):
        _, s = _storey()
        with pytest.raises(ValueError):
            s.add_wall((1, 1), (1, 1))

    def test_ids_monotonic(self):
        _, s = _storey()
        w1 = s.add_wall((0, 0), (4, 0))
        w2 = s.add_wall((4, 0), (4, 4))
        assert w2.id > w1.id


# ---------------------------------------------------------------------------
# add_opening
# ---------------------------------------------------------------------------


class TestAddOpening:
    def test_appends_to_wall(self):
        _, s = _storey()
        w = s.add_wall((0, 0), (4, 0))
        o = s.add_opening(
            w.id, OpeningKind.WINDOW, offset_m=1.0, width_m=1.2, sill_m=0.8, head_m=2.0
        )
        assert o in w.openings

    def test_door_defaults_sill_zero(self):
        _, s = _storey()
        w = s.add_wall((0, 0), (4, 0))
        o = s.add_opening(w.id, OpeningKind.DOOR, offset_m=1.0, width_m=0.9, head_m=2.0)
        assert o.sill_m == pytest.approx(0.0)

    def test_unknown_wall_id_raises_key_error(self):
        _, s = _storey()
        with pytest.raises(KeyError):
            s.add_opening(999, OpeningKind.DOOR, offset_m=0.0, width_m=0.9, head_m=2.0)

    def test_opening_beyond_wall_length_raises(self):
        _, s = _storey()
        w = s.add_wall((0, 0), (2, 0))  # length=2
        with pytest.raises(ValueError):
            s.add_opening(
                w.id, OpeningKind.WINDOW, offset_m=1.5, width_m=1.0, sill_m=0.8, head_m=2.0
            )

    def test_sill_above_head_raises(self):
        _, s = _storey()
        w = s.add_wall((0, 0), (4, 0))
        with pytest.raises(ValueError):
            s.add_opening(
                w.id, OpeningKind.WINDOW, offset_m=1.0, width_m=1.0, sill_m=2.0, head_m=1.5
            )

    def test_head_above_band_raises(self):
        _, s = _storey()  # height=3.0, slab=0.2 → band=2.8
        w = s.add_wall((0, 0), (4, 0))
        with pytest.raises(ValueError):
            s.add_opening(
                w.id, OpeningKind.WINDOW, offset_m=1.0, width_m=1.0, sill_m=0.8, head_m=3.0
            )

    def test_opening_id_allocated(self):
        _, s = _storey()
        w = s.add_wall((0, 0), (4, 0))
        o = s.add_opening(
            w.id, OpeningKind.WINDOW, offset_m=0.5, width_m=1.0, sill_m=0.8, head_m=2.0
        )
        assert o.id > w.id  # strictly later element id


# ---------------------------------------------------------------------------
# add_room
# ---------------------------------------------------------------------------


class TestAddRoom:
    def test_ccw_polygon_stored_as_is(self):
        _, s = _storey()
        poly = [(0, 0), (4, 0), (4, 4), (0, 4)]
        room = s.add_room(poly, tag="living")
        assert room.area_m2() == pytest.approx(16.0)
        assert room.tag == "living"
        assert room.auto is False

    def test_cw_polygon_reversed_to_ccw(self):
        _, s = _storey()
        cw = [(0, 0), (0, 4), (4, 4), (4, 0)]
        room = s.add_room(cw)
        assert room.area_m2() > 0.0  # shoelace positive = CCW

    def test_degenerate_polygon_raises(self):
        _, s = _storey()
        with pytest.raises(ValueError):
            s.add_room([(0, 0), (1, 0)])  # only 2 points

    def test_room_appended(self):
        _, s = _storey()
        room = s.add_room([(0, 0), (2, 0), (2, 2), (0, 2)])
        assert room in s.rooms

    def test_meta_stored(self):
        _, s = _storey()
        room = s.add_room([(0, 0), (2, 0), (2, 2), (0, 2)], meta={"style": "rustic"})
        assert room.meta == {"style": "rustic"}


# ---------------------------------------------------------------------------
# add_stairs
# ---------------------------------------------------------------------------


class TestAddStairs:
    def test_stub_returned_and_stored(self):
        b = _building()
        s0 = b.add_storey()
        b.add_storey()  # storey_to=1 must exist
        stub = s0.add_stairs(storey_to=1, anchor=(2.0, 3.0), direction_rad=0.0, width_m=1.0)
        assert stub in s0.stairs
        assert stub.storey_from == 0
        assert stub.storey_to == 1

    def test_direction_preserved(self):
        b = _building()
        s0 = b.add_storey()
        b.add_storey()
        stub = s0.add_stairs(storey_to=1, anchor=(0, 0), direction_rad=math.pi / 2, width_m=1.2)
        assert stub.direction_rad == pytest.approx(math.pi / 2)


# ---------------------------------------------------------------------------
# detect_rooms
# ---------------------------------------------------------------------------


class TestDetectRooms:
    def _square_storey(self) -> Storey:
        _, s = _storey()
        s.add_wall((0, 0), (4, 0))
        s.add_wall((4, 0), (4, 4))
        s.add_wall((4, 4), (0, 4))
        s.add_wall((0, 4), (0, 0))
        return s

    def test_closed_square_detects_one_room(self):
        cfg = Config()
        s = self._square_storey()
        rooms = s.detect_rooms(
            snap_eps_m=cfg.building_snap_eps_m,
            arc_segments_per_quarter=cfg.building_arc_segments_per_quarter,
        )
        assert len(rooms) == 1
        assert rooms[0].auto is True

    def test_detected_rooms_appended_to_storey(self):
        cfg = Config()
        s = self._square_storey()
        rooms = s.detect_rooms(
            snap_eps_m=cfg.building_snap_eps_m,
            arc_segments_per_quarter=cfg.building_arc_segments_per_quarter,
        )
        for r in rooms:
            assert r in s.rooms

    def test_re_detect_replaces_auto_keeps_explicit(self):
        cfg = Config()
        s = self._square_storey()
        explicit = s.add_room([(0, 0), (4, 0), (4, 4), (0, 4)], tag="living")
        s.detect_rooms(
            snap_eps_m=cfg.building_snap_eps_m,
            arc_segments_per_quarter=cfg.building_arc_segments_per_quarter,
        )
        # Re-detect replaces auto rooms but keeps explicit.
        first_auto_ids = {r.id for r in s.rooms if r.auto}
        s.detect_rooms(
            snap_eps_m=cfg.building_snap_eps_m,
            arc_segments_per_quarter=cfg.building_arc_segments_per_quarter,
        )
        assert explicit in s.rooms
        # Fresh auto room has a new id.
        new_auto_ids = {r.id for r in s.rooms if r.auto}
        assert new_auto_ids.isdisjoint(first_auto_ids)

    def test_open_storey_detects_no_rooms(self):
        cfg = Config()
        _, s = _storey()
        s.add_wall((0, 0), (4, 0))  # single unclosed wall
        rooms = s.detect_rooms(
            snap_eps_m=cfg.building_snap_eps_m,
            arc_segments_per_quarter=cfg.building_arc_segments_per_quarter,
        )
        assert rooms == []


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def _full_storey(self) -> tuple[Building, Storey]:
        b = _building()
        s = b.add_storey(height_m=3.5)
        w = s.add_wall((0, 0), (6, 0), thickness_m=0.4)
        s.add_wall((6, 0), (6, 4))
        s.add_wall((6, 4), (0, 4))
        s.add_wall((0, 4), (0, 0))
        s.add_opening(w.id, OpeningKind.DOOR, offset_m=2.0, width_m=1.0, head_m=2.2)
        s.add_room([(0, 0), (6, 0), (6, 4), (0, 4)], tag="main")
        b.add_storey()
        s.add_stairs(storey_to=1, anchor=(5, 3), direction_rad=math.pi, width_m=1.0)
        return b, s

    def test_wall_count_survives(self):
        b, s = self._full_storey()
        s2 = Storey.from_dict(b, s.to_dict())
        assert len(s2.walls) == len(s.walls)

    def test_room_count_survives(self):
        b, s = self._full_storey()
        s2 = Storey.from_dict(b, s.to_dict())
        assert len(s2.rooms) == len(s.rooms)

    def test_stairs_count_survives(self):
        b, s = self._full_storey()
        s2 = Storey.from_dict(b, s.to_dict())
        assert len(s2.stairs) == len(s.stairs)

    def test_height_survives(self):
        b, s = self._full_storey()
        s2 = Storey.from_dict(b, s.to_dict())
        assert s2.height_m == pytest.approx(s.height_m)

    def test_index_survives(self):
        b, s = self._full_storey()
        s2 = Storey.from_dict(b, s.to_dict())
        assert s2.index == s.index

    def test_opening_kind_survives(self):
        b, s = self._full_storey()
        s2 = Storey.from_dict(b, s.to_dict())
        openings = [o for w in s2.walls for o in w.openings]
        assert len(openings) == 1
        assert openings[0].kind is OpeningKind.DOOR
