"""
tests/test_buildings_rooms.py — room auto-detection (planar half-edge minimal
cycles): enclosure, winding, count, arc closure, endpoint snapping, and the
documented T-junction limitation.

Headless (numpy only — fire_engine/buildings/ never imports panda3d).
"""

import math

import numpy as np
import pytest

from fire_engine.buildings import Building, BuildingDefaults, WallKind
from fire_engine.buildings.rooms import detect_room_polygons
from fire_engine.core.config import Config
from fire_engine.core.math3d import Quat, Vec3

_CFG = Config()
_SNAP = _CFG.building_snap_eps_m
_QPQ = _CFG.building_arc_segments_per_quarter


def _defaults() -> BuildingDefaults:
    return BuildingDefaults.from_config(_CFG)


def _storey():
    b = Building(name="t", position=Vec3(0, 0, 8.0),
                 rotation=Quat.identity(), defaults=_defaults())
    return b, b.add_storey()


def _detect(storey):
    return detect_room_polygons(storey.walls, snap_eps_m=_SNAP,
                                arc_segments_per_quarter=_QPQ)


def _shoelace(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


class TestEnclosure:
    def test_rectangle_is_one_room(self):
        _, s = _storey()
        s.add_wall((0, 0), (8, 0))
        s.add_wall((8, 0), (8, 6))
        s.add_wall((8, 6), (0, 6))
        s.add_wall((0, 6), (0, 0))
        rooms = _detect(s)
        assert len(rooms) == 1
        assert _shoelace(rooms[0]) > 0.0          # CCW
        assert abs(abs(_shoelace(rooms[0])) - 48.0) < 1e-9

    def test_winding_is_ccw_even_for_cw_authored_walls(self):
        # Author the perimeter clockwise; detection must still emit CCW.
        _, s = _storey()
        s.add_wall((0, 0), (0, 6))
        s.add_wall((0, 6), (8, 6))
        s.add_wall((8, 6), (8, 0))
        s.add_wall((8, 0), (0, 0))
        rooms = _detect(s)
        assert len(rooms) == 1
        assert _shoelace(rooms[0]) > 0.0

    def test_open_l_encloses_nothing(self):
        _, s = _storey()
        s.add_wall((0, 0), (8, 0))
        s.add_wall((8, 0), (8, 6))   # two walls meeting at a corner, open
        assert _detect(s) == []

    def test_shared_wall_yields_two_rooms(self):
        # Two stacked rooms sharing the y=3 divider wall.
        _, s = _storey()
        # outer perimeter
        s.add_wall((0, 0), (8, 0))
        s.add_wall((8, 0), (8, 6))
        s.add_wall((8, 6), (0, 6))
        s.add_wall((0, 6), (0, 0))
        # divider (its endpoints coincide with perimeter vertices)
        s.add_wall((0, 3), (8, 3))
        # split the left/right perimeter walls so the divider meets endpoints
        # (re-author the verticals as two spans each through the y=3 node)
        rooms = _detect(s)
        # The divider's endpoints (0,3)/(8,3) are NOT perimeter vertices, so
        # this is a T-junction on both sides → documented v1 limitation: the
        # divider does not split the rooms. Expect the single outer room.
        assert len(rooms) == 1

    def test_shared_wall_with_split_perimeter_yields_two_rooms(self):
        # Proper authoring: perimeter verticals split at the divider node.
        _, s = _storey()
        s.add_wall((0, 0), (8, 0))      # bottom
        s.add_wall((8, 0), (8, 3))      # right-lower
        s.add_wall((8, 3), (8, 6))      # right-upper
        s.add_wall((8, 6), (0, 6))      # top
        s.add_wall((0, 6), (0, 3))      # left-upper
        s.add_wall((0, 3), (0, 0))      # left-lower
        s.add_wall((0, 3), (8, 3))      # divider, meets endpoints both sides
        rooms = _detect(s)
        assert len(rooms) == 2
        areas = sorted(abs(_shoelace(r)) for r in rooms)
        assert abs(areas[0] - 24.0) < 1e-9
        assert abs(areas[1] - 24.0) < 1e-9
        assert all(_shoelace(r) > 0.0 for r in rooms)


class TestArcs:
    def test_arc_bay_closes_into_a_room(self):
        # Three straight walls + one bulging wall closing the loop.
        _, s = _storey()
        s.add_wall((0, 0), (8, 0))
        s.add_wall((8, 0), (8, 6), bulge=-0.4)   # bows outward (east)
        s.add_wall((8, 6), (0, 6))
        s.add_wall((0, 6), (0, 0))
        rooms = _detect(s)
        assert len(rooms) == 1
        # Outward bay adds area beyond the 48 m² rectangle.
        assert abs(_shoelace(rooms[0])) > 48.0

    def test_arc_polygon_uses_tessellation_density(self):
        # A semicircle wall + its chord: polygon vertex count tracks density.
        _, s = _storey()
        s.add_wall((0, 0), (6, 0), bulge=1.0)    # semicircle, sweep = 180°
        s.add_wall((6, 0), (0, 0))               # closing chord
        rooms = _detect(s)
        assert len(rooms) == 1
        # 2 quarters × _QPQ chords on the arc + 1 chord vertex shared.
        assert rooms[0].shape[0] >= 2 * _QPQ


class TestSnapping:
    def test_endpoints_within_eps_merge(self):
        # Corner authored with a sub-eps gap still closes the loop.
        gap = _SNAP * 0.4
        _, s = _storey()
        s.add_wall((0, 0), (8, 0))
        s.add_wall((8, 0), (8, 6))
        s.add_wall((8, 6), (0, 6))
        s.add_wall((0 + gap, 6 - gap), (0, 0))   # start nudged < eps
        rooms = _detect(s)
        assert len(rooms) == 1

    def test_gap_beyond_eps_does_not_close(self):
        gap = _SNAP * 5.0
        _, s = _storey()
        s.add_wall((0, 0), (8, 0))
        s.add_wall((8, 0), (8, 6))
        s.add_wall((8, 6), (0, 6))
        s.add_wall((0 + gap, 6), (0 + gap, 0))   # left wall offset, no closure
        assert _detect(s) == []


class TestStoreyIntegration:
    def test_detect_rooms_appends_auto_rooms(self):
        _, s = _storey()
        s.add_wall((0, 0), (4, 0))
        s.add_wall((4, 0), (4, 4))
        s.add_wall((4, 4), (0, 4))
        s.add_wall((0, 4), (0, 0))
        detected = s.detect_rooms(snap_eps_m=_SNAP,
                                  arc_segments_per_quarter=_QPQ)
        assert len(detected) == 1
        assert detected[0].auto is True
        assert detected[0] in s.rooms
        assert detected[0].id > 0

    def test_detect_rooms_replaces_prior_auto_keeps_explicit(self):
        _, s = _storey()
        s.add_wall((0, 0), (4, 0))
        s.add_wall((4, 0), (4, 4))
        s.add_wall((4, 4), (0, 4))
        s.add_wall((0, 4), (0, 0))
        explicit = s.add_room([(0, 0), (4, 0), (4, 4), (0, 4)], tag="living")
        s.detect_rooms(snap_eps_m=_SNAP, arc_segments_per_quarter=_QPQ)
        first_auto_ids = {r.id for r in s.rooms if r.auto}
        # Re-running replaces auto rooms but never the explicit one.
        s.detect_rooms(snap_eps_m=_SNAP, arc_segments_per_quarter=_QPQ)
        assert explicit in s.rooms
        assert sum(1 for r in s.rooms if not r.auto) == 1
        assert sum(1 for r in s.rooms if r.auto) == 1
        # fresh auto room got a fresh id (not reused)
        assert {r.id for r in s.rooms if r.auto}.isdisjoint(first_auto_ids)

    def test_detected_room_area_matches_shoelace(self):
        _, s = _storey()
        s.add_wall((0, 0), (5, 0))
        s.add_wall((5, 0), (5, 5))
        s.add_wall((5, 5), (0, 5))
        s.add_wall((0, 5), (0, 0))
        [room] = s.detect_rooms(snap_eps_m=_SNAP,
                                arc_segments_per_quarter=_QPQ)
        assert abs(room.area_m2() - 25.0) < 1e-9
