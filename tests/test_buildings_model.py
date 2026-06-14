"""
tests/test_buildings_model.py — buildings data model: authoring API,
id allocation, elevation math, arc geometry, dict round-trips, world AABB.

Headless (numpy only — fire_engine/buildings/ never imports panda3d).
"""

import math

import numpy as np
import pytest

from fire_engine.buildings import (
    Building,
    BuildingDefaults,
    OpeningKind,
    WallKind,
)
from fire_engine.core.config import Config
from fire_engine.core.math3d import Quat, Vec3


def _defaults() -> BuildingDefaults:
    return BuildingDefaults.from_config(Config())


def _building(name: str = "test") -> Building:
    return Building(
        name=name, position=Vec3(0.0, 0.0, 8.0), rotation=Quat.identity(), defaults=_defaults()
    )


# ---------------------------------------------------------------------------
# Defaults from config
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_from_config_uses_building_fields(self):
        d = BuildingDefaults.from_config(Config())
        assert d.storey_height_m == 3.0
        assert d.wall_thickness_m == 0.3
        assert d.slab_thickness_m == 0.2
        assert d.foundation_depth_m == 0.5

    def test_dict_round_trip(self):
        d = _defaults()
        assert BuildingDefaults.from_dict(d.to_dict()) == d


# ---------------------------------------------------------------------------
# Element ids
# ---------------------------------------------------------------------------


class TestElementIds:
    def test_ids_are_monotonic_and_unique(self):
        b = _building()
        s = b.add_storey()
        w1 = s.add_wall((0, 0), (4, 0))
        w2 = s.add_wall((4, 0), (4, 4))
        o = s.add_opening(w1.id, OpeningKind.DOOR, offset_m=1.0, width_m=0.9, head_m=2.0)
        ids = [s.id, w1.id, w2.id, o.id]
        assert ids == sorted(ids)
        assert len(set(ids)) == 4

    def test_ids_stable_across_dict_round_trip(self):
        b = _building()
        s = b.add_storey()
        w = s.add_wall((0, 0), (4, 0))
        b2 = Building.from_dict(b.to_dict())
        assert b2.storeys[0].walls[0].id == w.id
        # The counter itself round-trips: new ids never collide with old.
        new_wall = b2.storeys[0].add_wall((0, 4), (4, 4))
        assert new_wall.id > w.id


# ---------------------------------------------------------------------------
# Storey stacking / elevation math
# ---------------------------------------------------------------------------


class TestElevations:
    def test_storey_base_z_stacks_heights(self):
        b = _building()
        b.add_storey(height_m=3.0)
        b.add_storey(height_m=2.5)
        b.add_storey()
        assert b.storey_base_z(0) == 0.0
        assert b.storey_base_z(1) == 3.0
        assert b.storey_base_z(2) == 5.5
        assert b.total_height_m == 8.5

    def test_storey_base_z_range_check(self):
        b = _building()
        b.add_storey()
        with pytest.raises(IndexError):
            b.storey_base_z(1)

    def test_defaults_fill_in(self):
        b = _building()
        s = b.add_storey()
        assert s.height_m == 3.0
        assert s.slab_m == 0.2
        w = s.add_wall((0, 0), (1, 0))
        assert w.thickness_m == 0.3


# ---------------------------------------------------------------------------
# Wall geometry — bulge arcs
# ---------------------------------------------------------------------------


class TestWallGeometry:
    def test_straight_wall_kind_and_length(self):
        b = _building()
        s = b.add_storey()
        w = s.add_wall((0, 0), (3, 4))
        assert w.kind is WallKind.SEGMENT
        assert w.length_m() == pytest.approx(5.0)
        np.testing.assert_allclose(w.tessellate(8), [[0, 0], [3, 4]])

    def test_semicircle_bulge_one(self):
        # |bulge| = 1 → included angle = 4·atan(1) = π (a semicircle).
        b = _building()
        s = b.add_storey()
        w = s.add_wall((0, 0), (2, 0), bulge=1.0)
        assert w.kind is WallKind.ARC
        (cx, cy), r, _, sweep = w.arc_params()
        assert (cx, cy) == pytest.approx((1.0, 0.0))
        assert r == pytest.approx(1.0)
        assert abs(sweep) == pytest.approx(math.pi)
        assert w.length_m() == pytest.approx(math.pi)

    def test_arc_bulges_left_of_chord_for_positive_bulge(self):
        b = _building()
        s = b.add_storey()
        w = s.add_wall((0, 0), (2, 0), bulge=0.5)
        pts = w.tessellate(8)
        # a→b runs +x, so "left" is +y: every interior point must have y > 0.
        assert np.all(pts[1:-1, 1] > 0.0)
        # The arc apex sits sagitta = bulge·chord/2 = 0.5 m off the chord
        # midpoint (tessellation midpoint lands exactly on it: 8 chords).
        assert pts[pts.shape[0] // 2] == pytest.approx((1.0, 0.5))

    def test_negative_bulge_bows_right(self):
        b = _building()
        s = b.add_storey()
        w = s.add_wall((0, 0), (2, 0), bulge=-0.5)
        pts = w.tessellate(8)
        assert np.all(pts[1:-1, 1] < 0.0)

    def test_tessellation_endpoints_are_exact(self):
        b = _building()
        s = b.add_storey()
        w = s.add_wall((0.123, 4.567), (8.9, 1.2), bulge=0.37)
        pts = w.tessellate(8)
        assert tuple(pts[0]) == w.a
        assert tuple(pts[-1]) == w.b

    def test_tessellation_segment_count_scales_with_sweep(self):
        b = _building()
        s = b.add_storey()
        quarter = s.add_wall((0, 0), (2, 2), bulge=math.tan(math.pi / 8))
        semi = s.add_wall((0, 0), (2, 0), bulge=1.0)
        assert quarter.tessellate(8).shape[0] == 9  # 1 quarter → 8 chords
        assert semi.tessellate(8).shape[0] == 17  # 2 quarters → 16 chords

    def test_arc_length_quarter_circle(self):
        # Quarter circle radius 2: chord from (2,0) to (0,2) about origin.
        b = _building()
        s = b.add_storey()
        w = s.add_wall((2, 0), (0, 2), bulge=math.tan(math.pi / 8))
        assert w.length_m() == pytest.approx(math.pi, rel=1e-9)

    def test_coincident_endpoints_rejected(self):
        b = _building()
        s = b.add_storey()
        with pytest.raises(ValueError):
            s.add_wall((1, 1), (1, 1))


# ---------------------------------------------------------------------------
# Openings — validation
# ---------------------------------------------------------------------------


class TestOpenings:
    def test_door_defaults_to_floor_level(self):
        b = _building()
        s = b.add_storey()
        w = s.add_wall((0, 0), (4, 0))
        o = s.add_opening(w.id, OpeningKind.DOOR, offset_m=1.0, width_m=0.9, head_m=2.0)
        assert o.sill_m == 0.0
        assert w.openings == [o]

    def test_opening_beyond_wall_length_rejected(self):
        b = _building()
        s = b.add_storey()
        w = s.add_wall((0, 0), (2, 0))
        with pytest.raises(ValueError):
            s.add_opening(
                w.id, OpeningKind.WINDOW, offset_m=1.5, width_m=1.0, sill_m=1.0, head_m=2.0
            )

    def test_opening_above_wall_band_rejected(self):
        b = _building()
        s = b.add_storey()  # band = 3.0 - 0.2 = 2.8 m
        w = s.add_wall((0, 0), (4, 0))
        with pytest.raises(ValueError):
            s.add_opening(
                w.id, OpeningKind.WINDOW, offset_m=1.0, width_m=1.0, sill_m=1.0, head_m=2.9
            )

    def test_sill_must_be_below_head(self):
        b = _building()
        s = b.add_storey()
        w = s.add_wall((0, 0), (4, 0))
        with pytest.raises(ValueError):
            s.add_opening(
                w.id, OpeningKind.WINDOW, offset_m=1.0, width_m=1.0, sill_m=2.0, head_m=2.0
            )

    def test_unknown_wall_id_rejected(self):
        b = _building()
        s = b.add_storey()
        with pytest.raises(KeyError):
            s.add_opening(999, OpeningKind.DOOR, offset_m=0.0, width_m=0.9, head_m=2.0)


# ---------------------------------------------------------------------------
# Rooms (explicit) / stairs / foundation / roof
# ---------------------------------------------------------------------------


class TestRoomsAndSlabs:
    def test_explicit_room_normalizes_winding_to_ccw(self):
        b = _building()
        s = b.add_storey()
        cw = [(0, 0), (0, 4), (4, 4), (4, 0)]  # clockwise square
        room = s.add_room(cw, tag="kitchen")
        assert room.area_m2() == pytest.approx(16.0)  # positive → CCW
        assert room.tag == "kitchen"
        assert room.auto is False

    def test_room_centroid(self):
        b = _building()
        s = b.add_storey()
        room = s.add_room([(0, 0), (4, 0), (4, 2), (0, 2)])
        assert room.centroid() == pytest.approx((2.0, 1.0))

    def test_degenerate_room_polygon_rejected(self):
        b = _building()
        s = b.add_storey()
        with pytest.raises(ValueError):
            s.add_room([(0, 0), (1, 1)])

    def test_stairs_stub_is_data_only(self):
        b = _building()
        s = b.add_storey()
        b.add_storey()
        stub = s.add_stairs(storey_to=1, anchor=(2.0, 2.0), direction_rad=0.0, width_m=1.0)
        assert stub.storey_from == 0 and stub.storey_to == 1
        assert s.stairs == [stub]

    def test_explicit_foundation_and_roof(self):
        b = _building()
        b.add_storey()
        poly = [(0, 0), (6, 0), (6, 4), (0, 4)]
        f = b.set_foundation(poly, depth_m=0.6)
        r = b.set_roof(poly, thickness_m=0.25)
        assert f.depth_m == 0.6
        assert r.thickness_m == 0.25
        np.testing.assert_allclose(f.polygon, poly)

    def test_auto_footprint_contains_walls(self):
        b = _building()
        s = b.add_storey()
        s.add_wall((0, 0), (6, 0))
        s.add_wall((6, 0), (6, 4))
        s.add_wall((6, 4), (0, 4))
        s.add_wall((0, 4), (0, 0))
        f = b.set_foundation()
        assert f.depth_m == 0.5
        # Hull padded by half wall thickness: extends past the centerlines.
        assert f.polygon[:, 0].min() <= -0.10
        assert f.polygon[:, 0].max() >= 6.10
        assert f.polygon[:, 1].min() <= -0.10
        assert f.polygon[:, 1].max() >= 4.10

    def test_auto_footprint_without_walls_rejected(self):
        b = _building()
        with pytest.raises(ValueError):
            b.set_foundation()


# ---------------------------------------------------------------------------
# Dict round-trip (the Saveable payload)
# ---------------------------------------------------------------------------


def _full_building() -> Building:
    b = Building(
        name="house",
        position=Vec3(-24.0, 10.0, 8.0),
        rotation=Quat.from_axis_angle(Vec3.UP, math.radians(15.0)),
        defaults=_defaults(),
        tags=["rural", "demo"],
    )
    s0 = b.add_storey()
    w_s = s0.add_wall((0, 0), (8, 0))
    s0.add_wall((8, 0), (8, 6), bulge=0.4, thickness_m=0.25)
    s0.add_wall((8, 6), (0, 6))
    s0.add_wall((0, 6), (0, 0))
    s0.add_opening(w_s.id, OpeningKind.DOOR, offset_m=3.5, width_m=0.9, head_m=2.0)
    s0.add_opening(w_s.id, OpeningKind.WINDOW, offset_m=1.0, width_m=1.2, sill_m=1.0, head_m=2.2)
    s0.add_room([(0, 0), (8, 0), (8, 6), (0, 6)], tag="living", meta={"style": "rustic"})
    s0.add_stairs(storey_to=1, anchor=(6.0, 5.0), direction_rad=math.pi / 2, width_m=1.0)
    s1 = b.add_storey(height_m=2.6)
    s1.add_wall((0, 0), (8, 0))
    b.set_foundation()
    b.set_roof()
    return b


class TestDictRoundTrip:
    def test_round_trip_is_exact(self):
        b = _full_building()
        d = b.to_dict()
        assert Building.from_dict(d).to_dict() == d

    def test_dict_is_plain_primitives(self):
        # No numpy scalars/arrays, no live objects — msgpack/json-safe.
        def check(node, path="root"):
            if isinstance(node, dict):
                for k, v in node.items():
                    assert isinstance(k, str), f"non-str key at {path}"
                    check(v, f"{path}.{k}")
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    check(v, f"{path}[{i}]")
            else:
                assert node is None or isinstance(node, (bool, int, float, str)), (
                    f"non-primitive {type(node)} at {path}"
                )

        check(_full_building().to_dict())

    def test_same_authoring_yields_identical_dicts(self):
        # Determinism of the authoring layer itself (no RNG involved).
        assert _full_building().to_dict() == _full_building().to_dict()


# ---------------------------------------------------------------------------
# World AABB under rotation
# ---------------------------------------------------------------------------


class TestWorldAabb:
    def test_identity_rotation_aabb(self):
        b = _building()
        s = b.add_storey()
        s.add_wall((0, 0), (6, 0))
        s.add_wall((6, 0), (6, 4))
        s.add_wall((6, 4), (0, 4))
        s.add_wall((0, 4), (0, 0))
        b.set_foundation()
        mn, mx = b.world_aabb()
        # Position (0,0,8); walls span x:[0,6] y:[0,4] padded by 0.15.
        assert mn[0] == pytest.approx(-0.15, abs=1e-6)
        assert mx[0] == pytest.approx(6.15, abs=1e-6)
        assert mn[2] == pytest.approx(8.0 - 0.5, abs=1e-6)  # foundation
        assert mx[2] == pytest.approx(8.0 + 3.0, abs=1e-6)  # storey top

    def test_yaw_90_swaps_extents(self):
        b = Building(
            name="t",
            position=Vec3(0, 0, 0),
            rotation=Quat.from_axis_angle(Vec3.UP, math.pi / 2),
            defaults=_defaults(),
        )
        s = b.add_storey()
        s.add_wall((0, 0), (6, 0))  # runs +x locally → +y after 90° yaw
        mn, mx = b.world_aabb()
        assert mx[1] - mn[1] == pytest.approx(6.3, abs=1e-5)
        assert mx[0] - mn[0] == pytest.approx(0.3, abs=1e-5)

    def test_roll_rotation_lifts_z_extent(self):
        # 90° roll about +Y: the wall's height now spans X, its length Z...
        b = Building(
            name="t",
            position=Vec3(0, 0, 0),
            rotation=Quat.from_axis_angle(Vec3.FORWARD, math.pi / 2),
            defaults=_defaults(),
        )
        s = b.add_storey()
        s.add_wall((0, 0), (6, 0))
        mn, mx = b.world_aabb()
        # Local x:[−0.15, 6.15] rotates into −z: z extent ≈ 6.3.
        assert mx[2] - mn[2] == pytest.approx(6.3, abs=1e-5)
