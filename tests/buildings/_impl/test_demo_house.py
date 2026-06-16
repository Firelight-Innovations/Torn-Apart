"""
tests/buildings/_impl/test_demo_house.py — DemoHouseDef (the concrete
procedural demo house implementation in buildings/_impl/demo_house.py).

These tests cover: class attributes, generate() output structure, determinism,
and re-export from buildings.defs.  All tests are headless (no panda3d).
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.buildings._impl.demo_house import DemoHouseDef
from fire_engine.buildings.defs import DemoHouseDef as DemoHouseDefFromDefs
from fire_engine.buildings.enums import OpeningKind, WallKind
from fire_engine.buildings.model import Building
from fire_engine.core.config import Config
from fire_engine.core.rng import set_world_seed
from fire_engine.procedural import clear_cache, get

_SEED = 42
_DEF_NAME = "building_demo_house"


def _make(seed: int = _SEED, **params) -> Building:
    set_world_seed(seed)
    clear_cache()
    return get(_DEF_NAME, **params)


# ---------------------------------------------------------------------------
# Class-level attributes
# ---------------------------------------------------------------------------


class TestClassAttributes:
    def test_name_attribute(self):
        assert DemoHouseDef.name == _DEF_NAME

    def test_is_subclass_of_building_def(self):
        from fire_engine.buildings.defs import BuildingDef

        assert issubclass(DemoHouseDef, BuildingDef)

    def test_re_exported_from_defs(self):
        # Both import paths resolve to the same class.
        assert DemoHouseDef is DemoHouseDefFromDefs

    def test_plan_constants_positive(self):
        assert DemoHouseDef._W > 0
        assert DemoHouseDef._D > 0
        assert DemoHouseDef._EXT_T > 0
        assert DemoHouseDef._INT_T > 0

    def test_chamfer_less_than_dim(self):
        assert DemoHouseDef._CHAMFER < DemoHouseDef._W
        assert DemoHouseDef._CHAMFER < DemoHouseDef._D


# ---------------------------------------------------------------------------
# generate() — structure
# ---------------------------------------------------------------------------


class TestGenerateStructure:
    def test_returns_building(self):
        b = _make()
        assert isinstance(b, Building)

    def test_two_storeys(self):
        b = _make()
        assert len(b.storeys) == 2

    def test_storey0_height_3_5(self):
        b = _make()
        assert b.storeys[0].height_m == pytest.approx(3.5)

    def test_storey1_height_2_5(self):
        b = _make()
        assert b.storeys[1].height_m == pytest.approx(2.5)

    def test_total_height_6(self):
        b = _make()
        assert b.total_height_m == pytest.approx(6.0)

    def test_storey0_rooms_are_three(self):
        b = _make()
        assert len(b.storeys[0].rooms) == 3

    def test_storey0_room_tags(self):
        b = _make()
        tags = {r.tag for r in b.storeys[0].rooms}
        assert tags == {"living", "dining", "kitchen"}

    def test_storey1_has_explicit_loft_room(self):
        b = _make()
        assert len(b.storeys[1].rooms) == 1
        assert b.storeys[1].rooms[0].tag == "loft"
        assert b.storeys[1].rooms[0].auto is False

    def test_foundation_and_roof_set(self):
        b = _make()
        assert b.foundation is not None
        assert b.roof is not None

    def test_stairs_stub_present(self):
        b = _make()
        assert len(b.storeys[0].stairs) == 1
        stub = b.storeys[0].stairs[0]
        assert stub.storey_from == 0
        assert stub.storey_to == 1

    def test_has_arc_wall_in_storey0(self):
        b = _make()
        arcs = [w for w in b.storeys[0].walls if w.kind is WallKind.ARC]
        assert len(arcs) == 1
        assert arcs[0].bulge == pytest.approx(-0.4)

    def test_has_arc_wall_in_storey1(self):
        b = _make()
        arcs = [w for w in b.storeys[1].walls if w.kind is WallKind.ARC]
        assert len(arcs) == 1

    def test_half_wall_with_height_override(self):
        b = _make()
        half_walls = [
            w for w in b.storeys[0].walls if w.height_m is not None and abs(w.height_m - 1.1) < 1e-9
        ]
        assert len(half_walls) == 1

    def test_exterior_door_present(self):
        b = _make()
        all_openings = [o for w in b.storeys[0].walls for o in w.openings]
        doors = [o for o in all_openings if o.kind is OpeningKind.DOOR]
        # 1 exterior + 2 spine doors = 3
        assert len(doors) == 3

    def test_yaw_is_nonzero(self):
        b = _make()
        # 18° yaw → w < 1 (identity quaternion has w=1)
        assert abs(b.rotation.w) < 1.0 - 1e-6

    def test_foundation_depth_matches_config(self):
        cfg = Config()
        b = _make()
        assert b.foundation.depth_m == pytest.approx(cfg.building_foundation_depth_m)

    def test_slab_thickness_matches_config(self):
        cfg = Config()
        b = _make()
        for s in b.storeys:
            assert s.slab_m == pytest.approx(cfg.building_slab_thickness_m)

    def test_exterior_wall_thickness_is_0_4(self):
        b = _make()
        ext_walls = [w for w in b.storeys[0].walls if abs(w.thickness_m - 0.4) < 1e-9]
        assert len(ext_walls) > 0

    def test_interior_wall_thickness_is_0_15(self):
        b = _make()
        int_walls = [w for w in b.storeys[0].walls if abs(w.thickness_m - 0.15) < 1e-9]
        assert len(int_walls) > 0

    def test_building_name_is_demo_house(self):
        b = _make()
        assert b.name == "demo_house"

    def test_tags_include_showcase(self):
        b = _make()
        assert "showcase" in b.tags


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_to_dict(self):
        b1 = _make(seed=100)
        d1 = b1.to_dict()
        clear_cache()
        b2 = _make(seed=100)
        d2 = b2.to_dict()
        assert d1 == d2

    def test_fixed_layout_same_across_seeds(self):
        """DemoHouseDef uses no RNG (fixed layout): two seeds give same structure."""
        b_a = _make(seed=1)
        b_b = _make(seed=999)
        assert b_a.to_dict()["storeys"] == b_b.to_dict()["storeys"]

    def test_ground_z_param_propagates(self):
        b = _make(ground_z=12.0)
        assert b.position.z == pytest.approx(12.0)

    def test_generate_directly_is_deterministic(self):
        defn = DemoHouseDef()
        rng = np.random.default_rng(0)
        b1 = defn.generate(rng, ground_z=8.0)
        b2 = defn.generate(rng, ground_z=8.0)
        assert b1.to_dict()["storeys"] == b2.to_dict()["storeys"]


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_to_dict_from_dict(self):
        b = _make()
        d = b.to_dict()
        b2 = Building.from_dict(d)
        assert b2.to_dict() == d

    def test_storey_count_survives_round_trip(self):
        b = _make()
        b2 = Building.from_dict(b.to_dict())
        assert len(b2.storeys) == len(b.storeys)
