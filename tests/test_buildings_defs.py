"""
tests/test_buildings_defs.py — characterization tests (golden-master) for
buildings/defs.py: BuildingDef abstract base and DemoHouseDef.

Pins CURRENT behaviour — do NOT fix bugs, only note suspicions in comments.

Headless (numpy only — fire_engine/buildings/ never imports panda3d).
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.buildings import (
    Building,
    BuildingDef,
    DemoHouseDef,
    OpeningKind,
    WallKind,
)
from fire_engine.core.config import Config
from fire_engine.core.rng import set_world_seed
from fire_engine.procedural import clear_cache, get

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEED = 12345
_DEF_NAME = "building_demo_house"


def _make_demo(seed: int = _SEED, **params) -> Building:
    """Generate the demo house under a fixed seed."""
    set_world_seed(seed)
    clear_cache()
    return get(_DEF_NAME, **params)


# ---------------------------------------------------------------------------
# BuildingDef abstract base
# ---------------------------------------------------------------------------


class TestBuildingDefAbstractBase:
    def test_base_generate_raises_not_implemented(self):
        """Directly instantiating BuildingDef is impossible (ABC); calling
        generate on a thin concrete subclass that doesn't override it must
        raise NotImplementedError per the stub contract."""

        class _MinimalDef(BuildingDef):
            name = "test_minimal_building"

            # Does NOT override generate — inherits BuildingDef.generate
            # which raises NotImplementedError.

        defn = _MinimalDef()
        rng = np.random.default_rng(0)
        with pytest.raises(NotImplementedError):
            defn.generate(rng)

    def test_base_generate_error_message_mentions_architecture(self):
        """The NotImplementedError message should reference ARCHITECTURE.md so
        future subclass authors know where to look."""

        class _Stub(BuildingDef):
            name = "test_stub_building"

        defn = _Stub()
        rng = np.random.default_rng(0)
        with pytest.raises(NotImplementedError, match="ARCHITECTURE"):
            defn.generate(rng)

    def test_subclass_without_name_attr_can_be_instantiated(self):
        """BuildingDef.name is declared as a class-level annotation on
        ProceduralDef.  A concrete subclass without it can still be
        instantiated — the error only surfaces on registration.
        Pin current behaviour: instantiation succeeds, but `name` may be
        missing as an instance attribute (AttributeError on access).
        SUSPICION: the ABC does not enforce `name` at construction time."""

        class _NoName(BuildingDef):
            def generate(self, rng, **params):
                raise NotImplementedError

        inst = _NoName()
        assert inst is not None


# ---------------------------------------------------------------------------
# DemoHouseDef: registration
# ---------------------------------------------------------------------------


class TestDemoHouseRegistration:
    def test_name_attribute_is_correct(self):
        defn = DemoHouseDef()
        assert defn.name == _DEF_NAME

    def test_registered_under_expected_name(self):
        """Importing fire_engine.buildings triggers @register_def on
        DemoHouseDef; get() must not raise KeyError."""
        set_world_seed(_SEED)
        clear_cache()
        result = get(_DEF_NAME)
        assert result is not None

    def test_get_returns_a_building_instance(self):
        b = _make_demo()
        assert isinstance(b, Building)

    def test_get_raises_key_error_for_unknown_name(self):
        with pytest.raises(KeyError):
            get("building_does_not_exist_xyz")


# ---------------------------------------------------------------------------
# DemoHouseDef: determinism
# ---------------------------------------------------------------------------


class TestDemoHouseDeterminism:
    def test_same_seed_same_to_dict(self):
        """DemoHouseDef uses no RNG (fixed layout), so two calls with the
        same seed and params must produce byte-identical to_dict() output."""
        b1 = _make_demo()
        d1 = b1.to_dict()
        # Clear cache so a second generate() call actually runs.
        clear_cache()
        b2 = _make_demo()
        d2 = b2.to_dict()
        assert d1 == d2

    def test_different_seed_same_layout_because_no_rng(self):
        """The demo house is hand-authored (no RNG calls), so changing the
        world seed must NOT change its structure.  Pin: to_dict() equal
        across seeds."""
        b_a = _make_demo(seed=1)
        d_a = b_a.to_dict()
        b_b = _make_demo(seed=999)
        d_b = b_b.to_dict()
        # Exclude the cache hit: both must produce the same structure.
        # NOTE: the registry returns a *cached* Building object — so the
        # same object may be returned from the cache under different seeds
        # unless the cache is cleared.  Pinning this structural equality.
        # SUSPICION: seed changes the cache key but the layout is fixed; the
        # dict should be identical regardless.
        assert d_a["storeys"] == d_b["storeys"]

    def test_registry_cache_returns_same_object_same_params(self):
        """After the first get(), a second get() with same seed+params
        returns the SAME Python object (identity, not just equality) per
        the registry's caching contract."""
        set_world_seed(_SEED)
        clear_cache()
        b1 = get(_DEF_NAME)
        b2 = get(_DEF_NAME)
        assert b1 is b2


# ---------------------------------------------------------------------------
# DemoHouseDef: storey count
# ---------------------------------------------------------------------------


class TestDemoHouseStoreys:
    def test_has_exactly_two_storeys(self):
        """The demo house is documented as 'two-storey'."""
        b = _make_demo()
        assert len(b.storeys) == 2

    def test_storey0_height_is_3_5m(self):
        """Ground storey: taller at 3.5 m (contrasts with upper 2.5 m)."""
        b = _make_demo()
        assert b.storeys[0].height_m == pytest.approx(3.5)

    def test_storey1_height_is_2_5m(self):
        """Upper storey: 2.5 m."""
        b = _make_demo()
        assert b.storeys[1].height_m == pytest.approx(2.5)

    def test_total_height_matches_storey_sum(self):
        b = _make_demo()
        assert b.total_height_m == pytest.approx(3.5 + 2.5)


# ---------------------------------------------------------------------------
# DemoHouseDef: storey 0 wall structure
# ---------------------------------------------------------------------------


class TestDemoHouseStorey0Walls:
    def test_storey0_wall_count(self):
        """Storey 0 has 11 walls: 8 perimeter (split at junctions) + 2 spine
        segments + 1 east half-wall.  Pin the exact count from defs.py."""
        b = _make_demo()
        # Counting add_wall calls in DemoHouseDef.generate for storey 0:
        # sw, se, bay0, eup, NE, nw, west, cham, spine_lo, spine_hi,
        # east half-wall = 11 walls.
        assert len(b.storeys[0].walls) == 11

    def test_storey0_has_exactly_one_arc_wall(self):
        """Storey 0 has exactly one arc wall: the east bay (bulge=-0.4)."""
        b = _make_demo()
        arc_walls = [w for w in b.storeys[0].walls if w.kind is WallKind.ARC]
        assert len(arc_walls) == 1

    def test_storey0_bay_wall_has_negative_bulge(self):
        """East bay bows right (outward on a CCW perimeter) — negative bulge."""
        b = _make_demo()
        arc_walls = [w for w in b.storeys[0].walls if w.kind is WallKind.ARC]
        bay = arc_walls[0]
        assert bay.bulge < 0.0
        assert bay.bulge == pytest.approx(-0.4)

    def test_storey0_has_exterior_and_interior_walls(self):
        """Exterior walls are 0.4 m thick; interior partitions are 0.15 m.
        Both thicknesses must appear in storey 0."""
        b = _make_demo()
        thicknesses = [w.thickness_m for w in b.storeys[0].walls]
        assert any(abs(t - 0.4) < 1e-9 for t in thicknesses), "No 0.4 m exterior wall found"
        assert any(abs(t - 0.15) < 1e-9 for t in thicknesses), "No 0.15 m interior wall found"

    def test_storey0_east_half_wall_has_height_override(self):
        """The open-plan half-wall is 1.1 m tall (explicit height_m override).
        Pin that exactly one storey-0 wall has height_m == 1.1."""
        b = _make_demo()
        half_walls = [
            w for w in b.storeys[0].walls if w.height_m is not None and abs(w.height_m - 1.1) < 1e-9
        ]
        assert len(half_walls) == 1


# ---------------------------------------------------------------------------
# DemoHouseDef: storey 0 openings
# ---------------------------------------------------------------------------


class TestDemoHouseStorey0Openings:
    def _all_openings(self, b: Building):
        return [o for w in b.storeys[0].walls for o in w.openings]

    def test_storey0_total_opening_count(self):
        """Storey 0: 1 exterior door + 2 interior doors + 5 windows = 8
        openings (counted from the add_opening calls in defs.py)."""
        b = _make_demo()
        openings = self._all_openings(b)
        assert len(openings) == 8

    def test_storey0_door_count(self):
        b = _make_demo()
        doors = [o for o in self._all_openings(b) if o.kind is OpeningKind.DOOR]
        # 1 exterior south door + 2 interior spine doors = 3.
        assert len(doors) == 3

    def test_storey0_window_count(self):
        b = _make_demo()
        windows = [o for o in self._all_openings(b) if o.kind is OpeningKind.WINDOW]
        # se, bay0 (curved), nw, west, cham = 5 windows.
        assert len(windows) == 5

    def test_storey0_curved_wall_carries_a_window(self):
        """The bay (arc) wall has exactly one window opening."""
        b = _make_demo()
        arc_walls = [w for w in b.storeys[0].walls if w.kind is WallKind.ARC]
        bay = arc_walls[0]
        assert len(bay.openings) == 1
        assert bay.openings[0].kind is OpeningKind.WINDOW


# ---------------------------------------------------------------------------
# DemoHouseDef: storey 0 rooms
# ---------------------------------------------------------------------------


class TestDemoHouseStorey0Rooms:
    def test_storey0_auto_detects_three_rooms(self):
        """The docstring and code both say storey 0 auto-detects three rooms
        (living / dining / kitchen).  Pin this count."""
        b = _make_demo()
        # detect_rooms is called in generate; rooms are stored on the storey.
        assert len(b.storeys[0].rooms) == 3

    def test_storey0_room_tags_cover_expected_set(self):
        """After centroid-based tagging, rooms are tagged 'living', 'dining',
        'kitchen' in some order."""
        b = _make_demo()
        tags = {r.tag for r in b.storeys[0].rooms}
        assert tags == {"living", "dining", "kitchen"}

    def test_storey0_rooms_are_auto_detected(self):
        """All storey-0 rooms come from detect_rooms (auto=True)."""
        b = _make_demo()
        assert all(r.auto for r in b.storeys[0].rooms)


# ---------------------------------------------------------------------------
# DemoHouseDef: storey 1
# ---------------------------------------------------------------------------


class TestDemoHouseStorey1:
    def test_storey1_wall_count(self):
        """Storey 1: 6 perimeter walls (south whole + bay + east-upper + north
        + west + chamfer). Pin from defs.py add_wall calls in storey 1."""
        b = _make_demo()
        assert len(b.storeys[1].walls) == 6

    def test_storey1_has_exactly_one_arc_wall(self):
        """Storey 1 bay mirrors storey 0 (bulge=-0.4)."""
        b = _make_demo()
        arc_walls = [w for w in b.storeys[1].walls if w.kind is WallKind.ARC]
        assert len(arc_walls) == 1
        assert arc_walls[0].bulge == pytest.approx(-0.4)

    def test_storey1_has_exactly_one_explicit_room(self):
        """Storey 1 has one explicit add_room call for 'loft'."""
        b = _make_demo()
        assert len(b.storeys[1].rooms) == 1
        assert b.storeys[1].rooms[0].tag == "loft"

    def test_storey1_loft_room_is_not_auto(self):
        """Loft room is explicitly authored (auto=False)."""
        b = _make_demo()
        assert b.storeys[1].rooms[0].auto is False

    def test_storey1_opening_count(self):
        """Storey 1: 3 openings (sbay window + n1 window + w1 window)."""
        b = _make_demo()
        openings = [o for w in b.storeys[1].walls for o in w.openings]
        assert len(openings) == 3


# ---------------------------------------------------------------------------
# DemoHouseDef: stairs
# ---------------------------------------------------------------------------


class TestDemoHouseStairs:
    def test_storey0_has_one_stairs_stub(self):
        b = _make_demo()
        assert len(b.storeys[0].stairs) == 1

    def test_stairs_stub_connects_storeys_0_and_1(self):
        b = _make_demo()
        stub = b.storeys[0].stairs[0]
        assert stub.storey_from == 0
        assert stub.storey_to == 1


# ---------------------------------------------------------------------------
# DemoHouseDef: foundation and roof
# ---------------------------------------------------------------------------


class TestDemoHouseFoundationRoof:
    def test_has_foundation(self):
        b = _make_demo()
        assert b.foundation is not None

    def test_foundation_depth_matches_config(self):
        """set_foundation() uses the config default (0.5 m)."""
        b = _make_demo()
        cfg = Config()
        assert b.foundation.depth_m == pytest.approx(cfg.building_foundation_depth_m)

    def test_has_flat_roof(self):
        b = _make_demo()
        assert b.roof is not None


# ---------------------------------------------------------------------------
# DemoHouseDef: dimensions from config
# ---------------------------------------------------------------------------


class TestDemoHouseDimensionsFromConfig:
    def test_storey_slab_thickness_matches_config(self):
        """Slabs use BuildingDefaults.from_config() — pin against the config
        default so a magic-number regression would be caught."""
        b = _make_demo()
        cfg = Config()
        for s in b.storeys:
            assert s.slab_m == pytest.approx(cfg.building_slab_thickness_m)

    def test_exterior_wall_thickness_is_explicitly_0_4_not_default(self):
        """The demo house uses EXT_T=0.4 m, not the config default (0.3 m).
        Confirm the two values differ so this test is non-trivial."""
        cfg = Config()
        assert cfg.building_default_wall_thickness_m == pytest.approx(0.3)
        b = _make_demo()
        ext_walls = [w for w in b.storeys[0].walls if abs(w.thickness_m - 0.4) < 1e-9]
        assert len(ext_walls) > 0


# ---------------------------------------------------------------------------
# DemoHouseDef: rotation
# ---------------------------------------------------------------------------


class TestDemoHouseRotation:
    def test_building_has_nonzero_yaw(self):
        """Demo house is yawed ~18° (default yaw_deg param) — rotation must
        not be the identity quaternion."""
        b = _make_demo()
        # Identity quaternion has w=1, xyz=0.  If yaw is 18°, w < 1.
        q = b.rotation
        assert abs(q.w) < 1.0 - 1e-6


# ---------------------------------------------------------------------------
# DemoHouseDef: to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


class TestDemoHouseRoundTrip:
    def test_to_dict_from_dict_gives_identical_dict(self):
        """Full dict round-trip: serialize and restore, then re-serialize.
        The two dicts must be equal (structural equality, not identity)."""
        b = _make_demo()
        d1 = b.to_dict()
        b2 = Building.from_dict(d1)
        d2 = b2.to_dict()
        assert d1 == d2

    def test_to_dict_storey_count_survives_round_trip(self):
        b = _make_demo()
        b2 = Building.from_dict(b.to_dict())
        assert len(b2.storeys) == len(b.storeys)

    def test_to_dict_room_count_survives_round_trip(self):
        b = _make_demo()
        b2 = Building.from_dict(b.to_dict())
        for i, (s_orig, s_rt) in enumerate(zip(b.storeys, b2.storeys)):
            assert len(s_rt.rooms) == len(s_orig.rooms), f"Room count mismatch on storey {i}"

    def test_to_dict_wall_count_survives_round_trip(self):
        b = _make_demo()
        b2 = Building.from_dict(b.to_dict())
        for i, (s_orig, s_rt) in enumerate(zip(b.storeys, b2.storeys)):
            assert len(s_rt.walls) == len(s_orig.walls), f"Wall count mismatch on storey {i}"
