"""
tests/test_buildings_manager.py — BuildingManager: clone-on-add, id assignment,
BuildingChangedEvent publishing, query ordering, and the Saveable delta
(round-trip via a fresh manager, msgpack encode/decode, baseline emptiness,
apply_delta republish).

Headless (numpy only — fire_engine/buildings/ never imports panda3d).
"""

import math

import pytest

from fire_engine.buildings import (
    Building,
    BuildingDefaults,
    BuildingManager,
    OpeningKind,
)
from fire_engine.core.config import Config
from fire_engine.core.event_bus import BuildingChangedEvent, EventBus
from fire_engine.core.math3d import Quat, Vec3
from fire_engine.save.save_manager import _decode_delta, _encode_delta

_CFG = Config()


def _spec(name: str = "hut") -> Building:
    b = Building(
        name=name,
        position=Vec3(-24.0, 10.0, 8.0),
        rotation=Quat.from_axis_angle(Vec3.UP, math.radians(15)),
        defaults=BuildingDefaults.from_config(_CFG),
        tags=["rural"],
    )
    s0 = b.add_storey()
    south = s0.add_wall((0, 0), (8, 0))
    s0.add_wall((8, 0), (8, 6))
    s0.add_wall((8, 6), (0, 6))
    s0.add_wall((0, 6), (0, 0))
    s0.add_opening(south.id, OpeningKind.DOOR, offset_m=3.5, width_m=0.9, head_m=2.0)
    b.set_foundation()
    b.set_roof()
    return b


class _Recorder:
    def __init__(self, bus: EventBus):
        self.events: list[BuildingChangedEvent] = []
        bus.subscribe(BuildingChangedEvent, self.events.append)


class TestMutation:
    def test_add_clones_and_assigns_id(self):
        mgr = BuildingManager(_CFG, bus=None)
        spec = _spec()
        managed = mgr.add(spec)
        assert managed is not spec  # clone, not the argument
        assert managed.id == 1
        assert spec.id == 0  # argument untouched
        assert mgr.get(1) is managed

    def test_add_same_spec_twice_independent(self):
        mgr = BuildingManager(_CFG, bus=None)
        spec = _spec()
        a = mgr.add(spec)
        b = mgr.add(spec)
        assert a.id == 1 and b.id == 2
        assert a is not b
        # Mutating one managed building does not touch the other or the spec.
        a.storeys[0].add_wall((1, 1), (2, 2))
        assert len(b.storeys[0].walls) == 4

    def test_remove(self):
        mgr = BuildingManager(_CFG, bus=None)
        managed = mgr.add(_spec())
        assert mgr.remove(managed.id) is True
        assert mgr.get(managed.id) is None
        assert mgr.remove(managed.id) is False

    def test_buildings_ordered_by_id(self):
        mgr = BuildingManager(_CFG, bus=None)
        a = mgr.add(_spec("a"))
        b = mgr.add(_spec("b"))
        assert [x.id for x in mgr.buildings()] == [a.id, b.id]

    def test_notify_changed_unknown_raises(self):
        mgr = BuildingManager(_CFG, bus=None)
        with pytest.raises(KeyError):
            mgr.notify_changed(99)


class TestEvents:
    def test_add_publishes_added_with_bounds(self):
        bus = EventBus()
        rec = _Recorder(bus)
        mgr = BuildingManager(_CFG, bus=bus)
        managed = mgr.add(_spec())
        assert len(rec.events) == 1
        e = rec.events[0]
        assert e.building_id == managed.id
        assert e.change == "added"
        # bounds match the building's world AABB
        mn, mx = managed.world_aabb()
        assert e.bounds_min == mn and e.bounds_max == mx
        assert all(a <= b for a, b in zip(e.bounds_min, e.bounds_max, strict=True))

    def test_modify_and_remove_publish(self):
        bus = EventBus()
        rec = _Recorder(bus)
        mgr = BuildingManager(_CFG, bus=bus)
        managed = mgr.add(_spec())
        mgr.notify_changed(managed.id)
        mgr.remove(managed.id)
        assert [e.change for e in rec.events] == ["added", "modified", "removed"]


class TestSaveable:
    def test_baseline_unchanged_is_empty_delta(self):
        mgr = BuildingManager(_CFG, bus=None)
        mgr.add(_spec())
        mgr.mark_baseline()
        assert mgr.get_delta() == {}

    def test_delta_nonempty_after_change(self):
        mgr = BuildingManager(_CFG, bus=None)
        mgr.add(_spec())
        mgr.mark_baseline()
        mgr.add(_spec("second"))
        delta = mgr.get_delta()
        assert delta != {}
        assert len(delta["buildings"]) == 2
        assert delta["next_id"] == 3

    def test_delta_round_trip_via_fresh_manager(self):
        mgr = BuildingManager(_CFG, bus=None)
        mgr.add(_spec())
        mgr.add(_spec("second"))
        delta = mgr.get_delta()

        fresh = BuildingManager(_CFG, bus=None)
        fresh.apply_delta(delta)
        assert [b.id for b in fresh.buildings()] == [1, 2]
        assert [b.to_dict() for b in fresh.buildings()] == [b.to_dict() for b in mgr.buildings()]
        assert fresh._next_id == mgr._next_id

    def test_delta_survives_msgpack_encode_decode(self):
        mgr = BuildingManager(_CFG, bus=None)
        mgr.add(_spec())
        delta = mgr.get_delta()
        decoded = _decode_delta(_encode_delta(delta))
        fresh = BuildingManager(_CFG, bus=None)
        fresh.apply_delta(decoded)
        assert fresh.get(1).to_dict() == mgr.get(1).to_dict()

    def test_apply_delta_republishes_added(self):
        mgr = BuildingManager(_CFG, bus=None)
        mgr.add(_spec())
        mgr.add(_spec("second"))
        delta = mgr.get_delta()

        bus = EventBus()
        rec = _Recorder(bus)
        fresh = BuildingManager(_CFG, bus=bus)
        fresh.apply_delta(delta)
        assert [e.change for e in rec.events] == ["added", "added"]
        assert {e.building_id for e in rec.events} == {1, 2}

    def test_empty_delta_apply_is_noop(self):
        mgr = BuildingManager(_CFG, bus=None)
        managed = mgr.add(_spec())
        mgr.apply_delta({})
        assert mgr.get(managed.id) is managed
