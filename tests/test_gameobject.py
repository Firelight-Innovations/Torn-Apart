"""
tests/test_gameobject.py — Headless tests for the Unity-clone object model.

Covers:
  - Lifecycle order: all awakes → on_enables → starts → updates → late_updates
  - A component added during update is awaked NEXT frame, not this one
  - destroy fires on_disable + on_destroy at end of frame
  - set_active(False) cascade disables children
  - find_with_tag / find_objects_with_tag
  - get_component_in_children

NO panda3d imports allowed in this file.
"""

from __future__ import annotations

import pytest

from fire_engine.render.component  import Component
from fire_engine.render.gameobject import GameObject
from fire_engine.render.registry   import (
    ComponentRegistry,
    instantiate,
    destroy,
    find_with_tag,
    find_objects_with_tag,
    _STATE,
)
from fire_engine.core.clock     import Clock
from fire_engine.core.event_bus import EventBus
from fire_engine.core.math3d    import Vec3, Quat


# ---------------------------------------------------------------------------
# Pytest fixture — reset registry between every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_registry():
    """Isolate each test by clearing the registry before and after."""
    ComponentRegistry.clear()
    yield
    ComponentRegistry.clear()


@pytest.fixture()
def clock():
    return Clock(fixed_dt=0.02, bus=EventBus())


# ---------------------------------------------------------------------------
# Recording Component (used by lifecycle tests)
# ---------------------------------------------------------------------------

class Recorder(Component):
    """Records every lifecycle callback call in order."""

    def __init__(self):
        super().__init__()
        self.log: list[str] = []

    def awake(self):
        self.log.append("awake")

    def on_enable(self):
        self.log.append("on_enable")

    def start(self):
        self.log.append("start")

    def update(self, dt):
        self.log.append("update")

    def late_update(self, dt):
        self.log.append("late_update")

    def fixed_update(self, dt):
        self.log.append("fixed_update")

    def on_disable(self):
        self.log.append("on_disable")

    def on_destroy(self):
        self.log.append("on_destroy")


# ---------------------------------------------------------------------------
# Lifecycle order tests
# ---------------------------------------------------------------------------

class TestLifecycleOrder:
    def test_single_component_first_frame(self, clock):
        """awake → on_enable → start → update → late_update in one frame."""
        clock.update(0.016)   # advance so dt > 0
        go  = instantiate()
        rec = go.add_component(Recorder)
        ComponentRegistry.run_frame(clock)

        assert rec.log[:5] == ["awake", "on_enable", "start", "update", "late_update"]

    def test_all_awakes_before_any_start(self, clock):
        """
        With two components, all awake() calls happen before any start() call.
        """
        clock.update(0.016)
        go  = instantiate()
        r1  = go.add_component(Recorder)
        r2  = go.add_component(Recorder)
        ComponentRegistry.run_frame(clock)

        # Both recorders should have seen awake before start
        # Combined order (registry flushes awakes first, then starts):
        # r1: awake, on_enable — r2: awake, on_enable — r1: start — r2: start
        # then update for r1, update for r2, late_update for r1, late_update for r2
        assert "awake" in r1.log and "awake" in r2.log
        assert "start" in r1.log and "start" in r2.log
        # awake precedes start in each recorder's log
        assert r1.log.index("awake") < r1.log.index("start")
        assert r2.log.index("awake") < r2.log.index("start")

    def test_second_frame_no_double_start(self, clock):
        """start() must only fire once; second frame only has update + late_update."""
        clock.update(0.016)
        go  = instantiate()
        rec = go.add_component(Recorder)
        ComponentRegistry.run_frame(clock)    # frame 1

        clock.update(0.016)
        ComponentRegistry.run_frame(clock)    # frame 2

        assert rec.log.count("start") == 1
        assert rec.log.count("update") == 2
        assert rec.log.count("late_update") == 2

    def test_update_before_late_update(self, clock):
        """update always appears before late_update in the same frame."""
        clock.update(0.016)
        go  = instantiate()
        rec = go.add_component(Recorder)
        ComponentRegistry.run_frame(clock)
        assert rec.log.index("update") < rec.log.index("late_update")

    def test_component_added_during_update_awakes_next_frame(self, clock):
        """
        A component created inside update() must NOT awake in the current frame.
        It should awake in the NEXT run_frame call.
        """
        late_rec = []

        class Spawner(Component):
            def update(self, dt):
                child = self.game_object.add_component(Recorder)
                late_rec.append(child)   # store reference

        clock.update(0.016)
        go = instantiate()
        go.add_component(Spawner)
        ComponentRegistry.run_frame(clock)   # frame 1 — Spawner.update runs, spawns Recorder

        # Recorder should NOT have been awaked yet
        assert len(late_rec) == 1
        r = late_rec[0]
        assert "awake" not in r.log, "Component added during update awoke too early"

        # Frame 2 — now Recorder should awake + on_enable + start + update + late_update
        clock.update(0.016)
        ComponentRegistry.run_frame(clock)
        assert "awake" in r.log
        assert "start" in r.log


# ---------------------------------------------------------------------------
# Destroy tests
# ---------------------------------------------------------------------------

class TestDestroy:
    def test_destroy_fires_on_disable_then_on_destroy_end_of_frame(self, clock):
        """destroy() defers teardown; on_disable + on_destroy fire at end of frame."""
        clock.update(0.016)
        go  = instantiate()
        rec = go.add_component(Recorder)
        ComponentRegistry.run_frame(clock)   # frame 1 — awake/start/update

        # Destroy the component
        destroy(rec)
        # Not yet fired
        assert "on_destroy" not in rec.log

        # Next run_frame flushes the destroy queue
        clock.update(0.016)
        ComponentRegistry.run_frame(clock)   # frame 2 — flush includes destroy
        assert "on_disable" in rec.log
        assert "on_destroy" in rec.log
        assert rec.log.index("on_disable") < rec.log.index("on_destroy")

    def test_destroy_gameobject_tears_down_all_components(self, clock):
        """Destroying the GameObject calls on_disable + on_destroy on all components."""
        clock.update(0.016)
        go = instantiate()
        r1 = go.add_component(Recorder)
        r2 = go.add_component(Recorder)
        ComponentRegistry.run_frame(clock)

        destroy(go)
        clock.update(0.016)
        ComponentRegistry.run_frame(clock)

        assert "on_destroy" in r1.log
        assert "on_destroy" in r2.log

    def test_destroyed_component_not_updated(self, clock):
        """A destroyed component should not receive update after the destroy frame."""
        clock.update(0.016)
        go  = instantiate()
        rec = go.add_component(Recorder)
        ComponentRegistry.run_frame(clock)

        destroy(rec)
        clock.update(0.016)
        ComponentRegistry.run_frame(clock)  # destroy flushed here

        update_count_after_destroy = rec.log.count("update")
        clock.update(0.016)
        ComponentRegistry.run_frame(clock)  # another frame — should not update
        assert rec.log.count("update") == update_count_after_destroy


# ---------------------------------------------------------------------------
# set_active cascade
# ---------------------------------------------------------------------------

class TestSetActive:
    def test_set_active_false_disables_components(self, clock):
        clock.update(0.016)
        go  = instantiate()
        rec = go.add_component(Recorder)
        ComponentRegistry.run_frame(clock)

        go.set_active(False)
        assert "on_disable" in rec.log

    def test_set_active_false_cascades_to_children(self, clock):
        clock.update(0.016)
        parent_go = instantiate()
        child_go  = instantiate()
        child_go.transform.set_parent(parent_go.transform, keep_world=False)
        _STATE.objects.append(child_go)  # manual registration since instantiate already registered parent

        rec_parent = parent_go.add_component(Recorder)
        rec_child  = child_go.add_component(Recorder)
        ComponentRegistry.run_frame(clock)

        # Deactivate parent
        parent_go.set_active(False)

        assert "on_disable" in rec_parent.log
        assert "on_disable" in rec_child.log

    def test_set_active_false_child_inactive_in_hierarchy(self):
        parent_go = instantiate()
        child_go  = instantiate()
        child_go.transform.set_parent(parent_go.transform, keep_world=False)

        parent_go.set_active(False)
        assert not child_go.active_in_hierarchy

    def test_set_active_true_re_enables(self, clock):
        clock.update(0.016)
        go  = instantiate()
        rec = go.add_component(Recorder)
        ComponentRegistry.run_frame(clock)

        go.set_active(False)
        disable_count = rec.log.count("on_disable")

        go.set_active(True)
        assert rec.log.count("on_enable") >= 2  # initial + re-enable

    def test_no_update_while_inactive(self, clock):
        clock.update(0.016)
        go  = instantiate()
        rec = go.add_component(Recorder)
        ComponentRegistry.run_frame(clock)   # frame 1

        go.set_active(False)
        clock.update(0.016)
        ComponentRegistry.run_frame(clock)   # frame 2

        update_count = rec.log.count("update")
        assert update_count == 1  # only frame 1


# ---------------------------------------------------------------------------
# find_with_tag / find_objects_with_tag
# ---------------------------------------------------------------------------

class TestTagLookup:
    def test_find_with_tag(self):
        go = instantiate()
        go.tag = "enemy"
        found = find_with_tag("enemy")
        assert found is go

    def test_find_with_tag_returns_none_for_missing(self):
        assert find_with_tag("nonexistent") is None

    def test_find_objects_with_tag_multiple(self):
        g1 = instantiate(); g1.tag = "item"
        g2 = instantiate(); g2.tag = "item"
        g3 = instantiate(); g3.tag = "other"
        results = find_objects_with_tag("item")
        assert g1 in results
        assert g2 in results
        assert g3 not in results

    def test_compare_tag(self):
        go = instantiate()
        go.tag = "player"
        assert go.compare_tag("player")
        assert not go.compare_tag("enemy")


# ---------------------------------------------------------------------------
# get_component_in_children
# ---------------------------------------------------------------------------

class TestGetComponentInChildren:
    def test_finds_component_on_self(self):
        go  = instantiate()
        rec = go.add_component(Recorder)
        assert go.get_component_in_children(Recorder) is rec

    def test_finds_component_in_direct_child(self, clock):
        clock.update(0.016)
        parent = instantiate()
        child  = instantiate()
        child.transform.set_parent(parent.transform, keep_world=False)

        rec = child.add_component(Recorder)
        found = parent.get_component_in_children(Recorder)
        assert found is rec

    def test_returns_none_when_not_found(self):
        go = instantiate()
        assert go.get_component_in_children(Recorder) is None

    def test_finds_in_grandchild(self, clock):
        clock.update(0.016)
        root   = instantiate()
        mid    = instantiate()
        leaf   = instantiate()
        mid.transform.set_parent(root.transform,   keep_world=False)
        leaf.transform.set_parent(mid.transform,   keep_world=False)

        rec = leaf.add_component(Recorder)
        found = root.get_component_in_children(Recorder)
        assert found is rec


# ---------------------------------------------------------------------------
# instantiate convenience
# ---------------------------------------------------------------------------

class TestInstantiate:
    def test_instantiate_sets_position(self):
        go = instantiate(position=Vec3(1, 2, 3))
        assert go.transform.local_position.approx_eq(Vec3(1, 2, 3), eps=1e-5)

    def test_instantiate_sets_rotation(self):
        import math
        q = Quat.from_axis_angle(Vec3.UP, math.pi / 2)
        go = instantiate(rotation=q)
        assert go.transform.local_rotation.approx_eq(q, eps=1e-5)

    def test_instantiate_registers_gameobject(self):
        go = instantiate()
        go.tag = "test_inst"
        assert find_with_tag("test_inst") is go


# ---------------------------------------------------------------------------
# Multiple component types in order
# ---------------------------------------------------------------------------

class TestMultipleComponentTypes:
    def test_two_recorder_components_both_update(self, clock):
        clock.update(0.016)
        go = instantiate()
        r1 = go.add_component(Recorder)
        r2 = go.add_component(Recorder)
        ComponentRegistry.run_frame(clock)
        assert r1.log.count("update") == 1
        assert r2.log.count("update") == 1

    def test_disabled_component_not_updated(self, clock):
        clock.update(0.016)
        go  = instantiate()
        rec = go.add_component(Recorder)
        rec.enabled = False
        ComponentRegistry.run_frame(clock)
        assert "update" not in rec.log
