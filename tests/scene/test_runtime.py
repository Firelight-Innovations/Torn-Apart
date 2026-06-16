"""Mirror tests for fire_engine/scene/runtime.py.

Covers: SceneRuntime construction (headless), rebuild() — object count,
names, tags, parent wiring, local transforms; on_rebuilt callback; spawn_position;
get_delta / apply_delta Saveable protocol round-trip; SCENE_TAG constant.

All tests are HEADLESS (visual_factory=None). No panda3d imports.

Categories: CORRECTNESS (known inputs → known outputs), ROUND-TRIP
(get_delta / apply_delta), DETERMINISM (same store → same rebuild result).
"""

from __future__ import annotations

from fire_engine.core.math3d import Vec3
from fire_engine.render.registry import ComponentRegistry
from fire_engine.scene.runtime import SCENE_TAG, SceneRuntime

_EPS = 1e-6


def _rt() -> SceneRuntime:
    """Fresh headless runtime with a cleared component registry."""
    ComponentRegistry.clear()
    return SceneRuntime(visual_factory=None)


# ---------------------------------------------------------------------------
# SCENE_TAG constant
# ---------------------------------------------------------------------------


class TestSceneTag:
    def test_scene_tag_value(self):
        assert SCENE_TAG == "editor_scene"


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_no_factory_by_default(self):
        rt = _rt()
        assert rt.visual_factory is None

    def test_objects_dict_initially_empty(self):
        rt = _rt()
        assert rt.objects == {}

    def test_on_rebuilt_none_by_default(self):
        rt = _rt()
        assert rt.on_rebuilt is None

    def test_store_initially_empty(self):
        rt = _rt()
        assert len(rt.store) == 0

    def test_spawn_position_none_on_fresh_runtime(self):
        rt = _rt()
        assert rt.spawn_position is None


# ---------------------------------------------------------------------------
# rebuild() — basic population
# ---------------------------------------------------------------------------


class TestRebuildPopulation:
    def test_empty_store_rebuild_gives_no_objects(self):
        rt = _rt()
        rt.rebuild()
        assert rt.objects == {}

    def test_single_object_creates_one_game_object(self):
        rt = _rt()
        rt.store.create("cube", name="Box")
        rt.rebuild()
        assert len(rt.objects) == 1

    def test_object_id_maps_to_game_object(self):
        rt = _rt()
        obj = rt.store.create("cube", name="Box")
        rt.rebuild()
        assert obj["id"] in rt.objects

    def test_name_applied_to_game_object(self):
        rt = _rt()
        obj = rt.store.create("cube", name="NamedBox")
        rt.rebuild()
        assert rt.objects[obj["id"]].name == "NamedBox"

    def test_scene_tag_applied(self):
        rt = _rt()
        obj = rt.store.create("cube", name="T")
        rt.rebuild()
        assert rt.objects[obj["id"]].tag == SCENE_TAG

    def test_multiple_objects_all_created(self):
        rt = _rt()
        rt.store.create("cube")
        rt.store.create("sphere")
        rt.store.create("light")
        rt.rebuild()
        assert len(rt.objects) == 3

    def test_all_objects_have_scene_tag(self):
        rt = _rt()
        rt.store.create("cube")
        rt.store.create("light")
        rt.store.create("spawn")
        rt.rebuild()
        for go in rt.objects.values():
            assert go.tag == SCENE_TAG


# ---------------------------------------------------------------------------
# rebuild() — transforms
# ---------------------------------------------------------------------------


class TestRebuildTransforms:
    def test_local_position_applied(self):
        rt = _rt()
        obj = rt.store.create("cube", position=(3.0, 5.0, 7.0))
        rt.rebuild()
        go = rt.objects[obj["id"]]
        pos = go.transform.local_position
        assert abs(pos.x - 3.0) < _EPS
        assert abs(pos.y - 5.0) < _EPS
        assert abs(pos.z - 7.0) < _EPS

    def test_zero_position_stays_origin(self):
        rt = _rt()
        obj = rt.store.create("empty")
        rt.rebuild()
        pos = rt.objects[obj["id"]].transform.local_position
        assert (pos - Vec3(0.0, 0.0, 0.0)).length < _EPS

    def test_parent_child_wiring(self):
        rt = _rt()
        parent_d = rt.store.create("empty", name="P")
        child_d = rt.store.create("cube", parent=parent_d["id"], name="C")
        rt.rebuild()
        parent_go = rt.objects[parent_d["id"]]
        child_go = rt.objects[child_d["id"]]
        assert child_go.transform.parent is parent_go.transform

    def test_root_objects_have_no_transform_parent(self):
        rt = _rt()
        a = rt.store.create("cube")
        b = rt.store.create("sphere")
        rt.rebuild()
        assert rt.objects[a["id"]].transform.parent is None
        assert rt.objects[b["id"]].transform.parent is None


# ---------------------------------------------------------------------------
# rebuild() — idempotency
# ---------------------------------------------------------------------------


class TestRebuildIdempotency:
    def test_double_rebuild_stable_count(self):
        rt = _rt()
        rt.store.create("cube")
        rt.store.create("sphere")
        rt.rebuild()
        first_count = len(rt.objects)
        rt.rebuild()
        assert len(rt.objects) == first_count

    def test_triple_rebuild_stable_count(self):
        rt = _rt()
        rt.store.create("cube")
        for _ in range(3):
            rt.rebuild()
        assert len(rt.objects) == 1

    def test_rebuild_after_adding_object_gives_new_total(self):
        rt = _rt()
        rt.store.create("cube")
        rt.rebuild()
        rt.store.create("sphere")
        rt.rebuild()
        assert len(rt.objects) == 2


# ---------------------------------------------------------------------------
# on_rebuilt callback
# ---------------------------------------------------------------------------


class TestOnRebuiltCallback:
    def test_callback_fires_once_per_rebuild(self):
        calls: list[int] = []
        rt = _rt()
        rt.on_rebuilt = lambda: calls.append(1)
        rt.rebuild()
        assert calls == [1]

    def test_callback_fires_each_rebuild(self):
        calls: list[int] = []
        rt = _rt()
        rt.on_rebuilt = lambda: calls.append(1)
        rt.rebuild()
        rt.rebuild()
        assert len(calls) == 2

    def test_callback_none_does_not_raise(self):
        rt = _rt()
        assert rt.on_rebuilt is None
        rt.rebuild()  # must not raise

    def test_callback_sees_populated_objects(self):
        observed: list[int] = []
        rt = _rt()
        rt.store.create("cube")
        rt.store.create("sphere")
        rt.on_rebuilt = lambda: observed.append(len(rt.objects))
        rt.rebuild()
        assert observed == [2]


# ---------------------------------------------------------------------------
# spawn_position
# ---------------------------------------------------------------------------


class TestSpawnPosition:
    def test_no_spawn_returns_none(self):
        rt = _rt()
        rt.store.create("cube")
        rt.rebuild()
        assert rt.spawn_position is None

    def test_spawn_at_known_position(self):
        rt = _rt()
        rt.store.create("spawn", position=(4.0, 5.0, 6.0))
        rt.rebuild()
        sp = rt.spawn_position
        assert sp is not None
        assert abs(sp.x - 4.0) < _EPS
        assert abs(sp.y - 5.0) < _EPS
        assert abs(sp.z - 6.0) < _EPS

    def test_spawn_before_rebuild_is_none(self):
        """Without rebuild(), objects dict is empty → spawn_position is None."""
        rt = _rt()
        rt.store.create("spawn", position=(1.0, 2.0, 3.0))
        assert rt.spawn_position is None

    def test_first_dfs_spawn_wins(self):
        rt = _rt()
        rt.store.create("spawn", name="First", position=(1.0, 0.0, 0.0))
        rt.store.create("spawn", name="Second", position=(99.0, 0.0, 0.0))
        rt.rebuild()
        sp = rt.spawn_position
        assert sp is not None
        assert abs(sp.x - 1.0) < _EPS

    def test_spawn_under_translated_parent_composes_world_position(self):
        rt = _rt()
        parent = rt.store.create("empty", position=(10.0, 0.0, 0.0))
        rt.store.create("spawn", parent=parent["id"], position=(1.0, 0.0, 0.0))
        rt.rebuild()
        sp = rt.spawn_position
        assert sp is not None
        assert abs(sp.x - 11.0) < _EPS


# ---------------------------------------------------------------------------
# Saveable protocol — get_delta / apply_delta
# ---------------------------------------------------------------------------


class TestSaveableProtocol:
    def test_get_delta_delegates_to_store(self):
        rt = _rt()
        rt.store.create("cube")
        assert rt.get_delta() == rt.store.get_delta()

    def test_empty_store_get_delta_is_empty_dict(self):
        rt = _rt()
        assert rt.get_delta() == {}

    def test_apply_delta_empty_dict_gives_empty_objects(self):
        rt = _rt()
        rt.apply_delta({})
        assert rt.objects == {}

    def test_apply_delta_builds_correct_count(self):
        rt = _rt()
        rt.store.create("cube")
        rt.store.create("sphere")
        rt.store.create("light")
        delta = rt.get_delta()

        rt2 = SceneRuntime(visual_factory=None)
        rt2.apply_delta(delta)
        assert len(rt2.objects) == 3

    def test_apply_delta_restores_local_position(self):
        rt = _rt()
        obj = rt.store.create("cube", position=(7.0, 8.0, 9.0))
        delta = rt.get_delta()

        rt2 = SceneRuntime(visual_factory=None)
        rt2.apply_delta(delta)
        pos = rt2.objects[obj["id"]].transform.local_position
        assert abs(pos.x - 7.0) < _EPS
        assert abs(pos.y - 8.0) < _EPS
        assert abs(pos.z - 9.0) < _EPS

    def test_apply_delta_fires_on_rebuilt(self):
        calls: list[int] = []
        rt = _rt()
        rt.store.create("cube")
        delta = rt.get_delta()

        rt2 = SceneRuntime(visual_factory=None, on_rebuilt=lambda: calls.append(1))
        rt2.apply_delta(delta)
        assert calls == [1]

    def test_double_apply_delta_stable_count(self):
        rt = _rt()
        rt.store.create("cube")
        rt.store.create("sphere")
        delta = rt.get_delta()

        rt.apply_delta(delta)
        rt.apply_delta(delta)
        assert len(rt.objects) == 2


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_store_same_object_names(self):
        def build():
            r = _rt()
            r.store.create("cube", name="A", position=(1.0, 2.0, 3.0))
            r.store.create("sphere", name="B")
            r.rebuild()
            return {oid: go.name for oid, go in r.objects.items()}

        assert build() == build()

    def test_apply_delta_round_trip_names(self):
        rt = _rt()
        rt.store.create("cube", name="Crate", position=(1.0, 0.0, 0.0))
        rt.store.create("spawn", name="Start")
        delta = rt.get_delta()

        rt2 = SceneRuntime(visual_factory=None)
        rt2.apply_delta(delta)
        names = {go.name for go in rt2.objects.values()}
        assert names == {"Crate", "Start"}
