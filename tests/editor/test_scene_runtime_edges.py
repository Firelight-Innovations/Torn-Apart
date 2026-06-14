"""Characterisation / golden-master tests for SceneRuntime edge cases.

These tests PIN the current behaviour of SceneRuntime (fire_engine/scene/runtime.py).
They do NOT fix bugs — they record what the code actually does so regressions are
caught. Where behaviour is surprising, a comment flags the suspicion.

All tests are HEADLESS (visual_factory=None). Any path that requires a real
visual factory / panda3d is skipped with an explanatory xfail/skip marker.

Construction pattern mirrors tests/editor/test_scene_roundtrip.py exactly:
  - ComponentRegistry.clear() in setup
  - SceneRuntime(visual_factory=None)
  - drive the store directly (no Daemon / EditorSession needed)
"""

from __future__ import annotations

from fire_engine.core.math3d import Vec3
from fire_engine.render.registry import ComponentRegistry
from fire_engine.scene import SceneRuntime
from fire_engine.scene.objects import SceneObjectStore

_EPS = 1e-6


# ---------------------------------------------------------------------------
# Helpers shared across test cases
# ---------------------------------------------------------------------------


def _make_runtime() -> SceneRuntime:
    """Fresh runtime with registry cleared; visual_factory=None."""
    ComponentRegistry.clear()
    return SceneRuntime(visual_factory=None)


def _populate_store(store: SceneObjectStore):
    """Create cube + child empty + spawn in *store*; returns (cube_id, child_id, spawn_id)."""
    cube = store.create("cube", name="Crate", position=(4.0, 2.0, 8.0))
    child = store.create("empty", name="Pivot", parent=cube["id"], position=(1.0, 0.0, 0.0))
    spawn = store.create("spawn", name="Start", position=(5.0, 5.0, 1.0))
    return cube["id"], child["id"], spawn["id"]


# ---------------------------------------------------------------------------
# 1. Headless construction
# ---------------------------------------------------------------------------


class TestHeadlessConstruction:
    def test_construct_no_factory(self):
        """SceneRuntime constructs without error when visual_factory is None."""
        rt = _make_runtime()
        assert rt.visual_factory is None

    def test_initial_objects_empty(self):
        """objects dict is empty before any rebuild."""
        rt = _make_runtime()
        assert rt.objects == {}

    def test_initial_spawn_position_is_none(self):
        """spawn_position is None on a fresh runtime (no objects, no store data)."""
        rt = _make_runtime()
        assert rt.spawn_position is None

    def test_rebuild_empty_store_produces_no_objects(self):
        """rebuild() on an empty store leaves objects dict empty."""
        rt = _make_runtime()
        rt.rebuild()
        assert rt.objects == {}

    def test_rebuild_populates_objects(self):
        """After rebuild(), runtime.objects contains an entry per store object."""
        rt = _make_runtime()
        _populate_store(rt.store)
        rt.rebuild()
        # cube + child + spawn  = 3
        assert len(rt.objects) == 3

    def test_rebuild_sets_correct_names(self):
        """GameObject.name matches the authored name for each object."""
        rt = _make_runtime()
        cube_id, child_id, spawn_id = _populate_store(rt.store)
        rt.rebuild()
        assert rt.objects[cube_id].name == "Crate"
        assert rt.objects[child_id].name == "Pivot"
        assert rt.objects[spawn_id].name == "Start"

    def test_rebuild_stamps_scene_tag(self):
        """Every built GameObject carries the SCENE_TAG ('editor_scene')."""
        from fire_engine.scene.runtime import SCENE_TAG

        rt = _make_runtime()
        _populate_store(rt.store)
        rt.rebuild()
        for go in rt.objects.values():
            assert go.tag == SCENE_TAG


# ---------------------------------------------------------------------------
# 2. rebuild() idempotency — no leaks, stable count
# ---------------------------------------------------------------------------


class TestRebuildIdempotency:
    def test_double_rebuild_object_count_stable(self):
        """Calling rebuild() twice leaves the same number of objects (no duplicates)."""
        rt = _make_runtime()
        _populate_store(rt.store)
        rt.rebuild()
        count_first = len(rt.objects)
        rt.rebuild()
        assert len(rt.objects) == count_first

    def test_double_rebuild_replaces_go_identities(self):
        """Second rebuild() creates new GameObjects — old Python ids are gone."""
        rt = _make_runtime()
        _populate_store(rt.store)
        rt.rebuild()
        ids_first = {id(go) for go in rt.objects.values()}
        rt.rebuild()
        ids_second = {id(go) for go in rt.objects.values()}
        # All object Python ids must change (old ones destroyed)
        assert ids_first.isdisjoint(ids_second), (
            "SUSPICION: old GameObjects survived a second rebuild() — possible leak"
        )

    def test_triple_rebuild_object_count_stable(self):
        """Three rebuilds in a row should not inflate the count."""
        rt = _make_runtime()
        _populate_store(rt.store)
        for _ in range(3):
            rt.rebuild()
        assert len(rt.objects) == 3

    def test_rebuild_clears_objects_before_repopulating(self):
        """runtime.objects is empty between teardown and repopulation inside rebuild."""
        # We cannot observe the interior of rebuild(), but we can verify that
        # adding an object to the store AFTER a rebuild, then rebuilding again,
        # gives exactly the new total (not double).
        rt = _make_runtime()
        rt.store.create("cube", name="A")
        rt.rebuild()
        assert len(rt.objects) == 1
        rt.store.create("sphere", name="B")
        rt.rebuild()
        assert len(rt.objects) == 2


# ---------------------------------------------------------------------------
# 3. on_rebuilt callback
# ---------------------------------------------------------------------------


class TestOnRebuiltCallback:
    def test_callback_fires_on_rebuild(self):
        """on_rebuilt is called exactly once per rebuild() invocation."""
        calls: list[int] = []
        rt = _make_runtime()
        rt.on_rebuilt = lambda: calls.append(1)
        rt.rebuild()
        assert calls == [1]

    def test_callback_fires_on_double_rebuild(self):
        """Two rebuild() calls → on_rebuilt fires twice."""
        calls: list[int] = []
        rt = _make_runtime()
        rt.on_rebuilt = lambda: calls.append(1)
        rt.rebuild()
        rt.rebuild()
        assert len(calls) == 2

    def test_callback_none_does_not_raise(self):
        """on_rebuilt=None (default) does not raise during rebuild()."""
        rt = _make_runtime()
        assert rt.on_rebuilt is None
        rt.rebuild()  # must not raise

    def test_callback_receives_correct_object_count_after_build(self):
        """When on_rebuilt fires, runtime.objects is already populated."""
        observed: list[int] = []
        rt = _make_runtime()
        _populate_store(rt.store)
        rt.on_rebuilt = lambda: observed.append(len(rt.objects))
        rt.rebuild()
        assert observed == [3]


# ---------------------------------------------------------------------------
# 4. spawn_position
# ---------------------------------------------------------------------------


class TestSpawnPosition:
    def test_no_spawn_returns_none(self):
        """spawn_position is None when the store has no spawn object."""
        rt = _make_runtime()
        rt.store.create("cube", name="Crate")
        rt.rebuild()
        assert rt.spawn_position is None

    def test_single_spawn_returns_world_position(self):
        """spawn_position equals the authored position for a root-level spawn."""
        rt = _make_runtime()
        rt.store.create("spawn", name="S", position=(7.0, 3.0, 1.5))
        rt.rebuild()
        sp = rt.spawn_position
        assert sp is not None
        assert (sp - Vec3(7.0, 3.0, 1.5)).length < _EPS

    def test_spawn_before_rebuild_returns_none(self):
        """spawn_position is None before rebuild() even if the store has a spawn.

        SUSPICION: spawn_position iterates store.tree() but looks up self.objects —
        if rebuild() has not been called, the store has data but objects is empty,
        so the get() returns None and the method returns None (current pinned behaviour).
        """
        rt = _make_runtime()
        rt.store.create("spawn", name="S", position=(1.0, 2.0, 3.0))
        # deliberately no rebuild()
        assert rt.spawn_position is None

    def test_spawn_parented_under_translated_parent_returns_world_position(self):
        """A spawn parented under a translated parent composes world position correctly."""
        rt = _make_runtime()
        parent = rt.store.create("empty", name="Group", position=(10.0, 0.0, 0.0))
        rt.store.create("spawn", name="S", parent=parent["id"], position=(1.0, 0.0, 0.0))
        rt.rebuild()
        sp = rt.spawn_position
        assert sp is not None
        # World position: parent (10,0,0) + child local (1,0,0) = (11,0,0)
        assert (sp - Vec3(11.0, 0.0, 0.0)).length < _EPS

    def test_first_dfs_spawn_wins_when_multiple_spawns_exist(self):
        """When there are multiple spawn objects, spawn_position returns the first DFS one."""
        rt = _make_runtime()
        rt.store.create("spawn", name="First", position=(1.0, 0.0, 0.0))
        rt.store.create("spawn", name="Second", position=(99.0, 0.0, 0.0))
        rt.rebuild()
        sp = rt.spawn_position
        assert sp is not None
        # Pinning: first root spawn (DFS) is at x=1; second at x=99.
        assert (sp - Vec3(1.0, 0.0, 0.0)).length < _EPS


# ---------------------------------------------------------------------------
# 5. apply_delta / get_delta round-trip
# ---------------------------------------------------------------------------


class TestDeltaRoundTrip:
    def test_empty_store_get_delta_is_empty_dict(self):
        """An empty SceneRuntime's store returns {} from get_delta."""
        rt = _make_runtime()
        assert rt.get_delta() == {}

    def test_apply_delta_empty_dict_gives_empty_objects(self):
        """apply_delta({}) produces an empty objects dict."""
        rt = _make_runtime()
        rt.apply_delta({})
        assert rt.objects == {}

    def test_apply_delta_rebuilds_correct_count(self):
        """apply_delta with a saved store delta builds the expected number of GameObjects."""
        rt = _make_runtime()
        _populate_store(rt.store)
        delta = rt.store.get_delta()

        rt2 = SceneRuntime(visual_factory=None)
        rt2.apply_delta(delta)
        assert len(rt2.objects) == 3

    def test_apply_delta_restores_transforms(self):
        """Positions survive a get_delta / apply_delta round-trip exactly."""
        rt = _make_runtime()
        cube_id, _, _ = _populate_store(rt.store)
        delta = rt.store.get_delta()

        rt2 = SceneRuntime(visual_factory=None)
        rt2.apply_delta(delta)

        go = rt2.objects[cube_id]
        assert (go.transform.local_position - Vec3(4.0, 2.0, 8.0)).length < _EPS

    def test_apply_delta_calls_rebuild_and_fires_on_rebuilt(self):
        """apply_delta internally calls rebuild(), so on_rebuilt fires."""
        calls: list[int] = []
        rt = _make_runtime()
        _populate_store(rt.store)
        delta = rt.store.get_delta()

        rt2 = SceneRuntime(visual_factory=None, on_rebuilt=lambda: calls.append(1))
        rt2.apply_delta(delta)
        assert calls == [1]

    def test_double_apply_delta_gives_stable_count(self):
        """Calling apply_delta twice does not leak GameObjects."""
        rt = _make_runtime()
        _populate_store(rt.store)
        delta = rt.get_delta()

        # apply twice
        rt.apply_delta(delta)
        rt.apply_delta(delta)
        assert len(rt.objects) == 3

    def test_get_delta_delegates_to_store(self):
        """runtime.get_delta() and runtime.store.get_delta() return identical dicts."""
        rt = _make_runtime()
        _populate_store(rt.store)
        assert rt.get_delta() == rt.store.get_delta()


# ---------------------------------------------------------------------------
# 6. Parent / child wiring in headless rebuild
# ---------------------------------------------------------------------------


class TestParentChildWiring:
    def test_child_transform_parent_matches_parent_go(self):
        """After rebuild(), child.transform.parent is the parent object's transform."""
        rt = _make_runtime()
        cube_id, child_id, _ = _populate_store(rt.store)
        rt.rebuild()
        child_go = rt.objects[child_id]
        cube_go = rt.objects[cube_id]
        assert child_go.transform.parent is cube_go.transform

    def test_root_objects_have_no_parent(self):
        """Root-level objects (parent=None in store) have no transform parent."""
        rt = _make_runtime()
        cube_id, _, spawn_id = _populate_store(rt.store)
        rt.rebuild()
        assert rt.objects[cube_id].transform.parent is None
        assert rt.objects[spawn_id].transform.parent is None
