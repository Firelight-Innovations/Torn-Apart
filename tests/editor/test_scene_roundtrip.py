"""Editor scene → game round-trip — placed objects become live GameObjects.

The headline integration for the scene pipeline: author objects in the editor
daemon (cube/light/spawn + a parented child) → save → the GAME's own load path
(SceneRuntime registered with a SaveManager, what `python main.py --load` does)
instantiates them with exact transforms. Headless throughout (visual_factory
None or a recording stub — never panda3d).
"""
from __future__ import annotations

import asyncio

import pytest

from fire_engine.core import Clock, EventBus, load_config
from fire_engine.core.math3d import Quat, Vec3
from fire_engine.core.rng import set_world_seed
from fire_engine.save import SaveManager
from fire_engine.scene import SceneRuntime
from fire_engine.scene.objects import SceneObjectStore
from fire_engine.world.terrain import ChunkManager
from fire_engine.render.registry import ComponentRegistry

from fire_editor import Daemon, EditorSession

_EPS = 1e-6


def _run(coro):
    return asyncio.run(coro)


def _daemon_with_world():
    cfg = load_config()
    daemon = Daemon()
    daemon.session = EditorSession(cfg)
    return daemon, cfg


async def _noop_broadcast(*args, **kwargs):
    return None


def _author_scene(daemon):
    """Create cube (TRS'd) + child empty + light + spawn via the RPC service."""
    daemon.server.broadcast_notification = _noop_broadcast  # no clients attached
    rot = Quat.from_axis_angle(Vec3(0.0, 0.0, 1.0), 0.7)
    cube = _run(daemon.scene.create(
        {"kind": "cube", "name": "Crate", "x": 4.0, "y": 2.0, "z": 8.0}))["object"]
    _run(daemon.scene.set_transform(
        {"id": cube["id"], "rw": rot.w, "rx": rot.x, "ry": rot.y, "rz": rot.z,
         "sx": 2.0, "sy": 1.0, "sz": 1.0}))
    child = _run(daemon.scene.create(
        {"kind": "empty", "name": "Pivot", "parent": cube["id"],
         "x": 1.0, "y": 0.0, "z": 0.0}))["object"]
    light = _run(daemon.scene.create(
        {"kind": "light", "x": -3.0, "y": 5.0, "z": 9.0}))["object"]
    spawn = _run(daemon.scene.create(
        {"kind": "spawn", "x": 1.0, "y": 2.0, "z": 3.0}))["object"]
    return cube, child, light, spawn, rot


def _game_load(cfg, save_path, factory=None) -> SceneRuntime:
    """The game's load path: fresh registry + SaveManager + SceneRuntime."""
    set_world_seed(cfg.world_seed)
    ComponentRegistry.clear()
    bus = EventBus()
    clock = Clock(fixed_dt=cfg.fixed_dt, bus=bus)
    cm = ChunkManager(cfg, bus)
    sm = SaveManager(cfg, clock)
    sm.register(cm)
    runtime = SceneRuntime(visual_factory=factory)
    sm.register(runtime)
    sm.load(str(save_path))
    return runtime


class TestSceneRoundTrip:
    def test_objects_round_trip_with_exact_transforms(self, tmp_path):
        daemon, cfg = _daemon_with_world()
        cube, child, light, spawn, rot = _author_scene(daemon)
        save_path = tmp_path / "scene.ta"
        daemon.session.save(str(save_path))

        runtime = _game_load(cfg, save_path)
        assert len(runtime.objects) == 4

        cube_go = runtime.objects[cube["id"]]
        assert cube_go.name == "Crate"
        assert cube_go.tag == "editor_scene"
        assert (cube_go.transform.position - Vec3(4.0, 2.0, 8.0)).length < _EPS
        q = cube_go.transform.rotation
        assert abs(abs(q.w * rot.w + q.x * rot.x + q.y * rot.y + q.z * rot.z) - 1.0) < 1e-5
        assert (cube_go.transform.local_scale - Vec3(2.0, 1.0, 1.0)).length < _EPS

        # Child is parented with its authored LOCAL offset intact; the world
        # position composes through the engine's own Transform math (rotated
        # off the parent's +X axis, not a plain translation).
        child_go = runtime.objects[child["id"]]
        assert child_go.transform.parent is cube_go.transform
        assert (child_go.transform.local_position - Vec3(1.0, 0.0, 0.0)).length < _EPS
        world_offset = child_go.transform.position - cube_go.transform.position
        assert world_offset.length > _EPS  # composed, not just copied
        assert abs(world_offset.z) < _EPS  # rotation was about Z

        # Spawn point surfaces as the world-space player start.
        assert runtime.spawn_position is not None
        assert (runtime.spawn_position - Vec3(1.0, 2.0, 3.0)).length < _EPS

    def test_visual_factory_contract(self, tmp_path):
        daemon, cfg = _daemon_with_world()
        _author_scene(daemon)
        save_path = tmp_path / "scene.ta"
        daemon.session.save(str(save_path))

        class RecordingFactory:
            def __init__(self):
                self.attached: list[tuple[str, str]] = []
                self.teardowns = 0

            def attach(self, go, kind, obj):
                self.attached.append((kind, go.name))

            def teardown(self):
                self.teardowns += 1

        factory = RecordingFactory()
        runtime = _game_load(cfg, save_path, factory=factory)
        # Every object reaches the factory exactly once (factory no-ops decide).
        assert sorted(k for k, _ in factory.attached) == [
            "cube", "empty", "light", "spawn"]
        assert factory.teardowns == 1  # the rebuild that loaded the scene

        # Reload (F9): one more teardown, attachments doubled, objects stable.
        runtime.apply_delta(runtime.store.get_delta())
        assert factory.teardowns == 2
        assert len(factory.attached) == 8
        assert len(runtime.objects) == 4

    def test_double_load_does_not_leak_objects(self, tmp_path):
        daemon, cfg = _daemon_with_world()
        _author_scene(daemon)
        save_path = tmp_path / "scene.ta"
        daemon.session.save(str(save_path))

        runtime = _game_load(cfg, save_path)
        first_ids = {id(go) for go in runtime.objects.values()}
        runtime.apply_delta(runtime.store.get_delta())
        assert len(runtime.objects) == 4
        assert {id(go) for go in runtime.objects.values()} != first_ids

    def test_empty_scene_loads_clean(self, tmp_path):
        daemon, cfg = _daemon_with_world()
        save_path = tmp_path / "empty.ta"
        daemon.session.save(str(save_path))
        runtime = _game_load(cfg, save_path)
        assert runtime.objects == {}
        assert runtime.spawn_position is None


def test_editor_and_engine_share_one_store_class():
    """Schema-drift guard: the shim must re-export the engine class itself."""
    from fire_editor import scene_objects
    assert scene_objects.SceneObjectStore is SceneObjectStore
