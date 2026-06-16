"""
tests/render/test_registry.py — Headless tests for render/registry.py.

Tests ComponentRegistry + instantiate/destroy lifecycle dispatch.  No panda3d.
"""

from __future__ import annotations

import pytest

from fire_engine.core.clock import Clock
from fire_engine.core.event_bus import EventBus
from fire_engine.render.component import Component
from fire_engine.render.registry import (
    ComponentRegistry,
    destroy,
    find_objects_with_tag,
    find_with_tag,
    instantiate,
)


def _make_clock(dt: float = 0.016, fixed_dt: float = 1.0) -> Clock:
    """Build a Clock with a large fixed_dt to avoid multiple fixed_update ticks."""
    clock = Clock(fixed_dt=fixed_dt, bus=EventBus())
    clock.update(dt)
    return clock


class _Ticker(Component):
    """Counts update() calls."""

    def __init__(self) -> None:
        super().__init__()
        self.ticks: int = 0
        self.log: list[str] = []

    def awake(self) -> None:
        self.log.append("awake")

    def on_enable(self) -> None:
        self.log.append("on_enable")

    def start(self) -> None:
        self.log.append("start")

    def update(self, dt: float) -> None:
        self.ticks += 1
        self.log.append(f"update:{dt:.3f}")

    def on_disable(self) -> None:
        self.log.append("on_disable")

    def on_destroy(self) -> None:
        self.log.append("on_destroy")


@pytest.fixture(autouse=True)
def _clear_registry() -> None:
    """Reset the registry singleton before every test."""
    ComponentRegistry.clear()
    yield
    ComponentRegistry.clear()


class TestInstantiate:
    def test_returns_game_object(self) -> None:
        from fire_engine.render.gameobject import GameObject

        go = instantiate()
        assert isinstance(go, GameObject)

    def test_default_position_is_zero(self) -> None:
        go = instantiate()
        pos = go.transform.local_position
        assert abs(pos.x) < 1e-6
        assert abs(pos.y) < 1e-6
        assert abs(pos.z) < 1e-6

    def test_custom_position(self) -> None:
        from fire_engine.core.math3d import Vec3

        go = instantiate(position=Vec3(1.0, 2.0, 3.0))
        pos = go.transform.local_position
        assert abs(pos.x - 1.0) < 1e-6
        assert abs(pos.y - 2.0) < 1e-6
        assert abs(pos.z - 3.0) < 1e-6

    def test_template_copies_name_and_tag(self) -> None:
        template = instantiate()
        template.name = "Bullet"
        template.tag = "projectile"

        clone = instantiate(template=template)
        assert clone.name == "Bullet"
        assert clone.tag == "projectile"


class TestAddComponentAndLifecycle:
    def test_add_component_returns_instance(self) -> None:
        go = instantiate()
        t = go.add_component(_Ticker)
        assert isinstance(t, _Ticker)

    def test_get_component_returns_added(self) -> None:
        go = instantiate()
        t = go.add_component(_Ticker)
        assert go.get_component(_Ticker) is t

    def test_run_frame_dispatches_awake_start_update_in_order(self) -> None:
        go = instantiate()
        t = go.add_component(_Ticker)
        clock = _make_clock(dt=0.016, fixed_dt=1.0)
        ComponentRegistry.run_frame(clock)
        # awake → on_enable → start → update (in order)
        assert t.log[:4] == ["awake", "on_enable", "start", "update:0.016"]

    def test_start_runs_only_once(self) -> None:
        go = instantiate()
        t = go.add_component(_Ticker)
        clock = _make_clock(dt=0.016, fixed_dt=1.0)
        ComponentRegistry.run_frame(clock)
        ComponentRegistry.run_frame(clock)
        starts = [e for e in t.log if e == "start"]
        assert len(starts) == 1

    def test_update_runs_each_frame(self) -> None:
        go = instantiate()
        t = go.add_component(_Ticker)
        clock = _make_clock(dt=0.016, fixed_dt=1.0)
        ComponentRegistry.run_frame(clock)
        ComponentRegistry.run_frame(clock)
        assert t.ticks == 2

    def test_disabled_component_skips_update(self) -> None:
        go = instantiate()
        t = go.add_component(_Ticker)
        clock = _make_clock(dt=0.016, fixed_dt=1.0)
        ComponentRegistry.run_frame(clock)  # awake + start + update
        t.enabled = False
        ComponentRegistry.run_frame(clock)  # disabled — no update
        assert t.ticks == 1  # only the first frame's update

    def test_inactive_gameobject_skips_update(self) -> None:
        go = instantiate()
        t = go.add_component(_Ticker)
        clock = _make_clock(dt=0.016, fixed_dt=1.0)
        ComponentRegistry.run_frame(clock)  # first frame: ticks=1
        go.set_active(False)
        ComponentRegistry.run_frame(clock)
        assert t.ticks == 1


class TestDestroy:
    def test_destroy_component_calls_on_destroy(self) -> None:
        go = instantiate()
        t = go.add_component(_Ticker)
        clock = _make_clock(dt=0.016, fixed_dt=1.0)
        ComponentRegistry.run_frame(clock)
        t.log.clear()
        destroy(t)
        ComponentRegistry.run_frame(clock)
        assert "on_destroy" in t.log

    def test_destroy_component_calls_on_disable_when_enabled(self) -> None:
        go = instantiate()
        t = go.add_component(_Ticker)
        clock = _make_clock(dt=0.016, fixed_dt=1.0)
        ComponentRegistry.run_frame(clock)
        t.log.clear()
        destroy(t)
        ComponentRegistry.run_frame(clock)
        assert "on_disable" in t.log

    def test_destroy_gameobject_calls_on_destroy(self) -> None:
        go = instantiate()
        t = go.add_component(_Ticker)
        clock = _make_clock(dt=0.016, fixed_dt=1.0)
        ComponentRegistry.run_frame(clock)
        t.log.clear()
        destroy(go)
        ComponentRegistry.run_frame(clock)
        assert "on_destroy" in t.log

    def test_destroy_wrong_type_raises(self) -> None:
        with pytest.raises(TypeError):
            ComponentRegistry.destroy("not_a_go_or_component")  # type: ignore[arg-type]


class TestFindWithTag:
    def test_find_with_tag_returns_matching_go(self) -> None:
        go = instantiate()
        go.tag = "player"
        result = find_with_tag("player")
        assert result is go

    def test_find_with_tag_returns_none_when_missing(self) -> None:
        assert find_with_tag("nonexistent") is None

    def test_find_objects_with_tag_returns_all(self) -> None:
        a = instantiate()
        a.tag = "enemy"
        b = instantiate()
        b.tag = "enemy"
        c = instantiate()
        c.tag = "player"
        enemies = find_objects_with_tag("enemy")
        assert set(enemies) == {a, b}
        assert c not in enemies


class TestRegistryClear:
    def test_clear_resets_state(self) -> None:
        go = instantiate()
        go.tag = "foo"
        ComponentRegistry.clear()
        assert find_with_tag("foo") is None
