"""
tests/render/test_component.py — Headless tests for render/component.py.

Tests the Component base class lifecycle defaults.  All panda3d-free.
"""

from __future__ import annotations

from fire_engine.render.component import Component


class _Recorder(Component):
    """Minimal subclass that records lifecycle call order."""

    def __init__(self) -> None:
        super().__init__()
        self.log: list[str] = []

    def awake(self) -> None:
        self.log.append("awake")

    def on_enable(self) -> None:
        self.log.append("on_enable")

    def start(self) -> None:
        self.log.append("start")

    def update(self, dt: float) -> None:
        self.log.append(f"update:{dt}")

    def late_update(self, dt: float) -> None:
        self.log.append(f"late_update:{dt}")

    def fixed_update(self, dt: float) -> None:
        self.log.append(f"fixed_update:{dt}")

    def on_disable(self) -> None:
        self.log.append("on_disable")

    def on_destroy(self) -> None:
        self.log.append("on_destroy")


class TestComponentDefaults:
    """Base Component has the right initial state."""

    def test_initial_enabled_is_true(self) -> None:
        c = Component()
        assert c.enabled is True

    def test_initial_game_object_is_none(self) -> None:
        c = Component()
        assert c.game_object is None

    def test_initial_transform_is_none(self) -> None:
        c = Component()
        assert c.transform is None

    def test_initial_started_is_false(self) -> None:
        c = Component()
        assert c._started is False

    def test_default_lifecycle_methods_are_noop(self) -> None:
        """All lifecycle no-ops must return None without raising."""
        c = Component()
        assert c.awake() is None
        assert c.on_enable() is None
        assert c.start() is None
        assert c.update(0.016) is None
        assert c.late_update(0.016) is None
        assert c.fixed_update(0.02) is None
        assert c.on_disable() is None
        assert c.on_destroy() is None

    def test_enabled_toggle(self) -> None:
        c = Component()
        c.enabled = False
        assert c.enabled is False
        c.enabled = True
        assert c.enabled is True


class TestRecorderSubclass:
    """Subclass overrides are invoked correctly by the caller."""

    def test_awake_recorded(self) -> None:
        r = _Recorder()
        r.awake()
        assert r.log == ["awake"]

    def test_lifecycle_sequence(self) -> None:
        r = _Recorder()
        r.awake()
        r.on_enable()
        r.start()
        r.update(0.016)
        r.late_update(0.016)
        r.fixed_update(0.02)
        r.on_disable()
        r.on_destroy()
        assert r.log == [
            "awake",
            "on_enable",
            "start",
            "update:0.016",
            "late_update:0.016",
            "fixed_update:0.02",
            "on_disable",
            "on_destroy",
        ]

    def test_update_receives_correct_dt(self) -> None:
        r = _Recorder()
        r.update(0.1)
        assert r.log == ["update:0.1"]

    def test_multiple_components_are_independent(self) -> None:
        a, b = _Recorder(), _Recorder()
        a.awake()
        b.awake()
        b.start()
        assert a.log == ["awake"]
        assert b.log == ["awake", "start"]
