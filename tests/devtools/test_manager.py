"""
tests/devtools/test_manager.py — tests for fire_engine/devtools/manager.py.

Covers DevToolsManager: tool registry, panel building, selectable add/remove/find,
pick_and_select, and enabled flag. Fully headless; no panda3d imports.
"""

from __future__ import annotations

import pytest

from fire_engine.core.math3d import Vec3
from fire_engine.devtools.manager import DevToolsManager
from fire_engine.devtools.picking import Selectable
from fire_engine.devtools.types import Panel, Section
from fire_engine.render.gameobject import GameObject
from fire_engine.render.registry import ComponentRegistry


@pytest.fixture(autouse=True)
def _clean_registry():
    ComponentRegistry.clear()
    yield
    ComponentRegistry.clear()


# ---------------------------------------------------------------------------
# Minimal fake DevTool (avoids importing tools.py to keep this unit-focused)
# ---------------------------------------------------------------------------


class _FakeTool:
    tool_id = "fake"
    title = "Fake"

    @property
    def revision(self) -> int:
        return 0

    def build(self) -> Panel:
        return Panel(self.tool_id, self.title, [Section("S", [])])


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def test_initial_state():
    mgr = DevToolsManager()
    assert mgr.tools == []
    assert mgr.selectables == []
    assert mgr.selection.current is None
    assert mgr.enabled is False


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


def test_register_tool_appends_and_returns():
    mgr = DevToolsManager()
    tool = _FakeTool()
    returned = mgr.register_tool(tool)  # type: ignore[arg-type]
    assert returned is tool
    assert mgr.tools == [tool]


def test_register_multiple_tools_ordered():
    mgr = DevToolsManager()
    t1, t2 = _FakeTool(), _FakeTool()
    mgr.register_tool(t1)  # type: ignore[arg-type]
    mgr.register_tool(t2)  # type: ignore[arg-type]
    assert mgr.tools[0] is t1
    assert mgr.tools[1] is t2


def test_panels_calls_build_on_each_tool():
    mgr = DevToolsManager()
    t1, t2 = _FakeTool(), _FakeTool()
    mgr.register_tool(t1)  # type: ignore[arg-type]
    mgr.register_tool(t2)  # type: ignore[arg-type]
    panels = mgr.panels()
    assert len(panels) == 2
    assert all(isinstance(p, Panel) for p in panels)


def test_panels_empty_when_no_tools():
    mgr = DevToolsManager()
    assert mgr.panels() == []


# ---------------------------------------------------------------------------
# add_selectable / find_selectable / remove_selectable
# ---------------------------------------------------------------------------


def test_add_selectable_appends_and_returns():
    mgr = DevToolsManager()
    go = GameObject(name="Box")
    sel = mgr.add_selectable(go, Vec3(0.5, 0.5, 0.5))
    assert isinstance(sel, Selectable)
    assert sel.game_object is go
    assert sel in mgr.selectables


def test_find_selectable_returns_correct_entry():
    mgr = DevToolsManager()
    go1 = GameObject(name="A")
    go2 = GameObject(name="B")
    mgr.add_selectable(go1, Vec3(1, 1, 1))
    mgr.add_selectable(go2, Vec3(1, 1, 1))
    assert mgr.find_selectable(go1).game_object is go1
    assert mgr.find_selectable(go2).game_object is go2


def test_find_selectable_none_for_unregistered():
    mgr = DevToolsManager()
    go = GameObject(name="Unknown")
    assert mgr.find_selectable(go) is None


def test_remove_selectable_drops_entry():
    mgr = DevToolsManager()
    go = GameObject(name="Box")
    mgr.add_selectable(go, Vec3(1, 1, 1))
    mgr.remove_selectable(go)
    assert mgr.selectables == []


def test_remove_selectable_clears_selection_if_selected():
    mgr = DevToolsManager()
    go = GameObject(name="Box")
    go.transform.local_position = Vec3(0, 5, 0)
    mgr.add_selectable(go, Vec3(1, 1, 1))
    mgr.selection.set(go)
    assert mgr.selection.current is go
    mgr.remove_selectable(go)
    assert mgr.selection.current is None


def test_remove_selectable_does_not_clear_other_selection():
    mgr = DevToolsManager()
    go1 = GameObject(name="A")
    go2 = GameObject(name="B")
    mgr.add_selectable(go1, Vec3(1, 1, 1))
    mgr.add_selectable(go2, Vec3(1, 1, 1))
    mgr.selection.set(go2)
    mgr.remove_selectable(go1)
    assert mgr.selection.current is go2


# ---------------------------------------------------------------------------
# pick / pick_and_select
# ---------------------------------------------------------------------------


def test_pick_returns_nearest():
    mgr = DevToolsManager()
    near = GameObject(name="near")
    near.transform.local_position = Vec3(0, 5, 0)
    far = GameObject(name="far")
    far.transform.local_position = Vec3(0, 20, 0)
    mgr.add_selectable(near, Vec3(1, 1, 1))
    mgr.add_selectable(far, Vec3(1, 1, 1))
    hit = mgr.pick(Vec3(0, 0, 0), Vec3(0, 1, 0))
    assert hit is near


def test_pick_returns_none_on_miss():
    mgr = DevToolsManager()
    go = GameObject(name="Box")
    go.transform.local_position = Vec3(0, 5, 0)
    mgr.add_selectable(go, Vec3(1, 1, 1))
    assert mgr.pick(Vec3(50, 0, 0), Vec3(0, 1, 0)) is None


def test_pick_and_select_updates_selection():
    mgr = DevToolsManager()
    go = GameObject(name="Box")
    go.transform.local_position = Vec3(0, 5, 0)
    mgr.add_selectable(go, Vec3(1, 1, 1))
    hit = mgr.pick_and_select(Vec3(0, 0, 0), Vec3(0, 1, 0))
    assert hit is go
    assert mgr.selection.current is go


def test_pick_and_select_deselects_on_miss():
    mgr = DevToolsManager()
    go = GameObject(name="Box")
    go.transform.local_position = Vec3(0, 5, 0)
    mgr.add_selectable(go, Vec3(1, 1, 1))
    mgr.selection.set(go)
    mgr.pick_and_select(Vec3(50, 0, 0), Vec3(0, 1, 0))
    assert mgr.selection.current is None


# ---------------------------------------------------------------------------
# enabled flag
# ---------------------------------------------------------------------------


def test_enabled_defaults_false():
    mgr = DevToolsManager()
    assert mgr.enabled is False


def test_enabled_can_be_set():
    mgr = DevToolsManager()
    mgr.enabled = True
    assert mgr.enabled is True
