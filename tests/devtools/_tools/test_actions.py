"""
tests/devtools/_tools/test_actions.py — tests for fire_engine/devtools/_tools/actions.py.

Covers ActionsTool: construction, initial build, add_action / revision bump,
button invocation. Fully headless; no panda3d imports.
"""

from __future__ import annotations

from fire_engine.devtools._tools.actions import ActionsTool
from fire_engine.devtools.types import Panel

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_title_and_tool_id():
    tool = ActionsTool()
    assert tool.title == "Actions"
    assert tool.tool_id == "actions"


def test_custom_title():
    tool = ActionsTool(title="World")
    assert tool.title == "World"


def test_initial_revision_zero():
    tool = ActionsTool()
    assert tool.revision == 0


def test_no_initial_actions():
    tool = ActionsTool()
    p = tool.build()
    assert p.buttons == []


def test_initial_actions_from_dict():
    fired = []
    tool = ActionsTool("T", {"Go": lambda: fired.append(1)})
    p = tool.build()
    assert len(p.buttons) == 1
    assert p.buttons[0].label == "Go"
    p.buttons[0].on_click()
    assert fired == [1]


# ---------------------------------------------------------------------------
# add_action
# ---------------------------------------------------------------------------


def test_add_action_bumps_revision():
    tool = ActionsTool()
    r0 = tool.revision
    tool.add_action("Spawn", lambda: None)
    assert tool.revision == r0 + 1


def test_add_action_appends_button():
    tool = ActionsTool("T", {"A": lambda: None})
    tool.add_action("B", lambda: None)
    p = tool.build()
    assert [b.label for b in p.buttons] == ["A", "B"]


def test_add_action_multiple_bumps_revision_each_time():
    tool = ActionsTool()
    for i in range(3):
        tool.add_action(f"X{i}", lambda: None)
    assert tool.revision == 3


def test_add_action_handler_invoked():
    fired = []
    tool = ActionsTool()
    tool.add_action("Boom", lambda: fired.append("boom"))
    tool.build().buttons[0].on_click()
    assert fired == ["boom"]


# ---------------------------------------------------------------------------
# Panel structure
# ---------------------------------------------------------------------------


def test_build_returns_panel_with_correct_ids():
    tool = ActionsTool("Ops")
    p = tool.build()
    assert isinstance(p, Panel)
    assert p.tool_id == "actions"
    assert p.title == "Ops"


def test_build_includes_revision_in_panel():
    tool = ActionsTool()
    tool.add_action("X", lambda: None)
    p = tool.build()
    assert p.revision == tool.revision


def test_build_sections_empty():
    tool = ActionsTool()
    p = tool.build()
    assert p.sections == []
