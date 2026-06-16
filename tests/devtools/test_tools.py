"""
tests/devtools/test_tools.py — tests for fire_engine/devtools/tools.py.

tools.py is a re-export shim that gathers all DevTool subclasses from _tools/.
Tests confirm each symbol is importable from this path AND exercise one real
behaviour. Fully headless; no panda3d imports.
"""

from __future__ import annotations

import pytest

from fire_engine.devtools.selection import Selection
from fire_engine.devtools.tools import (
    ActionsTool,
    CallbackTool,
    ClockTool,
    DevTool,
    InspectorTool,
    PerformanceTool,
)
from fire_engine.devtools.types import Button, Section

# ---------------------------------------------------------------------------
# __all__ re-export check
# ---------------------------------------------------------------------------


def test_dunder_all_contains_expected_symbols():
    import fire_engine.devtools.tools as mod

    assert hasattr(mod, "__all__")
    for name in (
        "ActionsTool",
        "CallbackTool",
        "ClockTool",
        "DevTool",
        "InspectorTool",
        "PerformanceTool",
    ):
        assert name in mod.__all__, f"{name!r} missing from __all__"


# ---------------------------------------------------------------------------
# DevTool base class
# ---------------------------------------------------------------------------


def test_dev_tool_base_revision_is_zero():
    assert DevTool().revision == 0


def test_dev_tool_base_build_raises():
    with pytest.raises(NotImplementedError):
        DevTool().build()


# ---------------------------------------------------------------------------
# PerformanceTool
# ---------------------------------------------------------------------------


def test_performance_tool_reads_providers_live():
    counter = {"n": 0}

    def tick():
        counter["n"] += 1
        return counter["n"]

    tool = PerformanceTool({"ticks": tick})
    p = tool.build()
    f = p.sections[0].fields[0]
    assert f.label == "ticks"
    assert f.read_only is True
    assert f.get() == 1
    assert f.get() == 2


def test_performance_tool_panel_ids():
    tool = PerformanceTool({"x": lambda: 1})
    p = tool.build()
    assert p.tool_id == "performance"
    assert p.title == "Performance"


# ---------------------------------------------------------------------------
# ActionsTool
# ---------------------------------------------------------------------------


def test_actions_tool_initial_build():
    fired = []
    tool = ActionsTool("World", {"A": lambda: fired.append("A")})
    p = tool.build()
    assert p.tool_id == "actions"
    assert len(p.buttons) == 1
    p.buttons[0].on_click()
    assert fired == ["A"]


def test_actions_tool_add_bumps_revision():
    tool = ActionsTool()
    r0 = tool.revision
    tool.add_action("X", lambda: None)
    assert tool.revision == r0 + 1


def test_actions_tool_add_appends_button():
    tool = ActionsTool("T", {"A": lambda: None})
    tool.add_action("B", lambda: None)
    p = tool.build()
    assert [b.label for b in p.buttons] == ["A", "B"]


# ---------------------------------------------------------------------------
# InspectorTool
# ---------------------------------------------------------------------------


def test_inspector_tool_shows_placeholder_when_nothing_selected():
    sel = Selection()
    tool = InspectorTool(sel)
    p = tool.build()
    assert "nothing selected" in p.sections[0].fields[0].label


def test_inspector_tool_revision_tracks_selection():
    sel = Selection()
    tool = InspectorTool(sel)
    assert tool.revision == sel.revision


# ---------------------------------------------------------------------------
# CallbackTool
# ---------------------------------------------------------------------------


def test_callback_tool_delegates_build():
    sec = Section("S", [])
    btn = Button("B", lambda: None)
    tool = CallbackTool("env", "Environment", lambda: ([sec], [btn]))
    p = tool.build()
    assert p.tool_id == "env"
    assert p.title == "Environment"
    assert p.sections == [sec]
    assert p.buttons == [btn]


def test_callback_tool_revision_fn():
    rev = {"n": 0}
    tool = CallbackTool("t", "T", lambda: ([], []), revision_fn=lambda: rev["n"])
    assert tool.revision == 0
    rev["n"] = 5
    assert tool.revision == 5


def test_callback_tool_default_revision_is_zero():
    tool = CallbackTool("t", "T", lambda: ([], []))
    assert tool.revision == 0


# ---------------------------------------------------------------------------
# ClockTool
# ---------------------------------------------------------------------------


def test_clock_tool_formats_time():
    class _FakeClock:
        game_day = 7
        game_time_of_day = 8 * 3600 + 30 * 60  # 08:30

    p = ClockTool(_FakeClock()).build()
    rows = {f.label: f.get() for f in p.sections[0].fields}
    assert rows["day"] == 7
    assert rows["time of day"] == "08:30"
