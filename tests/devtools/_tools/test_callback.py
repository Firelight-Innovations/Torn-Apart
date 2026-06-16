"""
tests/devtools/_tools/test_callback.py — tests for fire_engine/devtools/_tools/callback.py.

Covers CallbackTool: construction, build delegation, revision_fn, and defaults.
Fully headless; no panda3d imports.
"""

from __future__ import annotations

from fire_engine.devtools._tools.callback import CallbackTool
from fire_engine.devtools.types import Button, Panel, Section

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_tool_id_and_title_stored():
    tool = CallbackTool("env", "Environment", lambda: ([], []))
    assert tool.tool_id == "env"
    assert tool.title == "Environment"


def test_default_revision_is_zero():
    tool = CallbackTool("t", "T", lambda: ([], []))
    assert tool.revision == 0


def test_revision_fn_none_defaults_to_zero():
    tool = CallbackTool("t", "T", lambda: ([], []), revision_fn=None)
    assert tool.revision == 0


# ---------------------------------------------------------------------------
# revision_fn
# ---------------------------------------------------------------------------


def test_revision_fn_called_live():
    rev = {"n": 0}
    tool = CallbackTool("t", "T", lambda: ([], []), revision_fn=lambda: rev["n"])
    assert tool.revision == 0
    rev["n"] = 7
    assert tool.revision == 7


# ---------------------------------------------------------------------------
# build() delegation
# ---------------------------------------------------------------------------


def test_build_calls_build_fn():
    sec = Section("S", [])
    btn = Button("B", lambda: None)
    tool = CallbackTool("env", "Env", lambda: ([sec], [btn]))
    p = tool.build()
    assert isinstance(p, Panel)
    assert p.tool_id == "env"
    assert p.title == "Env"
    assert p.sections == [sec]
    assert p.buttons == [btn]


def test_build_fn_called_each_time():
    counter = {"n": 0}

    def build_fn():
        counter["n"] += 1
        return ([], [])

    tool = CallbackTool("t", "T", build_fn)
    tool.build()
    tool.build()
    assert counter["n"] == 2


def test_build_returns_empty_sections_and_buttons():
    tool = CallbackTool("t", "T", lambda: ([], []))
    p = tool.build()
    assert p.sections == []
    assert p.buttons == []


def test_build_includes_revision_in_panel():
    rev = {"n": 0}
    tool = CallbackTool("t", "T", lambda: ([], []), revision_fn=lambda: rev["n"])
    rev["n"] = 3
    p = tool.build()
    assert p.revision == 3


def test_build_fn_dynamic_sections():
    state = {"sections": []}
    sec_a = Section("A", [])

    def build_fn():
        return (state["sections"], [])

    tool = CallbackTool("t", "T", build_fn)
    assert tool.build().sections == []
    state["sections"] = [sec_a]
    assert tool.build().sections == [sec_a]
