"""
tests/devtools/_tools/test_base.py — tests for fire_engine/devtools/_tools/base.py.

Covers DevTool: tool_id/title class attributes, default revision, and that build()
raises NotImplementedError. Also verifies subclass overrides work correctly.
Fully headless; no panda3d imports.
"""

from __future__ import annotations

import pytest

from fire_engine.devtools._tools.base import DevTool
from fire_engine.devtools.types import Panel, Section

# ---------------------------------------------------------------------------
# Default class attributes
# ---------------------------------------------------------------------------


def test_default_tool_id():
    assert DevTool.tool_id == "tool"


def test_default_title():
    assert DevTool.title == "Tool"


def test_default_revision_is_zero():
    tool = DevTool()
    assert tool.revision == 0


# ---------------------------------------------------------------------------
# build() raises NotImplementedError
# ---------------------------------------------------------------------------


def test_build_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="DevTool subclasses must implement build"):
        DevTool().build()


# ---------------------------------------------------------------------------
# Subclass overrides
# ---------------------------------------------------------------------------


class _ConcreteFixedTool(DevTool):
    tool_id = "concrete"
    title = "Concrete"

    def build(self) -> Panel:
        return Panel(self.tool_id, self.title, [Section("S", [])])


def test_subclass_can_override_ids():
    t = _ConcreteFixedTool()
    assert t.tool_id == "concrete"
    assert t.title == "Concrete"


def test_subclass_build_returns_panel():
    p = _ConcreteFixedTool().build()
    assert isinstance(p, Panel)
    assert p.tool_id == "concrete"
    assert p.title == "Concrete"


def test_subclass_revision_still_zero_unless_overridden():
    assert _ConcreteFixedTool().revision == 0


class _DynamicRevisionTool(DevTool):
    tool_id = "dyn"
    title = "Dyn"

    def __init__(self) -> None:
        self._rev = 0

    @property
    def revision(self) -> int:
        return self._rev

    def build(self) -> Panel:
        return Panel(self.tool_id, self.title, [], revision=self._rev)


def test_subclass_dynamic_revision():
    t = _DynamicRevisionTool()
    assert t.revision == 0
    t._rev = 3
    assert t.revision == 3


# ---------------------------------------------------------------------------
# DevTool is a base class (not abstract — instantiable but build raises)
# ---------------------------------------------------------------------------


def test_dev_tool_instantiable():
    t = DevTool()
    assert t is not None


def test_revision_is_property():
    t = DevTool()
    # revision is a @property — it has no setter on the base class
    assert isinstance(type(t).revision, property)
