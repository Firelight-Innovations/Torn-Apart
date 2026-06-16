"""
tests/devtools/_tools/test_performance.py — tests for fire_engine/devtools/_tools/performance.py.

Covers PerformanceTool: construction, live provider reads, panel structure,
read-only field assertion, and ordering. Fully headless; no panda3d imports.
"""

from __future__ import annotations

from fire_engine.devtools._tools.performance import PerformanceTool
from fire_engine.devtools.enums import FieldKind
from fire_engine.devtools.types import Panel

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_tool_id_and_title():
    tool = PerformanceTool({})
    assert tool.tool_id == "performance"
    assert tool.title == "Performance"


def test_revision_is_zero():
    tool = PerformanceTool({"fps": lambda: 60})
    assert tool.revision == 0


# ---------------------------------------------------------------------------
# build() panel structure
# ---------------------------------------------------------------------------


def test_build_returns_panel():
    tool = PerformanceTool({"fps": lambda: 60})
    assert isinstance(tool.build(), Panel)


def test_build_single_stats_section():
    tool = PerformanceTool({"fps": lambda: 60})
    p = tool.build()
    assert len(p.sections) == 1
    assert p.sections[0].title == "Stats"


def test_build_empty_providers():
    tool = PerformanceTool({})
    p = tool.build()
    assert p.sections[0].fields == []


def test_fields_are_label_kind():
    tool = PerformanceTool({"fps": lambda: 60, "ms": lambda: 16.7})
    p = tool.build()
    for f in p.sections[0].fields:
        assert f.kind == FieldKind.LABEL


def test_fields_are_read_only():
    tool = PerformanceTool({"fps": lambda: 60})
    p = tool.build()
    for f in p.sections[0].fields:
        assert f.read_only is True
        assert f.set is None


# ---------------------------------------------------------------------------
# Live provider reads
# ---------------------------------------------------------------------------


def test_provider_called_live_each_get():
    counter = {"n": 0}

    def tick():
        counter["n"] += 1
        return counter["n"]

    tool = PerformanceTool({"ticks": tick})
    p = tool.build()
    f = p.sections[0].fields[0]
    assert f.get() == 1
    assert f.get() == 2
    assert f.get() == 3


def test_multiple_providers_all_present():
    tool = PerformanceTool({"fps": lambda: 60, "chunks": lambda: 100, "objs": lambda: 5})
    p = tool.build()
    labels = [f.label for f in p.sections[0].fields]
    assert "fps" in labels
    assert "chunks" in labels
    assert "objs" in labels


def test_provider_insertion_order_preserved():
    providers = {"A": lambda: 1, "B": lambda: 2, "C": lambda: 3}
    tool = PerformanceTool(providers)
    p = tool.build()
    labels = [f.label for f in p.sections[0].fields]
    assert labels == ["A", "B", "C"]


def test_provider_returns_non_numeric():
    tool = PerformanceTool({"status": lambda: "OK"})
    p = tool.build()
    assert p.sections[0].fields[0].get() == "OK"
