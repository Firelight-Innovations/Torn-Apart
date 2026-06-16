"""
tests/devtools/_tools/test_inspector.py — tests for fire_engine/devtools/_tools/inspector.py.

Covers InspectorTool: empty-selection placeholder, revision tracking, object routing
(GameObject vs chunk duck-type), and panel titles. Fully headless; no panda3d imports.
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.devtools._tools.inspector import InspectorTool
from fire_engine.devtools.selection import Selection
from fire_engine.render.gameobject import GameObject
from fire_engine.render.registry import ComponentRegistry


@pytest.fixture(autouse=True)
def _clean_registry():
    ComponentRegistry.clear()
    yield
    ComponentRegistry.clear()


# ---------------------------------------------------------------------------
# Minimal chunk duck-type
# ---------------------------------------------------------------------------


class _FakeChunk:
    def __init__(self, coord=(0, 0, 0)):
        self.coord = coord
        self.materials = np.zeros((32, 32, 32), dtype=np.uint8)
        self.chunk_meters = 16.0
        self.world_origin = type("V", (), {"x": 0.0, "y": 0.0, "z": 0.0})()
        self.dirty = False
        self.edited = False


# ---------------------------------------------------------------------------
# tool_id / title
# ---------------------------------------------------------------------------


def test_tool_id_and_title():
    tool = InspectorTool(Selection())
    assert tool.tool_id == "inspector"
    assert tool.title == "Inspector"


# ---------------------------------------------------------------------------
# Revision tracks selection
# ---------------------------------------------------------------------------


def test_revision_matches_selection_revision():
    sel = Selection()
    tool = InspectorTool(sel)
    assert tool.revision == sel.revision
    sel.set(object())  # type: ignore[arg-type]
    assert tool.revision == sel.revision


# ---------------------------------------------------------------------------
# Empty-selection placeholder
# ---------------------------------------------------------------------------


def test_placeholder_when_nothing_selected():
    sel = Selection()
    tool = InspectorTool(sel)
    p = tool.build()
    assert "nothing selected" in p.sections[0].fields[0].label


def test_placeholder_panel_id():
    sel = Selection()
    tool = InspectorTool(sel)
    p = tool.build()
    assert p.tool_id == "inspector"


# ---------------------------------------------------------------------------
# GameObject routing
# ---------------------------------------------------------------------------


def test_builds_gameobject_panel():
    sel = Selection()
    go = GameObject(name="Hero")
    sel.set(go)
    tool = InspectorTool(sel)
    p = tool.build()
    assert "Hero" in p.title


def test_panel_revision_in_sync():
    sel = Selection()
    go = GameObject(name="X")
    sel.set(go)
    tool = InspectorTool(sel)
    p = tool.build()
    assert p.revision == sel.revision


# ---------------------------------------------------------------------------
# Chunk routing
# ---------------------------------------------------------------------------


def test_routes_chunk_to_chunk_describer():
    sel = Selection()
    chunk = _FakeChunk(coord=(1, 2, 3))
    sel.set(chunk)  # type: ignore[arg-type]
    tool = InspectorTool(sel)
    p = tool.build()
    assert "Chunk" in p.title
    assert "(1, 2, 3)" in p.title


def test_chunk_panel_has_chunk_and_voxels_sections():
    sel = Selection()
    chunk = _FakeChunk(coord=(0, 0, 0))
    sel.set(chunk)  # type: ignore[arg-type]
    tool = InspectorTool(sel)
    p = tool.build()
    titles = [s.title for s in p.sections]
    assert "Chunk" in titles
    assert "Voxels" in titles


# ---------------------------------------------------------------------------
# Selection change causes revision bump (rendering cue)
# ---------------------------------------------------------------------------


def test_revision_increments_on_selection_change():
    sel = Selection()
    tool = InspectorTool(sel)
    r0 = tool.revision

    go1 = GameObject(name="A")
    sel.set(go1)
    r1 = tool.revision
    assert r1 > r0

    go2 = GameObject(name="B")
    sel.set(go2)
    r2 = tool.revision
    assert r2 > r1
