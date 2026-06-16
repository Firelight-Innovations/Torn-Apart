"""
fire_engine.devtools — headless engine for the in-game developer overlay.

This package is the brain of the dev/debug tool the owner flies the noclip
camera in: it decides *what* the overlay shows and *how* edits apply, but draws
nothing itself.  The Panda3D renderer lives in ``world/devtools_overlay.py``
(the only place allowed to import panda3d — CLAUDE.md hard rule 1); it consumes
the plain-data :class:`Panel` model produced here and turns mouse events into
the world-space rays :class:`DevToolsManager` picks against.

Pieces
------
- :mod:`fields`    — the declarative Panel/Section/Field/Button model.
- :mod:`selection` — current-selection state with a change counter.
- :mod:`picking`   — CPU ray/AABB object picking for click-to-select.
- :mod:`introspect`— reflect a GameObject into editable inspector sections.
- :mod:`tools`     — DevTool plugins (Performance, Inspector, Actions, Clock).
- :mod:`manager`   — the hub the renderer drives.

Adding a new tool is the whole design goal: subclass
:class:`~fire_engine.devtools.tools.DevTool`, implement ``build()``, and
``manager.register_tool(MyTool(...))``.  See ``docs/systems/devtools.md``.

Everything here is panda3d-free and unit-tested in ``tests/test_devtools.py``.

Quick-start
-----------
    from fire_engine.devtools import (
        DevToolsManager, PerformanceTool, InspectorTool, ActionsTool,
    )

    mgr = DevToolsManager()
    mgr.register_tool(PerformanceTool({"FPS": lambda: 60.0}))
    mgr.register_tool(InspectorTool(mgr.selection))
    panels = mgr.panels()        # hand these to the renderer
"""

from fire_engine.devtools.fields import (
    Button,
    Field,
    FieldKind,
    Panel,
    Section,
)
from fire_engine.devtools.gizmo import (
    DragState,
    Gizmo,
    GizmoMode,
    Handle,
    HandleType,
    update_drag,
)
from fire_engine.devtools.introspect import describe_chunk, describe_object, is_chunk
from fire_engine.devtools.manager import DevToolsManager
from fire_engine.devtools.picking import Selectable, pick, ray_aabb
from fire_engine.devtools.selection import Selection
from fire_engine.devtools.tools import (
    ActionsTool,
    CallbackTool,
    ClockTool,
    DevTool,
    InspectorTool,
    PerformanceTool,
)

__all__ = [
    # model
    "Field",
    "FieldKind",
    "Section",
    "Button",
    "Panel",
    # state
    "Selection",
    # picking
    "Selectable",
    "ray_aabb",
    "pick",
    # introspection
    "describe_object",
    "describe_chunk",
    "is_chunk",
    # tools
    "DevTool",
    "PerformanceTool",
    "InspectorTool",
    "ActionsTool",
    "ClockTool",
    "CallbackTool",
    # hub
    "DevToolsManager",
    # transform gizmo (headless math)
    "GizmoMode",
    "HandleType",
    "Handle",
    "DragState",
    "Gizmo",
    "update_drag",
]
