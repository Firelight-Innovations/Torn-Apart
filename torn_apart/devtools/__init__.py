"""
torn_apart.devtools — headless engine for the in-game developer overlay.

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
:class:`~torn_apart.devtools.tools.DevTool`, implement ``build()``, and
``manager.register_tool(MyTool(...))``.  See ``docs/systems/devtools.md``.

Everything here is panda3d-free and unit-tested in ``tests/test_devtools.py``.

Quick-start
-----------
    from torn_apart.devtools import (
        DevToolsManager, PerformanceTool, InspectorTool, ActionsTool,
    )

    mgr = DevToolsManager()
    mgr.register_tool(PerformanceTool({"FPS": lambda: 60.0}))
    mgr.register_tool(InspectorTool(mgr.selection))
    panels = mgr.panels()        # hand these to the renderer
"""

from torn_apart.devtools.fields import (
    Field,
    FieldKind,
    Section,
    Button,
    Panel,
)
from torn_apart.devtools.selection import Selection
from torn_apart.devtools.picking import Selectable, ray_aabb, pick
from torn_apart.devtools.introspect import describe_object, describe_chunk, is_chunk
from torn_apart.devtools.tools import (
    DevTool,
    PerformanceTool,
    InspectorTool,
    ActionsTool,
    ClockTool,
    CallbackTool,
)
from torn_apart.devtools.manager import DevToolsManager
from torn_apart.devtools.gizmo import (
    GizmoMode,
    HandleType,
    Handle,
    DragState,
    Gizmo,
    update_drag,
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
