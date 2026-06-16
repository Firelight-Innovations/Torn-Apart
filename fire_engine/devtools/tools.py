"""
devtools/tools.py — the dev-tool plugins that populate the debug overlay.

A :class:`DevTool` is a self-contained panel: it knows how to :meth:`~DevTool.build`
its current :class:`~fire_engine.devtools.fields.Panel`, and exposes a ``revision``
the renderer watches to know when the panel's *structure* changed (so it rebuilds
widgets only then).  New tools are added by subclassing :class:`DevTool` and
registering an instance with :class:`~fire_engine.devtools.manager.DevToolsManager`
— this is the extension point the whole system is built around.

Built-in tools
--------------
- :class:`PerformanceTool` — live engine stats from injected provider callables.
- :class:`InspectorTool`   — reflected, editable view of the selected GameObject.
- :class:`ActionsTool`     — a grid of one-shot action buttons (spawn, fire event…).
- :class:`ClockTool`       — read-out of the game calendar (seed for a future
                             day/night editor).

None of this imports panda3d (hard rule 1).  Panda3D-specific values (FPS, draw
counts) arrive as plain callables supplied by the renderer in ``world/``.

Class definitions live in :mod:`fire_engine.devtools._tools`; this module
re-exports them to preserve every historical import path.

Docs: docs/systems/devtools.md
"""

from __future__ import annotations

from fire_engine.devtools._tools.actions import ActionsTool
from fire_engine.devtools._tools.base import DevTool
from fire_engine.devtools._tools.callback import CallbackTool
from fire_engine.devtools._tools.clock import ClockTool
from fire_engine.devtools._tools.inspector import InspectorTool
from fire_engine.devtools._tools.performance import PerformanceTool

__all__ = [
    "ActionsTool",
    "CallbackTool",
    "ClockTool",
    "DevTool",
    "InspectorTool",
    "PerformanceTool",
]
