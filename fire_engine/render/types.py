"""
render/types.py — Shared plain-data types for the render package.

Grouping module (exempt from one-class rule per pyproject.toml [tool.firelight]).

Docs: docs/systems/render.md
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InputState:
    """
    Snapshot of the current input state passed to FlyController each frame.

    App populates this from Panda3D's key/mouse state before calling
    registry.run_frame.  FlyController reads it without importing panda3d.

    Attributes
    ----------
    move_forward  : bool — W key held
    move_backward : bool — S key held
    move_left     : bool — A key held
    move_right    : bool — D key held
    move_up       : bool — Space key held (or E)
    move_down     : bool — Ctrl key held (or Q)
    sprint        : bool — Shift held (5× speed multiplier)
    mouse_dx      : float — raw mouse delta X since last frame (pixels)
    mouse_dy      : float — raw mouse delta Y since last frame (pixels)
    mouse_captured: bool  — True when the cursor is locked to the window
    escape_pressed: bool  — True on the frame ESC was pressed (toggle mouse capture)

    Docs: docs/systems/render.md
    """

    move_forward: bool = False
    move_backward: bool = False
    move_left: bool = False
    move_right: bool = False
    move_up: bool = False
    move_down: bool = False
    sprint: bool = False
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0
    mouse_captured: bool = False
    escape_pressed: bool = False
