"""
render/_impl/app_input.py — Input-collection helpers for App.

Extracted from render/app.py to keep that module under 500 lines (C0302).
Functions take the App instance as their first argument (``self_obj``) and are
called from the class as ``_func(self, ...)``.  Not part of the public API.

Docs: docs/systems/render.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from panda3d.core import WindowProperties

if TYPE_CHECKING:
    from fire_engine.render.app import App


def collect_input(self_obj: App) -> None:
    """
    Read current Panda3D input state and write to self_obj.input_state.

    Mouse delta is read from the pointer and recentred.  This is the
    ONLY place panda3d input is read; everything else uses InputState.
    """
    ks = self_obj._key_state
    inp = self_obj.input_state

    inp.move_forward = bool(ks.get("w", False))
    inp.move_backward = bool(ks.get("s", False))
    inp.move_left = bool(ks.get("a", False))
    inp.move_right = bool(ks.get("d", False))
    inp.move_up = bool(ks.get("space", False) or ks.get("e", False))
    inp.move_down = bool(
        ks.get("lcontrol", False) or ks.get("rcontrol", False) or ks.get("q", False)
    )
    inp.sprint = bool(ks.get("lshift", False) or ks.get("rshift", False))

    # Toggle mouse capture
    if inp.escape_pressed:
        inp.mouse_captured = not inp.mouse_captured
        set_mouse_capture(self_obj, inp.mouse_captured)
        inp.escape_pressed = False

    # Mouse delta (only when captured).
    #
    # We read the RAW pixel pointer position (``win.get_pointer``) relative
    # to the window centre, then recentre the pointer every frame.  Reading
    # raw pixels (not the normalised mouseWatcher value) keeps BOTH axes
    # symmetric and avoids the edge-clamping that froze one axis under the
    # old confined-cursor + normalised-delta path.  The cursor is in
    # relative mode (see set_mouse_capture) so the OS never clamps it at a
    # screen edge.  Note: get_pointer Y is pixels-from-TOP (Y-down).
    inp.mouse_dx = 0.0
    inp.mouse_dy = 0.0
    if inp.mouse_captured:
        win_w = self_obj.win.get_x_size()
        win_h = self_obj.win.get_y_size()
        cx = win_w // 2
        cy = win_h // 2
        ptr = self_obj.win.get_pointer(0)
        if ptr.get_in_window() and not self_obj._skip_mouse_delta:
            inp.mouse_dx = float(ptr.get_x() - cx)
            inp.mouse_dy = float(ptr.get_y() - cy)
        # Always recentre so the next frame's delta is measured from centre.
        self_obj.win.move_pointer(0, cx, cy)
        self_obj._skip_mouse_delta = False


def set_mouse_capture(self_obj: App, captured: bool) -> None:
    """
    Lock/unlock the cursor for free-look.

    Captured → cursor hidden + **relative** mouse mode (the OS stops
    clamping the pointer at screen edges, so look never freezes on an axis).
    Released → cursor shown + absolute mode (normal desktop pointer).

    Re-enabling capture arms ``_skip_mouse_delta`` so the first post-capture
    frame doesn't snap the view by the pre-capture pointer offset.
    """
    props = WindowProperties()
    props.set_cursor_hidden(captured)
    props.set_mouse_mode(WindowProperties.M_relative if captured else WindowProperties.M_absolute)
    self_obj.win.request_properties(props)
    if captured:
        self_obj._skip_mouse_delta = True


def window_event(self_obj: App, win: Any) -> None:
    """
    Handle Panda3D window events (focus, resize, close).

    Extends ShowBase's default handling to fix a mouse-capture desync: when
    the window loses OS focus (alt-tab), the platform releases our hidden /
    relative-mouse cursor properties.  Panda3D does not re-apply them on
    focus regain, so the engine would think the mouse is captured while the
    OS shows a free, absolute-mode cursor — free-look stays dead until the
    next ESC toggle.  Here we detect the focus-regain edge and reassert
    whatever capture state we want, re-arming the first-frame delta skip so
    the view doesn't snap.

    Parameters
    ----------
    self_obj : App — the App instance
    win : panda3d.core.GraphicsWindow
        The window the event is about (ignored unless it is ``self_obj.win``).
    """
    if win is not self_obj.win:
        return
    has_focus = bool(win.get_properties().get_foreground())
    if has_focus and not self_obj._had_focus:
        # Regained focus — reapply the capture state the engine believes in.
        set_mouse_capture(self_obj, self_obj.input_state.mouse_captured)
    self_obj._had_focus = has_focus


def push_input_to_controllers(self_obj: App) -> None:
    """
    Forward the current InputState to all FlyController components.

    FlyController.set_input_state(inp) is called here — the controller
    stays panda3d-free and reads the state on its next update().
    """
    # Import lazily to avoid circular imports
    try:
        from fire_engine.simulation.player.fly_controller import FlyController
    except ImportError:
        return

    from fire_engine.render.registry import _STATE

    bucket = _STATE.buckets.get(FlyController, [])
    for ctrl in bucket:
        if isinstance(ctrl, FlyController):
            ctrl.set_input_state(self_obj.input_state)
