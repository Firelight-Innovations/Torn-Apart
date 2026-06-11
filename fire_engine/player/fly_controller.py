"""
player/fly_controller.py — Free-fly camera controller (Unity Component, pure Python).

FlyController drives a Transform using keyboard + mouse input.  It is
intentionally panda3d-free — input arrives via set_input_state(InputState)
called by App before each update().  The controller never imports panda3d.

Controls
--------
    W / S               Move forward / backward along transform.forward (horizontal)
    A / D               Strafe left / right along transform.right
    Space / E           Move up  (+Z)
    Ctrl / Q            Move down (−Z)
    Shift               5× speed multiplier (sprint)
    Mouse move          Look around (yaw + pitch)
    ESC                 Toggle mouse capture (handled by App; controller reads the
                        captured flag from InputState)

Mouse-look quaternion composition
-----------------------------------
Yaw and pitch are accumulated as floats (radians), NOT integrated into the
quaternion on every delta — this avoids the roll-drift trap described in
DEVELOPMENT_PLAN.md Known Traps.

Each frame:
    rotation = Quat.from_axis_angle(Vec3.UP, yaw) * Quat.from_axis_angle(Vec3.RIGHT, pitch)

Pitch is clamped to ±(π/2 − ε) to prevent gimbal flip.
Yaw wraps freely.

The result is assigned to transform.local_rotation each frame (replacing,
not accumulating).

Speed parameters
----------------
    move_speed  : float — meters per second (default 10.0 m/s)
    sprint_mult : float — multiplier when Shift is held (default 5.0)
    mouse_sensitivity : float — radians per pixel (default 0.003 rad/px)

Units
-----
    Positions: meters
    Speeds: meters per second
    Angles: radians

Example
-------
    from fire_engine.world.gameobject      import GameObject
    from fire_engine.player.fly_controller import FlyController

    camera_go = GameObject(name="Camera")
    ctrl = camera_go.add_component(FlyController, move_speed=15.0)

    # App calls this each frame before registry.run_frame:
    ctrl.set_input_state(input_state)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fire_engine.world.component  import Component
from fire_engine.core.math3d      import Vec3, Quat

if TYPE_CHECKING:
    from fire_engine.world.app import InputState

# Maximum pitch in radians before clamping (just under 90°)
_PITCH_LIMIT: float = math.radians(89.0)


class FlyController(Component):
    """
    Free-fly camera controller.

    Attach to the camera GameObject.  App.push_input_to_controllers() calls
    set_input_state each frame; update() consumes it.

    Parameters
    ----------
    move_speed        : float — movement speed in meters per second (default 10.0).
    sprint_mult       : float — multiplier applied when Shift is held (default 5.0).
    mouse_sensitivity : float — radians per pixel (default 0.003 rad/px).

    Attributes
    ----------
    yaw   : float — accumulated heading angle in radians (world Z rotation).
    pitch : float — accumulated pitch angle in radians (local X rotation),
                    clamped to ±89°.
    """

    __slots__ = (
        "move_speed",
        "sprint_mult",
        "mouse_sensitivity",
        "yaw",
        "pitch",
        "_input",
    )

    def __init__(
        self,
        move_speed:        float = 10.0,
        sprint_mult:       float = 5.0,
        mouse_sensitivity: float = 0.003,
    ) -> None:
        super().__init__()
        self.move_speed:        float = float(move_speed)
        self.sprint_mult:       float = float(sprint_mult)
        self.mouse_sensitivity: float = float(mouse_sensitivity)

        self.yaw:   float = 0.0  # radians, world Z
        self.pitch: float = 0.0  # radians, local X

        self._input: "InputState | None" = None

    # ------------------------------------------------------------------
    # Input setter (called by App before update)
    # ------------------------------------------------------------------

    def set_input_state(self, state: "InputState") -> None:
        """
        Provide the current frame's input state.

        Called by App._push_input_to_controllers() before registry.run_frame().

        Parameters
        ----------
        state : InputState — snapshot of keyboard/mouse state for this frame.
        """
        self._input = state

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def awake(self) -> None:
        """Read initial rotation from the transform (yaw/pitch from world rotation)."""
        if self.transform is not None:
            h, p, _r = self.transform.local_rotation.as_euler()
            self.yaw   = h
            self.pitch = p

    def update(self, dt: float) -> None:
        """
        Process one frame of input: update yaw/pitch, then move.

        Parameters
        ----------
        dt : float — real frame delta in seconds.
        """
        if self._input is None or self.transform is None:
            return

        inp = self._input
        speed = self.move_speed * (self.sprint_mult if inp.sprint else 1.0)

        # -- Mouse-look --------------------------------------------------
        if inp.mouse_captured:
            self.yaw   -= inp.mouse_dx * self.mouse_sensitivity
            self.pitch -= inp.mouse_dy * self.mouse_sensitivity
            self.pitch  = max(-_PITCH_LIMIT, min(_PITCH_LIMIT, self.pitch))

        # Compose rotation: yaw about world Z, then pitch about local X.
        # This matches DEVELOPMENT_PLAN.md §Known Traps and avoids roll drift.
        q_yaw   = Quat.from_axis_angle(Vec3.UP,   self.yaw)
        q_pitch = Quat.from_axis_angle(Vec3.RIGHT, self.pitch)
        self.transform.local_rotation = (q_yaw * q_pitch).normalized()

        # -- Movement ----------------------------------------------------
        # Horizontal: move along transform.forward / transform.right
        # (ignore the Z component so WASD doesn't fly you up/down)
        move = Vec3(0.0, 0.0, 0.0)

        fwd_horiz = self._horizontal(self.transform.forward)
        right_vec = self._horizontal(self.transform.right)

        if inp.move_forward:
            move = move + fwd_horiz
        if inp.move_backward:
            move = move - fwd_horiz
        if inp.move_right:
            move = move + right_vec
        if inp.move_left:
            move = move - right_vec

        # Vertical: absolute world Z
        if inp.move_up:
            move = move + Vec3.UP
        if inp.move_down:
            move = move - Vec3.UP

        if move.length_squared > 1e-10:
            move = move.normalized() * (speed * dt)
            self.transform.local_position = self.transform.local_position + move

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _horizontal(v: Vec3) -> Vec3:
        """Project a vector onto the XY plane (Z-up: kill the Z component)."""
        h = Vec3(v.x, v.y, 0.0)
        ls = h.length_squared
        if ls < 1e-10:
            return Vec3.FORWARD  # degenerate: return a safe default
        return h * (1.0 / math.sqrt(ls))
