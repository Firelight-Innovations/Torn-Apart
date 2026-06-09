"""
world/camera.py — Panda3D camera binding for the Torn Apart engine.

Binds a math3d Transform to the Panda3D camera NodePath so that wherever the
Transform moves/rotates, the Panda3D camera follows.  This is a ONE-WAY sync:
the Transform is always the authority; the NodePath is driven from it.

The FlyController (player/fly_controller.py) modifies the Transform; this
module mirrors the final state to ``base.camera`` each frame.

Panda3D type conversions (math3d ↔ Panda3D) happen ONLY in this file and
world/app.py — nowhere else.

Coordinate mapping
------------------
    math3d (Z-up, forward=+Y, right=+X, up=+Z)
    Panda3D (also Z-up, Y-forward) — no axis swap needed.

    Position: Vec3(x,y,z) → LPoint3f(x,y,z)
    Rotation: Quat(w,x,y,z) → LQuaternionf(w,x,y,z)   (Panda3D is also scalar-first)

Example
-------
    # In App.setup():
    camera_go = instantiate(name="Camera")
    cam = camera_go.add_component(CameraComponent)
    # Each frame, App syncs:
    #   cam.sync_to_panda()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Panda3D imports are allowed in world/ per ARCHITECTURE §3
from panda3d.core import LPoint3f, LQuaternionf  # type: ignore[import]

from torn_apart.world.component import Component

if TYPE_CHECKING:
    pass


class CameraComponent(Component):
    """
    Component that syncs its owning Transform to the Panda3D camera NodePath.

    Attach to the camera GameObject.  App calls sync_to_panda() each frame
    after the FlyController has updated the transform.

    Parameters
    ----------
    base : ShowBase — Panda3D application instance (passed at add_component time).

    Attributes
    ----------
    base : ShowBase — the Panda3D application; used to access base.camera.

    Example
    -------
        # In App.setup():
        self.camera_go = instantiate(name="MainCamera")
        self.camera_go.add_component(CameraComponent, base=self)
    """

    __slots__ = ("base",)

    def __init__(self, base=None) -> None:
        super().__init__()
        self.base = base  # Panda3D ShowBase instance

    def awake(self) -> None:
        """Disable Panda3D's default camera node so we drive it manually."""
        if self.base is not None:
            # Detach camera from default camera group to prevent conflicts
            pass  # Panda3D camera is already directly accessible via base.camera

    def sync_to_panda(self) -> None:
        """
        Copy the owning Transform's world position/rotation to the Panda3D camera.

        Call this once per frame from App, after all components have been updated.
        The conversion is Z-up → Z-up (no swap needed) and Quat scalar-first
        is the same convention in both math3d and Panda3D.

        Note
        ----
        This is the ONLY place math3d types are converted to Panda3D types in
        the camera pipeline.
        """
        if self.base is None or self.transform is None:
            return

        pos = self.transform.position
        rot = self.transform.rotation

        p3d_pos = LPoint3f(pos.x, pos.y, pos.z)
        # Panda3D LQuaternionf is (w, x, y, z) — same as our Quat
        p3d_rot = LQuaternionf(rot.w, rot.x, rot.y, rot.z)

        self.base.camera.set_pos(p3d_pos)
        self.base.camera.set_quat(p3d_rot)
