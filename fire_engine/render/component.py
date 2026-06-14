"""
world/component.py — Base class for all Torn Apart components (Unity API clone).

A Component is the building block of the Torn Apart object model.  It attaches
to a GameObject and implements the Unity lifecycle:

    awake()  →  on_enable()  →  start()  →  update(dt) / late_update(dt) /
    fixed_update(dt)  →  on_disable()  →  on_destroy()

Lifecycle guarantee (enforced by ComponentRegistry.run_frame):
    1. awake()       — called immediately after construction, before any
                       start() in the same frame.  Runs even when the component
                       is disabled.
    2. on_enable()   — called after awake() when the component is enabled.
    3. start()       — called once before the first update(), after ALL awake()
                       and on_enable() calls for that frame have completed.
    4. update(dt)    — called every frame while enabled and active_in_hierarchy.
    5. late_update(dt) — called after ALL update() calls that frame (camera
                        follow, IK post-processing, etc.).
    6. fixed_update(dt) — called at a fixed timestep (default 50 Hz) driven
                          by Clock.fixed_steps(); may be called 0–5 times per
                          real frame.
    7. on_disable()  — called when enabled → False, or when the GameObject is
                       deactivated.
    8. on_destroy()  — called at end-of-frame when the component (or its
                       owning GameObject) is destroyed.

No panda3d imports — this module is fully headless-testable.

Example
-------
    from fire_engine.render.component import Component

    class SpinComponent(Component):
        '''Rotate the owning object continuously.'''
        def __init__(self, speed: float = 1.0) -> None:
            super().__init__()
            self.speed = speed   # radians per second

        def update(self, dt: float) -> None:
            from fire_engine.core.math3d import Vec3, Quat
            from math import pi
            self.transform.rotate(
                Quat.from_axis_angle(Vec3.UP, self.speed * dt)
            )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fire_engine.render.gameobject import GameObject
    from fire_engine.render.transform import Transform


class Component:
    """
    Base class for all Torn Apart components (Unity API clone, snake_case).

    Subclass this to add behaviour to a GameObject.  Override any combination
    of the lifecycle methods; all default to no-ops.

    Attributes
    ----------
    game_object : GameObject
        The GameObject this component is attached to.  Set by
        GameObject.add_component() before awake() is called.
    transform : Transform
        Convenience alias — equivalent to self.game_object.transform.
        Set alongside game_object.
    enabled : bool
        When False the component is skipped for update/late_update/fixed_update.
        awake() and start() still run regardless of enabled state.

    Lifecycle order (guaranteed by ComponentRegistry)
    --------------------------------------------------
    1. awake
    2. on_enable   (only if enabled at the time)
    3. start
    4. update      (every enabled frame)
    5. late_update (every enabled frame, after all updates)
    6. fixed_update (fixed-dt, 0–5× per real frame)
    7. on_disable  (enabled → False, or object deactivated)
    8. on_destroy  (deferred to end of the frame that called destroy())
    """

    __slots__ = ("game_object", "transform", "enabled", "_started")

    def __init__(self) -> None:
        self.game_object: "GameObject | None" = None
        self.transform:   "Transform | None"  = None
        self.enabled: bool = True
        self._started: bool = False  # tracks whether start() has run

    # ------------------------------------------------------------------
    # Lifecycle — default no-ops
    # ------------------------------------------------------------------

    def awake(self) -> None:
        """
        Called once when the component is first created.

        Run before any start() calls this frame, even if the component is
        disabled.  Use for initialisation that doesn't depend on other
        components (self-contained state setup).

        Note: do NOT call base class awake() — the default body is a no-op
        and components are not guaranteed to have a super() chain.
        """

    def on_enable(self) -> None:
        """
        Called when the component transitions from disabled → enabled.

        Also called on the first enable (after awake) when the component is
        created in an enabled state.  May be called multiple times during
        the lifetime of the component.
        """

    def start(self) -> None:
        """
        Called once before the first update(), after all awake() calls for
        the frame have completed.

        Use for initialisation that depends on other components being awake.
        """

    def update(self, dt: float) -> None:
        """
        Called every frame while the component is enabled and its owning
        GameObject is active_in_hierarchy.

        Parameters
        ----------
        dt : float — real frame delta in **seconds** (e.g. 0.016 at 60 fps).
        """

    def late_update(self, dt: float) -> None:
        """
        Called after ALL update() calls for the current frame.

        Use for operations that depend on the final positions of other objects
        (camera follow, IK, etc.).

        Parameters
        ----------
        dt : float — real frame delta in **seconds**.
        """

    def fixed_update(self, dt: float) -> None:
        """
        Called at a fixed timestep, driven by Clock.fixed_steps().

        May be called 0–5 times per real frame (default 50 Hz = 0.02 s).
        Use for physics, AI ticks, or any logic that must run at a predictable
        rate regardless of rendering frame rate.

        Parameters
        ----------
        dt : float — fixed timestep in **seconds** (always clock.fixed_dt).
        """

    def on_disable(self) -> None:
        """
        Called when enabled transitions to False, or when the owning
        GameObject's active_in_hierarchy becomes False.

        May be called multiple times (paired with on_enable).
        """

    def on_destroy(self) -> None:
        """
        Called at end-of-frame when this component (or its owning GameObject)
        is destroyed via world.destroy().

        Perform cleanup: unsubscribe events, release resources, etc.
        Called after on_disable() for enabled components.
        """
