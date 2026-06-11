"""
devtools/manager.py — the headless hub the debug overlay drives.

:class:`DevToolsManager` owns everything the on-screen overlay needs that does
*not* require panda3d: the registered tools, the shared :class:`Selection`, the
list of pickable :class:`Selectable`s, and the enabled/visible flag.  The
renderer (``world/devtools_overlay.py``) holds one of these and, each frame,
asks it for the panels to draw and for pick results.

This split is the point: all the editor *logic* is here and unit-tested without
a window; ``world/`` only turns :class:`Panel` data into DirectGUI widgets and
turns mouse events into rays.  Swapping the renderer (e.g. to Dear ImGui later)
touches nothing in this package.

No panda3d imports — headless-testable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from fire_engine.core.math3d import Vec3
from fire_engine.devtools.selection import Selection
from fire_engine.devtools.picking import Selectable, pick
from fire_engine.devtools.fields import Panel
from fire_engine.devtools.tools import DevTool

if TYPE_CHECKING:
    from fire_engine.world.gameobject import GameObject


class DevToolsManager:
    """
    Central registry for dev tools, selection, and pickable objects.

    Attributes
    ----------
    selection : Selection
        Shared current-selection state (read by the Inspector + outline).
    tools : list[DevTool]
        Registered panels, in display order.
    selectables : list[Selectable]
        Objects the click-picker can hit.
    enabled : bool
        Whether the overlay is currently shown.  The renderer reads this; the
        owning app flips it on the toggle key.

    Example
    -------
        mgr = DevToolsManager()
        mgr.register_tool(PerformanceTool({...}))
        mgr.register_tool(InspectorTool(mgr.selection))
        mgr.add_selectable(cube_go, Vec3(0.5, 0.5, 0.5))
    """

    __slots__ = ("selection", "tools", "selectables", "enabled")

    def __init__(self) -> None:
        self.selection: Selection = Selection()
        self.tools: list[DevTool] = []
        self.selectables: list[Selectable] = []
        self.enabled: bool = False

    # ------------------------------------------------------------------
    # Tool registry
    # ------------------------------------------------------------------

    def register_tool(self, tool: DevTool) -> DevTool:
        """
        Register a dev tool (panel).  Returns the tool for convenient chaining.

        Parameters
        ----------
        tool : DevTool
        """
        self.tools.append(tool)
        return tool

    def panels(self) -> list[Panel]:
        """Build the current-frame :class:`Panel` for every registered tool."""
        return [t.build() for t in self.tools]

    # ------------------------------------------------------------------
    # Pickable objects
    # ------------------------------------------------------------------

    def add_selectable(self, go: "GameObject", half_extents: Vec3) -> Selectable:
        """
        Register ``go`` as click-pickable with the given local AABB half-extents.

        Parameters
        ----------
        go : GameObject
        half_extents : Vec3 — half the box size per axis, meters, pre-scale.

        Returns
        -------
        Selectable — the created entry (also appended to ``selectables``).
        """
        sel = Selectable(go, half_extents)
        self.selectables.append(sel)
        return sel

    def remove_selectable(self, go: "GameObject") -> None:
        """
        Drop ``go`` from the pickable set (e.g. when it is destroyed); clears the
        selection if it was the selected object.

        Parameters
        ----------
        go : GameObject
        """
        self.selectables = [s for s in self.selectables if s.game_object is not go]
        if self.selection.current is go:
            self.selection.clear()

    def find_selectable(self, go: "GameObject") -> Optional[Selectable]:
        """Return the :class:`Selectable` for ``go``, or ``None``."""
        for s in self.selectables:
            if s.game_object is go:
                return s
        return None

    def pick(self, origin: Vec3, direction: Vec3) -> "Optional[GameObject]":
        """
        Ray-pick the nearest selectable along ``origin``/``direction``.

        Parameters
        ----------
        origin : Vec3 — world-space ray origin (meters).
        direction : Vec3 — world-space ray direction (meters).

        Returns
        -------
        GameObject | None
        """
        return pick(origin, direction, self.selectables)

    def pick_and_select(self, origin: Vec3, direction: Vec3) -> "Optional[GameObject]":
        """
        Convenience: :meth:`pick` then write the result into :attr:`selection`.

        Returns the hit GameObject (or ``None``), and updates the selection to
        match (selecting ``None`` on a miss deselects).

        Parameters
        ----------
        origin : Vec3
        direction : Vec3
        """
        hit = self.pick(origin, direction)
        self.selection.set(hit)
        return hit
