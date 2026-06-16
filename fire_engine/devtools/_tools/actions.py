"""
devtools/_tools/actions.py — ActionsTool: panel of one-shot action buttons.

Docs: docs/systems/devtools.md
"""

from __future__ import annotations

from collections.abc import Callable

from fire_engine.devtools._tools.base import DevTool
from fire_engine.devtools.types import Button, Panel


class ActionsTool(DevTool):
    """
    A panel of one-shot action buttons — spawn props, fire events, reset state.

    Actions can be added at runtime via :meth:`add_action`; doing so bumps the
    revision so the renderer rebuilds the button row.  This is how gameplay
    systems hang their own dev verbs off the overlay without touching the
    renderer.

    Parameters
    ----------
    title : str — panel caption (default ``"Actions"``).
    actions : dict[str, Callable[[], None]] | None — initial label → handler map.

    Example
    -------
        tools = ActionsTool("World", {"Spawn Cube": spawn_cube})
        tools.add_action("Fire Explosion", explode_at_camera)

    Docs: docs/systems/devtools.md
    """

    tool_id = "actions"

    def __init__(
        self,
        title: str = "Actions",
        actions: dict[str, Callable[[], None]] | None = None,
    ) -> None:
        self.title = title
        self._actions: list[tuple[str, Callable[[], None]]] = list((actions or {}).items())
        self._revision = 0

    @property
    def revision(self) -> int:
        """
        Structure revision; bumps each time an action is added.

        Docs: docs/systems/devtools.md
        """
        return self._revision

    def add_action(self, label: str, handler: Callable[[], None]) -> None:
        """
        Append an action button and rebuild the panel next frame.

        Docs: docs/systems/devtools.md
        """
        self._actions.append((label, handler))
        self._revision += 1

    def build(self) -> Panel:
        """
        Build the current-frame Panel with all registered action buttons.

        Docs: docs/systems/devtools.md
        """
        buttons = [Button(label, fn) for label, fn in self._actions]
        return Panel(self.tool_id, self.title, [], buttons=buttons, revision=self._revision)
