"""
devtools/_tools/base.py — DevTool base class for dev-overlay panels.

Docs: docs/systems/devtools.md
"""

from __future__ import annotations

from fire_engine.devtools.types import Panel


class DevTool:
    """
    Base class for every dev-overlay panel.

    Subclasses set :attr:`tool_id` / :attr:`title` and implement :meth:`build`.
    Override :attr:`revision` if the panel's structure can change at runtime
    (the default 0 means "structure is fixed; only values change").

    Attributes
    ----------
    tool_id : str — stable id; the renderer keys persistent widgets off it.
    title   : str — panel caption.

    Docs: docs/systems/devtools.md
    """

    tool_id: str = "tool"
    title: str = "Tool"

    @property
    def revision(self) -> int:
        """
        Structure revision — bump when sections/fields/buttons appear or vanish.

        The renderer rebuilds its widgets whenever this value changes and only
        polls ``Field.get`` otherwise.  A fixed-structure tool can leave it at 0.

        Docs: docs/systems/devtools.md
        """
        return 0

    def build(self) -> Panel:
        """
        Return the panel to display this frame.  Implemented by subclasses.

        Docs: docs/systems/devtools.md
        """
        raise NotImplementedError(
            "DevTool subclasses must implement build(); see ARCHITECTURE.md §6."
        )
