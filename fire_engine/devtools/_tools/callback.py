"""
devtools/_tools/callback.py — CallbackTool: ad-hoc panel from a build function.

Docs: docs/systems/devtools.md
"""

from __future__ import annotations

from collections.abc import Callable

from fire_engine.devtools._tools.base import DevTool
from fire_engine.devtools.types import Button, Panel, Section


class CallbackTool(DevTool):
    """
    A panel whose contents come from a supplied ``build_fn`` each frame.

    The quickest way to surface a subsystem in the overlay without writing a
    dedicated :class:`DevTool` subclass — hand it a function that returns the
    sections/buttons.  Used (in ``world/devtools_overlay.py``) to expose the sky
    / weather / time-of-day environment controls, since that subsystem is
    panda3d-free but the binding lives in the renderer.

    Parameters
    ----------
    tool_id : str — stable id.
    title : str — panel caption.
    build_fn : Callable[[], tuple[list[Section], list[Button]]]
        Returns ``(sections, buttons)`` for the current frame.
    revision_fn : Callable[[], int] | None
        Optional structural-revision source (e.g. when the section layout can
        change).  Defaults to a constant 0 (fixed structure).

    Example
    -------
        CallbackTool("env", "Environment",
                     lambda: ([Section("Time", [...])], [Button("Noon", set_noon)]))

    Docs: docs/systems/devtools.md
    """

    def __init__(
        self,
        tool_id: str,
        title: str,
        build_fn: Callable[[], tuple[list[Section], list[Button]]],
        revision_fn: Callable[[], int] | None = None,
    ) -> None:
        self.tool_id = tool_id
        self.title = title
        self._build_fn = build_fn
        self._revision_fn = revision_fn

    @property
    def revision(self) -> int:
        """
        Structural revision sourced from the optional ``revision_fn``, or 0.

        Docs: docs/systems/devtools.md
        """
        return self._revision_fn() if self._revision_fn is not None else 0

    def build(self) -> Panel:
        """
        Invoke ``build_fn`` and wrap the result in a :class:`~fire_engine.devtools.types.Panel`.

        Docs: docs/systems/devtools.md
        """
        sections, buttons = self._build_fn()
        return Panel(self.tool_id, self.title, sections, buttons=buttons, revision=self.revision)
