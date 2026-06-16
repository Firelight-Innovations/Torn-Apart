"""
devtools/_tools/inspector.py — InspectorTool: editable inspector for the selected object.

Docs: docs/systems/devtools.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fire_engine.devtools._tools.base import DevTool
from fire_engine.devtools.enums import FieldKind
from fire_engine.devtools.introspect import describe_chunk, describe_object, is_chunk
from fire_engine.devtools.types import Field, Panel, Section

if TYPE_CHECKING:
    from fire_engine.devtools.selection import Selection


class InspectorTool(DevTool):
    """
    Editable inspector for the currently-selected GameObject.

    Rebuilds its structure (via the ``revision`` it borrows from the
    :class:`~fire_engine.devtools.selection.Selection`) whenever the selection
    changes; between changes it shows live values and applies edits straight to
    the object through :func:`~fire_engine.devtools.introspect.describe_object`.

    Parameters
    ----------
    selection : Selection — shared selection state.

    Docs: docs/systems/devtools.md
    """

    tool_id = "inspector"
    title = "Inspector"

    def __init__(self, selection: Selection) -> None:
        self._selection = selection

    @property
    def revision(self) -> int:
        """
        Mirrors the :class:`~fire_engine.devtools.selection.Selection` revision.

        Docs: docs/systems/devtools.md
        """
        # Selection changes are the only thing that reshapes this panel.
        return self._selection.revision

    def build(self) -> Panel:
        """
        Build the inspector Panel for the currently-selected object.

        Docs: docs/systems/devtools.md
        """
        go = self._selection.current
        if go is None:
            return Panel(
                self.tool_id,
                self.title,
                [
                    Section(
                        "",
                        [Field("(nothing selected)", FieldKind.LABEL, lambda: "click an object")],
                    )
                ],
                revision=self.revision,
            )
        # A picked terrain chunk is not a GameObject — route it to the chunk
        # describer (read-only voxel stats) instead of the component reflector.
        if is_chunk(go):
            title = f"Inspector — Chunk {tuple(go.coord)}"
            return Panel(self.tool_id, title, describe_chunk(go), revision=self.revision)
        title = f"Inspector — {go.name}"
        return Panel(self.tool_id, title, describe_object(go), revision=self.revision)
