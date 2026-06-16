"""
devtools/_tools/performance.py — PerformanceTool: live engine stats read-out.

Docs: docs/systems/devtools.md
"""

from __future__ import annotations

from collections.abc import Callable

from fire_engine.devtools._tools.base import DevTool
from fire_engine.devtools.enums import FieldKind
from fire_engine.devtools.types import Field, Panel, Section


class PerformanceTool(DevTool):
    """
    Live engine performance / state read-out.

    Parameters
    ----------
    providers : dict[str, Callable[[], object]]
        Ordered mapping of label → zero-arg callable returning the current
        value (rendered via ``str``).  The renderer supplies panda3d-backed
        callables (FPS, frame ms) plus engine ones (chunk count, object count),
        so this tool stays headless.

    Example
    -------
        PerformanceTool({
            "FPS":    lambda: round(globalClock.get_average_frame_rate(), 1),
            "chunks": lambda: len(chunk_manager.chunks),
        })

    Docs: docs/systems/devtools.md
    """

    tool_id = "performance"
    title = "Performance"

    def __init__(self, providers: dict[str, Callable[[], object]]) -> None:
        self._providers = dict(providers)

    def build(self) -> Panel:
        """
        Build the current-frame Panel with one LABEL row per provider.

        Docs: docs/systems/devtools.md
        """
        fields = [Field(label, FieldKind.LABEL, fn) for label, fn in self._providers.items()]
        return Panel(self.tool_id, self.title, [Section("Stats", fields)])
