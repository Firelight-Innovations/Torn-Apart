"""
devtools/_tools/clock.py — ClockTool: game calendar read-out for the dev overlay.

Docs: docs/systems/devtools.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fire_engine.devtools._tools.base import DevTool
from fire_engine.devtools.enums import FieldKind
from fire_engine.devtools.types import Field, Panel, Section

if TYPE_CHECKING:
    from fire_engine.core.clock import Clock


class ClockTool(DevTool):
    """
    Read-out of the game calendar — day number and time-of-day.

    This is the natural home for the upcoming day/night-cycle controls the owner
    described: when a sun-angle / time-scale system exists, add editable Fields
    here (e.g. a ``time_of_day`` slider) and they appear in the overlay with no
    renderer changes.

    Parameters
    ----------
    clock : Clock — the engine clock (duck-typed: ``game_day`` / ``game_time_of_day``).
    """

    tool_id = "clock"
    title = "Time"

    def __init__(self, clock: Clock) -> None:
        self._clock = clock

    @staticmethod
    def _fmt_tod(seconds: float) -> str:
        """Format seconds-within-day as HH:MM (24h)."""
        total_min = int(seconds // 60)
        return f"{(total_min // 60) % 24:02d}:{total_min % 60:02d}"

    def build(self) -> Panel:
        c = self._clock
        return Panel(
            self.tool_id,
            self.title,
            [
                Section(
                    "Calendar",
                    [
                        Field("day", FieldKind.LABEL, lambda: c.game_day),
                        Field(
                            "time of day",
                            FieldKind.LABEL,
                            lambda: self._fmt_tod(c.game_time_of_day),
                        ),
                    ],
                )
            ],
        )
