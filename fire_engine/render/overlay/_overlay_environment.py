"""
render/overlay/_overlay_environment.py — Environment + time-of-day panel helpers.

Extracted from devtools_overlay.py; called as free functions taking the overlay
instance as first argument (C0302 fat-class split pattern).

Docs: docs/systems/render.overlay.md
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from fire_engine.devtools import Button, Field, FieldKind, Section

if TYPE_CHECKING:
    from fire_engine.render.overlay.devtools_overlay import DevOverlay


def _fmt(value: object) -> str:
    """Compact display string for a scalar field value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def build_environment(
    self_obj: DevOverlay, sky: Any, clock: Any
) -> tuple[list[Section], list[Button]]:
    """
    Build the Environment panel: editable time-of-day / time-scale plus a
    live read-out of the current weather and sky parameters, with a single
    compact "Cycle Weather" button.

    All engine access is guarded (``getattr`` / ``try``) so a change in the
    concurrent sky API degrades to blanks rather than crashing the overlay.
    """

    def get_tod_hours() -> float:
        return float(getattr(clock, "game_time_of_day", 0.0)) / 3600.0

    def set_tod_hours(h: Any) -> None:
        with contextlib.suppress(Exception):
            clock.game_time_of_day = (float(h) % 24.0) * 3600.0

    def set_scale(v: Any) -> None:
        with contextlib.suppress(Exception):
            clock.game_time_scale = float(v)

    def state_attr(name: str) -> Any:
        st = getattr(sky, "state", None)
        return getattr(st, name, None) if st is not None else None

    weather = getattr(sky, "weather", None)

    def weather_name() -> str:
        cur = getattr(weather, "current", None)
        return str(getattr(cur, "value", "?")) if cur is not None else "?"

    sections = [
        Section(
            "Time",
            [
                Field(
                    "time of day",
                    FieldKind.FLOAT,
                    get_tod_hours,
                    set_tod_hours,
                    step=1.0,
                    units="h",
                ),
                Field(
                    "time scale",
                    FieldKind.FLOAT,
                    lambda: float(getattr(clock, "game_time_scale", 60.0)),
                    set_scale,
                    step=60.0,
                ),
                Field("day", FieldKind.LABEL, lambda: getattr(clock, "game_day", 0)),
            ],
        ),
        Section(
            "Sky",
            [
                Field("weather", FieldKind.LABEL, weather_name),
                Field(
                    "cloud cover",
                    FieldKind.LABEL,
                    lambda: _fmt(state_attr("cloud_coverage")),
                ),
                Field("fog /m", FieldKind.LABEL, lambda: _fmt(state_attr("fog_density"))),
                Field("rain", FieldKind.LABEL, lambda: _fmt(state_attr("rain_intensity"))),
            ],
        ),
    ]
    buttons: list[Button] = []
    if weather is not None and hasattr(weather, "force_weather"):
        _weather_ref = weather

        def _do_cycle() -> None:
            cycle_weather(self_obj, _weather_ref)

        buttons.append(Button("Cycle Weather", _do_cycle))
    return sections, buttons


def cycle_weather(self_obj: DevOverlay, weather: Any) -> None:
    """Advance the forced-weather override one step (last step = natural)."""
    if not self_obj._weather_types:
        return
    self_obj._wx = (self_obj._wx + 1) % len(self_obj._weather_types)
    with contextlib.suppress(Exception):
        weather.force_weather(self_obj._weather_types[self_obj._wx])
