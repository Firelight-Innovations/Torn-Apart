"""
weather/_impl/_summon.py — Summon API method-cluster extracted from WeatherSystem.

Internal helpers — do NOT import from outside fire_engine.world.weather.
Each function takes the WeatherSystem instance as its first argument (``ws``).

Docs: docs/systems/world.weather.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fire_engine.world.weather.cells import CellKind, StormCell

if TYPE_CHECKING:
    from fire_engine.world.weather.system import WeatherSystem

#: Game seconds per game day.
_DAY_S: float = 24.0 * 3600.0


def summon_cell(
    ws: WeatherSystem,
    kind: CellKind,
    *,
    time_abs: float,
    player_pos: tuple[float, float],
    radius_m: float | None = None,
    duration_s: float | None = None,
    peak_intensity: float | None = None,
    upwind_m: float | None = None,
) -> str:
    """
    Spawn a summoned :class:`StormCell` UPWIND of the player and return its id.

    The cell is placed ``upwind_m`` meters from *player_pos* in the opposite
    direction to the synoptic wind at *time_abs*, so it drifts toward the player
    on the steering current.  Radius/duration/intensity default to the per-kind
    ``weather_summon_*`` config values.

    Parameters
    ----------
    ws : WeatherSystem — the system to add the cell to.
    kind : CellKind — what the summoned cell does.
    time_abs : float — absolute game seconds = the cell's ``spawn_time``.
    player_pos : tuple[float, float] — world XY the cell is aimed at (m).
    radius_m, duration_s, peak_intensity : float | None — per-kind overrides.
    upwind_m : float | None — spawn distance upwind (default: config value).

    Returns
    -------
    str — the new cell's stable id (``"s:{n}"``).

    Example
    -------
    >>> cid = summon_cell(ws, CellKind.THUNDERSTORM, time_abs=3600.0,
    ...                   player_pos=(0.0, 0.0))
    >>> cid.startswith("s:")
    True
    """
    kind = CellKind(kind)
    r_def, d_def, p_def = ws._summon_defaults[kind]
    radius = float(radius_m) if radius_m is not None else r_def
    duration = float(duration_s) if duration_s is not None else d_def
    peak = float(peak_intensity) if peak_intensity is not None else p_def
    dist = float(upwind_m) if upwind_m is not None else ws._summon_upwind_m

    (ux, uy), _ = ws.synoptic.wind(float(time_abs))
    spawn_pos = (
        float(player_pos[0]) - dist * ux,
        float(player_pos[1]) - dist * uy,
    )

    cell_id = f"s:{ws._summon_seq}"
    ws._summon_seq += 1
    ws._summoned.append(
        StormCell(
            id=cell_id,
            kind=kind,
            spawn_time=float(time_abs),
            spawn_pos=spawn_pos,
            duration_s=duration,
            radius_m=radius,
            peak_intensity=peak,
            drift_bias=(0.0, 0.0),
        )
    )
    return cell_id


def suppress(ws: WeatherSystem, cell_id: str) -> None:
    """
    Hide a cell from all future samples.

    A natural-cell id (``"n:{day}:{slot}"``) is added to the suppression set;
    a summoned-cell id (``"s:{n}"``) is removed outright.  No-op for unknown id.
    """
    cid = str(cell_id)
    if cid.startswith("s:"):
        ws._summoned = [c for c in ws._summoned if c.id != cid]
        ws._release_front(cid)
    else:
        ws._suppressed.add(cid)


def clear_all(ws: WeatherSystem) -> None:
    """
    Clear every summoned cell and suppress every natural cell active *now*.

    Gives a dev a one-call "clear skies": summoned cells are dropped and the
    natural cells alive at the last update are added to the suppression set.
    Registered gust fronts are released cleanly.
    """
    ws._summoned.clear()
    t = ws._last_abs_t if ws._last_abs_t is not None else 0.0
    day = int(t // _DAY_S)
    for c in ws._cells_for_day(day - 1) + ws._cells_for_day(day):
        ws._suppressed.add(c.id)
    for cid in list(ws._active_fronts):
        ws._release_front(cid)
