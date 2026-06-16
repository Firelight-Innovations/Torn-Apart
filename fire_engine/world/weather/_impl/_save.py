"""
weather/_impl/_save.py — Saveable protocol method-cluster extracted from WeatherSystem.

Internal helpers — do NOT import from outside fire_engine.world.weather.
Each function takes the WeatherSystem instance as its first argument (``ws``).

Docs: docs/systems/world.weather.md
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from fire_engine.world.weather.cells import CellKind, StormCell
from fire_engine.world.weather.classify import WeatherType
from fire_engine.world.weather.types import LocalWeather

if TYPE_CHECKING:
    from fire_engine.world.weather.system import WeatherSystem


# ---------------------------------------------------------------------------
# LocalWeather serialisation
# ---------------------------------------------------------------------------


def local_to_dict(lw: LocalWeather) -> dict[str, Any]:
    """Serialise a LocalWeather to plain primitives (Saveable).

    Docs: docs/systems/world.weather._impl.md
    """
    return {
        "cloud_coverage": float(lw.cloud_coverage),
        "cloud_density": float(lw.cloud_density),
        "fog_density": float(lw.fog_density),
        "rain_intensity": float(lw.rain_intensity),
        "wind_dir": [float(lw.wind_dir[0]), float(lw.wind_dir[1])],
        "wind_speed": float(lw.wind_speed),
        "humidity": float(lw.humidity),
        "wetness": float(lw.wetness),
        "temperature_c": float(lw.temperature_c),
    }


def local_from_dict(d: dict[str, Any]) -> LocalWeather:
    """Inverse of :func:`local_to_dict`.

    Docs: docs/systems/world.weather._impl.md
    """
    return LocalWeather(
        cloud_coverage=float(d["cloud_coverage"]),
        cloud_density=float(d["cloud_density"]),
        fog_density=float(d["fog_density"]),
        rain_intensity=float(d["rain_intensity"]),
        wind_dir=(float(d["wind_dir"][0]), float(d["wind_dir"][1])),
        wind_speed=float(d["wind_speed"]),
        humidity=float(d.get("humidity", 0.5)),
        wetness=float(d.get("wetness", 0.0)),
        temperature_c=float(d.get("temperature_c", 12.0)),
    )


# ---------------------------------------------------------------------------
# StormCell serialisation
# ---------------------------------------------------------------------------


def cell_to_dict(c: StormCell) -> dict[str, Any]:
    """Serialise a summoned :class:`StormCell` to plain primitives (Saveable).

    Docs: docs/systems/world.weather._impl.md
    """
    return {
        "id": str(c.id),
        "kind": c.kind.value,
        "spawn_time": float(c.spawn_time),
        "spawn_pos": [float(c.spawn_pos[0]), float(c.spawn_pos[1])],
        "duration_s": float(c.duration_s),
        "radius_m": float(c.radius_m),
        "peak_intensity": float(c.peak_intensity),
        "drift_bias": [float(c.drift_bias[0]), float(c.drift_bias[1])],
    }


def cell_from_dict(d: dict[str, Any]) -> StormCell:
    """Inverse of :func:`cell_to_dict`.

    Raises ``KeyError``/``ValueError`` on malformed input — caller guards.

    Docs: docs/systems/world.weather._impl.md
    """
    return StormCell(
        id=str(d["id"]),
        kind=CellKind(d["kind"]),
        spawn_time=float(d["spawn_time"]),
        spawn_pos=(float(d["spawn_pos"][0]), float(d["spawn_pos"][1])),
        duration_s=float(d["duration_s"]),
        radius_m=float(d["radius_m"]),
        peak_intensity=float(d["peak_intensity"]),
        drift_bias=(float(d["drift_bias"][0]), float(d["drift_bias"][1])),
    )


# ---------------------------------------------------------------------------
# WeatherSystem saveable protocol
# ---------------------------------------------------------------------------


def get_delta(ws: WeatherSystem) -> dict[str, Any]:
    """
    Deviations from the procedural baseline (Saveable protocol).

    Returns ``{}`` when no summons, no suppressions, and no dev override (or
    release blend) exist.  Otherwise a small dict of plain primitives (no live
    objects, no pickle, Hard Rule 3):
    ``summoned``, ``summon_seq``, ``suppressed``, and legacy override keys.

    Docs: docs/systems/world.weather._impl.md
    """
    delta: dict[str, Any] = {}

    if ws._summoned:
        delta["summoned"] = [cell_to_dict(c) for c in ws._summoned]
        delta["summon_seq"] = int(ws._summon_seq)
    if ws._suppressed:
        delta["suppressed"] = sorted(ws._suppressed)

    if ws._override is not None:
        delta["override"] = ws._override.value
        if ws._override_start_abs_t is not None:
            delta["override_start_abs_t"] = float(ws._override_start_abs_t)
        if ws._override_from is not None:
            delta["override_from"] = local_to_dict(ws._override_from)
        if ws._last_state is not None:
            delta["last_state"] = ws._last_state.value
    elif ws._release_from is not None:
        delta["release_from"] = local_to_dict(ws._release_from)
        if ws._release_start_abs_t is not None:
            delta["release_start_abs_t"] = float(ws._release_start_abs_t)
        if ws._last_state is not None:
            delta["last_state"] = ws._last_state.value

    return delta


def apply_delta_summons(ws: WeatherSystem, delta: dict[str, Any]) -> None:
    """Rebuild summoned cells + suppression set from delta (M8).

    Docs: docs/systems/world.weather._impl.md
    """
    summoned: list[StormCell] = []
    for d in delta.get("summoned", ()) or ():
        try:
            summoned.append(cell_from_dict(d))
        except (KeyError, ValueError, TypeError, IndexError):
            continue
    if summoned:
        ws._summoned = summoned
    if "summon_seq" in delta:
        with contextlib.suppress(TypeError, ValueError):
            ws._summon_seq = int(delta["summon_seq"])
    for c in ws._summoned:
        if c.id.startswith("s:"):
            with contextlib.suppress(ValueError):
                ws._summon_seq = max(ws._summon_seq, int(c.id[2:]) + 1)
    supp = delta.get("suppressed")
    if isinstance(supp, (list, tuple)):
        ws._suppressed = {str(s) for s in supp}


def apply_delta_override(ws: WeatherSystem, delta: dict[str, Any]) -> None:
    """Restore legacy dev-override shim state from delta.

    Docs: docs/systems/world.weather._impl.md
    """
    if "override" in delta:
        ws._override = WeatherType(delta["override"])
        ws._override_start_abs_t = (
            float(delta["override_start_abs_t"]) if "override_start_abs_t" in delta else None
        )
        ws._override_from = (
            local_from_dict(delta["override_from"]) if "override_from" in delta else None
        )
    elif "release_from" in delta:
        ws._release_from = local_from_dict(delta["release_from"])
        ws._release_start_abs_t = (
            float(delta["release_start_abs_t"]) if "release_start_abs_t" in delta else None
        )
    if "last_state" in delta:
        ws._last_state = WeatherType(delta["last_state"])
        ws._committed_state = ws._last_state
