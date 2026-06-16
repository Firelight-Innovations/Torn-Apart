"""
weather/_impl/_update.py — Update-loop and gust-front helpers for WeatherSystem.

Internal helpers — do NOT import from outside fire_engine.world.weather.
Each function takes the WeatherSystem instance as its first argument (``ws``).

Docs: docs/systems/world.weather.md
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from fire_engine.core.event_bus import LightningStrikeEvent, WeatherChangedEvent
from fire_engine.world.weather.cells import StormCell
from fire_engine.world.weather.classify import WeatherType, classify
from fire_engine.world.weather.types import LocalWeather

if TYPE_CHECKING:
    from fire_engine.world.weather.system import WeatherSystem

#: Game seconds per game day.
_DAY_S: float = 24.0 * 3600.0


def _smoothstep(x: float, lo: float, hi: float) -> float:
    """Hermite smoothstep clamped to [0, 1]."""
    if hi <= lo:
        return 0.0 if x < lo else 1.0
    t = (x - lo) / (hi - lo)
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return t * t * (3.0 - 2.0 * t)


def _lerp_local(a: LocalWeather, b: LocalWeather, t: float) -> LocalWeather:
    """
    Component-wise lerp between two :class:`LocalWeather` (``t`` in [0, 1]).

    ``wind_dir`` is lerped then renormalised; endpoints short-circuit so a
    completed blend is bit-exact equal to its target.
    """
    if t <= 0.0:
        return a
    if t >= 1.0:
        return b
    wx = a.wind_dir[0] + (b.wind_dir[0] - a.wind_dir[0]) * t
    wy = a.wind_dir[1] + (b.wind_dir[1] - a.wind_dir[1]) * t
    norm = math.hypot(wx, wy)
    wind_dir = (wx / norm, wy / norm) if norm > 1e-6 else b.wind_dir

    def mix(x: float, y: float) -> float:
        return x + (y - x) * t

    return LocalWeather(
        cloud_coverage=mix(a.cloud_coverage, b.cloud_coverage),
        cloud_density=mix(a.cloud_density, b.cloud_density),
        fog_density=mix(a.fog_density, b.fog_density),
        rain_intensity=mix(a.rain_intensity, b.rain_intensity),
        wind_dir=wind_dir,
        wind_speed=mix(a.wind_speed, b.wind_speed),
        humidity=mix(a.humidity, b.humidity),
        wetness=mix(a.wetness, b.wetness),
        temperature_c=mix(a.temperature_c, b.temperature_c),
    )


def classified_state(ws: WeatherSystem, lw: LocalWeather, abs_t: float) -> WeatherType:
    """
    Hysteresis-stabilised :func:`classify`: a changed label must persist
    ``HYSTERESIS_SECONDS`` before it becomes the committed state.
    """
    from fire_engine.world.weather.system import HYSTERESIS_SECONDS

    raw = classify(lw)
    if ws._committed_state is None:
        ws._committed_state = raw
        ws._pending_state = raw
        ws._pending_since = abs_t
        return raw
    if raw is ws._committed_state:
        ws._pending_state = raw
        ws._pending_since = abs_t
        return raw
    if raw is not ws._pending_state:
        ws._pending_state = raw
        ws._pending_since = abs_t
    if abs_t - ws._pending_since >= HYSTERESIS_SECONDS:
        ws._committed_state = raw
    return ws._committed_state


def emit_lightning(ws: WeatherSystem, abs_t: float, cells: list[StormCell]) -> None:
    """
    Publish a :class:`LightningStrikeEvent` per scheduled strike since the
    last update, for every active THUNDERSTORM cell.

    Pure-schedule emission (load-resume safe — see lightning.py).
    """
    from fire_engine.world.weather.cells import CellKind
    from fire_engine.world.weather.lightning import cell_id_int, scheduled_strikes

    if ws._bus is None:
        return
    last = ws._last_strike_time
    if last is None:
        return
    if abs_t <= last:
        return

    cloud_base = float(ws._config.weather_lightning_cloud_base_m)
    ground_z = float(ws._config.weather_lightning_ground_z_m)

    for cell in cells:
        if cell.kind is not CellKind.THUNDERSTORM:
            continue
        strikes = scheduled_strikes(cell, last, abs_t, ws._config)
        if not strikes:
            continue
        cid = cell_id_int(cell.id)
        for s in strikes:
            center = cell.center(s.time_abs, ws.synoptic)
            wx = float(center[0] + s.pos_xy[0])
            wy = float(center[1] + s.pos_xy[1])
            ws._bus.publish_deferred(
                LightningStrikeEvent(
                    pos=(wx, wy, ground_z + cloud_base),
                    ground_pos=(wx, wy, ground_z),
                    seed=int(s.seed),
                    time_abs=float(s.time_abs),
                    cell_id=int(cid),
                    intensity=float(s.intensity),
                )
            )


def update_gust_fronts(ws: WeatherSystem, t: float, player_pos: tuple[float, float]) -> None:
    """Register / release gust-front wind modifiers for nearby storm cells."""
    if ws._wind_field is None:
        return
    from fire_engine.world.wind import GustFront

    (ux, uy), _ = ws.synoptic.wind(t)
    origin = np.array(player_pos, dtype=np.float64)
    active = ws._active_cells(t)
    near_ids: set[str] = set()
    for cell in active:
        center = cell.center(t, ws.synoptic)
        edge = float(np.hypot(*(center - origin))) - cell.radius(t)
        if edge <= ws._gustfront_range_m:
            near_ids.add(cell.id)
            if cell.id not in ws._active_fronts:
                front = GustFront(
                    seed_key=("weather", cell.id),
                    direction=(ux, uy),
                    speed=max(1.0, ws._gustfront_strength),
                    strength=ws._gustfront_strength * cell.intensity(t),
                    width_m=ws._gustfront_width_m,
                )
                ws._wind_field.add_modifier(front)
                ws._active_fronts[cell.id] = front
    for cid in list(ws._active_fronts):
        if cid not in near_ids:
            ws._release_front(cid)


def do_update(
    ws: WeatherSystem,
    game_day: int,
    game_time_of_day: float,
    player_pos: tuple[float, float] | None,
    blend_seconds: float,
    state_targets: dict[WeatherType, tuple[float, float, float, float, float]],
) -> LocalWeather:
    """
    Core update logic for :meth:`WeatherSystem.update`.

    Samples the weather, advances the override/label state, publishes events.
    Returns the local sample (override-blended if forced).
    """
    from fire_engine.world.weather._impl._sampling import sample_local, temperature

    day = int(game_day)
    tod = float(game_time_of_day) % _DAY_S
    abs_t = day * _DAY_S + tod
    pos = (float(player_pos[0]), float(player_pos[1])) if player_pos is not None else (0.0, 0.0)
    ws._last_player = pos

    natural = sample_local(ws, pos, abs_t)

    cells = ws._active_cells(abs_t)
    origin = np.array(pos, dtype=np.float64)
    cells.sort(key=lambda c: float(np.hypot(*(c.center(abs_t, ws.synoptic) - origin))))
    ws._cells = cells

    update_gust_fronts(ws, abs_t, pos)

    if ws._bus is not None:
        emit_lightning(ws, abs_t, cells)
    ws._last_strike_time = abs_t

    if ws._override is not None:
        if ws._override_start_abs_t is None:
            ws._override_start_abs_t = abs_t
            ws._override_from = ws._last_local if ws._last_local is not None else natural
        cov, den, fog, rain, wind_mult = state_targets[ws._override]
        wind_dir, syn_speed = ws.synoptic.wind(abs_t)
        tod_h = (abs_t % _DAY_S) / 3600.0
        target = LocalWeather(
            cloud_coverage=cov,
            cloud_density=den,
            fog_density=fog,
            rain_intensity=rain,
            wind_dir=wind_dir,
            wind_speed=syn_speed * wind_mult,
            humidity=0.5,
            wetness=0.0,
            temperature_c=temperature(ws, tod_h),
        )
        bt = _smoothstep(abs_t - ws._override_start_abs_t, 0.0, blend_seconds)
        override_from = ws._override_from if ws._override_from is not None else natural
        local = _lerp_local(override_from, target, bt)
        new_state = ws._override
    elif ws._release_from is not None:
        if ws._release_start_abs_t is None:
            ws._release_start_abs_t = abs_t
        bt = _smoothstep(abs_t - ws._release_start_abs_t, 0.0, blend_seconds)
        local = _lerp_local(ws._release_from, natural, bt)
        if bt >= 1.0:
            ws._release_from = None
            ws._release_start_abs_t = None
        new_state = classified_state(ws, local, abs_t)
    else:
        local = natural
        new_state = classified_state(ws, local, abs_t)

    if ws._last_state is not None and new_state is not ws._last_state and ws._bus is not None:
        ws._bus.publish_deferred(
            WeatherChangedEvent(
                previous=ws._last_state.value,
                current=new_state.value,
                day=day,
            )
        )
    ws._last_state = new_state
    ws._last_local = local
    ws._last_abs_t = abs_t
    return local
