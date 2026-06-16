"""
weather/_impl/_sampling.py — Sampling method-cluster extracted from WeatherSystem.

Internal helpers — do NOT import from outside fire_engine.world.weather.
Each function takes the WeatherSystem instance as its first argument (``ws``)
and is called from the matching method stub inside the class.

Docs: docs/systems/world.weather.md
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from fire_engine.world.weather.cells import CellKind, day_regime, regime_ambient
from fire_engine.world.weather.humidity import (
    emergent_fog as _emergent_fog_fn,
)
from fire_engine.world.weather.humidity import (
    humidity_base,
    relative_humidity,
)
from fire_engine.world.weather.types import LocalWeather

if TYPE_CHECKING:
    from fire_engine.world.weather.system import WeatherSystem

#: Game seconds per game day — local copy so this module stays sky-free.
_DAY_S: float = 24.0 * 3600.0

#: Per-kind coverage weights (mirrors system.py — kept in sync by the test suite).
_KIND_COV: dict[CellKind, float] = {
    CellKind.CLOUD_BANK: 0.90,
    CellKind.SHOWER: 0.85,
    CellKind.THUNDERSTORM: 1.00,
    CellKind.FOG_BANK: 0.00,
}
_KIND_DEN: dict[CellKind, float] = {
    CellKind.CLOUD_BANK: 0.60,
    CellKind.SHOWER: 0.80,
    CellKind.THUNDERSTORM: 0.95,
    CellKind.FOG_BANK: 0.00,
}
_KIND_RAIN: dict[CellKind, float] = {
    CellKind.CLOUD_BANK: 0.00,
    CellKind.SHOWER: 0.65,
    CellKind.THUNDERSTORM: 1.00,
    CellKind.FOG_BANK: 0.00,
}
_FOG_BASELINE: float = 0.0008
_FOG_BANK_GAIN: float = 0.027


def temperature(ws: WeatherSystem, tod_h: float) -> float:
    """Local air temperature (°C): daily cosine peaking at 15:00.

    Docs: docs/systems/world.weather._impl.md
    """
    return ws._temp_mean + ws._temp_amp * math.cos(2.0 * math.pi * (tod_h - 15.0) / 24.0)


def sample_core(
    ws: WeatherSystem,
    pts: np.ndarray,
    t: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Recursion-safe core fields (no emergent fog) at *pts*, absolute *t*.

    Returns ``(coverage, density, rain, fog_bank, storm_gust)`` where
    ``fog_bank`` is the FOG_BANK + baseline fog coefficient *before* the
    emergent humidity-condensation term is added.  Delegates to
    ``ws._active_cells`` so summoned cells are included automatically.
    ``pts`` must already be a ``float64`` ``(N, 2)`` array.

    Docs: docs/systems/world.weather._impl.md
    """
    n = pts.shape[0]
    day = int(t // _DAY_S)
    tod = t - day * _DAY_S

    cov_cur, den_cur = regime_ambient(day_regime(day))
    if day > 0:
        cov_prev, den_prev = regime_ambient(day_regime(day - 1))
    else:
        cov_prev, den_prev = cov_cur, den_cur
    blend = 0.5 - 0.5 * math.cos(math.pi * min(tod / 3600.0, 1.0))
    coverage = np.full(n, cov_prev + (cov_cur - cov_prev) * blend)
    density = np.full(n, den_prev + (den_cur - den_prev) * blend)
    rain = np.zeros(n)
    fog_extra = np.zeros(n)
    storm_gust = np.zeros(n)

    for cell in ws._active_cells(t):
        c = cell.contribution(pts, t, ws.synoptic)
        coverage += c * _KIND_COV[cell.kind]
        density += c * _KIND_DEN[cell.kind]
        rain += c * _KIND_RAIN[cell.kind]
        if cell.kind is CellKind.FOG_BANK:
            fog_extra += c
        elif cell.kind is CellKind.THUNDERSTORM:
            storm_gust += c

    np.clip(coverage, 0.0, 1.0, out=coverage)
    np.clip(density, 0.0, 1.0, out=density)
    np.clip(rain, 0.0, 1.0, out=rain)
    fog_bank = _FOG_BASELINE + fog_extra * _FOG_BANK_GAIN
    return coverage, density, rain, fog_bank, storm_gust


def local_wind_speed(
    ws: WeatherSystem,
    coverage: np.ndarray,
    storm_gust: np.ndarray,
    t: float,
) -> np.ndarray:
    """
    Vectorised local wind speed (m/s) at *t* for each point.

    ``syn_speed·(0.7 + 0.5·coverage) + storm_gust·storm_wind_max``.

    Docs: docs/systems/world.weather._impl.md
    """
    _, syn_speed = ws.synoptic.wind(t)
    return syn_speed * (0.7 + 0.5 * coverage) + storm_gust * ws._storm_wind_max


def emergent_fog(
    ws: WeatherSystem,
    pts: np.ndarray,
    coverage: np.ndarray,
    storm_gust: np.ndarray,
    t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Emergent fog coefficient (1/m) and relative humidity at each point.

    Closed-form condensation: local relative humidity (per-day baseline +
    recent rain + ground wetness) condenses into ground fog where it exceeds
    the temperature-dependent saturation humidity, gated off by wind.
    Returns ``(emergent_fog, humidity)`` — both shape ``(N,)``.

    Docs: docs/systems/world.weather._impl.md
    """
    day = int(t // _DAY_S)
    tod = t - day * _DAY_S
    tod_h = tod / 3600.0
    h_cur = humidity_base(day, ws._config)
    h_prev = humidity_base(day - 1, ws._config) if day > 0 else h_cur
    blend = 0.5 - 0.5 * math.cos(math.pi * min(tod / 3600.0, 1.0))
    h_base = h_prev + (h_cur - h_prev) * blend

    temp = np.full(pts.shape[0], temperature(ws, tod_h))
    wind_sp = local_wind_speed(ws, coverage, storm_gust, t)
    rain_recent = wetness_at(ws, pts, t, use_recent=True)
    wet = wetness_at(ws, pts, t, use_recent=False)
    hum = relative_humidity(rain_recent, wet, h_base, ws._config)
    fog = _emergent_fog_fn(hum, temp, wind_sp, ws._config)
    return fog, hum


def wetness_at(
    ws: WeatherSystem,
    pts: np.ndarray,
    t: float,
    *,
    use_recent: bool = False,
) -> np.ndarray:
    """
    Closed-form ground wetness (or recent-rain measure) at each query point.

    Fixed-offset exponential quadrature over the analytic rain history.  When
    ``use_recent=False`` (the default) uses the wetness window
    (``_wet_tau``/``_wet_step``/``_wet_samples``); when ``use_recent=True``
    uses the longer recent-rain window (``_recent_tau`` etc.) for the emergent
    humidity model.  Pure function of (seed, time, position).

    Docs: docs/systems/world.weather._impl.md
    """
    if use_recent:
        tau = ws._recent_tau
        step = ws._recent_step
        n_samples = ws._recent_samples
    else:
        tau = ws._wet_tau
        step = ws._wet_step
        n_samples = ws._wet_samples

    acc = np.zeros(pts.shape[0])
    for k in range(1, n_samples + 1):
        tk = t - k * step
        if tk < 0.0:
            break
        weight = (step / tau) * math.exp(-k * step / tau)
        _, _, rain_k, _, _ = sample_core(ws, pts, tk)
        acc += weight * rain_k
    np.clip(acc, 0.0, 1.0, out=acc)
    return acc


def sample_fields(
    ws: WeatherSystem,
    points_xy: np.ndarray,
    t_abs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised core sampling: the raster channels at every query point.

    Single source of truth for the spatial weather field — ``sample_local``
    calls it with one point; the weather-map raster (M3) over a grid.  Returns
    ``(coverage, density, rain, fog, storm_gust)``.

    Docs: docs/systems/world.weather._impl.md
    """
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    t = float(t_abs)
    coverage, density, rain, fog_bank, storm_gust = sample_core(ws, pts, t)
    ef, _ = emergent_fog(ws, pts, coverage, storm_gust, t)
    fog = np.minimum(fog_bank + ef, ws._fog_max)
    return coverage, density, rain, fog, storm_gust


def sample_local(
    ws: WeatherSystem,
    pos_xy: tuple[float, float],
    t_abs: float | None,
) -> LocalWeather:
    """
    Sample the full natural weather at world position *pos_xy* and *t_abs*.

    Single-point wrapper over :func:`sample_fields` that also resolves wind,
    wetness, and temperature into a complete :class:`LocalWeather`.

    Docs: docs/systems/world.weather._impl.md
    """
    t = (
        float(t_abs)
        if t_abs is not None
        else (ws._last_abs_t if ws._last_abs_t is not None else 0.0)
    )
    pt = np.array([[float(pos_xy[0]), float(pos_xy[1])]], dtype=np.float64)
    cov, den, rain, fog_bank, gust = sample_core(ws, pt, t)
    coverage = float(cov[0])

    ef, hum = emergent_fog(ws, pt, cov, gust, t)
    fog = float(min(fog_bank[0] + ef[0], ws._fog_max))

    wind_dir, _ = ws.synoptic.wind(t)
    wind_sp = float(local_wind_speed(ws, cov, gust, t)[0])
    tod_h = (t % _DAY_S) / 3600.0
    return LocalWeather(
        cloud_coverage=coverage,
        cloud_density=float(den[0]),
        fog_density=fog,
        rain_intensity=float(rain[0]),
        wind_dir=wind_dir,
        wind_speed=wind_sp,
        humidity=float(hum[0]),
        wetness=float(wetness_at(ws, pt, t)[0]),
        temperature_c=temperature(ws, tod_h),
    )
