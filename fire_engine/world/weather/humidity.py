"""
weather/humidity.py — Emergent humidity and condensation-driven ground fog.

Fog in this engine is **not** a selectable weather state — it *condenses* from
conditions.  Air holds moisture (relative humidity rises after rain and over wet
ground); cold air holds less than warm air (the saturation humidity rises with
temperature); when humidity climbs past saturation in calm air the excess
condenses into ground fog.  So a calm, humid night after an evening shower grows
ground fog through the cool pre-dawn hours, and the rising sun burns it off as
the warming air's saturation humidity climbs back above the actual humidity.

Every quantity here is a **closed-form pure function of (world_seed, game time,
position)** — vectorised over an ``(N,)`` batch — so the weather-map raster and
the per-frame local sample both pick up emergent fog for free, and it costs zero
save bytes (recompute on load, like all natural weather).

All formula tunables live in :class:`fire_engine.core.config.Config`
(``weather_humidity_*`` / ``weather_fog_*``).  The functions take the resolved
config so they stay free of magic numbers and importable headlessly.

Units: relative humidity 0–1; temperature °C; wind m/s; fog coefficient 1/m.

Example
-------
>>> import numpy as np
>>> from fire_engine.core import load_config
>>> cfg = load_config()
>>> T = np.array([5.0, 19.0])                 # cool pre-dawn / warm afternoon
>>> np.round(saturation_humidity(T, cfg), 3)  # warmer air saturates higher
array([0.63 , 0.784])
"""

from __future__ import annotations

import numpy as np

from fire_engine.core.config import Config
from fire_engine.core.rng import for_domain

__all__ = [
    "condense_fraction",
    "emergent_fog",
    "humidity_base",
    "relative_humidity",
    "saturation_humidity",
    "wind_gate",
]


def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Vectorised Hermite smoothstep clamped to [0, 1] (array ``x``).

    Returns ``0`` for ``x ≤ lo``, ``1`` for ``x ≥ hi``, a smooth S-curve
    between.  ``hi <= lo`` degenerates to a hard step at ``lo``.
    """
    x = np.asarray(x, dtype=np.float64)
    if hi <= lo:
        return (x >= lo).astype(np.float64)
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def humidity_base(day: int, config: Config) -> float:
    """
    The day's calm-air baseline relative humidity (0–1), seeded per day.

    Drawn from ``for_domain("weather", "humidity", day)`` uniformly in
    ``[weather_humidity_base_min, weather_humidity_base_max]`` — a gentle
    per-day band so even rain-free days vary a little in mugginess.  Pure
    function of (world_seed, day); identical across processes and reloads.

    Parameters
    ----------
    day : int — in-game day number.
    config : Config — reads ``weather_humidity_base_min`` / ``..._max``.

    Returns
    -------
    float — baseline relative humidity in [base_min, base_max].
    """
    lo = float(config.weather_humidity_base_min)
    hi = float(config.weather_humidity_base_max)
    rng = for_domain("weather", "humidity", int(day))
    return float(rng.uniform(lo, hi))


def relative_humidity(
    rain_recent: np.ndarray,
    wetness: np.ndarray,
    h_base: float,
    config: Config,
) -> np.ndarray:
    """
    Relative humidity 0–1 at each query point.

    ``humidity = clamp(h_base + rain_gain·rain_recent + wetness_gain·wetness, 0,
    1)`` — the calm per-day baseline plus moisture the air picked up from recent
    precipitation and from evaporating off wet ground.

    Parameters
    ----------
    rain_recent : np.ndarray — shape ``(N,)`` recent-precip measure 0–1 at each
        point (recent-rain quadrature; see :meth:`WeatherSystem.rain_recent_at`).
    wetness : np.ndarray — shape ``(N,)`` ground wetness 0–1 (see
        :meth:`WeatherSystem.wetness_at`).
    h_base : float — the day's calm-air baseline humidity (0–1); see
        :func:`humidity_base`.  The caller cosine-blends it across the midnight
        hand-off so humidity never snaps at 00:00.
    config : Config — reads ``weather_humidity_rain_gain`` /
        ``weather_humidity_wetness_gain``.

    Returns
    -------
    np.ndarray — shape ``(N,)`` relative humidity clamped to [0, 1].
    """
    rain_gain = float(config.weather_humidity_rain_gain)
    wet_gain = float(config.weather_humidity_wetness_gain)
    h = (
        float(h_base)
        + rain_gain * np.asarray(rain_recent, dtype=np.float64)
        + wet_gain * np.asarray(wetness, dtype=np.float64)
    )
    return np.clip(h, 0.0, 1.0)


def saturation_humidity(temperature_c: np.ndarray, config: Config) -> np.ndarray:
    """
    Saturation relative humidity 0.5–1.0 as a function of temperature.

    Warm air holds more moisture, so it must reach a *higher* relative humidity
    before the excess condenses: ``h_sat`` **rises with temperature**.  Linear in
    ``T`` about a reference, clamped to [0.5, 1.0]::

        h_sat = clamp(sat_base + sat_slope_per_c·(T − sat_ref_c), 0.5, 1.0)

    Tuned (default config) so a cool pre-dawn (T≈4–6 °C) saturates near
    humidity≈0.63 and a warm afternoon (T≈18–20 °C) needs ≈0.78 — i.e. an
    evening-rain-dampened point fogs over readily in the cold pre-dawn and
    rarely in the heat of the day.

    Parameters
    ----------
    temperature_c : np.ndarray — shape ``(N,)`` (or scalar) air temperature °C.
    config : Config — reads ``weather_fog_sat_base`` / ``weather_fog_sat_ref_c``
        / ``weather_fog_sat_slope_per_c``.

    Returns
    -------
    np.ndarray — shape ``(N,)`` saturation humidity clamped to [0.5, 1.0].
    """
    base = float(config.weather_fog_sat_base)
    ref = float(config.weather_fog_sat_ref_c)
    slope = float(config.weather_fog_sat_slope_per_c)
    t = np.asarray(temperature_c, dtype=np.float64)
    return np.clip(base + slope * (t - ref), 0.5, 1.0)


def condense_fraction(humidity: np.ndarray, h_sat: np.ndarray, config: Config) -> np.ndarray:
    """
    Condensation fraction 0–1: how far humidity has pushed past saturation.

    ``smoothstep(humidity − h_sat, 0, condense_band)`` — zero at/below
    saturation, ramping smoothly to full over a ``weather_fog_condense_band``
    overshoot so fog thickens gradually rather than snapping on.

    Parameters
    ----------
    humidity : np.ndarray — shape ``(N,)`` relative humidity 0–1.
    h_sat : np.ndarray — shape ``(N,)`` saturation humidity 0–1.
    config : Config — reads ``weather_fog_condense_band``.

    Returns
    -------
    np.ndarray — shape ``(N,)`` condensation fraction in [0, 1].
    """
    band = float(config.weather_fog_condense_band)
    over = np.asarray(humidity, dtype=np.float64) - np.asarray(h_sat, dtype=np.float64)
    return _smoothstep(over, 0.0, band)


def wind_gate(wind_speed: np.ndarray, config: Config) -> np.ndarray:
    """
    Wind gate 0–1: wind mixes ground fog away, so fog only survives in calm air.

    ``1 − smoothstep(wind_speed, fog_wind_full_ms, fog_wind_none_ms)`` — full
    (``1``) below ``weather_fog_wind_full_ms`` (~1 m/s), fading to none (``0``)
    above ``weather_fog_wind_none_ms`` (~3 m/s).

    Parameters
    ----------
    wind_speed : np.ndarray — shape ``(N,)`` (or scalar) local wind speed m/s.
    config : Config — reads ``weather_fog_wind_full_ms`` / ``..._none_ms``.

    Returns
    -------
    np.ndarray — shape ``(N,)`` gate in [0, 1].
    """
    full = float(config.weather_fog_wind_full_ms)
    none = float(config.weather_fog_wind_none_ms)
    return 1.0 - _smoothstep(np.asarray(wind_speed, dtype=np.float64), full, none)


def emergent_fog(
    humidity: np.ndarray,
    temperature_c: np.ndarray,
    wind_speed: np.ndarray,
    config: Config,
) -> np.ndarray:
    """
    Emergent ground-fog coefficient (1/m) condensed from the local conditions.

    ``emergent_fog = fog_emergent_max · condense · gate`` where ``condense`` is
    :func:`condense_fraction` (humidity past the temperature-dependent
    saturation) and ``gate`` is :func:`wind_gate` (calm air only).  This is added
    onto the existing fog channel (baseline + FOG_BANK contributions) and the sum
    is capped at ``weather_fog_max_density`` by the caller.

    Parameters
    ----------
    humidity : np.ndarray — shape ``(N,)`` relative humidity 0–1.
    temperature_c : np.ndarray — shape ``(N,)`` air temperature °C.
    wind_speed : np.ndarray — shape ``(N,)`` local wind speed m/s.
    config : Config — reads ``weather_fog_emergent_max`` plus the saturation,
        condense-band, and wind-gate fields.

    Returns
    -------
    np.ndarray — shape ``(N,)`` emergent fog coefficient (1/m), ≥ 0.

    Example
    -------
    >>> import numpy as np
    >>> from fire_engine.core import load_config
    >>> cfg = load_config()
    >>> # Humid, cold, calm → fog; same humidity but windy → none.
    >>> h = np.array([0.95, 0.95]); T = np.array([5.0, 5.0]); w = np.array([0.5, 5.0])
    >>> f = emergent_fog(h, T, w, cfg)
    >>> bool(f[0] > 0.0) and bool(f[1] == 0.0)
    True
    """
    f_max = float(config.weather_fog_emergent_max)
    h_sat = saturation_humidity(temperature_c, config)
    condense = condense_fraction(humidity, h_sat, config)
    gate = wind_gate(wind_speed, config)
    result: np.ndarray = f_max * condense * gate
    return result
