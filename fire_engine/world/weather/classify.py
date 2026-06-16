"""
weather/classify.py — Discrete weather label from a continuous local sample.

The spatial weather model produces a continuous :class:`LocalWeather` sample
at the player; the rest of the game (UI, audio, the dev override, save
payloads, :class:`WeatherChangedEvent`) still wants a single human label.
:func:`classify` is that bucketing — a pure function of the sample, with
thresholds ordered most-specific first (fog and storm win over plain rain,
rain over mere cloud cover).

:class:`WeatherType` keeps the **exact** string values the old Markov system
used (``"clear"``…``"storm"``) so every existing consumer — devtools, F6
cycling, save deltas — keeps working unchanged.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a runtime import cycle
    from fire_engine.world.weather.system import LocalWeather

__all__ = ["WeatherType", "classify"]


class WeatherType(str, Enum):
    """
    Discrete weather label.  ``str`` mixin so ``.value`` round-trips through
    saves and :class:`WeatherChangedEvent` payloads as a plain string.  Values
    are identical to the legacy Markov system — do not renumber.
    """

    CLEAR = "clear"
    CLOUDY = "cloudy"
    OVERCAST = "overcast"
    FOG = "fog"
    RAIN = "rain"
    STORM = "storm"


def classify(lw: LocalWeather) -> WeatherType:
    """
    Bucket a :class:`LocalWeather` sample into a :class:`WeatherType`.

    Order matters — the first matching rule wins:

    1. ``fog_density > 0.008``               → FOG   (visibility-limiting haze)
    2. ``rain_intensity > 0.55`` and
       ``wind_speed > 9 m/s``                → STORM (heavy rain + strong wind)
    3. ``rain_intensity > 0.05``             → RAIN
    4. ``cloud_coverage > 0.7``              → OVERCAST
    5. ``cloud_coverage > 0.3``              → CLOUDY
    6. otherwise                             → CLEAR

    Parameters
    ----------
    lw : LocalWeather — the local sample to label.

    Returns
    -------
    WeatherType

    Example
    -------
    >>> from fire_engine.world.weather.system import LocalWeather
    >>> classify(LocalWeather(0.9, 0.9, 0.0, 0.8, (1.0, 0.0), 11.0)).value
    'storm'
    """
    if lw.fog_density > 0.008:
        return WeatherType.FOG
    if lw.rain_intensity > 0.55 and lw.wind_speed > 9.0:
        return WeatherType.STORM
    if lw.rain_intensity > 0.05:
        return WeatherType.RAIN
    if lw.cloud_coverage > 0.7:
        return WeatherType.OVERCAST
    if lw.cloud_coverage > 0.3:
        return WeatherType.CLOUDY
    return WeatherType.CLEAR
