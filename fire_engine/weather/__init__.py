"""
fire_engine/weather — Spatial, volumetric weather simulation (headless).

Layer 1 service, peer of ``sky/`` and ``wind/``.  Owns the synoptic flow
(M1), storm cells / regimes (M2), the weather-map raster (M3), emergent
humidity/fog (M5) and lightning scheduling (M7).  Never imports panda3d;
render bridges live in ``fire_engine/world/``.

See ``docs/systems/weather.md`` for the system contract.
"""

from fire_engine.weather.cells import (
    CellKind,
    Regime,
    StormCell,
    day_regime,
    natural_cells,
    regime_ambient,
)
from fire_engine.weather.classify import WeatherType, classify
from fire_engine.weather.synoptic import Synoptic
from fire_engine.weather.system import LocalWeather, WeatherSystem

__all__ = [
    "Synoptic",
    "CellKind",
    "Regime",
    "StormCell",
    "day_regime",
    "natural_cells",
    "regime_ambient",
    "WeatherType",
    "classify",
    "LocalWeather",
    "WeatherSystem",
]
