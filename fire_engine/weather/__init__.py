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
from fire_engine.weather.clouds import (
    BAND_HIGH,
    BAND_LOW,
    BAND_MID,
    CloudBand,
    CloudGenus,
    CloudLayers,
    classify_genus,
    cloud_layers,
)
from fire_engine.weather.humidity import (
    condense_fraction,
    emergent_fog,
    humidity_base,
    relative_humidity,
    saturation_humidity,
    wind_gate,
)
from fire_engine.weather.synoptic import Synoptic
from fire_engine.weather.system import LocalWeather, WeatherSystem
from fire_engine.weather.weather_map import MAP_CHANNELS, WeatherMap

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
    "CloudGenus",
    "CloudBand",
    "CloudLayers",
    "classify_genus",
    "cloud_layers",
    "BAND_HIGH",
    "BAND_MID",
    "BAND_LOW",
    "humidity_base",
    "relative_humidity",
    "saturation_humidity",
    "condense_fraction",
    "wind_gate",
    "emergent_fog",
    "LocalWeather",
    "WeatherSystem",
    "WeatherMap",
    "MAP_CHANNELS",
]
