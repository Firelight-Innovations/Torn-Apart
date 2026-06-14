"""
fire_engine/world/weather — Spatial, volumetric weather simulation (headless).

Layer 1 service, peer of ``sky/`` and ``wind/``.  Owns the synoptic flow
(M1), storm cells / regimes (M2), the weather-map raster (M3), emergent
humidity/fog (M5) and lightning scheduling (M7).  Never imports panda3d;
render bridges live in ``fire_engine/render/``.

See ``docs/systems/weather.md`` for the system contract.
"""

from fire_engine.world.weather.bolt import BoltGeometry, generate_bolt
from fire_engine.world.weather.cells import (
    CellKind,
    Regime,
    StormCell,
    day_regime,
    natural_cells,
    regime_ambient,
)
from fire_engine.world.weather.classify import WeatherType, classify
from fire_engine.world.weather.clouds import (
    BAND_HIGH,
    BAND_LOW,
    BAND_MID,
    CloudBand,
    CloudGenus,
    CloudLayers,
    classify_genus,
    cloud_layers,
)
from fire_engine.world.weather.humidity import (
    condense_fraction,
    emergent_fog,
    humidity_base,
    relative_humidity,
    saturation_humidity,
    wind_gate,
)
from fire_engine.world.weather.lightning import (
    StrikeParams,
    cell_id_int,
    scheduled_strikes,
)
from fire_engine.world.weather.synoptic import Synoptic
from fire_engine.world.weather.system import LocalWeather, WeatherSystem
from fire_engine.world.weather.weather_map import MAP_CHANNELS, WeatherMap

__all__ = [
    "BAND_HIGH",
    "BAND_LOW",
    "BAND_MID",
    "MAP_CHANNELS",
    "BoltGeometry",
    "CellKind",
    "CloudBand",
    "CloudGenus",
    "CloudLayers",
    "LocalWeather",
    "Regime",
    "StormCell",
    "StrikeParams",
    "Synoptic",
    "WeatherMap",
    "WeatherSystem",
    "WeatherType",
    "cell_id_int",
    "classify",
    "classify_genus",
    "cloud_layers",
    "condense_fraction",
    "day_regime",
    "emergent_fog",
    "generate_bolt",
    "humidity_base",
    "natural_cells",
    "regime_ambient",
    "relative_humidity",
    "saturation_humidity",
    "scheduled_strikes",
    "wind_gate",
]
