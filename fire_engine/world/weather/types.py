"""
weather/types.py — Trivial support types and enums for the weather package.

Grouping module (exempt from the one-class rule): holds all @dataclass /
Enum / StrEnum support types that are shared across the weather sub-system.
Behavioural classes (WeatherSystem, StormCell, Synoptic, WeatherMap) live
in their own focused modules.

Docs: docs/systems/world.weather.md
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum

import numpy as np

__all__ = [
    "CellKind",
    "CloudBand",
    "CloudGenus",
    "CloudLayers",
    "LocalWeather",
    "Regime",
]


# ---------------------------------------------------------------------------
# Cell / regime enums  (moved from cells.py)
# ---------------------------------------------------------------------------


class CellKind(StrEnum):
    """
    The four kinds of storm cell.  ``str`` mixin so ``.value`` round-trips
    through saves and event payloads as a plain string.

    SHOWER
        Light-to-moderate rain, no lightning.
    THUNDERSTORM
        Heavy rain + lightning (M7) + a strong core gust; biased larger.
    CLOUD_BANK
        Clouds only — coverage/density, no precipitation.
    FOG_BANK
        Ground fog — raises the local fog coefficient, little cloud cover.

    Docs: docs/systems/world.weather.md
    """

    SHOWER = "shower"
    THUNDERSTORM = "thunderstorm"
    CLOUD_BANK = "cloud_bank"
    FOG_BANK = "fog_bank"


class Regime(StrEnum):
    """Per-day synoptic regime — sets ambient sky and the cell spawn mix.

    Docs: docs/systems/world.weather.md
    """

    HIGH_PRESSURE = "high_pressure"
    MIXED = "mixed"
    FRONTAL = "frontal"


# ---------------------------------------------------------------------------
# Cloud enums  (moved from clouds.py)
# ---------------------------------------------------------------------------


class CloudGenus(StrEnum):
    """
    The WMO cloud genera this model expresses.  ``str`` mixin so ``.value``
    round-trips through events/UI as a plain string (mirrors the
    :class:`CellKind` / :class:`WeatherType` convention).

    The genera are grouped by altitude band (see :data:`BAND_OF` in clouds.py):

    CIRRUS / CIRROSTRATUS
        **High** band (~6–8 km scaled): thin, ice-crystal clouds.  CIRRUS is
        wispy stretched streaks; CIRROSTRATUS is a full-sky thin veil.
    ALTOCUMULUS / ALTOSTRATUS
        **Mid** band (~2.5–4 km scaled): ALTOCUMULUS is broken lumpy patches,
        ALTOSTRATUS a featureless grey mid-level sheet.
    STRATOCUMULUS / STRATUS / CUMULUS / CUMULONIMBUS
        **Low** band (~0.5–2 km scaled): STRATOCUMULUS lumpy low sheet,
        STRATUS a flat low sheet (also the rain-layer / nimbostratus role at
        high precip), CUMULUS fair-weather → towering heaps, CUMULONIMBUS the
        storm tower with a dark rain base and an anvil top.

    Docs: docs/systems/world.weather.md
    """

    CIRRUS = "cirrus"
    CIRROSTRATUS = "cirrostratus"
    ALTOCUMULUS = "altocumulus"
    ALTOSTRATUS = "altostratus"
    STRATOCUMULUS = "stratocumulus"
    STRATUS = "stratus"
    CUMULUS = "cumulus"
    CUMULONIMBUS = "cumulonimbus"


class CloudBand(int, Enum):
    """Altitude band a genus lives in.  ``int`` so it indexes layer arrays.

    Docs: docs/systems/world.weather.md
    """

    HIGH = 0
    MID = 1
    LOW = 2


# ---------------------------------------------------------------------------
# Cloud layer result type  (moved from clouds.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CloudLayers:
    """
    Per-band layer parameters for one weather sample — what the renderer needs
    to draw three altitude bands with genus-appropriate look.

    All arrays are length-3, indexed by :class:`CloudBand` (0=high, 1=mid,
    2=low), so the whole struct is a fixed-shape, GPU-friendly bundle.  A band
    with ``coverage[b] == 0`` is simply not drawn.

    Attributes
    ----------
    genus_high, genus_mid, genus_low : CloudGenus
        The dominant genus chosen for each band (the band's *name*; the
        renderer reads the numeric params, this is for UI / tests / docs).
    base_altitude_m : np.ndarray, shape (3,), float64
        World-Z base altitude (meters) of each band's cloud slab.  Strictly
        increasing (low < mid < high) — an invariant the tests pin.
    thickness_m : np.ndarray, shape (3,), float64
        Vertical thickness (meters) of each band's slab.
    coverage : np.ndarray, shape (3,), float64
        0–1 sky-fill weight per band: how much of that band is filled.  This
        is the per-band coverage the renderer thresholds against.
    density : np.ndarray, shape (3,), float64
        0–1 opacity per band (high cirrus is thin → low; low storm → high).
    detail_scale : np.ndarray, shape (3,), float64
        Relative turbulence / detail frequency per band (cirrus stretched and
        smooth → low; cumulus billowy → high).  A multiplier on the renderer's
        base noise frequency.

    Example
    -------
    >>> from fire_engine.core import load_config
    >>> from fire_engine.world.weather.types import Regime
    >>> from fire_engine.world.weather.clouds import cloud_layers
    >>> L = cloud_layers(0.9, 0.9, 1.0, Regime.FRONTAL, load_config())
    >>> L.genus_low.value
    'cumulonimbus'
    >>> bool(L.base_altitude_m[0] > L.base_altitude_m[2])   # high above low
    True

    Docs: docs/systems/world.weather.md
    """

    genus_high: CloudGenus
    genus_mid: CloudGenus
    genus_low: CloudGenus
    base_altitude_m: np.ndarray
    thickness_m: np.ndarray
    coverage: np.ndarray
    density: np.ndarray
    detail_scale: np.ndarray

    def genus_for_band(self, band: CloudBand | int) -> CloudGenus:
        """The chosen genus for *band* (0=high, 1=mid, 2=low).

        Docs: docs/systems/world.weather.md
        """
        return (self.genus_high, self.genus_mid, self.genus_low)[int(band)]


# ---------------------------------------------------------------------------
# Local weather snapshot  (moved from system.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalWeather:
    """
    Continuous weather sampled at one world position and instant.

    The first six fields match the corresponding ``SkyState`` fields exactly
    (same names, units, meaning) so the sky composer fills ``SkyState``
    one-to-one.  ``humidity`` is the live emergent relative humidity (M5);
    ``wetness`` and ``temperature_c`` are already live.

    Attributes
    ----------
    cloud_coverage : float — 0–1 fraction of sky filled.
    cloud_density : float — 0–1 cloud opacity/darkness.
    fog_density : float — exponential fog coefficient, 1/m (0.0008 ≈ clear).
    rain_intensity : float — 0–1 (0 = dry, 1 = torrential).
    wind_dir : tuple[float, float] — unit XY direction the wind blows TOWARD.
    wind_speed : float — m/s.
    humidity : float — 0–1 emergent relative humidity (base + recent rain +
        ground wetness).
    wetness : float — 0–1 ground wetness.
    temperature_c : float — local air temperature, °C.

    Example
    -------
    >>> lw = LocalWeather(0.2, 0.4, 0.0008, 0.0, (1.0, 0.0), 3.0)
    >>> lw.temperature_c
    12.0

    Docs: docs/systems/world.weather.md
    """

    cloud_coverage: float
    cloud_density: float
    fog_density: float
    rain_intensity: float
    wind_dir: tuple[float, float]
    wind_speed: float
    humidity: float = 0.5
    wetness: float = 0.0
    temperature_c: float = 12.0
