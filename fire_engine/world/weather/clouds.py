"""
weather/clouds.py — WMO cloud genera as a layered appearance model (M9).

Real skies stack clouds at distinct altitudes with distinct shapes: thin
wispy cirrus high up, lumpy stratocumulus low down, a towering cumulonimbus
under a thunderstorm.  The volumetric weather sim
(:mod:`fire_engine.world.weather.system`) produces a *continuous* local state —
``coverage``, ``density``, ``precip`` (rain) and the day :class:`Regime` — but
the renderer historically drew one uniform cloud slab.  This module maps that
continuous state onto the **World Meteorological Organization genera** and
expresses each as an altitude band the cloud raymarcher can render distinctly,
so the sky reads as *weather* rather than one grey sheet.

Design — closed-form, save-free, GPU-mirror-able
------------------------------------------------
Everything here is a **pure, vectorised function of the already-sampled
weather fields** (coverage/density/precip + regime), which are themselves
closed-form functions of (seed, time, position).  It therefore:

* costs **zero save bytes** (re-derives from the sim like everything in
  ``weather/``);
* is **consistent between the weather-map raster and a local sample** by
  construction — the same fields in give the same layers out, whether the
  caller is :class:`~fire_engine.world.weather.WeatherMap` over a grid or
  :meth:`~fire_engine.world.weather.WeatherSystem.sample_local` at one point;
* is **mirrored on the GPU without new texture data**: the cloud shader
  (:file:`world/shaders/cloud_volumetric.frag`) derives the very same three
  altitude bands from the existing ``coverage``/``density``/``precip``
  weather-map channels plus the band-altitude/weight uniforms pushed from
  config.  We deliberately do **not** pack a genus code into a spare
  sub-channel — that would touch the M3/M4 weather-map 4-channel packing
  contract (locked, shared).  See ``docs/systems/weather.md`` "WMO genera".

Genus subset (8 — the tractable set the renderer can express)
-------------------------------------------------------------
======================  =====  ======================================
:class:`CloudGenus`     Band   Look
======================  =====  ======================================
CIRRUS                  high   thin wispy stretched streaks
CIRROSTRATUS            high   thin high veil (full-sky halo sheet)
ALTOCUMULUS             mid    mid-level lumpy patches
ALTOSTRATUS             mid    mid-level featureless grey sheet
STRATOCUMULUS           low    low lumpy sheet (broken to overcast)
STRATUS                 low    low flat featureless sheet / rain layer
CUMULUS                 low    low puffy heaps (fair → towering)
CUMULONIMBUS            low    storm tower, dark base, anvil top
======================  =====  ======================================

(NIMBOSTRATUS — the low thick rain layer — is folded into STRATUS with high
``precip``; it shares STRATUS's flat low band and only differs by rain, which
the existing precip channel already drives.  Folding it keeps the subset at a
size the three-band renderer can actually express without a ninth code.)

Units: altitudes/thicknesses in **meters (world Z)**, all weights/scales 0–1
or 1/m as noted.  Z-up.

Docs: docs/systems/world.weather.md
"""

from __future__ import annotations

from typing import cast

import numpy as np

from fire_engine.core.config import Config
from fire_engine.world.weather.types import (  # re-exported below
    CloudBand,
    CloudGenus,
    CloudLayers,
    Regime,
)

__all__ = [
    "BAND_HIGH",
    "BAND_LOW",
    "BAND_MID",
    "CloudBand",
    "CloudGenus",
    "CloudLayers",
    "classify_genus",
    "cloud_layers",
]

# CloudGenus, CloudBand, and CloudLayers are defined in types.py and
# re-exported here for backward-compatible imports.  The structure checker
# counts only ``class X:`` *definitions*; these imports do not count.

#: Stable band index constants (also the GPU band order: 0=high, 1=mid, 2=low).
BAND_HIGH: int = int(CloudBand.HIGH)
BAND_MID: int = int(CloudBand.MID)
BAND_LOW: int = int(CloudBand.LOW)

#: Which altitude band each genus renders in.
BAND_OF: dict[CloudGenus, CloudBand] = {
    CloudGenus.CIRRUS: CloudBand.HIGH,
    CloudGenus.CIRROSTRATUS: CloudBand.HIGH,
    CloudGenus.ALTOCUMULUS: CloudBand.MID,
    CloudGenus.ALTOSTRATUS: CloudBand.MID,
    CloudGenus.STRATOCUMULUS: CloudBand.LOW,
    CloudGenus.STRATUS: CloudBand.LOW,
    CloudGenus.CUMULUS: CloudBand.LOW,
    CloudGenus.CUMULONIMBUS: CloudBand.LOW,
}


def _smoothstep(x: float | np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Vectorised Hermite smoothstep clamped to [0, 1]."""
    if hi <= lo:
        return np.where(np.asarray(x) < lo, 0.0, 1.0)
    t = np.clip((np.asarray(x, dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def classify_genus(
    coverage: float | np.ndarray,
    density: float | np.ndarray,
    precip: float | np.ndarray,
    regime: Regime,
) -> tuple[CloudGenus, CloudGenus, CloudGenus] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Discrete dominant genus per band from a continuous weather sample.

    Pure function of the sampled fields + the day :class:`Regime`.  Returns,
    for the **scalar** call, a ``(CloudGenus, CloudGenus, CloudGenus)`` tuple
    ``(high, mid, low)``; for **array** inputs, three ndarrays of
    :class:`CloudGenus` (``dtype=object``) of the broadcast shape.  The genus
    is purely a *label* of which look each band wears — the continuous layer
    weights come from :func:`cloud_layers`.

    Band rules (first match wins per band; ``cov``/``den``/``precip`` are the
    sampled scalars, ``regime`` the strong hint):

    * **Low band** — the weather's "main event":
        - ``precip > 0.45``                 → CUMULONIMBUS (storm tower)
        - ``precip > 0.05``                 → STRATUS      (rain layer / nimbostratus)
        - ``regime is FRONTAL`` & cov>0.55  → STRATOCUMULUS (broken frontal deck)
        - ``cov > 0.55``                    → STRATOCUMULUS (overcast lumps)
        - ``cov > 0.18``                    → CUMULUS       (fair-weather heaps)
        - else                              → STRATUS       (degenerate thin)
    * **Mid band** — present once there is moderate cover:
        - ``cov > 0.55`` & ``den > 0.55``   → ALTOSTRATUS (grey sheet)
        - ``cov > 0.30``                    → ALTOCUMULUS (lumpy patches)
        - else                              → ALTOCUMULUS (thin patches)
    * **High band** — the thin residual / fair-weather cirrus:
        - ``cov > 0.55``                    → CIRROSTRATUS (thickening veil)
        - else                              → CIRRUS       (wispy streaks)

    HIGH_PRESSURE residual coverage therefore reads as CIRRUS up high (the
    classic fair-weather "mares' tails"); a FRONTAL overcast stacks
    cirrostratus → altostratus → stratocumulus; a THUNDERSTORM (high precip)
    puts a CUMULONIMBUS in the low band.

    Parameters
    ----------
    coverage, density, precip : float | np.ndarray — 0–1 sampled fields.
    regime : Regime — the day's synoptic regime (strong hint).

    Returns
    -------
    tuple | (ndarray, ndarray, ndarray) — (high, mid, low) genus per band.
    """
    cov = np.asarray(coverage, dtype=np.float64)
    den = np.asarray(density, dtype=np.float64)
    pre = np.asarray(precip, dtype=np.float64)
    scalar = cov.ndim == 0 and den.ndim == 0 and pre.ndim == 0
    cov, den, pre = np.broadcast_arrays(cov, den, pre)
    frontal = regime is Regime.FRONTAL

    # --- low band ---
    low = np.empty(cov.shape, dtype=object)
    low[...] = CloudGenus.STRATUS
    low[cov > 0.18] = CloudGenus.CUMULUS
    strato = (cov > 0.55) | (frontal & (cov > 0.45))
    low[strato] = CloudGenus.STRATOCUMULUS
    low[pre > 0.05] = CloudGenus.STRATUS  # rain layer
    low[pre > 0.45] = CloudGenus.CUMULONIMBUS  # storm tower

    # --- mid band ---
    mid = np.empty(cov.shape, dtype=object)
    mid[...] = CloudGenus.ALTOCUMULUS
    mid[(cov > 0.55) & (den > 0.55)] = CloudGenus.ALTOSTRATUS

    # --- high band ---
    high = np.empty(cov.shape, dtype=object)
    high[...] = CloudGenus.CIRRUS
    high[cov > 0.55] = CloudGenus.CIRROSTRATUS

    if scalar:
        return (CloudGenus(high.item()), CloudGenus(mid.item()), CloudGenus(low.item()))
    return high, mid, low


def _band_altitudes(config: Config) -> tuple[np.ndarray, np.ndarray]:
    """Per-band (base_altitude_m, thickness_m) arrays from config, len-3."""
    base = np.array(
        [
            float(config.cloud_genera_high_alt_m),
            float(config.cloud_genera_mid_alt_m),
            float(config.cloud_genera_low_alt_m),
        ],
        dtype=np.float64,
    )
    thick = np.array(
        [
            float(config.cloud_genera_high_thick_m),
            float(config.cloud_genera_mid_thick_m),
            float(config.cloud_genera_low_thick_m),
        ],
        dtype=np.float64,
    )
    return base, thick


def cloud_layers(
    coverage: float,
    density: float,
    precip: float,
    regime: Regime,
    config: Config,
) -> CloudLayers:
    """
    Map one (scalar) weather sample to per-band cloud-layer parameters.

    The continuous companion of :func:`classify_genus`: it picks the dominant
    genus per band (for naming) and resolves the **continuous** per-band
    coverage / density / detail weights the renderer marches.  Pure,
    deterministic function of the inputs (+ config tunables) — no randomness,
    no state, no save bytes.  Continuous in the inputs (built from smoothsteps
    and lerps), so a sweep of coverage/precip yields layer params with no
    jumps — required so a cell drifting overhead doesn't pop the sky.

    Band coverage shaping
    ---------------------
    * **Low** band carries the bulk of the weather: its coverage ≈ the sampled
      ``coverage`` (the main deck), boosted by precip so a storm fills the low
      sky.  Genus CUMULONIMBUS (precip→high) gets the deepest, densest slab.
    * **Mid** band fades in with moderate-to-high coverage (``smoothstep`` from
      ~0.30) — clear/fair-weather skies have little mid cloud; frontal overcast
      stacks a full altostratus deck.
    * **High** band carries a thin residual that is *present even in fair
      weather* (the HIGH_PRESSURE cirrus hint): a small floor plus a coverage
      term, but always **low density** (ice cloud is thin) and stretched
      (``detail_scale`` low → smooth streaks).

    Parameters
    ----------
    coverage : float — 0–1 sampled sky-fill.
    density : float — 0–1 sampled cloud opacity.
    precip : float — 0–1 sampled rain strength.
    regime : Regime — the day's synoptic regime (genus hint).
    config : Config — reads the ``cloud_genera_*`` band/weight tunables.

    Returns
    -------
    CloudLayers — fixed length-3 (high, mid, low) bundle.

    Example
    -------
    >>> from fire_engine.core import load_config
    >>> from fire_engine.world.weather.cells import Regime
    >>> L = cloud_layers(0.08, 0.30, 0.0, Regime.HIGH_PRESSURE, load_config())
    >>> float(L.coverage[2]) < 0.3        # near-clear low deck
    True
    >>> float(L.density[0]) < 0.5         # cirrus is always thin
    True
    """
    cov = float(np.clip(coverage, 0.0, 1.0))
    den = float(np.clip(density, 0.0, 1.0))
    pre = float(np.clip(precip, 0.0, 1.0))

    _genus = cast(tuple[CloudGenus, CloudGenus, CloudGenus], classify_genus(cov, den, pre, regime))
    g_high, g_mid, g_low = _genus
    base_alt, thick = _band_altitudes(config)

    w_high = float(config.cloud_genera_high_cov_weight)
    w_mid = float(config.cloud_genera_mid_cov_weight)
    floor_high = float(config.cloud_genera_high_cov_floor)
    den_high = float(config.cloud_genera_high_density)
    detail = np.array(
        [
            float(config.cloud_genera_high_detail_scale),
            float(config.cloud_genera_mid_detail_scale),
            float(config.cloud_genera_low_detail_scale),
        ],
        dtype=np.float64,
    )

    # Use scalar smoothsteps (the array helper works fine on 0-d too).
    def ss(x: float, lo: float, hi: float) -> float:
        return float(_smoothstep(np.float64(x), lo, hi))

    # --- LOW band: the main deck. Coverage ≈ sampled coverage, lifted by rain
    #     so a storm fills the low sky even if ambient coverage lags. Density
    #     rises with sampled density and precip (storm bases are dark/dense).
    low_cov = float(np.clip(cov + 0.35 * pre, 0.0, 1.0))
    low_den = float(np.clip(den + 0.25 * pre, 0.0, 1.0))
    # CUMULONIMBUS towers: deepen + darken the slab so it reads as a wall.
    cb = ss(pre, 0.30, 0.60)  # 0 → 1 as it becomes a storm
    low_thick = thick[BAND_LOW] * (1.0 + 0.8 * cb)
    low_den = float(np.clip(low_den + 0.10 * cb, 0.0, 1.0))

    # --- MID band: fades in with moderate+ coverage; altostratus (grey sheet)
    #     thickens with density.
    mid_present = ss(cov, 0.30, 0.65)
    mid_cov = float(np.clip(w_mid * cov * mid_present, 0.0, 1.0))
    mid_den = float(np.clip(0.45 * den + 0.35 * cov, 0.0, 1.0)) * mid_present

    # --- HIGH band: thin residual present even fair-weather (cirrus). A small
    #     floor + a coverage term, always low density, stretched detail.
    high_cov = float(np.clip(floor_high + w_high * cov, 0.0, 1.0))
    high_den = float(np.clip(den_high * (0.6 + 0.4 * cov), 0.0, 1.0))

    coverage_arr = np.array([high_cov, mid_cov, low_cov], dtype=np.float64)
    density_arr = np.array([high_den, mid_den, low_den], dtype=np.float64)
    thick_arr = thick.copy()
    thick_arr[BAND_LOW] = low_thick

    return CloudLayers(
        genus_high=g_high,
        genus_mid=g_mid,
        genus_low=g_low,
        base_altitude_m=base_alt,
        thickness_m=thick_arr,
        coverage=coverage_arr,
        density=density_arr,
        detail_scale=detail,
    )
