"""
sky/sky_state.py — SkyState snapshot dataclass and the SkySystem composer.

``SkySystem`` is the headless half of the sky/weather feature: once per frame
it reads the game clock, asks :class:`WeatherSystem` for blended weather
parameters, evaluates the celestial geometry and color ramps, and emits one
frozen :class:`SkyState` snapshot.  The render half (``world/``) consumes the
snapshot — this module never touches panda3d.

Art direction (Minecraft × Morrowind)
-------------------------------------
Slightly desaturated, moody, painterly.  All color keyframe tables in this
module key on **sun elevation** (``sun_dir.z`` in [-1, 1]), not on time, so
dawn and dusk share the same warm ramp and seasons can later shift the arc
without touching the palette:

- noon zenith ~(0.30, 0.46, 0.72), noon horizon ~(0.62, 0.72, 0.82)
- dawn/dusk horizon warm amber (1.0, 0.55, 0.28), zenith teal-green
- night zenith near-black indigo (0.02, 0.03, 0.07)
- sun warm amber at the horizon → near-white (1.0, 0.97, 0.90) at noon
- overcast/storm desaturate and darken everything toward gray

Example
-------
    from fire_engine.core import Clock, EventBus, load_config, set_world_seed
    from fire_engine.sky import SkySystem

    cfg = load_config()
    set_world_seed(cfg.world_seed)
    bus = EventBus()
    clock = Clock(fixed_dt=cfg.fixed_dt, bus=bus)

    sky = SkySystem(cfg, clock, bus)
    state = sky.update()                  # call once per frame
    print(state.sun_dir, state.daylight, state.rain_intensity)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fire_engine.core.clock import Clock
from fire_engine.core.config import Config
from fire_engine.core.event_bus import EventBus
from fire_engine.core.math3d import Vec3
from fire_engine.sky.celestial import (
    DAYLIGHT_Z_HI,
    DAYLIGHT_Z_LO,
    color_ramp,
    lerp_color,
    moon_direction,
    smoothstep,
    sun_direction,
)
from fire_engine.sky.weather import WeatherParams, WeatherSystem

__all__ = ["SkyState", "SkySystem", "MOON_CYCLE_DAYS"]


# ---------------------------------------------------------------------------
# Atmosphere lookup table (physical radiance, sampled per frame)
# ---------------------------------------------------------------------------
#
# The single-scattering integrals in sky/atmosphere.py cost milliseconds —
# fine at boot, not per frame.  Physics is seed-independent, so ONE module-
# level LUT over sun elevation serves every SkySystem: 56 elevations, each
# row holding sun_radiance / sky_ambient / zenith / horizon RGB.  Per frame
# we np.interp three scalars per channel — cheap and smooth.

_LUT_SUN_Z = np.linspace(-0.35, 1.0, 56)
_NIGHT_AMBIENT = (0.010, 0.012, 0.022)     # clear moonless-night skylight
_NIGHT_ZENITH = (0.012, 0.016, 0.035)      # display floor for the gradient
_NIGHT_HORIZON = (0.020, 0.026, 0.046)
_MOON_BASE_RADIANCE = (0.060, 0.070, 0.100)  # full moon at the zenith


class _AtmosphereLUT:
    """Boot-time table of atmosphere outputs keyed on ``sin(sun elevation)``."""

    def __init__(self) -> None:
        from fire_engine.sky import atmosphere as atmo
        zs = _LUT_SUN_Z
        self.sun = atmo.sun_radiance(zs)                       # (56, 3)
        # One sky_radiance call per elevation: 48 ambient hemisphere dirs +
        # 1 zenith + 8 just-above-horizon dirs, split afterwards.
        hemi = atmo._hemisphere_dirs(48)
        az = np.radians(np.arange(8) * 45.0)
        horizon_dirs = np.stack(
            [np.cos(az) * 0.998, np.sin(az) * 0.998,
             np.full(8, 0.06)], axis=1)
        dirs = np.vstack([hemi, [[0.0, 0.0, 1.0]], horizon_dirs])
        amb_rows, zen_rows, hor_rows = [], [], []
        for z in zs:
            sun_dir = np.array(
                [math.sqrt(max(1.0 - z * z, 0.0)), 0.0, float(z)])
            L = atmo.sky_radiance(dirs, sun_dir)               # (57, 3)
            amb = (2.0 * math.pi / 48.0) * np.sum(
                L[:48] * hemi[:, 2:3], axis=0) * atmo.AMBIENT_SCALE
            amb_rows.append(amb)
            zen_rows.append(L[48])
            hor_rows.append(L[49:].mean(axis=0))
        self.ambient = np.asarray(amb_rows)                    # (56, 3)
        self.zenith = np.asarray(zen_rows)
        self.horizon = np.asarray(hor_rows)

    def sample(self, table: np.ndarray, z: float) -> tuple[float, float, float]:
        """Linear interpolation of one (56, 3) table at sun elevation ``z``."""
        return (float(np.interp(z, _LUT_SUN_Z, table[:, 0])),
                float(np.interp(z, _LUT_SUN_Z, table[:, 1])),
                float(np.interp(z, _LUT_SUN_Z, table[:, 2])))


_ATMOSPHERE_LUT: _AtmosphereLUT | None = None


def _get_atmosphere_lut() -> _AtmosphereLUT:
    """Build (once per process) and return the shared atmosphere LUT."""
    global _ATMOSPHERE_LUT
    if _ATMOSPHERE_LUT is None:
        _ATMOSPHERE_LUT = _AtmosphereLUT()
    return _ATMOSPHERE_LUT

# ---------------------------------------------------------------------------
# Tuning constants (documented; palette keyframes key on sun_dir.z)
# ---------------------------------------------------------------------------

#: Length of one lunar cycle in game days (0 = new moon, day 15 = full).
MOON_CYCLE_DAYS: int = 30

#: Zenith color ramp — key: sun elevation z.  Keyframes (z, linear RGB):
#:   -1.00 night near-black indigo · -0.24 still full night (matches the
#:   daylight window) · 0.00 teal-green dusk zenith · 0.15/0.40 morning
#:   climb · ≥0.85 the noon blue (flat through noon so midday is stable).
_ZENITH_RAMP = (
    (-1.00, (0.02, 0.03, 0.07)),
    (-0.24, (0.02, 0.03, 0.07)),
    (-0.05, (0.05, 0.09, 0.12)),
    (0.00, (0.10, 0.17, 0.18)),
    (0.15, (0.16, 0.27, 0.38)),
    (0.40, (0.26, 0.41, 0.64)),
    (0.85, (0.30, 0.46, 0.72)),
    (1.00, (0.30, 0.46, 0.72)),
)

#: Horizon color ramp.  Keyframes (z, linear RGB):
#:   -1.00 night · -0.24 faint horizon lift at deep twilight · -0.10
#:   pre-dawn ember glow · 0.00 the warm amber sunrise/sunset band ·
#:   0.12/0.35 amber fading through haze · ≥0.85 noon pale blue-gray.
_HORIZON_RAMP = (
    (-1.00, (0.03, 0.04, 0.08)),
    (-0.24, (0.04, 0.05, 0.10)),
    (-0.10, (0.35, 0.20, 0.18)),
    (0.00, (1.00, 0.55, 0.28)),
    (0.12, (0.92, 0.62, 0.42)),
    (0.35, (0.72, 0.70, 0.72)),
    (0.85, (0.62, 0.72, 0.82)),
    (1.00, (0.62, 0.72, 0.82)),
)

#: Sun disc/light color ramp.  Keyframes (z, linear RGB):
#:   -0.10 dim ember below the horizon (lights nothing — intensity is 0) ·
#:   0.00 warm amber on the horizon · 0.15/0.45 warming toward white ·
#:   1.00 near-white noon.
_SUN_COLOR_RAMP = (
    (-0.10, (0.85, 0.42, 0.22)),
    (0.00, (1.00, 0.55, 0.28)),
    (0.15, (1.00, 0.72, 0.48)),
    (0.45, (1.00, 0.90, 0.76)),
    (1.00, (1.00, 0.97, 0.90)),
)

#: Terrain light scale ramp (clear-weather baseline, weather-dim applied on
#: top).  Keyframes (z, linear RGB multiplier):
#:   ≤-0.24 the night floor — dim cool blue (0.16, 0.19, 0.30) so night
#:   terrain reads as moonlit, never pitch black · -0.05/0.10 climb out of
#:   the blue · 0.30 the warm (1.0, 0.82, 0.62)-tinted dawn/dusk blend ·
#:   0.60 warm white · ≥0.85 full white so clear noon is exactly (1, 1, 1).
_TERRAIN_LIGHT_RAMP = (
    (-1.00, (0.16, 0.19, 0.30)),
    (-0.24, (0.16, 0.19, 0.30)),
    (-0.05, (0.28, 0.27, 0.33)),
    (0.10, (0.62, 0.52, 0.44)),
    (0.30, (0.95, 0.80, 0.62)),
    (0.60, (1.00, 0.96, 0.90)),
    (0.85, (1.00, 1.00, 1.00)),
    (1.00, (1.00, 1.00, 1.00)),
)

#: Weather "grayness" w = clamp(_WEATHER_GRAY_GAIN · coverage · density, 0, 1):
#: clear ≈ 0.05, overcast ≈ 0.78, storm ≈ 1.0.  Drives sky desaturation,
#: darkening and fog-color blending.
_WEATHER_GRAY_GAIN: float = 1.15

#: Sky desaturation amount at w = 1 (lerp toward luminance gray).
_SKY_DESAT: float = 0.75

#: Sky darkening amount at w = 1 (multiplicative).
_SKY_DARKEN: float = 0.40

#: Weather dim factor for terrain light: dim = 1 − a·(cov·den)^b, fitted so
#: clear ≈ 1.00, overcast ≈ 0.75, storm ≈ 0.55 (documented spec targets).
_TERRAIN_DIM_A: float = 0.52
_TERRAIN_DIM_B: float = 1.9

#: Fog base gray (linear RGB) before day/night brightness scaling.
_FOG_GRAY: tuple[float, float, float] = (0.62, 0.66, 0.70)


def _luminance_gray(c: tuple[float, float, float]) -> tuple[float, float, float]:
    """Rec.601 luminance of *c*, replicated to a gray RGB tuple."""
    lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
    return (lum, lum, lum)


def _weathered(
    c: tuple[float, float, float], w: float
) -> tuple[float, float, float]:
    """
    Desaturate + darken a sky color by weather grayness *w* in [0, 1].

    ``w = 0`` returns *c* unchanged; ``w = 1`` pulls 75 % toward luminance
    gray and darkens by 40 % — the overcast/storm "lead sky" look.
    """
    desat = lerp_color(c, _luminance_gray(c), _SKY_DESAT * w)
    k = 1.0 - _SKY_DARKEN * w
    return (desat[0] * k, desat[1] * k, desat[2] * k)


# ---------------------------------------------------------------------------
# SkyState
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SkyState:
    """
    Immutable per-frame sky/weather snapshot consumed by the render layer.

    All directions are unit ``Vec3`` (Z-up, forward = +Y), pointing FROM the
    scene TOWARD the body.  All colors are linear RGB tuples with components
    in 0–1.  Distances meters, time seconds, angles radians.

    Attributes
    ----------
    sun_dir : Vec3 — toward the sun; ``z > 0`` = above the horizon.
    moon_dir : Vec3 — toward the moon (roughly opposite the sun).
    sun_color : tuple — sun light color (warm amber at horizon → near-white).
    sun_intensity : float — 0–1; exactly 0 when the sun is below the horizon,
        dimmed by cloud cover.
    moon_phase : float — 0–1 lunar phase from ``game_day``
        (0 = new, 0.5 = full, cycle = ``MOON_CYCLE_DAYS`` days).
    daylight : float — smooth 0–1 day factor (0 night, 1 midday; 0.5 at
        sunrise/sunset, saturating ~1 game hour past the horizon).
    star_visibility : float — 0–1; 1 = clear night.  Clouds hide stars:
        ``(1 − daylight) · (1 − 0.85 · cloud_coverage)``.
    zenith_color : tuple — sky gradient top color (weather-graded).
    horizon_color : tuple — sky gradient horizon color (weather-graded).
    cloud_coverage : float — 0–1 fraction of cloud cells filled (blended).
    cloud_density : float — 0–1 cloud opacity/darkness (blended).
    fog_density : float — exponential fog coefficient, 1/m (0 = none;
        typical FOG weather ≈ 0.025 → ~120 m visibility).
    fog_color : tuple — horizon color blended toward weather gray.
    rain_intensity : float — 0–1.
    wind_dir : tuple[float, float] — unit XY direction wind blows toward.
    wind_speed : float — m/s.
    terrain_light_scale : tuple — RGB multiplier the renderer applies to
        terrain vertex light: clear day ≈ (1, 1, 1); night floor
        ≈ (0.16, 0.19, 0.30); warm-tinted at dawn/dusk; dimmed further by
        weather (overcast ~0.75×, storm ~0.55×).  Smooth everywhere.
        Used by the CPU lighting backend only; the GPU volumetric pipeline
        reads the three HDR radiance fields below instead.
    sun_radiance : tuple — **linear HDR RGB** direct-sun light reaching the
        ground (atmosphere-transmitted): clear noon ≈ (3.2, 3.0, 2.6);
        strongly orange and dimmer near sunset (the R/B ratio rises as the
        sun drops); smooth twilight tail to exactly 0 at −4° elevation;
        sharply attenuated by cloud cover.  Consumed by the GPU lighting
        pipeline as the sun injection color (`lighting/gpu.py`).
    moon_radiance : tuple — linear HDR RGB moonlight: pale blue-white, full
        moon high in a clear sky ≈ (0.06, 0.07, 0.10), scaled by the phase's
        illuminated fraction and elevation; 0 below the horizon.
    sky_ambient : tuple — linear HDR RGB hemispheric skylight irradiance
        (the GI skylight injection): clear noon ≈ (0.21, 0.40, 0.71);
        warm-gray at sunset; overcast = desaturated at similar luminance;
        clear moonless night ≈ (0.010, 0.012, 0.022) plus a small moonlight
        bump when the moon is up.

    Example
    -------
    >>> state = sky_system.update()
    >>> if state.sun_dir.z > 0.0:
    ...     render_sun(state.sun_dir, state.sun_color, state.sun_intensity)
    """

    # celestial
    sun_dir: Vec3
    moon_dir: Vec3
    sun_color: tuple[float, float, float]
    sun_intensity: float
    moon_phase: float
    daylight: float
    star_visibility: float
    # sky gradient
    zenith_color: tuple[float, float, float]
    horizon_color: tuple[float, float, float]
    # weather (already blended/smoothed)
    cloud_coverage: float
    cloud_density: float
    fog_density: float
    fog_color: tuple[float, float, float]
    rain_intensity: float
    wind_dir: tuple[float, float]
    wind_speed: float
    # lighting integration
    terrain_light_scale: tuple[float, float, float]
    # HDR radiance contract for the GPU volumetric lighting pipeline
    # (linear RGB; see the attribute docs above for calibrated ranges).
    sun_radiance: tuple[float, float, float] = (0.0, 0.0, 0.0)
    moon_radiance: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sky_ambient: tuple[float, float, float] = (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# SkySystem
# ---------------------------------------------------------------------------

class SkySystem:
    """
    Headless sky/weather composer (Layer 1 — Services, peer of lighting/).

    Owns a :class:`WeatherSystem` (exposed as the ``weather`` attribute —
    register it with SaveManager for delta saves) and turns clock time +
    weather params into a single :class:`SkyState` per frame.  Deterministic:
    two fresh systems with the same world seed and identical clock readings
    produce identical states.

    Parameters
    ----------
    config : Config — engine configuration (cloud geometry fields
        ``sky_cloud_altitude_m`` / ``sky_cloud_thickness_m`` /
        ``sky_cloud_cell_m`` are read by the render half; ``sky_star_count``
        feeds the "night_sky" texture).
    clock : Clock — source of ``game_day`` and ``game_time_of_day``.
    bus : EventBus — passed to the WeatherSystem for
        ``WeatherChangedEvent`` notifications.

    Example
    -------
    >>> sky = SkySystem(cfg, clock, bus)
    >>> state = sky.update()        # once per frame
    >>> sky.state is state          # cached last snapshot
    True
    """

    def __init__(self, config: Config, clock: Clock, bus: EventBus) -> None:
        self._config = config
        self._clock = clock
        self._bus = bus
        #: The weather sub-system (Saveable, ``save_key = "weather"``).
        self.weather: WeatherSystem = WeatherSystem(config, bus)
        self._state: SkyState | None = None
        # Physical-atmosphere lookup table (built once per process; the
        # physics is seed-independent).  ~0.2 s at first construction.
        self._lut = _get_atmosphere_lut()

    def update(self) -> SkyState:
        """
        Recompute the sky snapshot from the current clock time.

        Call once per frame (cheap: a handful of scalar ramps — no arrays,
        no events, no render calls).  Reads ``clock.game_day`` and
        ``clock.game_time_of_day``; advances the weather blend; caches and
        returns the new :class:`SkyState`.

        Returns
        -------
        SkyState — the freshly computed snapshot (also available via
        :attr:`state` until the next update).
        """
        day = int(self._clock.game_day)
        tod = float(self._clock.game_time_of_day)

        wp: WeatherParams = self.weather.update(day, tod)

        sun = sun_direction(tod)
        moon = moon_direction(tod)
        z = float(sun.z)

        daylight = smoothstep(z, DAYLIGHT_Z_LO, DAYLIGHT_Z_HI)
        # Weather grayness drives desaturation/darkening of the sky palette.
        w = min(1.0, _WEATHER_GRAY_GAIN * wp.cloud_coverage * wp.cloud_density)

        # Sun: 0 below the horizon, ramping up over the first ~14° of
        # elevation; heavy cloud dims direct sun toward a diffuse glow.
        sun_intensity = smoothstep(z, 0.0, 0.25) * (
            1.0 - 0.65 * wp.cloud_coverage * wp.cloud_density
        )
        sun_color = color_ramp(z, _SUN_COLOR_RAMP)

        moon_phase = (day % MOON_CYCLE_DAYS) / float(MOON_CYCLE_DAYS)
        star_visibility = (1.0 - daylight) * (1.0 - 0.85 * wp.cloud_coverage)

        # Sky gradient from the physical atmosphere (LUT over sun elevation)
        # with an artistic night floor, then the usual weather grading.
        # Clamped to [0, 1] — these two stay LDR display colors for the dome
        # gradient and legacy consumers; HDR lives in the radiance fields.
        night = 1.0 - daylight
        atmo_zen = self._lut.sample(self._lut.zenith, z)
        atmo_hor = self._lut.sample(self._lut.horizon, z)
        zenith = _weathered(tuple(
            min(1.0, atmo_zen[i] + _NIGHT_ZENITH[i] * night)
            for i in range(3)), w)
        horizon = _weathered(tuple(
            min(1.0, atmo_hor[i] + _NIGHT_HORIZON[i] * night)
            for i in range(3)), w)

        # --- HDR radiance contract for the GPU lighting pipeline ---------
        cloud_block = wp.cloud_coverage * wp.cloud_density
        sun_clear = self._lut.sample(self._lut.sun, z)
        sun_atten = 1.0 - 0.92 * cloud_block      # overcast ⇒ diffuse, no disc
        sun_radiance = tuple(c * sun_atten for c in sun_clear)

        moon_z = float(moon.z)
        illum = 0.5 * (1.0 - math.cos(2.0 * math.pi * moon_phase))
        moon_up = smoothstep(moon_z, 0.0, 0.25)
        moon_atten = 1.0 - 0.90 * cloud_block
        moon_radiance = tuple(
            b * illum * moon_up * moon_atten for b in _MOON_BASE_RADIANCE)

        # Skylight: physical ambient, desaturated (not darkened) by overcast,
        # plus the night floor and a small moonlit-sky bump.
        amb = self._lut.sample(self._lut.ambient, z)
        amb = lerp_color(amb, _luminance_gray(amb), 0.8 * w)
        sky_ambient = tuple(
            amb[i] + _NIGHT_AMBIENT[i] * night + moon_radiance[i] * 0.18
            for i in range(3))

        # Fog color: horizon hue pulled toward a neutral gray that itself
        # dims at night (fog should never glow in the dark).
        night_dim = 0.15 + 0.85 * daylight
        fog_gray = (
            _FOG_GRAY[0] * night_dim,
            _FOG_GRAY[1] * night_dim,
            _FOG_GRAY[2] * night_dim,
        )
        fog_color = lerp_color(horizon, fog_gray, 0.55)

        # Terrain light: elevation ramp × weather dim factor
        # (clear ≈ 1.00, overcast ≈ 0.75, storm ≈ 0.55 — see _TERRAIN_DIM_*).
        base_scale = color_ramp(z, _TERRAIN_LIGHT_RAMP)
        dim = 1.0 - _TERRAIN_DIM_A * (
            (wp.cloud_coverage * wp.cloud_density) ** _TERRAIN_DIM_B
        )
        terrain_light_scale = (
            base_scale[0] * dim,
            base_scale[1] * dim,
            base_scale[2] * dim,
        )

        self._state = SkyState(
            sun_dir=sun,
            moon_dir=moon,
            sun_color=sun_color,
            sun_intensity=float(sun_intensity),
            moon_phase=float(moon_phase),
            daylight=float(daylight),
            star_visibility=float(star_visibility),
            zenith_color=zenith,
            horizon_color=horizon,
            cloud_coverage=float(wp.cloud_coverage),
            cloud_density=float(wp.cloud_density),
            fog_density=float(wp.fog_density),
            fog_color=fog_color,
            rain_intensity=float(wp.rain_intensity),
            wind_dir=wp.wind_dir,
            wind_speed=float(wp.wind_speed),
            terrain_light_scale=terrain_light_scale,
            sun_radiance=sun_radiance,
            moon_radiance=moon_radiance,
            sky_ambient=sky_ambient,
        )
        return self._state

    @property
    def state(self) -> SkyState:
        """
        The last computed :class:`SkyState`.

        If :meth:`update` has never been called, it is invoked lazily once
        so the renderer can always read a valid snapshot at boot.
        """
        if self._state is None:
            return self.update()
        return self._state
