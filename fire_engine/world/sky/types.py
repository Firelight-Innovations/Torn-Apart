"""
Trivial support types for fire_engine.world.sky.

Contains :class:`SkyState`, the immutable per-frame sky/weather snapshot
consumed by the render layer.  Kept in a dedicated *types.py* grouping module
so that :mod:`fire_engine.world.sky.sky_state` remains a single-public-class
module (holding :class:`SkySystem` only).

Docs: docs/systems/world.sky.md
"""

from __future__ import annotations

from dataclasses import dataclass

from fire_engine.core.math3d import Vec3

__all__ = ["SkyState"]


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

    Docs: docs/systems/world.sky.md
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
