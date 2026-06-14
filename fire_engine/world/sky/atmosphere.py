"""
sky/atmosphere.py — Physically-based single-scattering atmosphere model.

Earth-like Rayleigh + Mie **single scattering**, evaluated on the CPU with
pure, deterministic, fully vectorized numpy — no panda3d, no RNG, no state.
The GLSL sky-dome shader (``world/sky_shaders.SKY_DOME_FRAGMENT``) re-implements
the *same* math per pixel; this module is the reference implementation used by
``SkySystem`` (lookup tables for ``sun_radiance`` / ``sky_ambient`` /
``zenith_color`` / ``horizon_color``) and by the headless tests.

Model
-----
Spherical planet (radius ``PLANET_RADIUS_M`` = 6371 km), exponential density
falloff with scale heights ``RAYLEIGH_SCALE_HEIGHT_M`` (8500 m) and
``MIE_SCALE_HEIGHT_M`` (1200 m), atmosphere top at ``ATMOSPHERE_TOP_M``
(60 km), observer ``OBSERVER_ALTITUDE_M`` (2 m) above ground.  Scattering
coefficients at sea level:

    Rayleigh  β_R = (5.8e-6, 13.5e-6, 33.1e-6)  1/m   (linear RGB channels)
    Mie       β_M = 3.9e-6                      1/m   (g = 0.76, extinction
                                                       = 1.1 × scattering)

Radiance is integrated along view rays with a fixed-step raymarch; per sample
the transmittance toward the sun uses a nested fixed-step march, with a
planet-occlusion test (this is what produces the earth-shadow / twilight
arch after sunset).  All marches are loops over a FIXED step count with bulk
numpy array expressions inside — never per-element Python loops.

Units & conventions
-------------------
* Distances in **meters**, directions are unit vectors, Z-up; ``dir.z`` is the
  sine of elevation.  The model is azimuth-symmetric about the sun, so the
  scalar entry points (``sun_radiance``, ``sky_ambient``) take a sun
  **elevation** ``sun_z = sin(elevation)`` instead of a full direction.
* All radiometric outputs are **linear HDR RGB** (unitless engine exposure —
  see the calibration constants), tuned so that with the v0 sun arc
  (noon ``sun_z ≈ 0.94``):

    - ``sun_radiance(0.94) ≈ (3.2, 3.0, 2.6)``  — direct sun at the ground
    - ``sky_ambient(0.94) ≈ (0.35, 0.45, 0.70)`` — hemispheric sky irradiance

Example
-------
::

    import numpy as np
    from fire_engine.world.sky import atmosphere

    noon = atmosphere.sun_radiance(0.94)        # ~ (3.2, 3.0, 2.6)
    dusk = atmosphere.sun_radiance(0.02)        # strongly orange, much dimmer
    assert dusk[0] / dusk[2] > noon[0] / noon[2]

    amb = atmosphere.sky_ambient(0.94)          # ~ (0.35, 0.45, 0.70)
    L = atmosphere.sky_radiance(
        np.array([[0.0, 0.0, 1.0]]),            # zenith view ray
        np.array([0.34, 0.0, 0.94]),            # sun direction
    )                                           # (1, 3) linear HDR radiance
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "BETA_RAYLEIGH",
    "BETA_MIE",
    "MIE_G",
    "RAYLEIGH_SCALE_HEIGHT_M",
    "MIE_SCALE_HEIGHT_M",
    "PLANET_RADIUS_M",
    "ATMOSPHERE_TOP_M",
    "OBSERVER_ALTITUDE_M",
    "SUN_TOA_RADIANCE",
    "SUN_GROUND_SCALE",
    "AMBIENT_SCALE",
    "SUN_FADE_LO_Z",
    "transmittance",
    "sun_radiance",
    "sky_radiance",
    "sky_ambient",
]

# ---------------------------------------------------------------------------
# Physical constants (Earth-like; shared verbatim with the GLSL dome shader)
# ---------------------------------------------------------------------------

#: Rayleigh scattering coefficients at sea level, 1/m, linear RGB channels.
BETA_RAYLEIGH: np.ndarray = np.array([5.8e-6, 13.5e-6, 33.1e-6], dtype=np.float64)

#: Mie scattering coefficient at sea level, 1/m (wavelength-independent).
BETA_MIE: float = 3.9e-6

#: Mie Henyey-Greenstein anisotropy (0.76 = strong forward lobe = sun halo).
MIE_G: float = 0.76

#: Mie extinction = scattering × this factor (absorption by aerosols).
_MIE_EXTINCTION_FACTOR: float = 1.1

#: Exponential density scale heights, meters.
RAYLEIGH_SCALE_HEIGHT_M: float = 8500.0
MIE_SCALE_HEIGHT_M: float = 1200.0

#: Planet radius / atmosphere shell top, meters.
PLANET_RADIUS_M: float = 6_371_000.0
ATMOSPHERE_TOP_M: float = 60_000.0

#: Observer altitude above the planet surface, meters (~ ground level).
OBSERVER_ALTITUDE_M: float = 2.0

# ---------------------------------------------------------------------------
# Calibration / exposure constants (tuned to the SkyState contract ranges)
# ---------------------------------------------------------------------------

#: Top-of-atmosphere sun radiance used inside ``sky_radiance`` (matches the
#: ``SUN_I`` constant in the GLSL dome shader).  Unitless engine exposure.
SUN_TOA_RADIANCE: float = 22.0

#: Scale applied to the sun transmittance for the *ground-level direct sun*
#: (``sun_radiance``).  3.45 puts clear noon (sun_z ≈ 0.94) at ≈ (3.2, 3.0, 2.6).
SUN_GROUND_SCALE: float = 3.45

#: Scale applied to the hemispheric irradiance integral (``sky_ambient``).
#: Calibrated so clear noon lands near the SkyState contract target
#: ≈ (0.35, 0.45, 0.70): the raw single-scatter integral at noon is
#: ≈ (0.52, 1.00, 1.77), so 0.4 puts blue at ≈ 0.71.
AMBIENT_SCALE: float = 0.4

#: Sun elevation (``sin``) below which the direct sun is fully extinguished.
#: ``sin(-4°)`` — the smooth twilight tail runs from 0° down to -4°.
SUN_FADE_LO_Z: float = math.sin(math.radians(-4.0))

_R0: float = PLANET_RADIUS_M + OBSERVER_ALTITUDE_M
_R_TOP: float = PLANET_RADIUS_M + ATMOSPHERE_TOP_M


# ---------------------------------------------------------------------------
# Internal helpers (vectorized)
# ---------------------------------------------------------------------------


def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Vectorized Hermite smoothstep of *x* between *lo* and *hi* (→ [0, 1])."""
    t = np.clip((np.asarray(x, dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _unit_rows(dirs: np.ndarray) -> np.ndarray:
    """Coerce *dirs* to a float64 ``(N, 3)`` array of unit row vectors."""
    d = np.atleast_2d(np.asarray(dirs, dtype=np.float64))
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    return d / np.maximum(norm, 1e-12)


def _exit_distance(radius: np.ndarray, cos_b: np.ndarray) -> np.ndarray:
    """
    Distance (m) along a ray to the atmosphere-top sphere ``_R_TOP``.

    Parameters
    ----------
    radius : (N,) float — ray origin distance from the planet centre, meters.
    cos_b : (N,) float — ``dot(origin, dir)`` = ``radius * cos(angle from up)``.

    Returns
    -------
    (N,) float — the positive root of the ray/sphere quadratic.
    """
    disc = cos_b * cos_b - (radius * radius - _R_TOP * _R_TOP)
    return -cos_b + np.sqrt(np.maximum(disc, 0.0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def transmittance(view_dirs: np.ndarray, samples: int = 64) -> np.ndarray:
    """
    Atmospheric transmittance from the ground observer toward each direction.

    Integrates Rayleigh + Mie optical depth along each ray from the observer
    (``OBSERVER_ALTITUDE_M`` above ground) to the top of the atmosphere with
    *samples* quadratically-spaced steps (denser near the observer, where
    density is highest — important for near-horizon rays).  Rays that strike
    the planet (``dir.z`` below the geometric horizon) return ``(0, 0, 0)``.

    Used for the color of the sun/moon disc: ``disc_rgb ∝ transmittance(dir)``.

    Parameters
    ----------
    view_dirs : array-like, shape (N, 3) or (3,)
        Unit view direction(s), Z-up, world space.  Normalized defensively.
    samples : int
        Fixed integration step count (default 64).  Pure function of inputs.

    Returns
    -------
    np.ndarray, shape (N, 3), float64
        Linear RGB transmittance in [0, 1] per ray.

    Example
    -------
    >>> T = transmittance(np.array([0.0, 0.0, 1.0]))   # zenith
    >>> bool(T[0, 0] > T[0, 2])                        # red survives best
    True
    """
    d = _unit_rows(view_dirs)
    n = d.shape[0]
    radius = np.full(n, _R0, dtype=np.float64)
    cos_b = _R0 * d[:, 2]  # dot((0,0,r0), d)

    t_max = _exit_distance(radius, cos_b)
    # Planet hit: both quadratic roots positive ⇔ cos_b < 0 and disc > 0.
    disc_p = cos_b * cos_b - (_R0 * _R0 - PLANET_RADIUS_M * PLANET_RADIUS_M)
    hit = (cos_b < 0.0) & (disc_p > 0.0)

    u = (np.arange(samples, dtype=np.float64) + 0.5) / samples  # (S,)
    t = t_max[:, None] * (u * u)[None, :]  # (N, S)
    w = t_max[:, None] * (2.0 * u / samples)[None, :]  # dt weights
    r = np.sqrt(_R0 * _R0 + t * t + 2.0 * t * cos_b[:, None])  # (N, S)
    h = np.maximum(r - PLANET_RADIUS_M, 0.0)
    od_r = np.sum(np.exp(-h / RAYLEIGH_SCALE_HEIGHT_M) * w, axis=1)  # (N,)
    od_m = np.sum(np.exp(-h / MIE_SCALE_HEIGHT_M) * w, axis=1)

    tau = (
        BETA_RAYLEIGH[None, :] * od_r[:, None] + (BETA_MIE * _MIE_EXTINCTION_FACTOR) * od_m[:, None]
    )
    out = np.exp(-tau)
    out[hit] = 0.0
    return out


def sun_radiance(sun_z: float | np.ndarray, samples: int = 64) -> np.ndarray:
    """
    Direct sun radiance reaching the ground for a given sun elevation.

    ``SUN_GROUND_SCALE × transmittance(sun_dir)``, evaluated at the sun
    elevation clamped to the horizon, then multiplied by a smooth twilight
    fade: 1 at elevation ≥ 0°, smoothstep down to exactly 0 at -4°
    (``SUN_FADE_LO_Z``).  Azimuth-symmetric, so only the elevation sine is
    needed.  Linear HDR RGB; clear noon (``sun_z ≈ 0.94``) ≈ (3.2, 3.0, 2.6);
    near the horizon the output is strongly orange and much dimmer (the R/B
    ratio rises monotonically as the sun drops).

    Parameters
    ----------
    sun_z : float or (N,) array — ``sin(sun elevation)``, i.e. ``sun_dir.z``.
    samples : int — transmittance integration steps (default 64).

    Returns
    -------
    np.ndarray — shape (3,) for scalar input, (N, 3) for array input.
        Linear HDR RGB, all zeros below -4° elevation.

    Example
    -------
    >>> noon = sun_radiance(0.94)
    >>> abs(noon[0] - 3.2) < 0.5 and abs(noon[2] - 2.6) < 0.5
    True
    >>> bool(np.all(sun_radiance(-0.1) == 0.0))    # below the -4° cutoff
    True
    """
    z = np.asarray(sun_z, dtype=np.float64)
    scalar = z.ndim == 0
    z1 = np.atleast_1d(z)
    zc = np.clip(z1, 0.0, 1.0)
    dirs = np.stack(
        [np.sqrt(np.maximum(1.0 - zc * zc, 0.0)), np.zeros_like(zc), zc],
        axis=1,
    )
    t = transmittance(dirs, samples=samples)  # (N, 3)
    fade = _smoothstep(z1, SUN_FADE_LO_Z, 0.0)  # (N,)
    out = SUN_GROUND_SCALE * t * fade[:, None]
    return out[0] if scalar else out


def sky_radiance(
    view_dirs: np.ndarray,
    sun_dir: np.ndarray,
    steps: int = 16,
    light_steps: int = 8,
) -> np.ndarray:
    """
    Single-scattered sky radiance for a set of view directions.

    For each view ray, marches *steps* fixed samples from the observer to the
    atmosphere top (or the ground, whichever is nearer); at every sample the
    transmittance toward the sun is integrated with *light_steps* nested
    samples, including a planet-occlusion test (samples in the planet's
    shadow contribute nothing — this produces the earth-shadow / twilight
    arch).  Rayleigh and Mie phase functions (HG, g = ``MIE_G``) weight the
    in-scattered light; the result is scaled by ``SUN_TOA_RADIANCE``.

    This is the exact math the GLSL dome shader runs per pixel; here it is
    used for ambient integration, the SkySystem color lookup tables, and the
    headless tests.

    Parameters
    ----------
    view_dirs : array-like, shape (N, 3) or (3,)
        Unit view direction(s), Z-up.  Normalized defensively.
    sun_dir : array-like, shape (3,)
        Unit direction toward the sun (may be below the horizon).
    steps : int — fixed view-ray march steps (default 16).
    light_steps : int — fixed sun-ray march steps per sample (default 8).

    Returns
    -------
    np.ndarray, shape (N, 3), float64
        Linear HDR RGB radiance per ray (≈ 0 when the sun is far below the
        horizon).  Deterministic: pure function of the inputs.

    Example
    -------
    >>> L = sky_radiance(np.array([0.0, 0.0, 1.0]), np.array([0.34, 0.0, 0.94]))
    >>> bool(L[0, 2] > L[0, 0])     # zenith is blue at midday
    True
    """
    d = _unit_rows(view_dirs)
    s = _unit_rows(sun_dir)[0]
    n = d.shape[0]

    cos_b = _R0 * d[:, 2]
    t_exit = _exit_distance(np.full(n, _R0), cos_b)
    disc_p = cos_b * cos_b - (_R0 * _R0 - PLANET_RADIUS_M * PLANET_RADIUS_M)
    hit = (cos_b < 0.0) & (disc_p > 0.0)
    t_ground = np.where(hit, -cos_b - np.sqrt(np.maximum(disc_p, 0.0)), np.inf)
    t_max = np.minimum(t_exit, t_ground)  # (N,)

    mu = d @ s  # (N,)
    phase_r = (3.0 / (16.0 * math.pi)) * (1.0 + mu * mu)
    g, g2 = MIE_G, MIE_G * MIE_G
    phase_m = (
        (3.0 / (8.0 * math.pi))
        * (1.0 - g2)
        * (1.0 + mu * mu)
        / ((2.0 + g2) * np.power(1.0 + g2 - 2.0 * g * mu, 1.5))
    )

    od_r_view = np.zeros(n)
    od_m_view = np.zeros(n)
    radiance = np.zeros((n, 3))

    for i in range(steps):  # fixed-count raymarch loop
        # Quadratic step spacing (t = t_max·u², weight dt = 2·t_max·u/steps):
        # near-horizon rays exit ~600 km away while all the density sits in
        # the first few km — linear steps would skip it entirely and render
        # the day horizon black.  Mirrored in the GLSL dome shader.
        u = (i + 0.5) / steps
        t = t_max * (u * u)  # (N,)
        dt = t_max * (2.0 * u / steps)  # (N,)
        r = np.sqrt(_R0 * _R0 + t * t + 2.0 * t * cos_b)  # sample radius
        h = np.maximum(r - PLANET_RADIUS_M, 0.0)
        dens_r = np.exp(-h / RAYLEIGH_SCALE_HEIGHT_M)
        dens_m = np.exp(-h / MIE_SCALE_HEIGHT_M)
        od_r_view += dens_r * dt
        od_m_view += dens_m * dt

        # Sun-ray geometry from the sample point P = (0,0,_R0) + t·d:
        # dot(P, s) = _R0·s.z + t·(d·s).
        cos_bl = _R0 * s[2] + t * mu  # (N,)
        disc_l = cos_bl * cos_bl - (r * r - PLANET_RADIUS_M * PLANET_RADIUS_M)
        blocked = (cos_bl < 0.0) & (disc_l > 0.0)  # planet shadow

        tl_exit = _exit_distance(r, cos_bl)
        dl = tl_exit / light_steps
        od_r_l = np.zeros(n)
        od_m_l = np.zeros(n)
        for j in range(light_steps):  # fixed-count nested march
            tl = (j + 0.5) * dl
            rl = np.sqrt(r * r + tl * tl + 2.0 * tl * cos_bl)
            hl = np.maximum(rl - PLANET_RADIUS_M, 0.0)
            od_r_l += np.exp(-hl / RAYLEIGH_SCALE_HEIGHT_M) * dl
            od_m_l += np.exp(-hl / MIE_SCALE_HEIGHT_M) * dl

        tau = (
            BETA_RAYLEIGH[None, :] * (od_r_view + od_r_l)[:, None]
            + (BETA_MIE * _MIE_EXTINCTION_FACTOR) * (od_m_view + od_m_l)[:, None]
        )
        trans = np.exp(-tau)
        trans[blocked] = 0.0
        scatter = (
            BETA_RAYLEIGH[None, :] * (dens_r * phase_r)[:, None]
            + BETA_MIE * (dens_m * phase_m)[:, None]
        )
        radiance += trans * scatter * dt[:, None]

    return SUN_TOA_RADIANCE * radiance


# Fixed cosine-weighted hemisphere sample set for sky_ambient (Fibonacci
# spiral over the upper hemisphere — deterministic, no RNG).
def _hemisphere_dirs(samples: int) -> np.ndarray:
    """Fixed Fibonacci-spiral unit directions on the upper hemisphere (N, 3)."""
    i = np.arange(samples, dtype=np.float64)
    z = (i + 0.5) / samples  # uniform in cos(theta)
    phi = i * (math.pi * (3.0 - math.sqrt(5.0)))  # golden angle
    rho = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=1)


def sky_ambient(
    sun_z: float,
    samples: int = 48,
    steps: int = 16,
    light_steps: int = 8,
) -> np.ndarray:
    """
    Hemispheric sky ambient irradiance (cosine-weighted) at the ground.

    ``E = ∫ L(ω) cosθ dω`` over the upper hemisphere, estimated with a FIXED
    Fibonacci-spiral direction set (*samples* directions, default 48 — fully
    deterministic, vectorized through :func:`sky_radiance`), scaled by
    ``AMBIENT_SCALE`` to compensate for the missing multiple scattering.

    Azimuth-symmetric in the sun, so only the sun elevation sine is needed
    (the sun is placed at azimuth 0 internally).  Linear HDR RGB:
    clear noon (``sun_z ≈ 0.94``) ≈ (0.35, 0.45, 0.70); warm gray near
    sunset; ≈ 0 once the sun is well below the horizon (the night floor and
    moonlight bump are added by ``SkySystem``, not here).

    Parameters
    ----------
    sun_z : float — ``sin(sun elevation)``.
    samples : int — hemisphere direction count (default 48).
    steps, light_steps : int — forwarded to :func:`sky_radiance`.

    Returns
    -------
    np.ndarray, shape (3,) — linear HDR RGB irradiance.

    Example
    -------
    >>> amb = sky_ambient(0.94)
    >>> bool(amb[2] > amb[1] > amb[0])     # blue-dominant at noon
    True
    """
    z = float(sun_z)
    sun = np.array([math.sqrt(max(1.0 - z * z, 0.0)), 0.0, z])
    dirs = _hemisphere_dirs(samples)
    L = sky_radiance(dirs, sun, steps=steps, light_steps=light_steps)  # (N,3)
    # Uniform-in-cos hemisphere sampling: E ≈ (2π / N) Σ L_i cosθ_i.
    irradiance = (2.0 * math.pi / samples) * np.sum(L * dirs[:, 2:3], axis=0)
    return AMBIENT_SCALE * irradiance
