"""
sky/celestial.py — Pure sun/moon geometry and color-ramp helpers.

Everything in this module is a **pure function of time-of-day** (and fixed
keyframe tables) — no state, no config, no RNG, no panda3d.  The SkySystem
composes these into a full ``SkyState``; tests call them directly.

Coordinate convention (engine-wide, see docs/systems/core.md)
-------------------------------------------------------------
Z-up, forward = +Y, right/east = +X.  All angles in **radians**, all times in
**seconds** of in-game time-of-day (``[0, 86400)``; 0 = midnight).

Sun schedule (v0 — fixed equinox day, no seasons yet)
-----------------------------------------------------
Sunrise 06:00 (21600 s), solar noon 12:00, sunset 18:00 (64800 s).  The sun
travels a great-circle arc tilted ``SUN_ARC_TILT_RAD`` (20°) toward -Y
(south), like a mid-latitude northern-hemisphere sky:

    06:00 — rises due east (+X), elevation 0
    12:00 — high in the southern sky, ``sun_dir.z = cos(20°) ≈ 0.94``
    18:00 — sets due west (−X), elevation 0
    00:00 — nadir, ``sun_dir.z ≈ −0.94``

The moon follows the same arc half a day out of phase, advanced by
``MOON_PHASE_OFFSET_RAD`` so sun and moon are briefly both visible at
twilight, on a slightly different tilt so the two arcs do not coincide.

Example
-------
    from fire_engine.world.sky.celestial import sun_direction, moon_direction

    noon = sun_direction(12 * 3600.0)     # Vec3, unit length, z ≈ 0.94
    mid  = sun_direction(0.0)             # z ≈ -0.94 (below horizon)
"""

from __future__ import annotations

import math

import numpy as np

from fire_engine.core.math3d import Vec3

__all__ = [
    "sun_direction",
    "moon_direction",
    "daylight_factor",
    "smoothstep",
    "color_ramp",
    "lerp_color",
    "SUN_ARC_TILT_RAD",
    "MOON_ARC_TILT_RAD",
    "MOON_PHASE_OFFSET_RAD",
    "DAYLIGHT_Z_LO",
    "DAYLIGHT_Z_HI",
    "GAME_SECONDS_PER_DAY",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Seconds in one in-game day (matches core/clock.py's calendar).
GAME_SECONDS_PER_DAY: float = 24.0 * 3600.0

#: Tilt of the sun's great-circle arc away from straight-overhead, toward −Y
#: (south), in radians.  20° keeps noon ``sun_dir.z = cos(20°) ≈ 0.94``.
SUN_ARC_TILT_RAD: float = math.radians(20.0)

#: The moon rides a slightly shallower arc (12°) so the two paths differ.
MOON_ARC_TILT_RAD: float = math.radians(12.0)

#: Moon arc phase lead relative to "exactly opposite the sun", radians.
#: 0.26 rad ≈ 1 game hour — at dusk the moon is already ~1 h above the
#: eastern horizon while the sun is still setting (both briefly visible).
MOON_PHASE_OFFSET_RAD: float = 0.26

#: Daylight smoothstep bounds on sun elevation (``sun_dir.z``).
#: ``z = ±0.24`` is the elevation roughly 1 game hour after sunrise/before-
#: -after sunset (sin(15°)·cos(tilt) ≈ 0.243), so ``daylight`` reaches fully
#: 1 about 1 h after sunrise and fully 0 about 1 h after sunset.
DAYLIGHT_Z_LO: float = -0.24
DAYLIGHT_Z_HI: float = 0.24


# ---------------------------------------------------------------------------
# Generic math helpers
# ---------------------------------------------------------------------------


def smoothstep(x: float, lo: float, hi: float) -> float:
    """
    Hermite smoothstep of *x* between *lo* and *hi*.

    Returns 0.0 for ``x <= lo``, 1.0 for ``x >= hi``, and the C1-continuous
    cubic ``3t² − 2t³`` in between.  Pure float math (scalar).

    Parameters
    ----------
    x : float — input value (any units; same units as lo/hi).
    lo, hi : float — edges; ``hi > lo`` required.

    Example
    -------
    >>> smoothstep(0.5, 0.0, 1.0)
    0.5
    """
    t = (x - lo) / (hi - lo)
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def lerp_color(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    t: float,
) -> tuple[float, float, float]:
    """
    Linear interpolation between two linear-RGB color tuples.

    Parameters
    ----------
    a, b : tuple[float, float, float] — linear RGB, components 0–1 (may exceed
           1 slightly for HDR-ish sun colors; no clamping here).
    t : float — blend factor; 0 → *a*, 1 → *b*.  Clamped to [0, 1].

    Example
    -------
    >>> lerp_color((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0.5)
    (0.5, 0.5, 0.5)
    """
    t = min(1.0, max(0.0, t))
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def color_ramp(
    x: float,
    keyframes: tuple[tuple[float, tuple[float, float, float]], ...],
) -> tuple[float, float, float]:
    """
    Evaluate a keyframe color ramp at *x* using ``np.interp`` per channel.

    Keyframes are ``(key, (r, g, b))`` pairs sorted ascending by key.  Below
    the first key the first color is returned; above the last key the last
    color — exactly ``np.interp`` edge behaviour.

    Parameters
    ----------
    x : float
        Ramp coordinate.  All sky ramps in this package key on sun elevation
        ``sun_dir.z`` in [-1, 1] (NOT time) so dawn and dusk share keyframes.
    keyframes : tuple of (float, (float, float, float))
        Ascending ``(key, linear_rgb)`` stops.

    Returns
    -------
    tuple[float, float, float] — interpolated linear RGB.

    Example
    -------
    >>> ramp = ((0.0, (0.0, 0.0, 0.0)), (1.0, (1.0, 0.5, 0.0)))
    >>> color_ramp(0.5, ramp)
    (0.5, 0.25, 0.0)
    """
    keys = np.array([k for k, _ in keyframes], dtype=np.float64)
    cols = np.array([c for _, c in keyframes], dtype=np.float64)  # (N, 3)
    r = float(np.interp(x, keys, cols[:, 0]))
    g = float(np.interp(x, keys, cols[:, 1]))
    b = float(np.interp(x, keys, cols[:, 2]))
    return (r, g, b)


# ---------------------------------------------------------------------------
# Celestial directions
# ---------------------------------------------------------------------------


def _arc_direction(phase: float, tilt_rad: float) -> Vec3:
    """
    Direction on a tilted great-circle arc at the given *phase* angle.

    The arc is the unit circle in the XZ plane (rises +X at phase 0, peaks
    +Z at phase π/2, sets −X at phase π) rotated by *tilt_rad* about the
    X axis toward −Y (the southern sky).

    Parameters
    ----------
    phase : float — radians; 0 = rising due east, π/2 = arc peak.
    tilt_rad : float — arc tilt away from overhead, radians.

    Returns
    -------
    Vec3 — unit direction FROM the scene TOWARD the body (Z-up).
    """
    s, c = math.sin(phase), math.cos(phase)
    # Untilted: (c, 0, s).  Tilt about X toward -Y: y' = -s·sin(tilt),
    # z' = s·cos(tilt).  x stays.
    return Vec3(c, -s * math.sin(tilt_rad), s * math.cos(tilt_rad))


def _sun_phase(time_of_day_s: float) -> float:
    """
    Sun arc phase angle (radians) for a time of day.

    Phase 0 at 06:00 (sunrise), π/2 at 12:00 (noon), π at 18:00 (sunset),
    3π/2 at 00:00 (midnight).  Linear in time — one full revolution per
    24-game-hour day, so the direction is continuous (no snaps).
    """
    t = time_of_day_s % GAME_SECONDS_PER_DAY
    return 2.0 * math.pi * (t - 6.0 * 3600.0) / GAME_SECONDS_PER_DAY


def sun_direction(time_of_day_s: float) -> Vec3:
    """
    Unit direction FROM the scene TOWARD the sun for a given time of day.

    Fixed v0 schedule: sunrise 06:00 due east (+X), noon high south
    (``z = cos(20°) ≈ 0.94``), sunset 18:00 due west (−X), midnight nadir
    (``z ≈ −0.94``).  Continuous and periodic in *time_of_day_s*.

    Parameters
    ----------
    time_of_day_s : float
        In-game seconds within the day, ``[0, 86400)`` (values outside wrap).
        Read from ``clock.game_time_of_day``.

    Returns
    -------
    Vec3 — unit-length direction, Z-up.  ``dir.z > 0`` means the sun is
    above the horizon.

    Example
    -------
    >>> sun_direction(12 * 3600.0).z >= 0.9     # noon, high in the sky
    True
    >>> abs(sun_direction(6 * 3600.0).z) < 0.15 # sunrise, on the horizon
    True
    """
    return _arc_direction(_sun_phase(time_of_day_s), SUN_ARC_TILT_RAD)


def moon_direction(time_of_day_s: float) -> Vec3:
    """
    Unit direction FROM the scene TOWARD the moon for a given time of day.

    The moon rides the sun's arc shifted by π (roughly opposite the sun)
    plus ``MOON_PHASE_OFFSET_RAD`` (~1 game hour lead), on a slightly
    shallower tilt (``MOON_ARC_TILT_RAD``).  At dusk both bodies are briefly
    above the horizon: the sun setting in the west, the moon ~1 h up in the
    east.  Continuous in time.

    Parameters
    ----------
    time_of_day_s : float — in-game seconds within the day (wraps at 86400).

    Returns
    -------
    Vec3 — unit-length direction, Z-up.

    Example
    -------
    >>> moon_direction(0.0).z > 0.9     # midnight: moon near its peak
    True
    """
    phase = _sun_phase(time_of_day_s) + math.pi + MOON_PHASE_OFFSET_RAD
    return _arc_direction(phase, MOON_ARC_TILT_RAD)


def daylight_factor(time_of_day_s: float) -> float:
    """
    Smooth 0–1 day factor for a time of day (0 = full night, 1 = full day).

    Smoothstep on sun elevation ``sun_dir.z`` between ``DAYLIGHT_Z_LO``
    (-0.24) and ``DAYLIGHT_Z_HI`` (+0.24): exactly 0.5 at sunrise/sunset,
    fully 1 about one game hour after sunrise, fully 0 about one game hour
    after sunset.

    Parameters
    ----------
    time_of_day_s : float — in-game seconds within the day.

    Returns
    -------
    float in [0, 1].

    Example
    -------
    >>> daylight_factor(12 * 3600.0)    # noon
    1.0
    >>> daylight_factor(0.0)            # midnight
    0.0
    """
    z = sun_direction(time_of_day_s).z
    return smoothstep(float(z), DAYLIGHT_Z_LO, DAYLIGHT_Z_HI)
