"""
wind/debug.py — Headless physics integrator for the wind-field debug ball.

This is the **panda3d-free, pure-function** heart of the dev-only "wind ball"
seam proof (the rendered component lives in ``world/wind_debug.py``, which the
import rule keeps out of the headless suite).  Splitting the integrator out here
lets a headless test step a ball with a synthetic :class:`~fire_engine.world.wind.WindField`
and assert it accelerates downwind, with no window and no GPU.

The model is intentionally minimal — it is a *visible proof that physics can
sample the wind field*, not a rigid-body simulation:

- horizontal drag toward the local wind velocity (the ball is dragged along by
  the air; a gust crossing it scoots it, a storm rolls it hard),
- gravity + a hard clamp to the ground plane with simple ground friction (the
  ball rests on flat ground and never sinks or flies away),
- everything vectorised / closed-form — no per-step Python object churn beyond
  the single state tuple.

Determinism: a pure function of ``(state, wind_velocity, dt, params)``.  The
only randomness anywhere in the wind system is the spectral gust basis (drawn
once at :class:`WindField` construction); this integrator adds none.

No panda3d.  No per-element Python loops.

Example
-------
>>> import numpy as np
>>> from fire_engine.core.config import Config
>>> from fire_engine.core.rng import set_world_seed
>>> from fire_engine.world.wind import WindField
>>> set_world_seed(1337)
>>> field = WindField(Config())
>>> params = BallParams(ground_z=8.0)
>>> pos = np.array([0.0, 0.0, 8.0], dtype=np.float64)   # resting on ground
>>> vel = np.zeros(3)
>>> field.update(0.02, 0.0, None, pos)                  # publish a snapshot
>>> v_wind = field.sample(pos[None])[0]
>>> pos, vel = debug_ball_step(pos, vel, v_wind, dt=0.02, params=params)
>>> pos.shape
(3,)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BallParams", "debug_ball_step"]


@dataclass(frozen=True)
class BallParams:
    """
    Tuning for :func:`debug_ball_step` (all SI: meters, seconds, m/s²).

    Attributes
    ----------
    ground_z : float
        World Z of the flat ground surface the ball rests on (meters).  In the
        demo this is ``config.ground_height_m`` (the terrain top, z = 8).
    radius_m : float
        Ball radius (meters); its centre is clamped to ``ground_z + radius_m``.
    drag : float
        Horizontal coupling to the wind, per second.  ``vel_xy`` relaxes toward
        the local wind ``vel_xy`` at rate ``drag`` — higher = the ball chases
        gusts more eagerly.  A light, draggy ball (high drag, low mass) is what
        makes the gust response *visible*.
    gravity : float
        Downward acceleration magnitude (m/s²); pins the ball to the ground.
    friction : float
        Ground rolling/sliding friction per second applied to horizontal
        velocity while resting on the ground (0 = frictionless ice, 1 ≈ stops in
        ~1 s) — so the ball settles when a gust passes instead of drifting
        forever.
    max_speed : float
        Horizontal speed clamp (m/s); keeps a storm from launching the ball off
        the visible plane.
    """

    ground_z: float = 8.0
    radius_m: float = 0.4
    drag: float = 2.5
    gravity: float = 9.81
    friction: float = 1.5
    max_speed: float = 25.0


def debug_ball_step(
    pos: np.ndarray,
    vel: np.ndarray,
    wind_velocity: np.ndarray,
    dt: float,
    params: BallParams,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Advance the debug ball one fixed step — a pure function.

    Integrates the minimal wind-drag model (see the module docstring) and
    returns the **new** ``(pos, vel)`` as fresh arrays (the inputs are not
    mutated, so callers may keep the old state for comparison/tests).

    The horizontal velocity is dragged toward the local wind, gravity pulls the
    ball down, and the ball is then clamped to rest on the ground plane
    (``z = ground_z + radius_m``) with ground friction and a speed clamp applied
    while it rests there.  A gust crossing the ball pushes ``vel_xy`` toward the
    gust velocity → the ball visibly scoots downwind; a storm's stronger,
    choppier field rolls it hard.

    Parameters
    ----------
    pos : numpy.ndarray
        ``(3,)`` world position ``[x, y, z]`` in meters (float).
    vel : numpy.ndarray
        ``(3,)`` world velocity ``[vx, vy, vz]`` in m/s (float).
    wind_velocity : numpy.ndarray
        ``(3,)`` local wind velocity in m/s — typically
        ``WindField.sample(pos[None])[0]``.
    dt : float
        Fixed timestep in seconds (the component feeds ``clock.fixed_dt``).
    params : BallParams
        Tuning (ground height, drag, gravity, friction, clamps).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The new ``(pos, vel)`` arrays (``float64``, shape ``(3,)`` each).

    Example
    -------
    >>> import numpy as np
    >>> p = np.array([0.0, 0.0, 8.4]); v = np.zeros(3)
    >>> wind = np.array([6.0, 0.0, 0.0])              # a +X gust
    >>> p2, v2 = debug_ball_step(p, v, wind, 0.02, BallParams())
    >>> bool(v2[0] > 0.0)                              # accelerated downwind
    True
    """
    p = np.asarray(pos, dtype=np.float64).copy()
    v = np.asarray(vel, dtype=np.float64).copy()
    w = np.asarray(wind_velocity, dtype=np.float64)
    dt = float(dt)
    rest_z = float(params.ground_z) + float(params.radius_m)

    # --- Horizontal: relax velocity toward the local wind (drag coupling) ----
    # Exponential-ish relaxation clamped to [0,1] so a large drag*dt can't
    # overshoot the wind velocity (stable at any dt).
    k = min(max(float(params.drag) * dt, 0.0), 1.0)
    v[0] += (float(w[0]) - v[0]) * k
    v[1] += (float(w[1]) - v[1]) * k

    # --- Vertical: gravity ---------------------------------------------------
    v[2] -= float(params.gravity) * dt

    # --- Integrate position --------------------------------------------------
    p += v * dt

    # --- Ground clamp + friction + speed clamp (resting on flat ground) ------
    if p[2] <= rest_z:
        p[2] = rest_z
        if v[2] < 0.0:
            v[2] = 0.0                       # no bounce — it just rests
        # Ground friction bleeds horizontal speed so the ball settles in calm.
        f = min(max(float(params.friction) * dt, 0.0), 1.0)
        v[0] -= v[0] * f
        v[1] -= v[1] * f
        # Speed clamp keeps a storm from flinging the ball off the plane.
        speed = float(np.hypot(v[0], v[1]))
        if speed > float(params.max_speed):
            scale = float(params.max_speed) / speed
            v[0] *= scale
            v[1] *= scale
    return p, v
