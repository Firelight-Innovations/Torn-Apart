"""
fire_engine.wind — Spatially-varying, time-evolving wind field (headless).

The single source of truth for everything wind-driven (grass, future
flags/cloth/hair/water, ambient dust/leaf particles, physics pushes, future
procedural wind audio).  A player-centred 2.5-D field: a 64×64-cell × 4 m
(256 m) grid of horizontal wind velocity summed from ~12 seeded spectral gust
modes that **advect downwind** (so gust bands visibly travel), plus an analytic
vertical boundary-layer profile.

The field is a **pure function of (world_seed, wind_time, weather, player
cell)** — bit-reproducible, **zero save bytes** (no Saveable, same ethos as
``sky/weather.py``), and free to recenter as the player moves.  This package is
headless: numpy + core only, **no panda3d** (the upload/render half lives in
``world/``, added by WP3).

Public API summary
------------------
WindField
    The field.  ``update(dt, wind_time, sky_state, player_pos, chunks=None)``
    once per frame; ``sample(positions (N,3)) -> (N,3)`` m/s for physics/audio;
    ``snapshot`` for the current state; ``add_modifier`` / ``remove_modifier``.
WindSnapshot
    Frozen atomically-published field state (``field``, ``origin_m``,
    ``cell_m``, ``cells``, ``wind_time``).
WindModifier, GustFront
    In-place modifier protocol (the volumetric-weather seam) + a working moving
    gust-front example.
pack_wind_field
    Pack a snapshot to Panda3D 2-D-texture RAM bytes (float16, ``(y, x)``
    BGRA).
vertical_profile
    Analytic boundary-layer wind-speed multiplier vs. height above ground.
VenturiWorker, VenturiJob, VenturiResult, solve_venturi
    Off-thread terrain-funneling solver (mirror of the lighting assembly
    worker): wind speeds up through gaps/canyons and rises over windward
    obstacles.  ``WindField(cfg, worker)`` consumes a started worker; pass
    ``chunks`` into ``update()`` on recenter / terrain edits to drive it.
    ``solve_venturi(job)`` is the pure on-thread core (also used in tests).

Quick-start example
-------------------
::

    import numpy as np
    from fire_engine.core import load_config, set_world_seed
    from fire_engine.wind import WindField

    cfg = load_config()
    set_world_seed(cfg.world_seed)
    field = WindField(cfg)
    field.update(dt=0.016, wind_time=12.0, sky_state=None,
                 player_pos=(0.0, 0.0, 0.0))           # once per frame
    v = field.sample(np.array([[0.0, 0.0, 1.0]]))      # wind at a point, m/s
"""

from fire_engine.wind.debug import BallParams, debug_ball_step
from fire_engine.wind.field import (
    WindField,
    WindSnapshot,
    pack_wind_field,
    vertical_profile,
)
from fire_engine.wind.modifiers import GustFront, WindModifier
from fire_engine.wind.venturi import solve_venturi
from fire_engine.wind.worker import VenturiJob, VenturiResult, VenturiWorker

__all__ = [
    "WindField",
    "WindSnapshot",
    "WindModifier",
    "GustFront",
    "pack_wind_field",
    "vertical_profile",
    "VenturiWorker",
    "VenturiJob",
    "VenturiResult",
    "solve_venturi",
    "BallParams",
    "debug_ball_step",
]
