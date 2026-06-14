"""
wind/gusts.py — Seeded spectral "Brownian-band" gust modes (pure, headless).

The wind field's spatial/temporal variation is **not** an accumulated random
walk.  It is a sum of ``cfg.wind_gust_modes`` (~12) sinusoidal modes whose
wavevectors, phases and intrinsic frequencies are drawn **once** from
``core.rng.for_domain("wind", "gusts")`` and then evaluated analytically at
any (world position, time).  This makes the whole field a pure function of
``(world_seed, wind_time, world_position)``: bit-reproducible, zero save
bytes, free to recenter (no history to carry), and visually indistinguishable
from real Brownian gusting at 20–120 m wavelengths.

The amplitude spectrum is **red noise** (amplitude ∝ 1/wavelength, normalised
to sum 1), so a few big slow gusts dominate and small ripples are faint — the
look of wind crossing an open field.

The **advection term is the whole point**: each mode's phase advances not only
with its intrinsic frequency ``omega`` but with ``k · mean`` (the wavevector
dotted with the mean wind velocity), so gust crests *travel downwind at wind
speed*.  Neighbouring cells therefore sample genuinely different velocities and
a gust band visibly sweeps across a grass field.

Units
-----
- Wavelengths, positions: meters.  Wavevectors ``k``: rad/m.
- ``omega``: rad/s.  ``t_eff``: seconds (storm-frequency-scaled game time).
- ``mean``: m/s (the mean wind velocity vector, XY).
- Returned gust components are dimensionless shape values in roughly
  ``[-1, 1]`` (per-mode amplitudes sum to 1); the caller scales them to m/s by
  a weather-derived ``gust_gain``.

No panda3d.  No per-cell Python loops — everything is numpy broadcasting over
the ``(cells, cells, modes)`` mode axis.

Example
-------
>>> from fire_engine.core.config import Config
>>> from fire_engine.core.rng import set_world_seed
>>> import numpy as np
>>> set_world_seed(1337)
>>> cfg = Config()
>>> modes = build_modes(cfg)
>>> X, Y = np.meshgrid(np.arange(4.0), np.arange(4.0), indexing="ij")
>>> gx, gy = eval_gusts(modes, X, Y, t_eff=0.0, mean=(2.0, 0.0))
>>> gx.shape
(4, 4)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fire_engine.core.config import Config
from fire_engine.core.rng import for_domain

__all__ = ["GustModes", "build_modes", "eval_gusts"]


@dataclass(frozen=True)
class GustModes:
    """
    The drawn-once constants of the spectral gust basis.

    Every array is 1-D of length ``M = cfg.wind_gust_modes``.  Held frozen and
    reused across all evaluations and all recenters for the life of a
    :class:`~fire_engine.world.wind.field.WindField` — they depend only on the world
    seed and config, never on time or position.

    Attributes
    ----------
    kx, ky : numpy.ndarray
        ``float32 (M,)`` — wavevector components in rad/m (``2π/wavelength``
        times the mode's unit propagation direction).
    omega : numpy.ndarray
        ``float32 (M,)`` — intrinsic temporal angular frequency, rad/s.
    phase0 : numpy.ndarray
        ``float32 (M,)`` — per-mode phase offset, radians.
    pux, puy : numpy.ndarray
        ``float32 (M,)`` — unit push-direction the mode displaces the air
        along (the velocity contribution direction).
    amp : numpy.ndarray
        ``float32 (M,)`` — per-mode amplitude, red-noise weighted
        (∝ 1/wavelength) and normalised to ``sum == 1``.
    """

    kx: np.ndarray
    ky: np.ndarray
    omega: np.ndarray
    phase0: np.ndarray
    pux: np.ndarray
    puy: np.ndarray
    amp: np.ndarray


def build_modes(cfg: Config) -> GustModes:
    """
    Draw the spectral gust basis once from ``for_domain("wind", "gusts")``.

    Deterministic: the same world seed + config always yields the same
    :class:`GustModes` (in-process and cross-process — the draw goes through
    the engine's blake2b-seeded RNG service).  Call once at
    :class:`~fire_engine.world.wind.field.WindField` construction and reuse forever.

    The draws, in order (the order is part of the determinism contract — do not
    reorder, or saved worlds would regenerate different gusts):

    1. ``wavelength`` — uniform in ``[wind_gust_wavelen_min, _max]`` meters.
    2. ``angle`` — wavevector direction, uniform on the circle.
    3. ``phase0`` — uniform in ``[0, 2π)``.
    4. ``omega`` — uniform in ``[wind_gust_omega_min, _max]`` rad/s.
    5. ``push_angle`` — velocity-contribution direction, uniform on the circle.

    Amplitudes are ``1/wavelength`` normalised to sum 1 (red-noise: long
    wavelengths dominate).

    Parameters
    ----------
    cfg : Config
        Engine config; reads ``wind_gust_modes``, ``wind_gust_wavelen_min``,
        ``wind_gust_wavelen_max``, ``wind_gust_omega_min``,
        ``wind_gust_omega_max``.

    Returns
    -------
    GustModes

    Example
    -------
    >>> from fire_engine.core.rng import set_world_seed
    >>> set_world_seed(1337)
    >>> modes = build_modes(Config())
    >>> float(modes.amp.sum())
    1.0
    """
    m = int(cfg.wind_gust_modes)
    rng = for_domain("wind", "gusts")

    wavelength = rng.uniform(float(cfg.wind_gust_wavelen_min), float(cfg.wind_gust_wavelen_max), m)
    angle = rng.uniform(0.0, 2.0 * np.pi, m)
    phase0 = rng.uniform(0.0, 2.0 * np.pi, m)
    omega = rng.uniform(float(cfg.wind_gust_omega_min), float(cfg.wind_gust_omega_max), m)
    push_angle = rng.uniform(0.0, 2.0 * np.pi, m)

    k_mag = (2.0 * np.pi) / wavelength  # rad/m
    kx = k_mag * np.cos(angle)
    ky = k_mag * np.sin(angle)
    pux = np.cos(push_angle)
    puy = np.sin(push_angle)

    # Red-noise amplitude spectrum: big slow gusts dominate.  Normalise to 1.
    amp = 1.0 / wavelength
    amp = amp / amp.sum()

    return GustModes(
        kx=kx.astype(np.float32),
        ky=ky.astype(np.float32),
        omega=omega.astype(np.float32),
        phase0=phase0.astype(np.float32),
        pux=pux.astype(np.float32),
        puy=puy.astype(np.float32),
        amp=amp.astype(np.float32),
    )


def eval_gusts(
    modes: GustModes,
    X: np.ndarray,
    Y: np.ndarray,
    t_eff: float,
    mean: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the gust velocity shape at world points ``(X, Y)`` and time.

    Vectorised core of the wind field.  For each mode the phase is::

        theta = (kx*X + ky*Y) - (kx*mean_x + ky*mean_y + omega) * t_eff + phase0

    The ``-(k · mean) * t_eff`` term is the **downwind advection**: a crest at
    position ``p`` at time ``t`` reappears at ``p + mean*dt`` at time ``t+dt``,
    so gust bands travel with the mean wind.  ``-omega * t_eff`` adds the
    mode's own intrinsic pulsing on top.

    The per-mode contributions ``amp * push_dir * sin(theta)`` are summed over
    the mode axis to give the 2-D gust velocity shape (dimensionless, ~[-1, 1];
    the caller multiplies by a weather ``gust_gain`` to get m/s).

    Parameters
    ----------
    modes : GustModes
        The drawn-once basis from :func:`build_modes`.
    X, Y : numpy.ndarray
        World coordinates (meters) of the sample points, any matching shape
        ``S`` (typically ``(cells, cells)`` cell-centre meshes).
    t_eff : float
        Effective time in seconds (game time, optionally storm-frequency
        scaled by the caller).  Pure function of this — no internal state.
    mean : tuple[float, float]
        Mean wind velocity ``(mean_x, mean_y)`` in m/s (drives advection).

    Returns
    -------
    (gust_x, gust_y) : tuple[numpy.ndarray, numpy.ndarray]
        Two arrays of shape ``S``, the gust velocity shape components.

    Example
    -------
    >>> import numpy as np
    >>> from fire_engine.core.rng import set_world_seed
    >>> set_world_seed(1337)
    >>> modes = build_modes(Config())
    >>> X, Y = np.meshgrid(np.arange(8.0), np.arange(8.0), indexing="ij")
    >>> gx, gy = eval_gusts(modes, X, Y, 1.0, (3.0, 0.0))
    >>> gx.shape == X.shape
    True
    """
    # Broadcast the (..., 1) sample grid against the (M,) mode constants → a
    # trailing mode axis we sum over.  No Python loop over cells or modes.
    gx_ = X[..., None]
    gy_ = Y[..., None]
    kx = modes.kx
    ky = modes.ky
    mx = float(mean[0])
    my = float(mean[1])

    # Phase advance: spatial term minus (advection + intrinsic) * time, + offset.
    drift = (kx * mx + ky * my + modes.omega) * float(t_eff)
    theta = (kx * gx_ + ky * gy_) - drift + modes.phase0
    s = np.sin(theta)

    gust_x = (modes.amp * modes.pux * s).sum(axis=-1)
    gust_y = (modes.amp * modes.puy * s).sum(axis=-1)
    return gust_x.astype(np.float32), gust_y.astype(np.float32)
