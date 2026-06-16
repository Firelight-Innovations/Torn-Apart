"""
weather/synoptic.py — Closed-form synoptic (large-scale) wind flow.

The synoptic flow is the slow, hours-scale "steering current" that carries
air masses — and, from M2 on, storm cells — across the world.  It is the
weather system's analogue of the wind package's spectral gust modes: a
**closed-form pure function of (world_seed, game time)**, never integrated
state.  That buys the same things it bought the wind field:

* zero save bytes (recompute from seed at any time),
* bit-reproducibility across processes and reloads,
* free evaluation at *any* time — past, future, or out of order.

Model
-----
The wind **vector** is a constant prevailing term plus a small set of seeded
vector sinusoids (per axis)::

    W(t) = C + Σ_i a_i · sin(ω_i t + φ_i)        [m/s, world XY, game time t]

and the air-mass **displacement** is its exact analytic integral::

    D(t) = C·t − Σ_i (a_i / ω_i) · cos(ω_i t + φ_i) + D0     [m], D(0) = 0

so ``dD/dt ≡ W(t)`` holds to machine precision — anything advected by
``D(t2) − D(t1)`` moves exactly with the synoptic wind.  Storm-cell tracks
built on ``D`` therefore *bend automatically* when the wind direction
drifts, with no per-frame integration anywhere.

Speed band guarantee
--------------------
With ``v_mean = (v_min + v_max) / 2`` and the per-axis ripple amplitudes
budgeted to ``Σ_i |a_i| ≤ (v_max − v_min) / (2·√2)`` per axis, the triangle
inequality bounds the speed to ``|W(t)| ∈ [v_min, v_max]`` for all t.
Direction meanwhile swings up to ±~50° around the prevailing heading over
periods of a few game hours — enough for "the storm was passing by, then
the wind shifted" moments, while keeping a believable per-world prevailing
direction.

Units: meters, meters/second, game seconds (1 game hour = 3600 game s).

Example
-------
    from fire_engine.core import load_config, set_world_seed
    from fire_engine.world.weather.synoptic import Synoptic

    set_world_seed(1337)
    syn = Synoptic(load_config())
    (dx, dy), speed = syn.wind(t_abs=3 * 86400.0 + 8.5 * 3600.0)
    track_shift = syn.displacement(7200.0) - syn.displacement(3600.0)  # (2,)

Docs: docs/systems/world.weather.md
"""

from __future__ import annotations

import math

import numpy as np

from fire_engine.core.config import Config
from fire_engine.core.rng import for_domain

__all__ = ["Synoptic"]

#: Game seconds per game hour (clock-independent constant).
_HOUR_S: float = 3600.0


class Synoptic:
    """
    Seeded closed-form synoptic wind ``W(t)`` and displacement ``D(t)``.

    All parameters are drawn once from ``for_domain("weather", "synoptic")``
    at construction; every method is then a pure function of game time.
    Construction is cheap (a few dozen scalars) — systems may freely build
    their own instance; identical seeds yield identical flows.

    Parameters
    ----------
    config : Config
        Reads ``weather_synoptic_components``, ``weather_synoptic_speed_min_ms``
        / ``..._max_ms`` (the guaranteed speed band, m/s) and
        ``weather_synoptic_period_min_h`` / ``..._max_h`` (sinusoid period
        band, game hours).

    Example
    -------
    >>> syn = Synoptic(load_config())
    >>> (ux, uy), v = syn.wind(86400.0)     # noon-ish day 1
    >>> abs(math.hypot(ux, uy) - 1.0) < 1e-9 and v > 0.0
    True

    Docs: docs/systems/world.weather.md
    """

    def __init__(self, config: Config) -> None:
        rng = for_domain("weather", "synoptic")
        n = int(config.weather_synoptic_components)
        v_min = float(config.weather_synoptic_speed_min_ms)
        v_max = float(config.weather_synoptic_speed_max_ms)
        if not (0.0 <= v_min < v_max):
            raise ValueError(f"weather synoptic speed band invalid: [{v_min}, {v_max}]")
        v_mean = 0.5 * (v_min + v_max)

        # Prevailing heading: uniform per world.
        theta0 = float(rng.uniform(0.0, 2.0 * math.pi))
        self._c: np.ndarray = v_mean * np.array(
            [math.cos(theta0), math.sin(theta0)], dtype=np.float64
        )

        # Periods log-uniform in the configured band (game seconds).
        p_min = float(config.weather_synoptic_period_min_h) * _HOUR_S
        p_max = float(config.weather_synoptic_period_max_h) * _HOUR_S
        periods = np.exp(rng.uniform(math.log(p_min), math.log(p_max), size=n))
        self._omega: np.ndarray = 2.0 * math.pi / periods  # (n,)

        # Per-axis amplitudes: random weights normalised so Σ|a| per axis
        # equals the ripple budget → |W| stays inside [v_min, v_max] (see
        # module docstring).  Shape (2, n): row 0 = x, row 1 = y.
        ripple = (v_max - v_min) / (2.0 * math.sqrt(2.0))
        raw = rng.uniform(0.5, 1.0, size=(2, n))
        self._amp: np.ndarray = raw / raw.sum(axis=1, keepdims=True) * ripple

        # Per-axis phases, uniform.  Shape (2, n).
        self._phase: np.ndarray = rng.uniform(0.0, 2.0 * math.pi, size=(2, n))

        # Integration constant so that D(0) == (0, 0).
        self._d0: np.ndarray = ((self._amp / self._omega) * np.cos(self._phase)).sum(axis=1)

    # ------------------------------------------------------------------
    # Evaluation (scalar t → shape (2,);  (M,) array t → shape (M, 2))
    # ------------------------------------------------------------------

    def wind_vec(self, t_abs: float | np.ndarray) -> np.ndarray:
        """
        Synoptic wind vector(s) at absolute game time ``t_abs``.

        Parameters
        ----------
        t_abs : float | np.ndarray — absolute game seconds (day·86400 + tod).

        Returns
        -------
        np.ndarray — shape ``(2,)`` for scalar input, ``(M, 2)`` for an
        ``(M,)`` array.  Units m/s, world XY, direction the wind blows TOWARD.

        Docs: docs/systems/world.weather.md
        """
        t = np.asarray(t_abs, dtype=np.float64)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)  # (M,)
        # (M, 1, n) phase argument broadcast against (2, n) coefficients.
        arg = self._omega[None, None, :] * t[:, None, None] + self._phase[None, :, :]
        w = self._c[None, :] + (self._amp[None, :, :] * np.sin(arg)).sum(axis=2)
        result: np.ndarray = w[0] if scalar else w
        return result

    def displacement(self, t_abs: float | np.ndarray) -> np.ndarray:
        """
        Air-mass displacement ``D(t)`` since t = 0, in meters.

        ``D(t2) − D(t1)`` is exactly how far the synoptic flow carries an air
        mass between two instants (``dD/dt ≡ wind_vec``).  Storm-cell centers
        are ``spawn_pos + D(t) − D(spawn_time)``.

        Parameters / Returns — same shapes and conventions as
        :meth:`wind_vec`; units meters.

        Docs: docs/systems/world.weather.md
        """
        t = np.asarray(t_abs, dtype=np.float64)
        scalar = t.ndim == 0
        t = np.atleast_1d(t)
        arg = self._omega[None, None, :] * t[:, None, None] + self._phase[None, :, :]
        ripple = -((self._amp / self._omega)[None, :, :] * np.cos(arg)).sum(axis=2)
        d = self._c[None, :] * t[:, None] + ripple + self._d0[None, :]
        result: np.ndarray = d[0] if scalar else d
        return result

    def wind(self, t_abs: float) -> tuple[tuple[float, float], float]:
        """
        Convenience scalar form: (unit XY direction, speed m/s) at ``t_abs``.

        The direction is the unit vector of :meth:`wind_vec`; the speed is
        its magnitude, guaranteed inside the configured band.

        Example
        -------
        >>> (ux, uy), v = syn.wind(12 * 3600.0)
        >>> 1.5 <= v <= 11.0       # with default config band
        True

        Docs: docs/systems/world.weather.md
        """
        w = self.wind_vec(float(t_abs))
        speed = float(math.hypot(w[0], w[1]))
        if speed < 1e-9:  # unreachable with v_min > 0
            return (1.0, 0.0), 0.0
        return (float(w[0] / speed), float(w[1] / speed)), speed
