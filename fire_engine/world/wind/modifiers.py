"""
wind/modifiers.py — Pluggable in-place wind-field modifiers.

A :class:`WindModifier` is a small object that mutates the per-cell wind
velocity / turbulence arrays **in place** after the base spectral field is
composed but before it is published.  This is the extension seam for systems
that add *localised, non-ambient* wind on top of the global field:

- the future **volumetric weather** system (a storm gust front approaching
  from a distance — :class:`GustFront` below is its working stand-in), and
- a future full-3-D corrector that registers as a modifier without touching
  any wind API (the 2.5-D → 3-D upgrade path called out in the plan).

Modifiers are applied in registration order each frame.  Because a modifier
that is a **pure function of (seed_key, time)** adds nothing to saves and is
bit-reproducible, the field keeps its zero-save-bytes / determinism guarantee
as long as modifiers obey that discipline.  :class:`GustFront` does.

Units
-----
Positions/extents meters, velocities m/s, time seconds, ``turb`` dimensionless
(~0..3).  The ``X, Y`` meshes passed to :meth:`WindModifier.apply` are the
cell-centre world coordinates (from :class:`~fire_engine.world.wind.region.WindRegion`).

No panda3d.  No per-cell Python loops.

Example
-------
>>> import numpy as np
>>> from fire_engine.world.wind.modifiers import GustFront
>>> X, Y = np.meshgrid(np.arange(0.0, 64.0, 4.0),
...                    np.arange(0.0, 64.0, 4.0), indexing="ij")
>>> vx = np.zeros_like(X); vy = np.zeros_like(X); turb = np.zeros_like(X)
>>> front = GustFront(seed_key=("demo",), direction=(1.0, 0.0), speed=12.0,
...                   strength=6.0, width_m=20.0, period_m=400.0)
>>> front.apply(X, Y, t=3.0, vx=vx, vy=vy, turb=turb)   # mutates in place
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["WindModifier", "GustFront"]


@runtime_checkable
class WindModifier(Protocol):
    """
    In-place modifier of the composed wind field, applied before publish.

    Implementations mutate ``vx``, ``vy`` and ``turb`` (all same-shaped
    ``float32`` ``(cells, cells)`` arrays, indexed ``[x, y]``) in place; the
    return value is ignored.  ``X`` / ``Y`` are the matching cell-centre world
    coordinate meshes (meters); ``t`` is the field's evaluation time (seconds).

    Keep implementations a **pure function of their own seed/config and ``t``**
    (no accumulated state) to preserve the field's determinism and
    zero-save-bytes guarantee.

    Example
    -------
    >>> class Calm:                          # zero out all wind in a region
    ...     def apply(self, X, Y, t, vx, vy, turb):
    ...         mask = (X**2 + Y**2) < 100.0
    ...         vx[mask] = 0.0; vy[mask] = 0.0
    """

    def apply(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        t: float,
        vx: np.ndarray,
        vy: np.ndarray,
        turb: np.ndarray,
    ) -> None:
        """Mutate ``vx`` / ``vy`` / ``turb`` in place for time ``t``."""
        ...


class GustFront:
    """
    A moving line-front gust — the volumetric-weather seam's working example.

    Models a band of stronger wind sweeping across the region along a fixed
    direction: a velocity ramp plus a turbulence spike, peaking on a line that
    advances at ``speed`` m/s and repeats every ``period_m`` meters (so the
    demo loops indefinitely).  It is a **pure function of (seed_key, t)** — no
    internal state advances between calls — so it adds zero save bytes and is
    bit-reproducible, exactly like the spectral base field.

    When the real volumetric-weather system arrives it will drive fronts with
    physically-meaningful position/strength; this class proves the seam and is
    handy as a devtools toy ("blow a gust through the field now").

    Parameters
    ----------
    seed_key : tuple
        Identity key (e.g. ``("storm", front_id)``).  Hashed to a stable phase
        offset so two fronts with different keys are out of step; reserved for
        future per-front jitter.  Part of the determinism contract.
    direction : tuple[float, float]
        Unit-ish XY direction the front travels toward; normalised internally.
    speed : float
        Front travel speed in m/s (how fast the band sweeps across).
    strength : float
        Peak added wind speed (m/s) along the front line, in ``direction``.
    width_m : float
        Half-width of the front band in meters (the Gaussian falloff sigma).
    period_m : float, default 400.0
        Spatial repeat distance in meters — the front re-enters this far behind
        itself, so the band loops across the region forever.
    turb_gain : float, default 0.6
        Turbulence added at the front peak (dimensionless).

    Example
    -------
    >>> front = GustFront(("demo",), (1.0, 0.0), speed=12.0, strength=6.0,
    ...                   width_m=20.0)
    >>> # apply() pushes air toward +X in a moving 20 m-wide band.
    """

    def __init__(
        self,
        seed_key: tuple,
        direction: tuple[float, float],
        speed: float,
        strength: float,
        width_m: float,
        period_m: float = 400.0,
        turb_gain: float = 0.6,
    ) -> None:
        self.seed_key = tuple(seed_key)
        dx, dy = float(direction[0]), float(direction[1])
        norm = math.hypot(dx, dy)
        if norm < 1e-9:
            dx, dy, norm = 1.0, 0.0, 1.0
        self.dir = (dx / norm, dy / norm)
        self.speed = float(speed)
        self.strength = float(strength)
        self.width_m = max(1e-3, float(width_m))
        self.period_m = max(1.0, float(period_m))
        self.turb_gain = float(turb_gain)
        # Stable per-key phase offset (meters) so distinct fronts don't overlap.
        # Pure: derived from the key text, not from any process-salted hash.
        digest = abs(hash(repr(self.seed_key))) % 100000
        self._phase_m = (digest / 100000.0) * self.period_m

    def apply(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        t: float,
        vx: np.ndarray,
        vy: np.ndarray,
        turb: np.ndarray,
    ) -> None:
        """
        Add the moving front's velocity ramp + turbulence spike in place.

        The signed distance of each cell along ``direction`` is wrapped into
        ``[-period/2, period/2)`` around the front's current position
        ``speed * t``; a Gaussian of half-width ``width_m`` shapes the band.

        Parameters
        ----------
        X, Y : numpy.ndarray
            Cell-centre world coordinate meshes (meters), shape ``(cells, cells)``.
        t : float
            Field evaluation time (seconds).
        vx, vy, turb : numpy.ndarray
            Field arrays mutated in place (``float32 (cells, cells)``).
        """
        dx, dy = self.dir
        # Distance of each cell along the travel direction (meters).
        along = X * dx + Y * dy
        front_pos = self.speed * float(t) + self._phase_m
        # Wrap the relative distance into a centred period so the band loops.
        rel = (along - front_pos + 0.5 * self.period_m) % self.period_m - 0.5 * self.period_m
        band = np.exp(-0.5 * (rel / self.width_m) ** 2).astype(np.float32)

        vx += (self.strength * dx) * band
        vy += (self.strength * dy) * band
        turb += self.turb_gain * band
