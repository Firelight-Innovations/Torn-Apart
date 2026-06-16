"""
weather/cells.py — Spatial storm cells and the natural daily spawn schedule.

A :class:`StormCell` is a single weather feature at a world position — a
shower, thunderstorm, cloud bank, or fog bank.  It is **not** an integrated
particle: its center, radius, and intensity are all closed-form pure
functions of game time, riding the synoptic displacement ``D(t)`` (see
:mod:`fire_engine.world.weather.synoptic`).  Because the cell track is
``spawn_pos + (D(t) − D(spawn_time)) + drift_bias·(t − spawn_time)``, a
drifting synoptic wind **bends every cell's path automatically** — "the wind
shifted and the storm turned toward you" — with zero per-frame integration
and zero save bytes.

Natural weather is a per-day draw:

* a **regime** (HIGH_PRESSURE / MIXED / FRONTAL) sets the day's ambient cloud
  cover/density, how likely each spawn slot fires, and which cell kinds it
  tends to produce — ``for_domain("weather", "regime", day)``;
* up to ``weather_spawn_slots_per_day`` candidate cells, each accepted against
  the regime's spawn probability — ``for_domain("weather", "cell", day, slot)``.

Everything here is a pure function of (world_seed, day): two processes with
the same seed spawn bit-identical cells, so saves store nothing for natural
weather.  Summoned cells (M8) are the only saved deviation.

Units: meters, game seconds (1 game hour = 3600 s), m/s.

Docs: docs/systems/world.weather.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from fire_engine.core.config import Config
from fire_engine.core.rng import for_domain
from fire_engine.world.weather.types import CellKind, Regime  # re-exported below

if TYPE_CHECKING:
    from fire_engine.world.weather.synoptic import Synoptic

__all__ = [
    "CellKind",
    "Regime",
    "StormCell",
    "day_regime",
    "natural_cells",
    "regime_ambient",
]

#: Game seconds per game day (matches ``sky.celestial.GAME_SECONDS_PER_DAY``;
#: defined locally so ``weather/`` never imports ``sky/`` — sky imports
#: weather, and a cycle would break the layering).
_DAY_S: float = 24.0 * 3600.0

#: Footprint sharpness: ``contribution = intensity·exp(−(d/radius)²·k)`` with
#: ``k = ln(50)`` so the influence has fallen to 1/50 of peak at exactly one
#: ``radius(t)`` from the center — a crisp but soft-edged Gaussian disc.
_FOOTPRINT_K: float = math.log(50.0)


def _smoothstep(x: float, lo: float, hi: float) -> float:
    """Hermite smoothstep clamped to [0, 1] (local copy — no sky import)."""
    if hi <= lo:
        return 0.0 if x < lo else 1.0
    t = (x - lo) / (hi - lo)
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return t * t * (3.0 - 2.0 * t)


# CellKind and Regime are defined in types.py and re-exported here for
# backward-compatible imports (``from fire_engine.world.weather.cells import CellKind``).
# The structure checker counts only ``class X:`` *definitions*, so these
# re-exports do not count toward the one-class limit.

#: Stable kind order for the per-regime spawn distributions below.
_KIND_ORDER: tuple[CellKind, ...] = (
    CellKind.CLOUD_BANK,
    CellKind.SHOWER,
    CellKind.THUNDERSTORM,
    CellKind.FOG_BANK,
)

#: Stable regime order for the per-day regime draw.
_REGIME_ORDER: tuple[Regime, ...] = (
    Regime.HIGH_PRESSURE,
    Regime.MIXED,
    Regime.FRONTAL,
)

#: Per-day regime probabilities (sums to 1): calm days dominate, fronts rare.
_REGIME_P: tuple[float, ...] = (0.40, 0.40, 0.20)

#: Regime → (ambient cloud_coverage, ambient cloud_density) away from cells.
#: The coverage values sit cleanly inside the :func:`classify` buckets so a
#: cell-free day reads as its regime: HIGH ≈ CLEAR (cov < 0.3), MIXED ≈ CLOUDY
#: (0.3 < cov ≤ 0.7), FRONTAL ≈ OVERCAST (cov > 0.7).  Don't nudge these onto a
#: threshold boundary — it collapses distinct regimes to the same label.
_REGIME_AMBIENT: dict[Regime, tuple[float, float]] = {
    Regime.HIGH_PRESSURE: (0.08, 0.30),
    Regime.MIXED: (0.40, 0.52),
    Regime.FRONTAL: (0.75, 0.72),
}

#: Regime → probability that any one spawn slot actually produces a cell.
_REGIME_SPAWN_PROB: dict[Regime, float] = {
    Regime.HIGH_PRESSURE: 0.12,
    Regime.MIXED: 0.45,
    Regime.FRONTAL: 0.75,
}

#: Regime → cell-kind distribution in ``_KIND_ORDER`` order (each sums to 1).
_REGIME_KIND_P: dict[Regime, tuple[float, ...]] = {
    Regime.HIGH_PRESSURE: (0.70, 0.30, 0.00, 0.00),
    Regime.MIXED: (0.35, 0.45, 0.15, 0.05),
    Regime.FRONTAL: (0.20, 0.45, 0.35, 0.00),
}


@dataclass(frozen=True)
class StormCell:
    """
    One spatial weather feature with a closed-form analytic track.

    Attributes
    ----------
    id : str — stable identifier: ``"n:{day}:{slot}"`` for a natural cell,
        ``"s:{n}"`` for a summoned one.  Used to suppress/dedupe in saves.
    kind : CellKind — what the cell does (see :class:`CellKind`).
    spawn_time : float — absolute game seconds when the cell appears.
    spawn_pos : tuple[float, float] — world XY of the cell center at
        ``spawn_time`` (meters).
    duration_s : float — total lifetime (grow + plateau + decay), game seconds.
    radius_m : float — footprint radius at peak (meters); the cell is smaller
        while growing and decaying (see :meth:`radius`).
    peak_intensity : float — 0–1 strength multiplier at plateau.
    drift_bias : tuple[float, float] — a small steady velocity (m/s) added to
        the synoptic drift, so co-spawned cells don't move in lockstep.

    The cell's center, radius, intensity, and footprint are all pure functions
    of time; nothing here is mutated after construction.

    Example
    -------
    >>> from fire_engine.world.weather.synoptic import Synoptic
    >>> from fire_engine.core import load_config
    >>> syn = Synoptic(load_config())
    >>> cell = StormCell("s:0", CellKind.SHOWER, 0.0, (0.0, 0.0),
    ...                   3600.0, 500.0, 0.8, (0.0, 0.0))
    >>> float(cell.intensity(1800.0))            # mid-life: at plateau
    0.8
    """

    id: str
    kind: CellKind
    spawn_time: float
    spawn_pos: tuple[float, float]
    duration_s: float
    radius_m: float
    peak_intensity: float
    drift_bias: tuple[float, float]

    # -- envelope -------------------------------------------------------

    def _envelope(self, t: float) -> float:
        """
        0–1 life-cycle shape at absolute time *t*: smoothstep grow over the
        first 20 % of life, flat plateau, smoothstep decay over the last 30 %.
        Zero outside ``[spawn_time, spawn_time + duration_s]``.
        """
        u = (t - self.spawn_time) / self.duration_s
        if u <= 0.0 or u >= 1.0:
            return 0.0
        grow = _smoothstep(u, 0.0, 0.20)
        decay = 1.0 - _smoothstep(u, 0.70, 1.0)
        return grow * decay

    def intensity(self, t: float) -> float:
        """Peak-scaled strength 0–``peak_intensity`` at absolute time *t*."""
        return self.peak_intensity * self._envelope(t)

    def radius(self, t: float) -> float:
        """
        Footprint radius (m) at *t*: 55 % of ``radius_m`` at birth/death,
        growing to full ``radius_m`` at plateau (cells spread as they mature).
        """
        return self.radius_m * (0.55 + 0.45 * self._envelope(t))

    def active(self, t: float) -> bool:
        """True while the cell is alive (``spawn_time < t < end``)."""
        return 0.0 < (t - self.spawn_time) < self.duration_s

    def center(self, t: float, synoptic: Synoptic) -> np.ndarray:
        """
        World-XY center at absolute time *t* (meters), shape ``(2,)``.

        Rides the **raw** synoptic displacement ``D(t) − D(spawn_time)`` plus
        the per-cell ``drift_bias``.  Pass the system's :class:`Synoptic`
        instance; never the gameplay-multiplied wind (see synoptic.py gotcha).
        """
        dt = t - self.spawn_time
        shift = synoptic.displacement(t) - synoptic.displacement(self.spawn_time)
        return np.array(
            [
                self.spawn_pos[0] + shift[0] + self.drift_bias[0] * dt,
                self.spawn_pos[1] + shift[1] + self.drift_bias[1] * dt,
            ],
            dtype=np.float64,
        )

    def contribution(self, points_xy: np.ndarray, t: float, synoptic: Synoptic) -> np.ndarray:
        """
        Influence 0–``peak_intensity`` of this cell at each query point.

        Parameters
        ----------
        points_xy : np.ndarray — shape ``(N, 2)`` world-XY query points (m).
        t : float — absolute game seconds.
        synoptic : Synoptic — the flow the cell rides.

        Returns
        -------
        np.ndarray — shape ``(N,)``: ``intensity(t)·exp(−(d/radius(t))²·k)``,
        the soft Gaussian footprint.  Zero everywhere when the cell is dead.
        """
        pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        amp = self.intensity(t)
        if amp <= 0.0:
            return np.zeros(pts.shape[0], dtype=np.float64)
        c = self.center(t, synoptic)
        d2 = ((pts - c[None, :]) ** 2).sum(axis=1)
        r = self.radius(t)
        result: np.ndarray = amp * np.exp(-(d2 / (r * r)) * _FOOTPRINT_K)
        return result


# ---------------------------------------------------------------------------
# Natural daily spawn schedule — pure function of (world_seed, day)
# ---------------------------------------------------------------------------


def day_regime(day: int) -> Regime:
    """
    The synoptic regime for *day* — pure function of (world_seed, day).

    Drawn from ``for_domain("weather", "regime", day)`` against
    ``_REGIME_P``; identical across processes and restarts.
    """
    rng = for_domain("weather", "regime", int(day))
    idx = int(rng.choice(len(_REGIME_ORDER), p=list(_REGIME_P)))
    return _REGIME_ORDER[idx]


def regime_ambient(regime: Regime) -> tuple[float, float]:
    """Ambient ``(cloud_coverage, cloud_density)`` away from any cell."""
    return _REGIME_AMBIENT[regime]


def natural_cells(day: int, config: Config) -> list[StormCell]:
    """
    All naturally-spawned cells whose ``spawn_time`` falls within *day*.

    Pure function of (world_seed, day): iterates ``weather_spawn_slots_per_day``
    candidate slots, accepting each against the day regime's spawn probability,
    and drawing kind/position/radius/duration/intensity/drift from
    ``for_domain("weather", "cell", day, slot)``.

    A cell may live on into the next day (up to ``weather_cell_duration_max_s``);
    callers that sample a given instant should union the cells of *day* and
    *day − 1* and filter by :meth:`StormCell.active`.

    Parameters
    ----------
    day : int — in-game day number.
    config : Config — reads the ``weather_*`` cell tuning fields.

    Returns
    -------
    list[StormCell] — possibly empty; deterministic for (seed, day).
    """
    day = int(day)
    regime = day_regime(day)
    spawn_prob = _REGIME_SPAWN_PROB[regime]
    kind_p = list(_REGIME_KIND_P[regime])

    domain = float(config.weather_domain_m)
    r_min = float(config.weather_cell_radius_min_m)
    r_max = float(config.weather_cell_radius_max_m)
    d_min = float(config.weather_cell_duration_min_s)
    d_max = float(config.weather_cell_duration_max_s)
    slots = int(config.weather_spawn_slots_per_day)
    day_start = day * _DAY_S

    cells: list[StormCell] = []
    for slot in range(slots):
        rng = for_domain("weather", "cell", day, slot)
        # Draw the accept/reject die first so the slot's remaining draws are
        # only consumed when it spawns — keeps the stream short and readable.
        if float(rng.random()) >= spawn_prob:
            continue
        kind = _KIND_ORDER[int(rng.choice(len(_KIND_ORDER), p=kind_p))]
        spawn_time = day_start + float(rng.uniform(0.0, _DAY_S))
        spawn_pos = (
            float(rng.uniform(-domain, domain)),
            float(rng.uniform(-domain, domain)),
        )
        if kind is CellKind.THUNDERSTORM:
            radius = float(rng.uniform(0.5 * (r_min + r_max), r_max))
        else:
            radius = float(rng.uniform(r_min, r_max))
        duration = float(rng.uniform(d_min, d_max))
        peak = float(rng.uniform(0.6, 1.0))
        drift_ang = float(rng.uniform(0.0, 2.0 * math.pi))
        drift_mag = float(rng.uniform(0.0, 0.6))
        drift_bias = (drift_mag * math.cos(drift_ang), drift_mag * math.sin(drift_ang))
        cells.append(
            StormCell(
                id=f"n:{day}:{slot}",
                kind=kind,
                spawn_time=spawn_time,
                spawn_pos=spawn_pos,
                duration_s=duration,
                radius_m=radius,
                peak_intensity=peak,
                drift_bias=drift_bias,
            )
        )
    return cells
