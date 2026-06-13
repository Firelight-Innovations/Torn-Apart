"""
weather/lightning.py — Deterministic lightning strike schedule (M7, headless).

A THUNDERSTORM :class:`~fire_engine.weather.StormCell` fires strikes on a
**deterministic Poisson process**: inter-arrival gaps are exponential at a peak
rate (``config.weather_lightning_strikes_per_min``), thinned by the cell's
live intensity so a growing/decaying storm strikes less than one at its plateau.
:func:`scheduled_strikes` returns every strike whose time falls in a half-open
window ``[t0, t1)``.

Load-resume safety (the headline invariant)
--------------------------------------------
The schedule is a **pure function of the cell** (its id, spawn time, lifetime) —
NOT of how the window is sliced.  All strike times are drawn from a single
deterministic stream anchored at the cell's spawn time and walked forward; a
window just selects the strikes that land inside it.  Therefore::

    scheduled_strikes(cell, t0, t1) ==
        scheduled_strikes(cell, t0, tm) + scheduled_strikes(cell, tm, t1)

for any split ``tm`` — no double-count, no gap.  This is what makes a save/load
mid-storm identical: the system recomputes the schedule from the seed and simply
does not re-emit strikes from before the resume time (it advances
``last_strike_time`` across the gap).  Thinning uses the cell intensity at each
candidate's own time, so the decision for a given candidate is independent of the
window too.

All randomness flows through ``for_domain("weather", "lightning", cell_id)``
(Hard Rule 2).  No panda3d (Hard Rule 1).

Units: meters, game seconds, strikes/minute.

Example
-------
    from fire_engine.core import load_config, set_world_seed
    from fire_engine.weather.cells import natural_cells, CellKind
    from fire_engine.weather.lightning import scheduled_strikes

    set_world_seed(1337)
    cfg = load_config()
    cell = next(c for c in natural_cells(5, cfg) if c.kind is CellKind.THUNDERSTORM)
    strikes = scheduled_strikes(cell, cell.spawn_time, cell.spawn_time + 600.0, cfg)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from fire_engine.core.config import Config
from fire_engine.core.rng import for_domain
from fire_engine.weather.cells import StormCell

__all__ = ["StrikeParams", "cell_id_int", "scheduled_strikes"]

#: Game seconds per minute (rate is configured per game minute).
_MIN_S: float = 60.0


@dataclass(frozen=True)
class StrikeParams:
    """
    One scheduled lightning strike (deterministic, before rendering).

    Attributes
    ----------
    time_abs : float — absolute game time of the strike, seconds.
    pos_xy : tuple[float, float] — world XY of the strike within the cell
        footprint (meters); the renderer lifts this to the cloud base and drops
        it to the ground/roof Z.
    intensity : float — 0–1 strike brightness/scale (the cell intensity at the
        strike time).
    seed : int — per-strike bolt RNG seed (deterministic geometry); derived from
        the cell id and the strike index so each strike's channel is distinct
        yet reproducible.
    """

    time_abs: float
    pos_xy: tuple[float, float]
    intensity: float
    seed: int


def cell_id_int(cell_id: str) -> int:
    """
    Deterministic non-negative int digest of a cell's string id.

    The :class:`~fire_engine.core.event_bus.LightningStrikeEvent` carries a
    ``cell_id: int`` and the RNG domain keys want a stable scalar, so the cell's
    string id (``"n:5:2"`` / ``"s:3"``) is hashed to an int with the same
    cross-process-stable digest the RNG service uses (blake2b — never Python's
    salted ``hash()``).

    Parameters
    ----------
    cell_id : str — the cell's stable string id.

    Returns
    -------
    int — a 31-bit non-negative digest (fits a signed shader/event int).
    """
    import hashlib
    d = hashlib.blake2b(str(cell_id).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(d, "big", signed=False) % (2 ** 31)


def scheduled_strikes(
    cell: StormCell,
    t0: float,
    t1: float,
    config: Config,
) -> list[StrikeParams]:
    """
    All lightning strikes the ``cell`` fires in the window ``[t0, t1)``.

    Deterministic Poisson schedule (see the module docstring): exponential
    inter-arrival gaps at the peak rate ``config.weather_lightning_strikes_per_min``,
    each candidate kept with probability equal to the cell's intensity at that
    time (thinning — a growing/decaying storm strikes less than at its plateau).
    Empty unless ``cell`` is a THUNDERSTORM.

    The result for any window equals the concatenation of the results for any
    partition of that window (load-resume safe) — strike times come from one
    stream anchored at the cell spawn time, and each candidate's keep decision
    uses only its own time.

    Parameters
    ----------
    cell : StormCell — the source cell (only THUNDERSTORM cells strike).
    t0 : float — window start, inclusive (absolute game seconds).
    t1 : float — window end, exclusive (absolute game seconds).
    config : Config — reads ``weather_lightning_strikes_per_min`` and the bolt
        cap (for the per-strike seed range).

    Returns
    -------
    list[StrikeParams] — strikes with ``t0 <= time_abs < t1``, in time order.
    """
    from fire_engine.weather.cells import CellKind

    if cell.kind is not CellKind.THUNDERSTORM:
        return []
    if t1 <= t0:
        return []

    rate_per_min = float(config.weather_lightning_strikes_per_min)
    if rate_per_min <= 0.0:
        return []
    rate_per_s = rate_per_min / _MIN_S        # peak λ (strikes per game second)

    # Anchor the stream at the cell's spawn time and clamp the scan to the cell's
    # lifetime — strikes only happen while the cell is alive.
    cid = cell_id_int(cell.id)
    rng = for_domain("weather", "lightning", cid)

    t_start = float(cell.spawn_time)
    t_end = t_start + float(cell.duration_s)
    scan_hi = min(float(t1), t_end)
    if scan_hi <= t_start:
        return []

    out: list[StrikeParams] = []
    t = t_start
    idx = 0
    # Walk the exponential gaps forward from spawn; select those in [t0, t1).
    # We always start from spawn (not t0) so the stream is window-independent —
    # the cost is O(strikes since spawn), tiny at ~2.5/min over a storm lifetime.
    while True:
        gap = float(rng.exponential(1.0 / rate_per_s))
        t += gap
        if t >= scan_hi:
            break
        idx += 1
        # Per-candidate keep draw (drawn for EVERY candidate so the stream stays
        # aligned regardless of the window) — thinned by the live cell intensity.
        keep = float(rng.random())
        intensity = float(cell.intensity(t))
        if keep >= intensity:
            continue
        if t < float(t0):
            continue
        # Strike XY: a gaussian offset within the cell footprint (relative to the
        # cell center at `t`).  Drawn from a SEPARATE per-strike stream keyed by
        # the strike index — NOT the gap stream — so the gap/keep stream stays
        # perfectly window-independent (a kept strike before t0 must not consume
        # offset draws from the main stream, or splitting the window would
        # misalign it).  Returned footprint-relative so the schedule needs no
        # live wind; the consumer adds the synoptic-advected center for world XY.
        xy_rng = for_domain("weather", "lightning", cid, "xy", idx)
        pos_xy = _strike_offset(cell, t, xy_rng)
        seed = (cid * 1_000_003 + idx) % (2 ** 31)
        out.append(StrikeParams(
            time_abs=t,
            pos_xy=pos_xy,
            intensity=intensity,
            seed=int(seed),
        ))
    return out


def _strike_offset(cell: StormCell, t: float, rng) -> tuple[float, float]:
    """
    Strike XY offset within the cell footprint (gaussian, clamped to radius).

    Sampled in the footprint-relative frame so the schedule needs no live wind:
    the returned ``(dx, dy)`` is relative to the cell center at ``t``.  The
    consumer adds the (synoptic-advected) center to get the world XY.
    """
    r = float(cell.radius(t))
    # Gaussian within ~one radius (σ = r/2), clamped to the footprint.
    dx = float(rng.normal(0.0, 0.5 * r))
    dy = float(rng.normal(0.0, 0.5 * r))
    d = math.hypot(dx, dy)
    if d > r and d > 1e-9:
        dx *= r / d
        dy *= r / d
    return (dx, dy)
