"""
wind/types.py — Frozen support types for the wind package.

Groups the trivial ``@dataclass(frozen=True)`` support types shared across
wind modules: :class:`WindSnapshot`, :class:`VenturiJob`, and
:class:`VenturiResult`.  Behavioural classes (``WindField``,
``VenturiWorker``) stay in their own modules.

Docs: docs/systems/world.wind.md
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "VenturiJob",
    "VenturiResult",
    "WindSnapshot",
]


@dataclass(frozen=True)
class WindSnapshot:
    """
    Atomically-published immutable snapshot of the wind field at one instant.

    The main thread builds a new snapshot each :meth:`WindField.update` and
    publishes it by a single reference assignment (atomic in CPython, no
    locks); :meth:`WindField.sample` and :func:`pack_wind_field` always read
    the current snapshot, so a reader never sees a half-written field.

    Attributes
    ----------
    field : numpy.ndarray
        ``float32 (cells, cells, 4)`` indexed ``[x, y]``: channels
        ``vx, vy, turb, reserved`` (m/s, m/s, dimensionless ~0..3, 0).
    origin_m : tuple[float, float]
        World XY (meters) of cell ``(0, 0)``'s corner.
    cell_m : float
        Cell edge in meters (4.0).
    cells : int
        Cells per axis (64).
    wind_time : float
        Seconds the field was evaluated at (the shared clock value).

    Example
    -------
    >>> snap = field.snapshot
    >>> snap.field.shape
    (64, 64, 4)

    Docs: docs/systems/world.wind.md
    """

    field: np.ndarray
    origin_m: tuple[float, float]
    cell_m: float
    cells: int
    wind_time: float


@dataclass(frozen=True)
class VenturiJob:
    """
    One terrain-venturi solve request.

    Attributes
    ----------
    origin_cell : tuple[int, int]
        Wind-region origin (cell ``(0, 0)``'s corner, in wind cells) the solve
        is for.  Committed to the field only when a result with this origin
        drains — a result whose ``origin_cell`` no longer matches the region's
        current origin is discarded (the field re-submits on recenter).
    cells : int
        Wind cells per axis (e.g. 64).
    cell_m : float
        Wind cell edge in meters (e.g. 4.0).
    chunk_size : int
        Voxels per chunk edge (``config.chunk_size``, 32).
    voxel_size : float
        Meters per voxel (``config.voxel_size``, 0.5).  Each wind cell of
        ``cell_m`` therefore covers ``cell_m / voxel_size`` voxels per axis
        (8 at the defaults).
    ground_band : tuple[float, float]
        World Z band ``(z_lo, z_hi)`` over which column occupancy is folded
        (``[ground, ground + wind_layer_m]``).
    materials : dict[tuple[int, int, int], numpy.ndarray]
        Snapshot of ``uint8 (S, S, S)`` material arrays for the chunks the solve
        will read (coord → array).  Built on the main thread; referenced, not
        copied (like ``AssemblyJob``).  Cells with no loaded chunk data are
        treated as fully open.
    venturi_iters : int
        Jacobi flux-relaxation sweeps (``config.wind_venturi_iters``, ~8).
    venturi_max : float
        Clamp on the speed-up multiplier (``config.wind_venturi_max``, 2.2).
    deflect_gain : float
        Sideways-deflection gain (``config.wind_deflect_gain``, 0.15).
    seq : int
        Monotonic id; lets the consumer drop a superseded result.

    Docs: docs/systems/world.wind.md
    """

    origin_cell: tuple[int, int]
    cells: int
    cell_m: float
    chunk_size: int
    voxel_size: float
    ground_band: tuple[float, float]
    materials: dict[tuple[int, int, int], np.ndarray]
    venturi_iters: int
    venturi_max: float
    deflect_gain: float
    seq: int


@dataclass(frozen=True)
class VenturiResult:
    """
    A finished terrain-venturi correction grid.

    Attributes
    ----------
    origin_cell : tuple[int, int]
        The wind-region origin the solve was for (commit to the field only when
        it still matches the region's current origin).
    speedup : numpy.ndarray
        ``float32 (cells, cells)`` indexed ``[x, y]`` — multiplicative wind
        speed-up per cell, in ``[1.0, venturi_max]`` (1.0 = no funneling).
    deflect : numpy.ndarray
        ``float32 (cells, cells, 2)`` indexed ``[x, y, :]`` — additive sideways
        deflection (openness gradient × ``deflect_gain``), pushing flow around
        walls.  Scaled by the mean-wind magnitude when applied.
    seq : int
        Echoes the job's ``seq``.

    Docs: docs/systems/world.wind.md
    """

    origin_cell: tuple[int, int]
    speedup: np.ndarray
    deflect: np.ndarray
    seq: int
