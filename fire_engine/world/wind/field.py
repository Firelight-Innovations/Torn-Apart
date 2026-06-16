"""
wind/field.py — The wind field: WindField + sample + pack.

This is the headless heart of the wind system: a player-centred, time-evolving
2.5-D wind velocity field that is the single source of truth for everything
wind-driven (grass, future flags/cloth/hair/water, ambient particles, physics
pushes, procedural wind audio).

Core design decision — spectral modes, not a random walk
--------------------------------------------------------
The field is **not** an accumulated Brownian random walk integrated frame by
frame.  It is a **sum of seeded spectral gust modes** (``wind/gusts.py``) whose
phases advance with the **wind clock** and advect downwind — a **pure function
of (world_seed, wind_time, world_position)**.  The wind clock runs at
``config.wind_time_scale`` seconds per REAL second, independent of the game
clock's timescale (gusts are an aesthetic real-time effect — at 60× game
pacing they would sweep the world 60× too fast); the render component
accumulates it from real frame ``dt`` and hands it to :meth:`WindField.update`.

The tradeoff: the *spectral sum* is stateless and bit-reproducible (zero save
bytes, free recenter); a random walk would be stateful and save-dependent.  See
``_field_helpers.py`` module docstring for full tradeoff discussion.

The field is **2.5-D**: a 2-D horizontal velocity grid plus an analytic
vertical boundary-layer profile (:func:`vertical_profile`).  ``vz`` is 0 for
now; WP2 adds analytic obstacle updraft in :meth:`WindField.sample`.

Determinism
-----------
Same ``world_seed`` + ``wind_time`` + ``SkyState`` + player cell ⇒ a
bit-identical :class:`WindSnapshot`.  All randomness flows through
``for_domain("wind", ...)``.  No Saveable — the field costs 0 save bytes by
construction.

Units & conventions
-------------------
- World meters, Z-up.  Velocities m/s.  ``turb`` dimensionless (~0..3).
- ``field`` array is indexed ``[x, y]`` (matching ``WindRegion.X/Y`` meshgrid
  ``ij`` order).  Channels are ``vx, vy, turb, reserved``.
- ``origin_m`` is the world XY of cell ``(0, 0)``'s corner (texel-(0,0)-corner
  convention, what the GPU binds as ``u_wind_origin``).

No panda3d.  No per-cell Python loops.

Example
-------
>>> from fire_engine.core.config import Config
>>> from fire_engine.core.rng import set_world_seed
>>> import numpy as np
>>> set_world_seed(1337)
>>> field = WindField(Config())
>>> field.update(dt=0.016, wind_time=10.0, sky_state=None, player_pos=(0, 0, 0))
>>> v = field.sample(np.array([[0.0, 0.0, 1.0]]))   # one point at z=1 m
>>> v.shape
(1, 3)

Docs: docs/systems/world.wind.md
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from fire_engine.core.config import Config
from fire_engine.world.wind._field_helpers import (
    _venturi_step,
    pack_wind_field,
    vertical_profile,
)
from fire_engine.world.wind.gusts import GustModes, build_modes, eval_gusts
from fire_engine.world.wind.protocols import WindModifier
from fire_engine.world.wind.region import WindRegion
from fire_engine.world.wind.types import WindSnapshot

if TYPE_CHECKING:
    from fire_engine.world.wind.worker import VenturiWorker

__all__ = [
    "WindField",
    "WindSnapshot",
    "pack_wind_field",
    "vertical_profile",
]


class WindField:
    """
    The player-centred, time-evolving 2.5-D wind velocity field.

    Build once at boot, call :meth:`update` once per frame (sub-millisecond,
    main thread — it only evaluates a 64×64 closed-form gust sum and publishes
    a snapshot), then :meth:`sample` it from anywhere (physics, audio) and pack
    it for the GPU with :func:`pack_wind_field`.

    Deterministic / Saveable-free: the field is a pure function of the world
    seed, game time, weather and player cell, so it costs **zero save bytes**.

    Parameters
    ----------
    config : Config
        Engine config (all ``wind_*`` fields).
    worker : VenturiWorker | None, default None
        Off-thread terrain-venturi solver
        (:class:`~fire_engine.world.wind.worker.VenturiWorker`).  ``None`` ⇒ the
        venturi correction stays identity (no funneling, ``vz == 0``); pass a
        started worker and feed ``chunks`` into :meth:`update` to enable wind
        speeding up through gaps and rising over windward obstacles.

    Example
    -------
    >>> from fire_engine.core.rng import set_world_seed
    >>> set_world_seed(1337)
    >>> field = WindField(Config())
    >>> field.update(0.016, 5.0, sky_state=None, player_pos=(0.0, 0.0, 0.0))
    >>> field.snapshot.cells
    64

    Docs: docs/systems/world.wind.md
    """

    # Class-level attribute annotations required by mypy --strict for the
    # venturi-orchestration helpers in _field_helpers.py that read/write these.
    _cfg: Config
    _worker: VenturiWorker | None
    _region: WindRegion
    _modes: GustModes
    _modifiers: list[WindModifier]
    _front: WindSnapshot | None
    _z_ground: float
    _venturi_speedup: np.ndarray
    _venturi_deflect: np.ndarray
    _updraft_gain_grid: np.ndarray
    _venturi_origin: tuple[int, int] | None
    _venturi_seq: int
    _venturi_ever_submitted: bool

    def __init__(self, config: Config, worker: VenturiWorker | None = None) -> None:
        self._cfg = config
        self._worker = worker  # VenturiWorker | None (terrain funneling)
        self._region = WindRegion(
            cells=int(config.wind_cells),
            cell_m=float(config.wind_cell_m),
            snap_cells=int(config.wind_snap_cells),
            margin_cells=int(config.wind_margin_cells),
        )
        # Spectral gust basis, drawn once (depends only on seed + config).
        self._modes: GustModes = build_modes(config)
        self._modifiers: list[WindModifier] = []
        # The atomically-published current snapshot (single-assignment publish).
        self._front: WindSnapshot | None = None
        # Ground height for the vertical profile (flat world baseline).
        self._z_ground = float(config.ground_height_m)

        # --- Venturi (terrain-funneling) state ------------------------------
        # The applied correction grids and the origin they were solved for.
        # Identity (no funneling) until the worker returns a matching-origin
        # result; re-submitted on every recenter (see update()).
        cells = int(config.wind_cells)
        self._venturi_speedup = np.ones((cells, cells), dtype=np.float32)
        self._venturi_deflect = np.zeros((cells, cells, 2), dtype=np.float32)
        # vz updraft gain per cell: wind_updraft_gain * max(speedup - 1, 0).
        self._updraft_gain_grid = np.zeros((cells, cells), dtype=np.float32)
        # Origin the currently-applied correction is valid for (None = identity).
        self._venturi_origin: tuple[int, int] | None = None
        # Monotonic job id + whether any job has ever been submitted.
        self._venturi_seq = 0
        self._venturi_ever_submitted = False

    # ------------------------------------------------------------------
    # Modifiers
    # ------------------------------------------------------------------

    def add_modifier(self, m: WindModifier) -> None:
        """
        Register an in-place :class:`~fire_engine.world.wind.protocols.WindModifier`,
        applied (in registration order) on every subsequent :meth:`update`.
        """
        self._modifiers.append(m)

    def remove_modifier(self, m: WindModifier) -> None:
        """
        Unregister a previously-added modifier.  Removing the last-added
        modifier and re-running :meth:`update` restores the base field exactly
        (modifiers are pure / additive).  No-op if ``m`` is not registered.
        """
        with contextlib.suppress(ValueError):
            self._modifiers.remove(m)

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(
        self,
        dt: float,
        wind_time: float,
        sky_state: Any,
        player_pos: Sequence[float],
        chunks: dict[tuple[int, int, int], Any] | None = None,
    ) -> None:
        """
        Recompute and atomically publish the wind field for this frame.

        Steps (all numpy-bulk, no per-cell loops):

        1. **Recenter** the region to the player (snap + hysteresis); the
           field is analytic in position, so this only rebuilds the cached
           cell-centre meshes — no resampling, free.
        2. **Weather scaling** from the (duck-typed) ``sky_state`` (duck-typed:
           reads ``wind_dir``, ``wind_speed``, ``rain_intensity``,
           ``cloud_coverage``, ``cloud_density``; ``None`` ⇒ calm defaults).
        3. **Compose** mean wind + gusts + turbulence over the grid.
        4. **Venturi** terrain-funneling correction via :func:`_venturi_step`;
           then run **modifiers** and publish ``self._front``.

        Parameters
        ----------
        dt : float
            Frame delta in seconds (currently unused — the field is a pure
            function of ``wind_time``; accepted for API symmetry).
        wind_time : float
            The **wind clock** in seconds — monotonic, advancing at
            ``config.wind_time_scale`` real seconds per second.
        sky_state : object | None
            Weather source (duck-typed) or ``None`` for calm defaults.
        player_pos : sequence of floats
            Player/camera world position; only XY is used.
        chunks : dict | None, default None
            Loaded chunks for the venturi solver (passed only on a
            recenter / terrain-edit event; ``None`` ⇒ keep last correction).
        """
        cfg = self._cfg

        # --- 1. Recenter (free: analytic field, just rebuild meshes) --------
        recentered = self._region.maybe_recenter(player_pos)
        assert self._region.X is not None and self._region.Y is not None
        X: np.ndarray = self._region.X
        Y: np.ndarray = self._region.Y

        # --- 2. Weather scaling (duck-typed sky_state; None => calm) --------
        if sky_state is None:
            wind_dir = (1.0, 0.0)
            wind_speed = 2.0
            rain = 0.0
            cov = 0.0
            den = 0.0
        else:
            wind_dir = tuple(sky_state.wind_dir)
            wind_speed = float(sky_state.wind_speed)
            rain = float(sky_state.rain_intensity)
            cov = float(sky_state.cloud_coverage)
            den = float(sky_state.cloud_density)

        storminess = float(np.clip(rain * 0.6 + cov * den * 0.4, 0.0, 1.0))
        gust_gain = (cfg.wind_gust_base + cfg.wind_gust_storm_gain * storminess) * (
            0.4 + 0.6 * wind_speed / max(cfg.wind_speed_ref, 1e-6)
        )
        turb_amt = cfg.wind_turb_base + cfg.wind_turb_storm_gain * storminess
        t_eff = float(wind_time) * (1.0 + cfg.wind_storm_freq_gain * storminess)

        mean_x = wind_dir[0] * wind_speed
        mean_y = wind_dir[1] * wind_speed

        # --- 3. Compose mean + gusts + turbulence ---------------------------
        gust_x, gust_y = eval_gusts(self._modes, X, Y, t_eff, (mean_x, mean_y))
        vx = (mean_x + gust_gain * gust_x).astype(np.float32)
        vy = (mean_y + gust_gain * gust_y).astype(np.float32)
        # Turbulence rises where gusting is strong (hypot of the gust shape).
        turb = (turb_amt * (0.5 + 0.5 * np.hypot(gust_x, gust_y))).astype(np.float32)

        # --- 4. Venturi terrain-funneling correction ------------------------
        _venturi_step(self, recentered, chunks, (mean_x, mean_y))
        if self._venturi_origin == self._region.origin_cell:
            mean_mag = float(np.hypot(mean_x, mean_y))
            vx = (vx * self._venturi_speedup + self._venturi_deflect[..., 0] * mean_mag).astype(
                np.float32
            )
            vy = (vy * self._venturi_speedup + self._venturi_deflect[..., 1] * mean_mag).astype(
                np.float32
            )

        # --- Modifiers (volumetric-weather seam), then atomic publish -------
        for mod in self._modifiers:
            mod.apply(X, Y, float(wind_time), vx, vy, turb)

        field = np.empty((self._region.cells, self._region.cells, 4), dtype=np.float32)
        field[..., 0] = vx
        field[..., 1] = vy
        field[..., 2] = turb
        field[..., 3] = 0.0  # reserved

        self._front = WindSnapshot(
            field=field,
            origin_m=self._region.origin_m,
            cell_m=self._region.cell_m,
            cells=self._region.cells,
            wind_time=float(wind_time),
        )

    # ------------------------------------------------------------------
    # Read paths
    # ------------------------------------------------------------------

    @property
    def snapshot(self) -> WindSnapshot:
        """
        The current atomically-published :class:`WindSnapshot`.

        Raises ``RuntimeError`` if :meth:`update` has never run (no field yet).
        """
        snap = self._front
        if snap is None:
            raise RuntimeError("WindField.update() never called")
        return snap

    def sample(self, positions: np.ndarray) -> np.ndarray:
        """
        Sample wind velocity at world ``positions`` — vectorised, no loops.

        Bilinear-gathers ``(vx, vy)`` from the current snapshot at each
        point's XY (4-corner fancy indexing, indices clamped to the grid so
        out-of-region points clamp to the nearest edge values), scales the
        horizontal velocity by :func:`vertical_profile` at each point's Z, and
        adds an analytic obstacle **updraft** to ``vz`` (bilinear-sampled from
        the venturi updraft-gain grid × the local horizontal speed; ``0`` where
        there is no funnelling / no worker).

        Parameters
        ----------
        positions : numpy.ndarray
            ``(N, 3)`` world positions in meters ``[x, y, z]``.

        Returns
        -------
        numpy.ndarray
            ``(N, 3)`` wind velocities in m/s ``[vx, vy, vz]``.  No NaNs.

        Example
        -------
        >>> import numpy as np
        >>> v = field.sample(np.array([[0.0, 0.0, 1.0], [10.0, 5.0, 2.0]]))
        >>> v.shape
        (2, 3)
        """
        snap = self.snapshot
        P = np.asarray(positions, dtype=np.float32)
        if P.ndim != 2 or P.shape[1] != 3:
            raise ValueError(f"positions must be (N, 3); got {P.shape}")
        n = P.shape[0]
        out = np.zeros((n, 3), dtype=np.float32)
        if n == 0:
            return out

        cell_m = snap.cell_m
        cells = snap.cells
        ox, oy = snap.origin_m
        field = snap.field

        # Continuous cell-centre coordinates: cell (i,j) centre sits at
        # (origin + i + 0.5) * cell_m, so the inverse map subtracts the 0.5.
        fx = (P[:, 0] - ox) / cell_m - 0.5
        fy = (P[:, 1] - oy) / cell_m - 0.5

        # Lower corner indices + fractional offsets; clamp so out-of-region
        # points read the nearest edge value (no wrap, no NaN).
        i0 = np.floor(fx).astype(np.int64)
        j0 = np.floor(fy).astype(np.int64)
        tx = (fx - i0).astype(np.float32)
        ty = (fy - j0).astype(np.float32)
        i0c = np.clip(i0, 0, cells - 1)
        j0c = np.clip(j0, 0, cells - 1)
        i1c = np.clip(i0 + 1, 0, cells - 1)
        j1c = np.clip(j0 + 1, 0, cells - 1)

        # 4-corner bilinear gather of vx, vy (channels 0, 1).
        v00 = field[i0c, j0c, :2]
        v10 = field[i1c, j0c, :2]
        v01 = field[i0c, j1c, :2]
        v11 = field[i1c, j1c, :2]
        tx_ = tx[:, None]
        ty_ = ty[:, None]
        top = v00 * (1.0 - tx_) + v10 * tx_
        bot = v01 * (1.0 - tx_) + v11 * tx_
        horiz = top * (1.0 - ty_) + bot * ty_  # (N, 2) m/s

        # Vertical boundary-layer profile scales the horizontal speed.
        prof = vertical_profile(P[:, 2], self._z_ground, self._cfg)  # (N,)
        out[:, 0] = horiz[:, 0] * prof
        out[:, 1] = horiz[:, 1] * prof

        # Vertical updraft: wind funnelled by a windward obstacle (high venturi
        # speed-up) rises so motes/leaves lift over it.  vz is the bilinear
        # sample of updraft_gain_grid (wind_updraft_gain * max(speedup-1, 0),
        # set only for the current origin; zeros = identity / no obstacle)
        # multiplied by horizontal speed and divided by prof (so the rise tapers
        # with height).  Kept intentionally simple — particles just need to
        # rise over a constriction.  Gated on origin agreement so a snapshot
        # from a just-recentered frame never reads a stale updraft grid.
        if self._venturi_origin is not None and self._venturi_origin == self._region.origin_cell:
            g = self._updraft_gain_grid
            u00 = g[i0c, j0c]
            u10 = g[i1c, j0c]
            u01 = g[i0c, j1c]
            u11 = g[i1c, j1c]
            utop = u00 * (1.0 - tx) + u10 * tx
            ubot = u01 * (1.0 - tx) + u11 * tx
            updraft_gain = utop * (1.0 - ty) + ubot * ty  # (N,)
            horiz_speed = np.hypot(horiz[:, 0], horiz[:, 1])  # (N,) m/s
            # Rise is strongest near the ground and tapers with the same shear
            # profile that boosts horizontal wind aloft (1/prof falls with z).
            out[:, 2] = (updraft_gain * horiz_speed / np.maximum(prof, 1e-3)).astype(np.float32)
        else:
            out[:, 2] = 0.0
        return out
