"""
wind/field.py — The wind field: WindField + WindSnapshot + sample + pack.

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

The tradeoff, recorded here because it shapes every guarantee below:

- A *random walk* would be the "physically obvious" choice but is **stateful**:
  it must be integrated every frame, carried in saves (or it diverges on
  reload), cannot be recentered without resampling history, and is not
  reproducible for bug repro.  Its only advantage is a marginally more
  "organic" low-frequency wander.
- The *spectral sum* is visually indistinguishable from Brownian gusting at the
  20–120 m wavelengths that matter, yet is **stateless**: bit-reproducible from
  the seed, costs **zero save bytes** (no Saveable — same ethos as
  ``sky/weather.py``), recenters for free (the field is analytic in position,
  so moving the window just recomputes coordinate meshes), and survives
  save/load identically. We accept the (negligible) loss of true-random
  low-frequency wander to gain determinism + zero-byte saves + free recenter.

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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from fire_engine.core.config import Config
from fire_engine.world.wind.gusts import GustModes, build_modes, eval_gusts
from fire_engine.world.wind.modifiers import WindModifier
from fire_engine.world.wind.region import WindRegion
from fire_engine.world.wind.worker import VenturiJob

if TYPE_CHECKING:
    from fire_engine.world.wind.worker import VenturiWorker

__all__ = [
    "WindSnapshot",
    "WindField",
    "pack_wind_field",
    "vertical_profile",
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
    """

    field: np.ndarray
    origin_m: tuple[float, float]
    cell_m: float
    cells: int
    wind_time: float


def vertical_profile(z: np.ndarray, z_ground: float, cfg: Config) -> np.ndarray:
    """
    Analytic boundary-layer wind-speed multiplier vs. height above ground.

    A power-law wind-shear profile clamped to a floor and a cap::

        m = clamp( ( max(z - z_ground, 0) / z_ref ) ** shear, floor, cap )

    So wind never fully dies at ground level (``floor``, default 0.35 — grass
    still sways), grows with height to 1.0 at ``z_ref`` (default 10 m), and
    saturates at ``cap`` (default 1.6) high up.  Monotonically non-decreasing
    in ``z`` between the floor and cap.

    Parameters
    ----------
    z : numpy.ndarray
        World heights in meters (any shape).
    z_ground : float
        Ground height in meters at the sample (the profile is 0-anchored here).
    cfg : Config
        Reads ``wind_shear``, ``wind_profile_z_ref``, ``wind_profile_floor``,
        ``wind_profile_cap``.

    Returns
    -------
    numpy.ndarray
        Same shape as ``z``: the per-height speed multiplier, in
        ``[floor, cap]``.

    Example
    -------
    >>> import numpy as np
    >>> from fire_engine.core.config import Config
    >>> m = vertical_profile(np.array([0.0, 10.0, 100.0]), 0.0, Config())
    >>> bool(m[0] == Config().wind_profile_floor)   # floor at ground
    True
    >>> bool(m[1] >= m[0] and m[2] >= m[1])          # monotone
    True
    """
    shear = float(cfg.wind_shear)
    z_ref = float(cfg.wind_profile_z_ref)
    floor = float(cfg.wind_profile_floor)
    cap = float(cfg.wind_profile_cap)
    above = np.maximum(np.asarray(z, dtype=np.float32) - float(z_ground), 0.0)
    prof = (above / z_ref) ** shear
    return np.clip(prof, floor, cap).astype(np.float32)


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
    """

    def __init__(self, config: Config, worker: "VenturiWorker | None" = None) -> None:
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
        Register an in-place :class:`~fire_engine.world.wind.modifiers.WindModifier`,
        applied (in registration order) on every subsequent :meth:`update`.
        """
        self._modifiers.append(m)

    def remove_modifier(self, m: WindModifier) -> None:
        """
        Unregister a previously-added modifier.  Removing the last-added
        modifier and re-running :meth:`update` restores the base field exactly
        (modifiers are pure / additive).  No-op if ``m`` is not registered.
        """
        try:
            self._modifiers.remove(m)
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(
        self,
        dt: float,
        wind_time: float,
        sky_state: object | None,
        player_pos,
        chunks: dict | None = None,
    ) -> None:
        """
        Recompute and atomically publish the wind field for this frame.

        Steps (all numpy-bulk, no per-cell loops):

        1. **Recenter** the region to the player (snap + hysteresis); the
           field is analytic in position, so this only rebuilds the cached
           cell-centre meshes — no resampling, free.
        2. **Weather scaling** from the (duck-typed) ``sky_state``::

               storminess = clip(rain*0.6 + cov*den*0.4, 0, 1)
               gust_gain  = (base + storm_gain*storminess)
                            * (0.4 + 0.6*wind_speed/speed_ref)
               turb       = turb_base + turb_storm_gain*storminess
               t_eff      = wind_time * (1 + storm_freq_gain*storminess)

        3. **Compose** mean wind + gusts + turbulence over the grid.
        4. **Venturi** terrain-funneling correction: orchestrate the off-thread
           :class:`~fire_engine.world.wind.worker.VenturiWorker` (submit on
           recenter / when ``chunks`` is passed, drain + commit the newest
           matching-origin result), apply ``vx *= speedup;
           vx += deflect_x*|mean|`` (same for ``y``), then run **modifiers**
           and publish ``self._front``.  Identity when no worker / no result.

        Determinism note on ``t_eff``: scaling the wind clock by storminess means a
        *changing* storminess slightly chirps the gust frequency (the phase
        argument's time-derivative shifts as storminess blends).  This is
        deliberate and harmless: weather storminess only moves over the sky
        system's 20-game-minute blends (very slow), so the chirp is far below
        perceptual and the field stays a pure function of
        ``(wind_time, storminess)``.  We keep this closed form rather than an
        accumulated effective-time integral precisely because the integral
        would make the field history-dependent and break determinism /
        zero-byte saves.

        Parameters
        ----------
        dt : float
            Frame delta in seconds (currently unused — the field is a pure
            function of ``wind_time``; accepted for API symmetry and future
            modifiers that want it).
        wind_time : float
            The **wind clock** in seconds — monotonic, and advancing at
            ``config.wind_time_scale`` seconds per REAL second regardless of
            the game-clock timescale (``Clock.game_time_scale``: 60 today, 30
            later, 1800 on the F7 dev toggle).  Gust travel and oscillation
            are an aesthetic real-time effect: at game-time pacing a 60×
            timescale would sweep crests across the grass 60× too fast.  The
            render component accumulates this clock from real frame ``dt``
            (``wind_renderer.py``); headless callers may pass any monotonic
            value — the field is a pure function of whatever it is handed.
        sky_state : object | None
            Weather source, duck-typed: reads ``wind_dir`` (unit XY tuple),
            ``wind_speed`` (m/s), ``rain_intensity``, ``cloud_coverage``,
            ``cloud_density`` (all 0..1).  ``None`` ⇒ calm defaults (a light
            +X breeze) so headless tests need no sky package.
        player_pos : sequence of floats
            Player/camera world position; only ``[0], [1]`` (XY) are used.
        chunks : dict | None, default None
            Loaded chunks for the venturi solver — ``coord -> Chunk`` (or bare
            ``materials`` ndarray).  The renderer passes this **only on a
            recenter or terrain-edit (dirty) event**, so a non-``None`` value is
            itself the recompute request; ``None`` (the common per-frame case)
            keeps the previously-committed correction.  Ignored when ``worker``
            is ``None``.
        """
        cfg = self._cfg

        # --- 1. Recenter (free: analytic field, just rebuild meshes) --------
        recentered = self._region.maybe_recenter(player_pos)
        X = self._region.X
        Y = self._region.Y

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
        # Orchestrate the off-thread solver: (a) re-submit on recenter, (b)
        # submit the first time `chunks` is available, or (c) submit whenever
        # the caller passes `chunks` (the renderer only passes them on a
        # dirty/recenter event — see the system doc), then drain + apply the
        # newest result whose origin matches the CURRENT region origin.  The
        # correction is the identity until such a result lands.
        self._venturi_step(recentered, chunks, (mean_x, mean_y))
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
    # Venturi orchestration (off-thread terrain funneling)
    # ------------------------------------------------------------------

    def _venturi_step(
        self,
        recentered: bool,
        chunks: dict | None,
        mean: tuple[float, float],
    ) -> None:
        """
        Submit / drain the venturi worker and update the applied correction.

        Pure orchestration (no field math): decide whether a fresh
        :class:`~fire_engine.world.wind.worker.VenturiJob` is warranted, submit it,
        then drain finished results and commit the newest one whose
        ``origin_cell`` still matches the region's current origin.

        Submit when (and only when) there is a worker AND any of:

        - the region **recentered** this update (the old grid is for a stale
          origin — the renderer signals dirt by re-passing ``chunks`` too, but
          recenter alone is enough to re-solve), OR
        - ``chunks`` is available and **no job has ever been submitted** (first
          terrain solve), OR
        - ``chunks`` is not ``None`` — the renderer passes ``chunks`` *only* on
          a recenter or terrain-edit (dirty) event, so a non-``None`` ``chunks``
          is itself the recompute request (keeps ``wind/`` bus-free).

        Origin-match discipline (a Gotcha): a result solved for a previous
        origin is **discarded**, never shift-applied — the field re-submits on
        recenter and applies identity in the meantime.  This keeps the applied
        grid and the cells it scales perfectly aligned with zero index math.
        """
        worker = self._worker
        if worker is None:
            return

        want_submit = (
            recentered
            or (chunks is not None and not self._venturi_ever_submitted)
            or (chunks is not None)
        )
        if want_submit and chunks is not None:
            self._venturi_seq += 1
            self._venturi_ever_submitted = True
            assert self._region.origin_cell is not None
            ground = self._z_ground
            job = VenturiJob(
                origin_cell=self._region.origin_cell,
                cells=int(self._region.cells),
                cell_m=float(self._region.cell_m),
                chunk_size=int(self._cfg.chunk_size),
                voxel_size=float(self._cfg.voxel_size),
                ground_band=(ground, ground + float(self._cfg.wind_layer_m)),
                materials=self._snapshot_materials(chunks),
                venturi_iters=int(self._cfg.wind_venturi_iters),
                venturi_max=float(self._cfg.wind_venturi_max),
                deflect_gain=float(self._cfg.wind_deflect_gain),
                seq=self._venturi_seq,
            )
            worker.submit(job)

        # Drain all finished results; keep only the newest (highest seq).
        newest = None
        for res in worker.drain_results():
            if newest is None or res.seq >= newest.seq:
                newest = res
        if newest is not None:
            self._commit_venturi(newest)

        # A correction solved for an origin we have since moved away from must
        # not be applied — drop back to identity until a matching result lands.
        if self._venturi_origin != self._region.origin_cell:
            self._venturi_speedup.fill(1.0)
            self._venturi_deflect.fill(0.0)
            self._updraft_gain_grid.fill(0.0)
            self._venturi_origin = None

    @staticmethod
    def _snapshot_materials(chunks: dict) -> dict:
        """
        Build the ``coord -> uint8 materials`` snapshot the worker reads.

        Accepts either chunk objects (reads ``.materials``) or bare ndarrays
        (mirrors ``lighting`` assembly-worker's dual acceptance).  References,
        not copies — the arrays are treated as immutable for the solve's life.
        """
        out: dict = {}
        for coord, ch in chunks.items():
            out[coord] = getattr(ch, "materials", ch)
        return out

    def _commit_venturi(self, res) -> None:
        """
        Apply a drained :class:`~fire_engine.world.wind.worker.VenturiResult`.

        Only commits if the result's ``origin_cell`` matches the region's
        current origin (else it is a stale result for a window we have left —
        discard it, identity holds).  Derives the vz updraft-gain grid from the
        committed speed-up: ``wind_updraft_gain * clip(speedup - 1, 0, None)``.
        """
        if res.origin_cell != self._region.origin_cell:
            return  # stale — discard (origin-match discipline)
        self._venturi_speedup = res.speedup
        self._venturi_deflect = res.deflect
        self._venturi_origin = res.origin_cell
        self._updraft_gain_grid = (
            float(self._cfg.wind_updraft_gain) * np.clip(res.speedup - 1.0, 0.0, None)
        ).astype(np.float32)

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
        # speed-up) rises so motes/leaves lift over it.  vz =
        #   bilinear(updraft_gain_grid) * horizontal_speed * height_falloff,
        # where updraft_gain_grid = wind_updraft_gain * max(speedup-1, 0) (set
        # only for the current origin; zeros = identity / no obstacle).  Kept
        # intentionally simple — it just needs particles to rise over a
        # windward constriction, not be physically exact.  Gated on origin
        # agreement so a snapshot from a just-recentered frame never reads a
        # stale (other-origin) updraft grid.
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


def pack_wind_field(snap: WindSnapshot) -> bytes:
    """
    Pack a :class:`WindSnapshot` into Panda3D 2-D-texture RAM bytes.

    Produces a **float16** buffer in Panda3D's 2-D RAM layout: **row-major
    ``(y, x)``** (the field is stored ``[x, y]``, so it is transposed) with
    **BGRA** channel order — i.e. ``B = turb, G = vy, R = vx, A = horizontal
    speed`` (``hypot(vx, vy)``).  This mirrors
    ``lighting/volume.pack_volume``'s transpose + channel-swap convention so an
    upload is just ``Texture.set_ram_image(bytes)`` on the render thread.  Pure
    and thread-safe (no shared state) — safe to call off the main thread.

    LAYOUT IS PINNED (a test asserts it): if you change the transpose order or
    channel mapping you must update the GPU uniform contract
    (``u_wind_tex`` R=vx G=vy B=turb A=speed) and the shader decode together.

    Parameters
    ----------
    snap : WindSnapshot
        The field snapshot to pack.

    Returns
    -------
    bytes
        ``cells * cells * 4 * 2`` bytes of little-endian float16, ready for
        ``Texture(F_rgba16).set_ram_image``.

    Example
    -------
    >>> data = pack_wind_field(field.snapshot)
    >>> len(data) == field.snapshot.cells ** 2 * 4 * 2
    True
    """
    f = snap.field  # (cells, cells, 4) [x, y]: vx, vy, turb, reserved
    vx = f[..., 0]
    vy = f[..., 1]
    turb = f[..., 2]
    speed = np.hypot(vx, vy)

    # Build the RGBA-in-shader buffer in the texel's channel order, then
    # transpose [x, y] -> [y, x] (Panda3D 2-D RAM is row-major y outer) and
    # swap RGBA -> BGRA.  Mirrors pack_volume's transpose+swap discipline.
    rgba = np.stack([vx, vy, turb, speed], axis=-1)  # R, G, B, A
    bgra = rgba[..., [2, 1, 0, 3]]  # B, G, R, A
    data = np.ascontiguousarray(np.transpose(bgra, (1, 0, 2)).astype(np.float16))  # (y, x, 4) fp16
    return data.tobytes()
