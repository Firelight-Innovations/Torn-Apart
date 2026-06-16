"""
wind/venturi.py — Low-fidelity terrain "venturi" funneling for the wind field.

Wind speeds up where it is squeezed through a gap, canyon or tunnel, and bends
around solid walls.  This module computes that effect at the wind field's coarse
resolution (4 m cells) from the loaded voxel terrain, as a **per-cell speed-up
multiplier + a sideways deflection vector** that ``wind/field.py`` folds into the
gust field.  It is deliberately *low fidelity*: a 2-D, direction-agnostic
flux-relaxation over a column-occupancy map — enough that grass leans harder in a
trench and motes stream through a gap, not a real CFD solve.

Pipeline (all numpy bulk; the only Python loop is the bounded Jacobi count and
the bounded per-chunk dict iteration — both Hard-Rule-4-clean):

1. **Column solid fraction.**  For each wind cell, intersect the loaded chunk
   ``materials`` arrays with the cell's 8×8 voxel footprint over the vertical
   band ``[ground, ground + wind_layer_m]`` and fold (the reshape idiom from
   ``lighting/volume._downsample_chunk_block``) to ``solid (cells, cells)`` in
   ``0..1`` — the fraction of the column that is solid.  Cells with no loaded
   chunk data are fully **open** (``solid == 0``): unknown terrain never
   fabricates a wall.
2. **Blockage-crowding relaxation.**  ``passw = clip(1 - solid, .05, 1)`` is
   each cell's pass-ability.  A few Jacobi sweeps diffuse the *solid* field
   freely outward (``crowd = ½·crowd + ½·neighbour-mean(crowd)``, seeded with
   ``solid``), so a wall's blockage smears into the open cells beside it — the
   neighbourhood "crowding" an open cell sits in.  Neighbour means use
   **edge-replicate padded slicing, never ``np.roll``** (roll wraps the world
   edges and would leak crowding across the region).
3. **Speed-up + deflection.**  An open cell sitting in a crowded neighbourhood
   is a pinch (a gap, canyon or tunnel), so it speeds up::

       speedup = clip(1 + crowd_gain * crowd * passw, 1, venturi_max)

   (``× passw`` so the speed-up applies to the *open* cell, not the wall
   itself), then a 3×3 box blur to soften cell-edge stepping.  ``deflect`` is
   the openness gradient × ``deflect_gain`` (flow pushed from solid toward open
   — around walls).

Deviation note (recorded because it shapes the result): the plan sketched a
``flux = passw·neighbour-mean + (1-passw)·flux`` Jacobi sweep with
``speedup = flux / passw``.  That formulation is a pure Laplace smoothing that
relaxes a fully-open gap *toward its zero-flux wall neighbours* — it produces
``speedup ≤ 1`` in a gap (no acceleration) and cannot meet the WP2 acceptance
threshold (gap ``speedup > 1.3``).  This blockage-crowding model keeps the same
ingredients (column-occupancy fold → bounded padded-slice Jacobi relaxation →
``speedup`` from pass-ability → 3×3 blur → openness-gradient deflection) and the
same identity behaviour on open terrain, but actually funnels: at the defaults a
gap-in-a-wall reaches ``speedup ≈ 1.44`` while open field stays ``≈ 1.00`` and
fully-open terrain is exactly ``1.0``.

Determinism: ``solve_venturi`` is a **pure function of its job** — no RNG, no
shared state, no time.  Same chunks + origin + config ⇒ bit-identical result, on
the worker thread or called inline in a test.

No panda3d.  No per-voxel Python loops.
"""

from __future__ import annotations

import numpy as np

from fire_engine.world.wind.types import VenturiJob, VenturiResult

__all__ = ["column_solid_fraction", "solve_venturi"]

# How strongly neighbourhood crowding accelerates an open pinch cell.  Tuned so
# a gap-in-a-wall reaches speedup ≈ 1.44 (well past the WP2 > 1.3 acceptance
# floor) while open field stays ≈ 1.0 and the clamp to ``wind_venturi_max``
# handles the extreme (tunnel) case.  Not a config knob: it is an internal
# shape constant of the crowding model, like the 3×3 blur radius.
_CROWD_GAIN = 3.0
# Per-sweep diffusion rate for the crowding smear (0..1); ½ spreads a wall's
# blockage roughly one cell per sweep, saturating after a few sweeps.
_CROWD_DIFFUSE = 0.5


def _pad_replicate(a: np.ndarray) -> np.ndarray:
    """Edge-replicate pad a 2-D array by 1 on every side (``(N+2, N+2)``)."""
    return np.pad(a, 1, mode="edge")


def _neighbor_mean4(a: np.ndarray) -> np.ndarray:
    """
    Mean of the 4 axis-neighbours of every cell, edge-replicate at the border.

    Uses padded slicing (NOT ``np.roll`` — roll wraps the world edges and would
    funnel flux from one side of the region to the other).
    """
    p = _pad_replicate(a)
    up = p[:-2, 1:-1]
    down = p[2:, 1:-1]
    left = p[1:-1, :-2]
    right = p[1:-1, 2:]
    return np.asarray((up + down + left + right) * 0.25)


def _box_blur3(a: np.ndarray) -> np.ndarray:
    """3×3 box blur with edge-replicate borders (padded slicing, no roll)."""
    p = _pad_replicate(a)
    acc = np.zeros_like(a)
    for di in (0, 1, 2):
        for dj in (0, 1, 2):
            acc = acc + p[di : di + a.shape[0], dj : dj + a.shape[1]]
    return acc * (1.0 / 9.0)


def column_solid_fraction(job: VenturiJob) -> np.ndarray:
    """
    Fold the job's chunk terrain into a per-wind-cell solid column fraction.

    For each wind cell (``cell_m`` square, e.g. 4 m → 8×8 voxels per axis at the
    0.5 m voxel size) compute the fraction of the vertical band
    ``[z_lo, z_hi] = job.ground_band`` that is solid, averaged over the cell's
    8×8 voxel columns.  Cells over unloaded terrain are **open** (``0.0``).

    The grid is aligned to the field's ``[x, y]`` cell layout exactly: wind cell
    ``(i, j)`` spans world meters
    ``[(origin_cell + (i, j)) * cell_m, … + cell_m)`` (the same corner
    convention as :class:`~fire_engine.world.wind.region.WindRegion`), so the returned
    array drops straight onto ``vx``/``vy``.

    Python iterates the chunk dict only (bounded — a few intersecting chunks);
    all per-voxel work is numpy slicing + a reshape-fold.

    Parameters
    ----------
    job : VenturiJob
        Supplies ``origin_cell``, ``cells``, ``cell_m``, ``chunk_size``,
        ``voxel_size``, ``ground_band`` and the ``materials`` snapshot.

    Returns
    -------
    numpy.ndarray
        ``float32 (cells, cells)`` indexed ``[x, y]`` in ``0..1``.
    """
    n = int(job.cells)
    cell_m = float(job.cell_m)
    voxel = float(job.voxel_size)
    S = int(job.chunk_size)

    vpc = cell_m / voxel  # voxels per wind-cell edge
    if abs(vpc - round(vpc)) > 1e-9 or vpc < 1:
        raise ValueError(f"cell_m ({cell_m}) must be an integer multiple of voxel_size ({voxel})")
    vpc = round(vpc)  # 8 at the defaults

    ox_cell, oy_cell = job.origin_cell  # wind cells
    # World corner of wind cell (0,0), in voxel indices on the global voxel
    # grid (voxel v spans world [v*voxel, (v+1)*voxel)).
    vx0 = round(ox_cell * cell_m / voxel)
    vy0 = round(oy_cell * cell_m / voxel)

    z_lo, z_hi = job.ground_band
    # Global voxel-z range covering the band (inclusive of partial top voxel).
    vz_lo = int(np.floor(float(z_lo) / voxel))
    vz_hi = int(np.ceil(float(z_hi) / voxel))
    if vz_hi <= vz_lo:
        vz_hi = vz_lo + 1

    # Accumulate solid-voxel count and total-voxel count per wind cell over the
    # whole region's voxel footprint; ratio = column solid fraction.
    region_vx = n * vpc  # voxels spanning the region, X
    region_vy = n * vpc
    solid_vox = np.zeros((region_vx, region_vy), dtype=np.float64)
    total_vox = np.zeros((region_vx, region_vy), dtype=np.float64)

    materials = job.materials
    if materials:
        # Chunk index range intersecting the region's XY voxel footprint.
        cclo_x = int(np.floor(vx0 / S))
        cchi_x = int(np.floor((vx0 + region_vx - 1) / S))
        cclo_y = int(np.floor(vy0 / S))
        cchi_y = int(np.floor((vy0 + region_vy - 1) / S))
        cclo_z = int(np.floor(vz_lo / S))
        cchi_z = int(np.floor((vz_hi - 1) / S))

        for ccx in range(cclo_x, cchi_x + 1):
            for ccy in range(cclo_y, cchi_y + 1):
                # Region-relative voxel extent of this chunk column (XY), clipped.
                gx0 = ccx * S
                gy0 = ccy * S
                ax = max(gx0, vx0)
                bx = min(gx0 + S, vx0 + region_vx)
                ay = max(gy0, vy0)
                by = min(gy0 + S, vy0 + region_vy)
                if bx <= ax or by <= ay:
                    continue
                # Per-(x,y) solid count over the z-band, summed across the
                # intersecting chunks in Z.
                col_solid = None
                col_total = 0
                for ccz in range(cclo_z, cchi_z + 1):
                    arr = materials.get((ccx, ccy, ccz))
                    gz0 = ccz * S
                    az = max(gz0, vz_lo)
                    bz = min(gz0 + S, vz_hi)
                    if bz <= az:
                        continue
                    nz = bz - az
                    col_total += nz
                    if arr is None:
                        continue  # missing Z chunk → that band slice is air
                    sub = arr[ax - gx0 : bx - gx0, ay - gy0 : by - gy0, az - gz0 : bz - gz0]
                    cs = (sub > 0).sum(axis=2).astype(np.float64)
                    col_solid = cs if col_solid is None else (col_solid + cs)
                if col_total == 0:
                    continue
                # Scatter into the region voxel grid.
                rx0, rx1 = ax - vx0, bx - vx0
                ry0, ry1 = ay - vy0, by - vy0
                total_vox[rx0:rx1, ry0:ry1] += col_total
                if col_solid is not None:
                    solid_vox[rx0:rx1, ry0:ry1] += col_solid

    # Fold the per-voxel-column region grid (region_vx, region_vy) down to wind
    # cells: each cell is vpc×vpc voxel columns (reshape-fold idiom).
    s = solid_vox.reshape(n, vpc, n, vpc)
    t = total_vox.reshape(n, vpc, n, vpc)
    solid_sum = s.sum(axis=(1, 3))
    total_sum = t.sum(axis=(1, 3))
    frac = np.zeros((n, n), dtype=np.float32)
    nz = total_sum > 0
    frac[nz] = (solid_sum[nz] / total_sum[nz]).astype(np.float32)
    return frac


def solve_venturi(job: VenturiJob) -> VenturiResult:
    """
    Solve the terrain-venturi correction for ``job`` — a pure function.

    Returns a :class:`~fire_engine.world.wind.worker.VenturiResult` carrying a
    per-cell speed-up multiplier and a sideways deflection vector aligned to the
    job's ``origin_cell`` and the field's ``[x, y]`` cell layout.  Callable on
    the worker thread or inline (tests / first-frame path).

    Steps
    -----
    1. ``solid = column_solid_fraction(job)``; ``passw = clip(1 - solid, .05, 1)``.
    2. Blockage-crowding relaxation: ``venturi_iters`` Jacobi sweeps of
       ``crowd = ½·crowd + ½·mean4(crowd)`` seeded with ``solid`` (edge-replicate
       padded slicing — never ``np.roll``), spreading wall blockage into the
       open cells beside it.
    3. ``speedup = clip(1 + crowd_gain·crowd·passw, 1, venturi_max)`` then a 3×3
       box blur; ``deflect = stack(np.gradient(1 - solid), -1) * deflect_gain``.

    See the module docstring for why this replaces the plan's literal
    ``flux/passw`` Laplace sketch (which cannot accelerate an open gap).

    Parameters
    ----------
    job : VenturiJob
        The solve request (see :class:`~fire_engine.world.wind.worker.VenturiJob`).

    Returns
    -------
    VenturiResult
        ``speedup`` ``(cells, cells)`` in ``[1, venturi_max]``, ``deflect``
        ``(cells, cells, 2)``, both ``float32`` and finite.

    Example
    -------
    >>> # res = solve_venturi(job)
    >>> # res.speedup.shape == (job.cells, job.cells)
    """
    solid = column_solid_fraction(job).astype(np.float32)
    open_ = (1.0 - solid).astype(np.float32)
    passw = np.clip(open_, 0.05, 1.0).astype(np.float32)

    # --- Blockage-crowding relaxation -----------------------------------
    # Diffuse the solid field freely outward so a wall's blockage smears into
    # the open cells beside it (the neighbourhood an open cell sits in).  Pure
    # padded-slice Jacobi neighbour mean — no np.roll (would wrap world edges).
    crowd = solid.copy()
    iters = max(int(job.venturi_iters), 0)
    for _ in range(iters):
        blended = (1.0 - _CROWD_DIFFUSE) * crowd + _CROWD_DIFFUSE * _neighbor_mean4(crowd)
        crowd = np.asarray(blended, dtype=np.float32)
    crowd = crowd.astype(np.float32)

    # --- Speed-up: an OPEN cell in a crowded neighbourhood is a pinch -------
    speedup = np.clip(1.0 + _CROWD_GAIN * crowd * passw, 1.0, float(job.venturi_max))
    speedup = _box_blur3(speedup).astype(np.float32)
    # The blur can nudge a near-1 cell a hair below 1; re-clamp the floor.
    speedup = np.clip(speedup, 1.0, float(job.venturi_max)).astype(np.float32)

    # --- Deflection: openness gradient pushes flow around walls -------------
    gx, gy = np.gradient(open_)
    deflect = np.stack([gx, gy], axis=-1).astype(np.float32) * float(job.deflect_gain)

    return VenturiResult(
        origin_cell=job.origin_cell,
        speedup=speedup,
        deflect=deflect,
        seq=job.seq,
    )
