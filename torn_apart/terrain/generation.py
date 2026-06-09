"""
terrain/generation.py — Pure, deterministic voxel chunk generation.

``generate_chunk(coord, config)`` returns the ``uint8[32,32,32]`` material
array for a chunk, as a **pure function of (world_seed, chunk coord)**.  The
same inputs always yield a byte-identical array, on any process or machine —
this is what makes delta saves and reproducible bugs possible (CLAUDE.md Hard
Rule 2).

Pipeline
--------
1. **Heightmap** — a continuous 2-D value-noise surface sampled in *world XY*
   coordinates, amplitude ≈ 24 m, plus a low-frequency ridge term.  A voxel at
   world height ``z`` is solid iff ``z < surface_height(world_x, world_y)``.
2. **3-D carve pass** — a continuous 3-D value-noise field (also sampled in
   world coordinates) removes solid voxels where the field exceeds a threshold,
   producing occasional overhangs / shallow caves (needed to *see* lighting in
   Phase 4).  A minimum solid floor near the surface base is protected so the
   world is never see-through.

Seamlessness
------------
Both noise fields are sampled in **continuous WORLD coordinates** from a single
global coarse-noise grid (not per-chunk noise).  Neighbouring chunks therefore
agree exactly at their shared faces — no cliffs or gaps at chunk borders.  The
coarse grids are drawn from ``for_domain("terrain", ...)`` once and indexed by
world position, so the field is globally consistent and deterministic.

Everything is fully vectorised — no per-voxel Python loops (Hard Rule 4).
"""

from __future__ import annotations

import numpy as np

from torn_apart.core import Config, for_domain

# ---------------------------------------------------------------------------
# Tunable generation constants (world space, meters).
# Kept here as documented module constants rather than scattered magic numbers.
# ---------------------------------------------------------------------------

# Heightmap
_BASE_GROUND_Z_M: float = 8.0       # mean terrain surface height (meters, world Z)
_HEIGHT_AMPLITUDE_M: float = 24.0   # peak-to-mean amplitude of the detail noise
_RIDGE_AMPLITUDE_M: float = 14.0    # amplitude of the low-frequency ridge term
_DETAIL_WAVELENGTH_M: float = 96.0  # world-meters per coarse cell of the detail field
_RIDGE_WAVELENGTH_M: float = 320.0  # world-meters per coarse cell of the ridge field
_HEIGHT_OCTAVES: int = 5

# 3-D carve pass
_CARVE_WAVELENGTH_M: float = 22.0   # world-meters per coarse cell of the carve field
_CARVE_OCTAVES: int = 3
_CARVE_THRESHOLD: float = 0.62      # carve where field > threshold (higher = fewer caves)
_CARVE_FLOOR_DEPTH_M: float = 6.0   # protect this many meters of solid below the surface

_DEFAULT_MATERIAL: int = 1          # material id written for solid ground


# ===========================================================================
# Global continuous value-noise samplers (world-coordinate, seamless)
# ===========================================================================

def _coarse_grid_2d(domain: tuple, cells: int) -> np.ndarray:
    """
    Draw a global ``(cells+1, cells+1)`` coarse random grid in ``[0,1)``.

    The grid is deterministic per ``domain`` (a ``for_domain`` key tuple).  It
    tiles the entire world: world position is mapped into ``[0, cells)`` by a
    modulo so any chunk can sample the same global field seamlessly.
    """
    rng = for_domain(*domain)
    return rng.random((cells + 1, cells + 1)).astype(np.float64)


def _coarse_grid_3d(domain: tuple, cells: int) -> np.ndarray:
    """Draw a global ``(cells+1,)*3`` coarse random grid in ``[0,1)``."""
    rng = for_domain(*domain)
    return rng.random((cells + 1, cells + 1, cells + 1)).astype(np.float64)


def _sample_noise_2d_world(
    wx: np.ndarray,
    wy: np.ndarray,
    *,
    domain_prefix: tuple,
    wavelength_m: float,
    octaves: int,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
) -> np.ndarray:
    """
    Sample seamless layered 2-D value noise at world XY positions.

    Parameters
    ----------
    wx, wy : numpy.ndarray
        Broadcastable arrays of world X / Y coordinates (meters).
    domain_prefix : tuple
        ``for_domain`` key prefix; each octave appends its index for an
        independent coarse grid.
    wavelength_m : float
        World meters spanned by one coarse cell of the base octave.
    octaves : int
        Number of summed octaves.

    Returns
    -------
    numpy.ndarray
        ``float32`` noise in ``[0, 1]`` with the broadcast shape of (wx, wy).

    Notes
    -----
    A fixed global period of ``GLOBAL_CELLS`` coarse cells tiles the world; the
    world coordinate is wrapped into ``[0, GLOBAL_CELLS)`` per octave so that
    neighbouring chunks index the *same* grid and meet seamlessly.
    """
    GLOBAL_CELLS = 256  # coarse cells before the field tiles (large → no visible repeat nearby)
    acc = np.zeros(np.broadcast(wx, wy).shape, dtype=np.float64)
    weight_total = 0.0
    amplitude = 1.0
    wl = float(wavelength_m)
    for o in range(octaves):
        grid = _coarse_grid_2d(domain_prefix + (o,), GLOBAL_CELLS)
        # continuous cell coordinate in world units, wrapped to the global period
        u = np.mod(wx / wl, GLOBAL_CELLS)
        v = np.mod(wy / wl, GLOBAL_CELLS)
        acc += amplitude * _bilinear(grid, u, v, GLOBAL_CELLS)
        weight_total += amplitude
        amplitude *= persistence
        wl /= lacunarity
    out = np.clip(acc / weight_total, 0.0, 1.0).astype(np.float32)
    return out


def _sample_noise_3d_world(
    wx: np.ndarray,
    wy: np.ndarray,
    wz: np.ndarray,
    *,
    domain_prefix: tuple,
    wavelength_m: float,
    octaves: int,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
) -> np.ndarray:
    """
    Sample seamless layered 3-D value noise at world XYZ positions.

    Same contract as :func:`_sample_noise_2d_world` but trilinear over a 3-D
    coarse grid.  Used by the carve pass.  Fully vectorised.
    """
    GLOBAL_CELLS = 64  # 3-D grid is memory-heavier; a smaller global period is fine for caves
    acc = np.zeros(np.broadcast(wx, wy, wz).shape, dtype=np.float64)
    weight_total = 0.0
    amplitude = 1.0
    wl = float(wavelength_m)
    for o in range(octaves):
        grid = _coarse_grid_3d(domain_prefix + (o,), GLOBAL_CELLS)
        u = np.mod(wx / wl, GLOBAL_CELLS)
        v = np.mod(wy / wl, GLOBAL_CELLS)
        w = np.mod(wz / wl, GLOBAL_CELLS)
        acc += amplitude * _trilinear(grid, u, v, w, GLOBAL_CELLS)
        weight_total += amplitude
        amplitude *= persistence
        wl /= lacunarity
    out = np.clip(acc / weight_total, 0.0, 1.0)
    return out


def _bilinear(grid: np.ndarray, u: np.ndarray, v: np.ndarray, period: int) -> np.ndarray:
    """
    Bilinearly sample ``grid`` (shape ``(period+1, period+1)``) at fractional
    coordinates ``u, v`` in ``[0, period)``.  Vectorised; no Python loops.
    """
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    fu = u - u0
    fv = v - v0
    # wrap into [0, period]; grid has period+1 entries so index period == index 0 value
    u0 = np.mod(u0, period)
    v0 = np.mod(v0, period)
    u1 = u0 + 1
    v1 = v0 + 1
    g00 = grid[u0, v0]
    g10 = grid[u1, v0]
    g01 = grid[u0, v1]
    g11 = grid[u1, v1]
    return (
        g00 * (1 - fu) * (1 - fv)
        + g10 * fu * (1 - fv)
        + g01 * (1 - fu) * fv
        + g11 * fu * fv
    )


def _trilinear(
    grid: np.ndarray, u: np.ndarray, v: np.ndarray, w: np.ndarray, period: int
) -> np.ndarray:
    """
    Trilinearly sample ``grid`` (shape ``(period+1,)*3``) at fractional coords
    ``u, v, w`` in ``[0, period)``.  Vectorised; no Python loops.
    """
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    w0 = np.floor(w).astype(np.int64)
    fu = u - u0
    fv = v - v0
    fw = w - w0
    u0 = np.mod(u0, period)
    v0 = np.mod(v0, period)
    w0 = np.mod(w0, period)
    u1 = u0 + 1
    v1 = v0 + 1
    w1 = w0 + 1
    c000 = grid[u0, v0, w0]
    c100 = grid[u1, v0, w0]
    c010 = grid[u0, v1, w0]
    c110 = grid[u1, v1, w0]
    c001 = grid[u0, v0, w1]
    c101 = grid[u1, v0, w1]
    c011 = grid[u0, v1, w1]
    c111 = grid[u1, v1, w1]
    c00 = c000 * (1 - fu) + c100 * fu
    c10 = c010 * (1 - fu) + c110 * fu
    c01 = c001 * (1 - fu) + c101 * fu
    c11 = c011 * (1 - fu) + c111 * fu
    c0 = c00 * (1 - fv) + c10 * fv
    c1 = c01 * (1 - fv) + c11 * fv
    return c0 * (1 - fw) + c1 * fw


# ===========================================================================
# Surface height (world XY → meters)
# ===========================================================================

def surface_height(world_x: np.ndarray, world_y: np.ndarray) -> np.ndarray:
    """
    Continuous terrain surface height (world Z, meters) at world XY positions.

    Pure function of world coordinates and the global seed.  Used both by
    :func:`generate_chunk` and by callers that need the ground height at a
    point (e.g. spawn placement).

    Parameters
    ----------
    world_x, world_y : numpy.ndarray
        Broadcastable world X / Y coordinate arrays (meters).

    Returns
    -------
    numpy.ndarray
        ``float32`` surface height in meters (world Z), broadcast shape of
        ``(world_x, world_y)``.

    Example
    -------
    >>> import numpy as np
    >>> from torn_apart.core.rng import set_world_seed
    >>> set_world_seed(1337)
    >>> h = surface_height(np.array([0.0, 8.0]), np.array([0.0, 0.0]))
    >>> h.shape
    (2,)
    """
    detail = _sample_noise_2d_world(
        world_x, world_y,
        domain_prefix=("terrain", "height", "detail"),
        wavelength_m=_DETAIL_WAVELENGTH_M,
        octaves=_HEIGHT_OCTAVES,
    )
    ridge = _sample_noise_2d_world(
        world_x, world_y,
        domain_prefix=("terrain", "height", "ridge"),
        wavelength_m=_RIDGE_WAVELENGTH_M,
        octaves=2,
    )
    # ridge term: fold around 0.5 to make sharper crests (|2x-1| style)
    ridge_folded = 1.0 - np.abs(2.0 * ridge - 1.0)
    h = (
        _BASE_GROUND_Z_M
        + (detail - 0.5) * 2.0 * _HEIGHT_AMPLITUDE_M
        + ridge_folded * _RIDGE_AMPLITUDE_M
    )
    return h.astype(np.float32)


# ===========================================================================
# Chunk generation
# ===========================================================================

def generate_chunk(coord: tuple[int, int, int], config: Config) -> np.ndarray:
    """
    Generate the material array for a chunk — pure function of (seed, coord).

    Parameters
    ----------
    coord : tuple[int, int, int]
        Integer chunk coordinate ``(cx, cy, cz)``.
    config : Config
        Engine config (provides ``chunk_size`` and ``voxel_size``).  The world
        seed is read globally via ``core.rng`` (set with ``set_world_seed``).

    Returns
    -------
    numpy.ndarray
        ``uint8`` array of shape ``(chunk_size,)*3`` indexed ``[x, y, z]``.
        ``0`` = air, ``1`` = solid ground.  Byte-identical for identical
        ``(world_seed, coord)``.

    Determinism & seamlessness
    --------------------------
    All noise is sampled in continuous world coordinates from global coarse
    grids, so the result is deterministic and seamless across chunk borders:
    a voxel column straddling two vertically adjacent chunks is continuous.

    Example
    -------
    >>> from torn_apart.core import load_config
    >>> from torn_apart.core.rng import set_world_seed
    >>> set_world_seed(1337)
    >>> cfg = load_config()
    >>> m = generate_chunk((0, 0, 0), cfg)
    >>> m.shape, m.dtype
    ((32, 32, 32), dtype('uint8'))
    """
    n = int(config.chunk_size)
    vs = float(config.voxel_size)
    chunk_m = n * vs
    ox = coord[0] * chunk_m
    oy = coord[1] * chunk_m
    oz = coord[2] * chunk_m

    # World-space centre coordinate of every voxel along each local axis.
    # Voxel (x,y,z) centre world coord = origin + (idx + 0.5) * voxel_size.
    lin = (np.arange(n, dtype=np.float64) + 0.5) * vs
    wx_axis = ox + lin            # (n,) along local X
    wy_axis = oy + lin            # (n,) along local Y
    wz_axis = oz + lin            # (n,) along local Z

    # --- 1. Heightmap (depends on world X, Y only) -> shape (n, n) over (x, y)
    WX2 = wx_axis[:, None]        # (n, 1)
    WY2 = wy_axis[None, :]        # (1, n)
    surf = surface_height(WX2, WY2)              # (n, n) float32, world-Z meters

    # Solid where voxel-centre world Z < surface height.
    WZ = wz_axis[None, None, :]                  # (1, 1, n)
    solid = WZ < surf[:, :, None]                # (n, n, n) bool, indexed [x, y, z]

    # --- 2. 3-D carve pass (overhangs / shallow caves) ---
    WX3 = wx_axis[:, None, None]
    WY3 = wy_axis[None, :, None]
    WZ3 = wz_axis[None, None, :]
    carve_field = _sample_noise_3d_world(
        WX3, WY3, WZ3,
        domain_prefix=("terrain", "carve"),
        wavelength_m=_CARVE_WAVELENGTH_M,
        octaves=_CARVE_OCTAVES,
    )  # (n, n, n) in [0,1]

    # Protect a solid floor: only carve voxels that are at least
    # _CARVE_FLOOR_DEPTH_M below the local surface, so the ground base stays
    # intact and the world is never see-through.
    depth_below_surface = surf[:, :, None] - WZ   # meters; >0 means below surface
    carveable = depth_below_surface > _CARVE_FLOOR_DEPTH_M
    carved = (carve_field > _CARVE_THRESHOLD) & carveable

    solid = solid & ~carved

    materials = np.zeros((n, n, n), dtype=np.uint8)
    materials[solid] = _DEFAULT_MATERIAL
    return materials
