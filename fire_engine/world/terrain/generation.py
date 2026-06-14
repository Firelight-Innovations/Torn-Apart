"""
terrain/generation.py — Flat, bounded baseline terrain generation.

``generate_chunk(coord, config)`` returns the ``uint8[32,32,32]`` material
array for a chunk.  As of the flat-world rework (DECISIONS.md 2026-06-09) the
baseline terrain is **completely flat and seed-independent**:

- Solid ground fills everything below ``config.ground_height_m`` (world Z):
  the topmost solid voxel layer is :data:`MATERIAL_GRASS` (2), everything
  deeper is :data:`MATERIAL_DIRT` (1).
- The world has a finite square footprint of ``config.world_size_m`` meters,
  **centred on the origin** — i.e. solid only where world X and Y both lie in
  ``[-world_size_m/2, +world_size_m/2)``.  Outside that footprint the chunk is
  pure air.
- There are no hills, no noise, and no caves/overhangs.

Why flat (not procedural)
-------------------------
The world is no longer fully procedurally generated.  Terrain is authored
*semi-procedurally* — set up by humans plus rule-based / AI agents working from
parameters — so the baseline this module emits is a blank, flat canvas they
build on.  ``config.world_seed`` is still the global RNG seed, but it now drives
**other** procedural systems (textures, ambient noise, NPC behaviour, …), NOT
the terrain heightfield.  ``generate_chunk`` therefore ignores the seed entirely
and is a pure function of ``(coord, config)``.

Determinism
-----------
Trivially deterministic: identical ``(coord, config)`` always yields a
byte-identical array, on any process or machine.  This is what keeps delta saves
and reproducible bugs working (CLAUDE.md Hard Rule 2).

Everything is fully vectorised — no per-voxel Python loops (Hard Rule 4).
"""

from __future__ import annotations

import numpy as np

from fire_engine.core import Config

# ---------------------------------------------------------------------------
# Flat-world defaults (used only when no Config is supplied — e.g. the bare
# surface_height() helper).  The authoritative values live in config.toml /
# core.config.Config (ground_height_m, world_size_m); keep these in sync.
# ---------------------------------------------------------------------------

_FLAT_GROUND_Z_M: float = 8.0  # default flat ground surface height (world Z, meters)
_DEFAULT_WORLD_SIZE_M: float = 1000.0  # default square footprint side length (meters)

# ---------------------------------------------------------------------------
# Material ids (uint8 values stored in Chunk.materials; 0 = air).
# The renderer maps these to procedural textures (world/app.py):
#   MATERIAL_DIRT  -> "dirt_ground"
#   MATERIAL_GRASS -> "grass_ground"
# ---------------------------------------------------------------------------

MATERIAL_DIRT: int = 1  # bulk solid ground (exposed by digging)
MATERIAL_GRASS: int = 2  # topmost solid voxel layer of the baseline


def _ground_height(config: Config | None) -> float:
    """Flat ground surface height (world Z, meters) from config, or the default."""
    if config is not None and getattr(config, "ground_height_m", None) is not None:
        return float(config.ground_height_m)
    return _FLAT_GROUND_Z_M


def _world_half_extent(config: Config | None) -> float:
    """Half the square world footprint (meters); the footprint is centred on origin."""
    if config is not None and getattr(config, "world_size_m", None) is not None:
        return float(config.world_size_m) * 0.5
    return _DEFAULT_WORLD_SIZE_M * 0.5


# ===========================================================================
# Surface height (world XY → meters)
# ===========================================================================


def surface_height(
    world_x: np.ndarray,
    world_y: np.ndarray,
    config: Config | None = None,
) -> np.ndarray:
    """
    Flat terrain surface height (world Z, meters) at world XY positions.

    The baseline world is flat, so this returns a constant array equal to the
    configured ground height for every input position (the world footprint
    bound is applied in :func:`generate_chunk`, not here).

    Parameters
    ----------
    world_x, world_y : numpy.ndarray
        Broadcastable world X / Y coordinate arrays (meters).
    config : Config | None
        Engine config; ``config.ground_height_m`` is the surface height.  When
        ``None`` the module default (8.0 m) is used.

    Returns
    -------
    numpy.ndarray
        ``float32`` surface height in meters (world Z), broadcast shape of
        ``(world_x, world_y)``.

    Example
    -------
    >>> import numpy as np
    >>> h = surface_height(np.array([0.0, 8.0]), np.array([0.0, 0.0]))
    >>> h.shape
    (2,)
    >>> bool(np.all(h == 8.0))
    True
    """
    ground_z = _ground_height(config)
    shape = np.broadcast(world_x, world_y).shape
    return np.full(shape, ground_z, dtype=np.float32)


# ===========================================================================
# Chunk generation
# ===========================================================================


def generate_chunk(coord: tuple[int, int, int], config: Config) -> np.ndarray:
    """
    Generate the material array for a chunk — flat baseline terrain.

    Parameters
    ----------
    coord : tuple[int, int, int]
        Integer chunk coordinate ``(cx, cy, cz)``.
    config : Config
        Engine config.  Provides ``chunk_size``, ``voxel_size``,
        ``ground_height_m`` (flat surface Z) and ``world_size_m`` (square
        footprint side, centred on origin).  The world seed is **not** used —
        baseline terrain is seed-independent.

    Returns
    -------
    numpy.ndarray
        ``uint8`` array of shape ``(chunk_size,)*3`` indexed ``[x, y, z]``.
        ``0`` = air, :data:`MATERIAL_GRASS` (2) for the topmost solid voxel
        layer (the surface skin), :data:`MATERIAL_DIRT` (1) for everything
        solid below it.  Byte-identical for identical ``(coord, config)``.

    Behaviour
    ---------
    A voxel is solid iff BOTH:
      - its centre world Z is below ``config.ground_height_m`` (flat ground), AND
      - its centre world X and Y both lie within the world footprint
        ``[-world_size_m/2, +world_size_m/2)``.
    Everything else is air.  No hills, no caves.

    The single topmost solid layer (the voxel whose centre is below the ground
    height but whose +Z neighbour centre is not) is :data:`MATERIAL_GRASS`;
    all deeper solid voxels are :data:`MATERIAL_DIRT`.  The grass test is a
    pure function of world Z, so it is identical across chunk borders (a chunk
    fully buried under another solid chunk gets no grass).  Digging therefore
    exposes dirt, and rebuilding with the default brush material adds dirt.

    Example
    -------
    >>> from fire_engine.core import load_config
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

    ground_z = _ground_height(config)
    half = _world_half_extent(config)

    # World-space centre coordinate of every voxel along each local axis.
    # Voxel (x,y,z) centre world coord = origin + (idx + 0.5) * voxel_size.
    lin = (np.arange(n, dtype=np.float64) + 0.5) * vs
    wx_axis = ox + lin  # (n,) along local X
    wy_axis = oy + lin  # (n,) along local Y
    wz_axis = oz + lin  # (n,) along local Z

    # Inside the square world footprint (centred on origin) along X and Y.
    in_x = (wx_axis >= -half) & (wx_axis < half)  # (n,)
    in_y = (wy_axis >= -half) & (wy_axis < half)  # (n,)
    in_bounds = in_x[:, None] & in_y[None, :]  # (n, n) over (x, y)

    # Flat ground: solid below the ground height.
    below = wz_axis < ground_z  # (n,) over z

    # Topmost solid layer: solid here, but the voxel above would be air.
    # Pure function of world Z → identical across chunk borders.
    top_layer = below & (wz_axis + vs >= ground_z)  # (n,) over z

    # Solid where in-bounds AND below the ground surface.
    solid = in_bounds[:, :, None] & below[None, None, :]  # (n, n, n) [x, y, z]
    grass = in_bounds[:, :, None] & top_layer[None, None, :]

    materials = np.zeros((n, n, n), dtype=np.uint8)
    materials[solid] = MATERIAL_DIRT
    materials[grass] = MATERIAL_GRASS
    return materials
