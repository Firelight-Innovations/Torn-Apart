"""
zones/grass_placement.py — headless math for GPU-only grass placement.

The grass renderer (``world/grass_renderer.py``) draws every blade tuft via
hardware instancing: the CPU stores **no per-blade data** — each instance
derives its world position, rotation, scale and sway phase in the vertex
shader from ``gl_InstanceID`` through an integer hash.  This module is the
testable, panda3d-free half of that contract:

- :func:`hash_lowbias32` / :func:`instance_attribs` — a **line-for-line
  python mirror of the GLSL hash chain** in ``world/grass_shaders.py``.
  The two must stay byte-identical; the headless tests pin this mirror so
  any GLSL edit that forgets the mirror fails review.
- :func:`grass_hash_seed` — per-volume hash seed via ``core.rng.for_domain``
  (Hard Rule 2: all randomness through for_domain).
- :func:`grass_instance_count` — blade count from volume area × density.
- :func:`bake_grass_height_field` — the one CPU-baked texture per volume:
  a small RGBA8 grid (1 texel per voxel, 0.5 m) whose R channel encodes the
  terrain surface height inside the volume's Z window (255 = no surface →
  the shader collapses those blades).  Re-baked on terrain edits, so grass
  vanishes inside craters.

Units: meters, Z-up.  All bulk work is numpy (Hard Rule 4).

Example
-------
    from fire_engine.zones import grass_placement as gp

    count = gp.grass_instance_count(vol, cfg)            # blades to instance
    seed  = gp.grass_hash_seed(vol)                      # uniform u_hash_seed
    attrs = gp.instance_attribs(np.arange(count), seed,
                                vol.min_corner, vol.max_corner)
    field = gp.bake_grass_height_field(vol, chunk_manager.chunks, cfg)
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from fire_engine.core import Config
from fire_engine.core.rng import for_domain
from fire_engine.zones.volume import ZoneVolume

__all__ = [
    "hash_lowbias32",
    "instance_attribs",
    "grass_hash_seed",
    "grass_instance_count",
    "bake_grass_height_field",
    "HEIGHT_SENTINEL",
]

# R-channel byte meaning "no terrain surface in the volume's Z window here" —
# the vertex shader culls (collapses) blades whose footprint texel holds it.
HEIGHT_SENTINEL: int = 255

# Hash-chain XOR constants — MUST match world/grass_shaders.py exactly.
_K1 = np.uint32(0x9E3779B9)
_K2 = np.uint32(0x85EBCA6B)
_K3 = np.uint32(0xC2B2AE35)
_K4 = np.uint32(0x27D4EB2F)

# Per-instance scale jitter range (unitless multipliers on blade size).
_SCALE_MIN = 0.7
_SCALE_SPAN = 0.6


def hash_lowbias32(x: np.ndarray) -> np.ndarray:
    """
    Vectorized lowbias32 integer hash (Chris Wellons) over uint32.

    Line-for-line mirror of ``lowbias32(uint x)`` in
    ``world/grass_shaders.py`` — keep both in sync.

    Parameters
    ----------
    x : numpy.ndarray
        Any integer array; treated as uint32 (wrapping arithmetic).

    Returns
    -------
    numpy.ndarray
        uint32 array of hashed values, same shape.
    """
    x = np.asarray(x).astype(np.uint32, copy=True)
    x ^= x >> np.uint32(16)
    x *= np.uint32(0x7FEB352D)
    x ^= x >> np.uint32(15)
    x *= np.uint32(0x846CA68B)
    x ^= x >> np.uint32(16)
    return x


def instance_attribs(
    indices: np.ndarray,
    seed: int,
    min_corner: tuple[float, float, float],
    max_corner: tuple[float, float, float],
) -> dict[str, np.ndarray]:
    """
    Per-instance placement attributes — python mirror of the GLSL chain.

    The vertex shader computes EXACTLY this from ``gl_InstanceID``; this
    mirror exists so placement is testable headlessly (determinism, bounds,
    distribution) without a GPU.

    Parameters
    ----------
    indices : numpy.ndarray
        Instance ids (``gl_InstanceID`` values), any integer dtype.
    seed : int
        Per-volume hash seed (``grass_hash_seed``), 0 <= seed < 2**31.
    min_corner / max_corner : tuple[float, float, float]
        The grass volume's AABB corners (world meters).

    Returns
    -------
    dict[str, numpy.ndarray]
        ``"x"``/``"y"`` world-space blade base positions (float32, meters),
        ``"rot"`` yaw in radians [0, 2π), ``"scale"`` size multiplier
        [0.7, 1.3), ``"phase"`` sway phase in radians [0, 2π).
    """
    i = np.asarray(indices).astype(np.uint32, copy=False)
    h0 = hash_lowbias32(i ^ np.uint32(seed))
    h1 = hash_lowbias32(h0 ^ _K1)
    h2 = hash_lowbias32(h1 ^ _K2)
    h3 = hash_lowbias32(h2 ^ _K3)
    h4 = hash_lowbias32(h3 ^ _K4)

    inv = np.float32(1.0 / 4294967296.0)   # 1 / 2^32 — matches GLSL u2f()
    fx = h0.astype(np.float32) * inv
    fy = h1.astype(np.float32) * inv
    size_x = np.float32(max_corner[0] - min_corner[0])
    size_y = np.float32(max_corner[1] - min_corner[1])
    two_pi = np.float32(2.0 * math.pi)
    return {
        "x": np.float32(min_corner[0]) + fx * size_x,
        "y": np.float32(min_corner[1]) + fy * size_y,
        "rot": h2.astype(np.float32) * inv * two_pi,
        "scale": np.float32(_SCALE_MIN)
        + h3.astype(np.float32) * inv * np.float32(_SCALE_SPAN),
        "phase": h4.astype(np.float32) * inv * two_pi,
    }


def grass_hash_seed(volume: ZoneVolume) -> int:
    """
    Deterministic per-volume hash seed for the instance chain.

    Derived through ``for_domain("zones", "grass", volume.id)`` (Hard Rule 2)
    so the same world seed + volume id always places identical grass.
    Bounded to [0, 2**31) — Panda3D shader inputs pass it as a signed int.
    """
    return int(for_domain("zones", "grass", volume.id).integers(0, 2 ** 31))


def grass_instance_count(volume: ZoneVolume, config: Config) -> int:
    """
    Number of blade instances a grass volume spawns.

    ``density × footprint area``, where density (blades/m²) comes from the
    volume's ``params["density"]`` or ``config.grass_density_per_m2``,
    clamped to ``config.grass_max_instances``.

    Example
    -------
    >>> from fire_engine.core import Config
    >>> from fire_engine.zones import ZoneVolume
    >>> v = ZoneVolume(1, "grass", (0.0, 0.0, 0.0), (10.0, 10.0, 4.0),
    ...                params={"density": 8.0})
    >>> grass_instance_count(v, Config())
    800
    """
    density = float(volume.params.get("density", config.grass_density_per_m2))
    count = int(volume.area_xy_m2 * max(density, 0.0))
    return max(0, min(count, int(config.grass_max_instances)))


def bake_grass_height_field(
    volume: ZoneVolume,
    chunks: Mapping[tuple[int, int, int], object],
    config: Config,
) -> np.ndarray:
    """
    Bake the per-volume terrain height field the grass shader stands on.

    One texel per voxel column (0.5 m).  For each texel the topmost solid
    voxel whose **top face Z lies inside the volume's Z window** is found;
    its height is encoded into the R channel:

    - ``R in [0, 254]`` → surface Z = ``min_z + R/254 × (max_z − min_z)``
    - ``R == 255`` (:data:`HEIGHT_SENTINEL`) → no surface in the window
      (carved crater, unloaded chunk, off-world) → the shader culls the blade.

    G/B are reserved (zero), A is 255 for debug viewing.  Texel row index 0
    is the volume's min-Y edge and column 0 its min-X edge — uploaded WITHOUT
    a vertical flip (``texture_bridge.to_field_texture``) so the shader can
    sample at ``uv = (world_xy − min) / size`` directly.

    Parameters
    ----------
    volume : ZoneVolume
        The grass volume (its AABB defines both the grid and the Z window).
    chunks : Mapping[tuple[int, int, int], object]
        Loaded chunks (``ChunkManager.chunks``); each value exposes a
        ``materials`` uint8 ``(n, n, n)`` array indexed ``[x, y, z]``.
    config : Config
        Provides ``chunk_size`` and ``voxel_size``.

    Returns
    -------
    numpy.ndarray
        ``uint8 (H, W, 4)`` with ``H`` texel rows along +Y and ``W`` columns
        along +X.  Deterministic for identical inputs.
    """
    n = int(config.chunk_size)
    vs = float(config.voxel_size)
    x0, y0, z0 = volume.min_corner
    x1, y1, z1 = volume.max_corner

    W = max(1, int(math.ceil((x1 - x0) / vs)))
    H = max(1, int(math.ceil((y1 - y0) / vs)))

    # Global voxel index of each texel column (sampled at texel centres).
    vox_x = np.floor((x0 + (np.arange(W) + 0.5) * vs) / vs).astype(np.int64)
    vox_y = np.floor((y0 + (np.arange(H) + 0.5) * vs) / vs).astype(np.int64)

    # Chunk ranges the texel grid and Z window can touch.  A candidate voxel's
    # top face Z = (kz_global + 1) * vs must lie in [z0, z1], so kz_global
    # spans [z0/vs − 1, z1/vs − 1].
    cxs = np.unique(vox_x // n)
    cys = np.unique(vox_y // n)
    kz_lo = int(math.floor(z0 / vs)) - 1
    kz_hi = int(math.ceil(z1 / vs))
    czs = range(kz_lo // n, kz_hi // n + 1)

    best_z = np.full((H, W), -np.inf, dtype=np.float64)
    for cx in cxs:
        sel_x = np.nonzero((vox_x >= cx * n) & (vox_x < (cx + 1) * n))[0]
        if sel_x.size == 0:
            continue
        local_x = (vox_x[sel_x] - cx * n).astype(np.intp)
        for cy in cys:
            sel_y = np.nonzero((vox_y >= cy * n) & (vox_y < (cy + 1) * n))[0]
            if sel_y.size == 0:
                continue
            local_y = (vox_y[sel_y] - cy * n).astype(np.intp)
            for cz in czs:
                chunk = chunks.get((int(cx), int(cy), int(cz)))
                if chunk is None:
                    continue
                # (sx, sy, n) material sub-block for the selected columns.
                sub = chunk.materials[np.ix_(local_x, local_y)]
                solid = sub != 0
                # Topmost solid voxel index per column (−1 = none).
                kz = np.where(solid, np.arange(n)[None, None, :], -1).max(axis=2)
                z_top = (cz * n + kz + 1) * vs
                valid = (kz >= 0) & (z_top >= z0 - 1e-6) & (z_top <= z1 + 1e-6)
                z_val = np.where(valid, z_top, -np.inf)
                # best_z is (H=y, W=x); z_val is (sx, sy) → transpose.
                region = best_z[np.ix_(sel_y, sel_x)]
                best_z[np.ix_(sel_y, sel_x)] = np.maximum(region, z_val.T)

    out = np.zeros((H, W, 4), dtype=np.uint8)
    has_surface = np.isfinite(best_z)
    z_span = max(z1 - z0, 1e-6)
    frac = np.clip((best_z - z0) / z_span, 0.0, 1.0)
    encoded = np.clip(np.round(frac * 254.0), 0, 254).astype(np.uint8)
    out[..., 0] = np.where(has_surface, encoded, np.uint8(HEIGHT_SENTINEL))
    out[..., 3] = 255
    return out
