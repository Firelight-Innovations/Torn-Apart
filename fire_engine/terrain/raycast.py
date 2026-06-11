"""
terrain/raycast.py — Voxel DDA raycast (Amanatides & Woo).

Casts a world-space ray and returns the first SOLID voxel it enters.  Used to
turn a mouse click (camera ray) into a brush centre — e.g. left-click → fire a
``SphereBrush(REMOVE)`` explosion at the hit point.

This is the ONE place a short Python loop is allowed (CLAUDE.md Hard Rule 4):
the DDA steps voxel-by-voxel, bounded to ≤200 steps, and runs once per click —
not per voxel of the world.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fire_engine.core.math3d import Vec3

_MAX_STEPS: int = 200


@dataclass(frozen=True)
class Hit:
    """
    Result of a successful voxel raycast.

    Attributes
    ----------
    point : Vec3
        World-space hit point in meters (where the ray crossed into the solid
        voxel).  Suitable as a brush centre.
    voxel : tuple[int, int, int]
        Global voxel coordinate hit (integer voxel grid; voxel = 0.5 m).
    chunk_coord : tuple[int, int, int]
        Chunk coordinate containing the hit voxel ``(cx, cy, cz)``.
    normal : Vec3
        World-space face normal of the entered voxel (the axis the ray stepped
        across to enter it), pointing back toward the ray origin.  Use this to
        place an ADD brush against the surface, or to nudge a REMOVE brush.
    distance : float
        Distance from the ray origin to ``point`` in meters.
    """

    point: Vec3
    voxel: tuple[int, int, int]
    chunk_coord: tuple[int, int, int]
    normal: Vec3
    distance: float


def _voxel_to_chunk(vx: int, vy: int, vz: int, n: int) -> tuple[int, int, int]:
    """Map a global voxel coord to its chunk coord (floor division by ``n``)."""
    return (vx // n, vy // n, vz // n)


def raycast_voxel(
    origin: Vec3,
    direction: Vec3,
    chunk_provider,
    max_distance_m: float = 100.0,
    *,
    chunk_size: int = 32,
    voxel_size: float = 0.5,
) -> Hit | None:
    """
    Cast a ray through the voxel field; return the first solid voxel hit.

    Parameters
    ----------
    origin : Vec3
        Ray start in world meters (e.g. the camera position).
    direction : Vec3
        Ray direction (need not be normalised; it is normalised internally).
    chunk_provider : Callable[[tuple[int,int,int]], Chunk]
        ``chunk_provider(coord) -> Chunk`` returning (creating on demand) the
        chunk for a coordinate — same contract as ``apply_brush``'s provider.
    max_distance_m : float, default 100.0
        Maximum ray length in meters.  The DDA also stops after 200 steps.
    chunk_size : int, default 32
        Voxels per chunk edge.
    voxel_size : float, default 0.5
        Meters per voxel edge.

    Returns
    -------
    Hit | None
        The first solid-voxel hit, or ``None`` if no solid voxel is encountered
        within ``max_distance_m`` / 200 steps.

    Example
    -------
    >>> # camera at (8, -4, 10) looking down toward terrain:
    >>> hit = raycast_voxel(Vec3(8, -4, 10), Vec3(0, 0, -1), provider)
    >>> if hit:
    ...     print(hit.chunk_coord, hit.point)
    """
    o = origin.to_numpy().astype(np.float64)
    d = direction.to_numpy().astype(np.float64)
    dlen = math.sqrt(float(d[0] ** 2 + d[1] ** 2 + d[2] ** 2))
    if dlen == 0.0:
        return None
    d = d / dlen

    vs = float(voxel_size)
    n = int(chunk_size)

    # Current voxel (global integer voxel coordinate) at the ray origin.
    voxel = np.floor(o / vs).astype(np.int64)

    # Step direction per axis (+1 / -1 / 0).
    step = np.sign(d).astype(np.int64)

    # tMax: distance (in ray-length units, here meters since d is unit) to the
    # next voxel boundary on each axis. tDelta: distance between boundaries.
    t_max = np.empty(3, dtype=np.float64)
    t_delta = np.empty(3, dtype=np.float64)
    for i in range(3):  # 3-iteration setup loop, not per-voxel
        if d[i] == 0.0:
            t_max[i] = math.inf
            t_delta[i] = math.inf
        else:
            # world coordinate of the next boundary in the step direction
            if step[i] > 0:
                next_boundary = (voxel[i] + 1) * vs
            else:
                next_boundary = voxel[i] * vs
            t_max[i] = (next_boundary - o[i]) / d[i]
            t_delta[i] = vs / abs(d[i])

    t = 0.0
    # Track which axis we last stepped across, to compute the entry normal.
    last_axis = -1

    for _ in range(_MAX_STEPS):
        if t > max_distance_m:
            return None

        cx, cy, cz = _voxel_to_chunk(int(voxel[0]), int(voxel[1]), int(voxel[2]), n)
        chunk = chunk_provider((cx, cy, cz))
        # local index within the chunk
        lx = int(voxel[0]) - cx * n
        ly = int(voxel[1]) - cy * n
        lz = int(voxel[2]) - cz * n
        if chunk.materials[lx, ly, lz] != 0:
            # Hit. Compute entry point and normal.
            point = o + d * t
            if last_axis < 0:
                # Origin was already inside a solid voxel.
                normal_np = -d
            else:
                normal_np = np.zeros(3, dtype=np.float64)
                normal_np[last_axis] = -float(step[last_axis])
            return Hit(
                point=Vec3.from_numpy(point.astype(np.float32)),
                voxel=(int(voxel[0]), int(voxel[1]), int(voxel[2])),
                chunk_coord=(cx, cy, cz),
                normal=Vec3.from_numpy(normal_np.astype(np.float32)),
                distance=float(t),
            )

        # Advance to the next voxel along the smallest t_max axis.
        axis = int(np.argmin(t_max))
        t = float(t_max[axis])
        voxel[axis] += step[axis]
        t_max[axis] += t_delta[axis]
        last_axis = axis

    return None
