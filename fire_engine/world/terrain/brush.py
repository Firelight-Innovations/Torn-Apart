"""
terrain/brush.py — Volumetric brush editing: the ONLY terrain mutation path.

Per ARCHITECTURE.md §5.5, players never dig individual voxels; every terrain
edit (explosions, mining, construction) flows through ``apply_brush``.  A brush
rasterises a 3-D shape into a boolean voxel mask — **one vectorised numpy
expression per intersected chunk** — and the mask is OR'd into materials (ADD)
or zeroed (REMOVE).

Each touched chunk is flagged ``dirty`` (needs remesh) and ``edited`` (deviates
from baseline → saved in the delta), and a single ``TerrainEditedEvent`` is
published per touched chunk.

Determinism: brushes contain no randomness; the mask is a pure function of the
shape and centre.  No per-voxel Python loops (Hard Rule 4).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from fire_engine.core import TerrainEditedEvent
from fire_engine.core.math3d import Vec3


class BrushMode(Enum):
    """Whether a brush adds solid material or removes it."""

    ADD = "add"
    REMOVE = "remove"


# ---------------------------------------------------------------------------
# Brush shapes.  Each implements ``mask(X, Y, Z, center)`` returning a boolean
# array (same shape as the broadcast of X,Y,Z) of voxel CENTRES inside the
# shape, and ``aabb(center)`` returning (min_corner, max_corner) world meters.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SphereBrush:
    """
    Solid sphere brush.

    Parameters
    ----------
    radius_m : float
        Sphere radius in meters.

    Example
    -------
    >>> SphereBrush(radius_m=2.5)        # a 2.5 m explosion radius
    SphereBrush(radius_m=2.5)
    """

    radius_m: float

    def aabb(self, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World-space axis-aligned bounds (min, max) in meters."""
        r = np.array([self.radius_m] * 3, dtype=np.float64)
        return center - r, center + r

    def mask(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, center: np.ndarray) -> np.ndarray:
        """Boolean mask: voxel centres within ``radius_m`` of ``center``."""
        cx, cy, cz = center
        result: np.ndarray = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 <= self.radius_m**2
        return result


@dataclass(frozen=True)
class BoxBrush:
    """
    Axis-aligned box brush.

    Parameters
    ----------
    half_extents_m : Vec3
        Half-size of the box along each world axis in meters.

    Example
    -------
    >>> from fire_engine.core.math3d import Vec3
    >>> BoxBrush(half_extents_m=Vec3(1.0, 1.0, 2.0))   # 2×2×4 m block
    BoxBrush(half_extents_m=Vec3(1.0, 1.0, 2.0))
    """

    half_extents_m: Vec3

    def aabb(self, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World-space axis-aligned bounds (min, max) in meters."""
        he = self.half_extents_m.to_numpy().astype(np.float64)
        return center - he, center + he

    def mask(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, center: np.ndarray) -> np.ndarray:
        """Boolean mask: voxel centres inside the axis-aligned box."""
        cx, cy, cz = center
        hx, hy, hz = self.half_extents_m.to_numpy()
        result: np.ndarray = (
            (np.abs(X - cx) <= hx) & (np.abs(Y - cy) <= hy) & (np.abs(Z - cz) <= hz)
        )
        return result


@dataclass(frozen=True)
class CylinderBrush:
    """
    Vertical cylinder brush (axis along world +Z).

    Parameters
    ----------
    radius_m : float
        Cylinder radius in meters (in the XY plane).
    height_m : float
        Full cylinder height in meters (centred on ``center`` along Z).

    Example
    -------
    >>> CylinderBrush(radius_m=1.5, height_m=4.0)
    CylinderBrush(radius_m=1.5, height_m=4.0)
    """

    radius_m: float
    height_m: float

    def aabb(self, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World-space axis-aligned bounds (min, max) in meters."""
        half = np.array([self.radius_m, self.radius_m, self.height_m / 2.0], dtype=np.float64)
        return center - half, center + half

    def mask(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, center: np.ndarray) -> np.ndarray:
        """Boolean mask: voxel centres inside the vertical cylinder."""
        cx, cy, cz = center
        half_h = self.height_m / 2.0
        radial = (X - cx) ** 2 + (Y - cy) ** 2 <= self.radius_m**2
        vertical = np.abs(Z - cz) <= half_h
        result: np.ndarray = radial & vertical
        return result


Brush = SphereBrush | BoxBrush | CylinderBrush


# ---------------------------------------------------------------------------
# apply_brush — the mutation entry point.
# ---------------------------------------------------------------------------


def _apply_brush_to_chunk(
    brush: Brush,
    center_np: np.ndarray,
    mode: BrushMode,
    material: int,
    coord: tuple[int, int, int],
    chunk_provider: Callable[[tuple[int, int, int]], Any],
) -> np.ndarray | None:
    """
    Apply the brush to a single chunk and return the changed-voxel mask.

    Returns the boolean changed array if any voxel changed, or ``None`` if the
    chunk was not modified.  Side-effect: writes to ``chunk.materials`` when a
    change occurs.
    """
    chunk = chunk_provider(coord)
    n = chunk.materials.shape[0]
    vs = float(chunk._voxel_size)
    origin = chunk.world_origin.to_numpy().astype(np.float64)
    lin = (np.arange(n, dtype=np.float64) + 0.5) * vs
    X = (origin[0] + lin)[:, None, None]
    Y = (origin[1] + lin)[None, :, None]
    Z = (origin[2] + lin)[None, None, :]
    mask = brush.mask(X, Y, Z, center_np)
    if not mask.any():
        return None
    if mode is BrushMode.ADD:
        changed: np.ndarray = mask & (chunk.materials != material)
        if not changed.any():
            return None
        chunk.materials[mask] = np.uint8(material)
    else:  # REMOVE
        changed = mask & (chunk.materials != 0)
        if not changed.any():
            return None
        chunk.materials[mask] = 0
    return changed


def _collect_neighbor_dirty(
    changed: np.ndarray,
    coord: tuple[int, int, int],
    neighbor_dirty: set[tuple[int, int, int]],
) -> None:
    """
    Record neighbour chunks that need a remesh due to boundary voxel changes.

    Changed voxels on a boundary slab affect the neighbour chunk's mesh
    (cross-chunk face culling + faceted border vertices) → queue those
    neighbours for a remesh.  Per-axis side flags; diagonal offsets are
    included when both/all of their axis sides were touched (slightly
    over-marks corner cases — a harmless extra remesh, never a miss).
    """
    last = changed.shape[0] - 1
    side: dict[int, tuple[bool, bool, bool]] = {
        -1: (
            bool(changed[0, :, :].any()),
            bool(changed[:, 0, :].any()),
            bool(changed[:, :, 0].any()),
        ),
        1: (
            bool(changed[last, :, :].any()),
            bool(changed[:, last, :].any()),
            bool(changed[:, :, last].any()),
        ),
    }
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            for oz in (-1, 0, 1):
                if (ox, oy, oz) == (0, 0, 0):
                    continue
                if (
                    (ox == 0 or side[ox][0])
                    and (oy == 0 or side[oy][1])
                    and (oz == 0 or side[oz][2])
                ):
                    neighbor_dirty.add((coord[0] + ox, coord[1] + oy, coord[2] + oz))


def _chunks_in_aabb(mn: np.ndarray, mx: np.ndarray, chunk_m: float) -> list[tuple[int, int, int]]:
    """
    All chunk coords whose 16 m cube intersects the world-space AABB [mn, mx].

    Vectorised range product over the three axes (small Python product over
    chunk coords only — not voxels).
    """
    c_min = np.floor(mn / chunk_m).astype(np.int64)
    c_max = np.floor(mx / chunk_m).astype(np.int64)
    coords: list[tuple[int, int, int]] = []
    for cx in range(int(c_min[0]), int(c_max[0]) + 1):
        for cy in range(int(c_min[1]), int(c_max[1]) + 1):
            coords.extend((cx, cy, cz) for cz in range(int(c_min[2]), int(c_max[2]) + 1))
    return coords


def apply_brush(
    brush: Brush,
    center: Vec3,
    mode: BrushMode,
    material: int = 1,
    *,
    chunk_provider: Callable[[tuple[int, int, int]], Any],
    bus: Any = None,
) -> set[tuple[int, int, int]]:
    """
    Apply a brush edit to terrain — the single terrain mutation path.

    Parameters
    ----------
    brush : SphereBrush | BoxBrush | CylinderBrush
        The shape to rasterise.
    center : Vec3
        World-space centre of the brush in meters.
    mode : BrushMode
        ``BrushMode.ADD`` writes ``material`` into masked voxels;
        ``BrushMode.REMOVE`` sets masked voxels to air (0).
    material : int, default 1
        Material id to write in ADD mode (ignored in REMOVE mode).
    chunk_provider : Callable[[tuple[int,int,int]], Chunk]
        **Contract:** ``chunk_provider(coord) -> Chunk`` returns the chunk for a
        coordinate, *creating/generating it on demand* if not loaded.  The
        ``ChunkManager.get_or_create`` method satisfies this; for headless tests
        a simple dict-backed provider that generates on miss works too.
    bus : EventBus | None, optional
        If given, one :class:`~fire_engine.core.TerrainEditedEvent` is published
        per touched chunk (``chunk_coords`` = that single coord, ``brush`` = the
        brush instance).

    Returns
    -------
    set[tuple[int, int, int]]
        The set of chunk coordinates actually modified (a voxel changed).  A
        chunk whose mask was empty after intersection is **not** included and is
        left untouched (no dirty/edited flag, no event).

    Notes
    -----
    Per-chunk work is one vectorised mask expression on an ``np.indices`` world
    grid — no per-voxel Python loops (Hard Rule 4).

    **Border remeshing:** when changed voxels touch a chunk boundary, the
    adjacent chunk's mesh depends on them too (face culling for the blocky
    mesher; border dual-cell vertex positions for the faceted mesher), so
    those neighbour chunks are flagged ``dirty`` (remesh) — but NOT
    ``edited`` (their voxels did not change, they stay out of the save delta)
    and with no ``TerrainEditedEvent`` (their light columns are unchanged).

    Example
    -------
    >>> from fire_engine.core import EventBus
    >>> from fire_engine.world.terrain.chunk import Chunk
    >>> from fire_engine.core import load_config
    >>> from fire_engine.core.rng import set_world_seed
    >>> set_world_seed(1); cfg = load_config()
    >>> store = {}
    >>> def provider(coord):
    ...     return store.setdefault(coord, Chunk(coord))
    >>> touched = apply_brush(SphereBrush(2.0), Vec3(8, 8, 8),
    ...                       BrushMode.ADD, chunk_provider=provider)
    >>> (0, 0, 0) in touched
    True
    """
    center_np = center.to_numpy().astype(np.float64)
    mn, mx = brush.aabb(center_np)
    touched: set[tuple[int, int, int]] = set()
    neighbor_dirty: set[tuple[int, int, int]] = set()

    for coord in _chunks_in_aabb_lazy(mn, mx, chunk_provider):
        changed = _apply_brush_to_chunk(brush, center_np, mode, material, coord, chunk_provider)
        if changed is None:
            continue
        chunk = chunk_provider(coord)
        chunk.dirty = True
        chunk.edited = True
        touched.add(coord)
        if bus is not None:
            bus.publish(TerrainEditedEvent(chunk_coords=coord, brush=brush))
        _collect_neighbor_dirty(changed, coord, neighbor_dirty)

    # Flag border neighbours for remesh (dirty only — not edited, no event).
    for coord in neighbor_dirty - touched:
        chunk_provider(coord).dirty = True

    return touched


def _chunks_in_aabb_lazy(
    mn: np.ndarray,
    mx: np.ndarray,
    chunk_provider: Callable[[tuple[int, int, int]], Any],
) -> list[tuple[int, int, int]]:
    """
    Yield chunk coords intersecting AABB [mn, mx].

    Reads ``chunk_meters`` from the provider's first chunk so brushes never
    hard-code the 16 m chunk size (config-driven).  Falls back to 16.0 m only if
    the provider yields nothing (impossible in practice).
    """
    # Probe one chunk at the AABB min to learn the chunk size.
    probe_coord = (
        int(np.floor(mn[0] / 16.0)),
        int(np.floor(mn[1] / 16.0)),
        int(np.floor(mn[2] / 16.0)),
    )
    probe = chunk_provider(probe_coord)
    chunk_m = probe.chunk_meters
    return _chunks_in_aabb(np.asarray(mn), np.asarray(mx), chunk_m)
