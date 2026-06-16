"""
buildings/occlusion.py — splat building solids into the lighting cascades.

`BuildingOccupancyRasterizer` is a structural
:class:`~fire_engine.lighting.volume.GeometryOccupancyProvider` (it implements
``rasterize_occupancy`` with the matching signature, but imports nothing from
``lighting/`` — the coupling is duck-typed on purpose, so the dependency rule
holds in both directions).  Registered with the GPU lighting pipeline via
``GpuLightingPipeline.register_geometry_provider``, it lets buildings shadow
the sun and bounce GI exactly like terrain and trees do, with **zero shader
changes** (the payoff of the unified geometry-volume contract).

**v1 status: documented NO-OP.**  `rasterize_occupancy` returns immediately, so
buildings are lit by the cascades but do not yet *occlude* them — interiors get
no wall-cast shadow / GI darkening (the visible v1 limitation called out in the
plan's verification step).  The reason is thread-safety, not difficulty: the
cascade assembly runs on an async worker, so a live `BuildingManager` cannot be
read from it safely.  Turning this on means (1) snapshotting the relevant
buildings' meshes/voxelization into an immutable struct on the main thread when
they change, and (2) threading that snapshot through `AssemblyJob` the way
`TreeOccluderSet` already is.  That snapshot plumbing is the next commit's
scope; this class nails down the seam and the algorithm so it is a fill-in.

Intended algorithm (when enabled)
---------------------------------
For each building overlapping the window ``[origin_cell, origin_cell+cells)`` at
``cell_m`` resolution:

1. Transform the window's cell-centre grid into building-LOCAL space (inverse
   of ``position`` + ``rotation``) — vectorized over the overlapping sub-box,
   never per cell.
2. For each storey, mark cells whose local (x, y) lies within ``thickness/2`` of
   any wall centerline (point-to-segment / point-to-arc distance, vectorized)
   AND whose local z is in the wall band — minus the cells inside an opening's
   (s, z) rect (windows/doors let light through).  Slabs (floor/foundation/
   roof) mark their full polygon × thickness.
3. Max-combine that occupancy into ``albedo_occ[..., 3]`` (so terrain/tree
   solids already there are never lowered) and write a plaster bounce albedo
   into ``albedo_occ[..., :3]`` only where occupancy rises — matching
   ``splat_tree_occluders``.  Coarse cascades scale a wall cell's contribution
   by (solid sub-cell volume / cell volume) so a thin wall in an 8 m cell reads
   as the wisp it is, not a solid block.

All of step 1-3 is numpy bulk work over the overlapping sub-box (Hard Rule 4);
the only Python loop is over buildings/storeys (a handful).

Docs: docs/systems/buildings.md
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fire_engine.core import get_logger

__all__ = ["BuildingOccupancyRasterizer"]

_log = get_logger("buildings.occlusion")


class BuildingOccupancyRasterizer:
    """
    Geometry-occupancy provider that splats a :class:`BuildingManager`'s
    buildings into the lighting cascades.  **v1: no-op** (see module docstring).

    Example
    -------
        from fire_engine.buildings.occlusion import BuildingOccupancyRasterizer

        rasterizer = BuildingOccupancyRasterizer(building_manager)
        gpu_lighting.register_geometry_provider(rasterizer)   # store-only (v1)

    Docs: docs/systems/buildings.md
    """

    def __init__(self, manager: Any) -> None:
        """
        Parameters
        ----------
        manager : BuildingManager
            Source of the buildings to rasterize (read via ``buildings()``).
            Typed loosely to avoid coupling lighting wiring to the concrete
            type at construction sites.
        """
        self._manager = manager
        self._warned = False

    def rasterize_occupancy(
        self,
        origin_cell: tuple[int, int, int],
        cells: int,
        cell_m: float,
        albedo_occ: np.ndarray,
        emission: np.ndarray,
    ) -> None:
        """
        Splat building solids into ``albedo_occ`` (no-op in v1).

        Signature matches
        :meth:`fire_engine.lighting.volume.GeometryOccupancyProvider.rasterize_occupancy`.
        Returns without touching the arrays, so a window assembled with this
        provider is byte-identical to one assembled without it (pinned by the
        lighting-volume test).  Logs once that occlusion is stubbed.

        Docs: docs/systems/buildings.md
        """
        if not self._warned:
            _log.debug(
                "building light occlusion is a v1 no-op — buildings "
                "are lit but do not yet shadow the cascades"
            )
            self._warned = True
        return  # v1: intentional no-op; see module docstring for the algorithm
