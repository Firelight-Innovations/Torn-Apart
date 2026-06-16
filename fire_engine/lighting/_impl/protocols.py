"""
Protocol definitions for fire_engine.lighting.

Grouping module — may define more than one public Protocol.  All protocols are
re-exported from the originating parent modules so historical import paths
remain unchanged.

Docs: docs/systems/lighting.md
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["GeometryOccupancyProvider"]


@runtime_checkable
class GeometryOccupancyProvider(Protocol):
    """
    Structural hook letting a NON-terrain geometry system (buildings, future
    props) splat its solids into the lighting cascades so the GPU marches see
    and shadow them — without lighting importing that system or vice versa
    (the Protocol is structural; nothing imports across the boundary).

    A provider rasterizes into the already-assembled volume arrays in place,
    **max-combining** occupancy the same way :func:`splat_tree_occluders` does:
    write a cell's occupancy alpha (and bounce albedo) only where it would
    *raise* the existing value, so terrain solids always win over a building
    cell that happens to overlap a hill.

    Implementations must be deterministic and must touch only cells inside the
    window (``origin_cell`` … ``origin_cell + cells`` per axis); cells outside
    their own geometry are left untouched (so ``providers=()`` — and providers
    whose geometry misses the window — leave the output byte-identical).

    Thread-safety: a provider may be called from the async cascade-assembly
    worker, so it must read an immutable snapshot of its geometry, never live
    mutable state.  (v1's building provider is a documented no-op; live
    snapshot wiring is future scope — see ``buildings/occlusion.py``.)
    """

    def rasterize_occupancy(
        self,
        origin_cell: tuple[int, int, int],
        cells: int,
        cell_m: float,
        albedo_occ: np.ndarray,
        emission: np.ndarray,
    ) -> None:
        """
        Splat this provider's geometry into ``albedo_occ`` / ``emission``.

        Parameters
        ----------
        origin_cell : tuple[int, int, int]
            Window origin in light cells (integer cell coords).
        cells : int
            Window edge length in cells (arrays are ``(cells,)*3 (+,4)``).
        cell_m : float
            Cell edge in meters (cascade resolution).
        albedo_occ : np.ndarray
            ``uint8 (cells, cells, cells, 4)`` — RGB bounce albedo + A
            occupancy; mutate in place, max-combining occupancy.
        emission : np.ndarray
            ``uint8 (cells, cells, cells, 4)`` — emissive RGB (÷EMISSION_SCALE)
            for self-lit surfaces (e.g. future glowing windows); usually
            untouched.
        """
        ...  # pragma: no cover
