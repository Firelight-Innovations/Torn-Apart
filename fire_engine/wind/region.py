"""
wind/region.py — 2-D player-centred recenter window for the wind field.

The wind field covers a finite ``cells × cells`` grid of ``cell_m`` cells that
follows the player.  :class:`WindRegion` owns the **window math**: where the
grid sits in the world, when it re-snaps as the player moves (with hysteresis
so it doesn't crawl), and the cached world-coordinate meshes of the cell
**centres** that the gust evaluator samples at.

It is the 2-D analogue of ``lighting/volume.VolumeWindow`` (XY only, no Z), and
it additionally **caches the X/Y centre meshes** because the wind field is
analytic in position — recentering only needs to recompute those meshes, never
re-read any geometry.  This is what makes recentering free.

Units & conventions
-------------------
- World space meters, Z-up; the region is a horizontal XY tile.
- ``origin_cell`` is the integer world-cell index ``(ix, iy)`` of cell
  ``(0, 0)``'s **corner**: cell ``(i, j)`` spans world meters
  ``[(origin_cell + (i, j)) * cell_m, … + cell_m)`` and its **centre** is at
  ``(origin_cell + (i, j) + 0.5) * cell_m``.
- ``origin_m`` is that corner in meters — exactly what the GPU upload binds as
  ``u_wind_origin`` (texel-(0,0)-corner convention) in later work packages.

No panda3d.  No per-cell Python loops.

Example
-------
>>> region = WindRegion(cells=64, cell_m=4.0, snap_cells=8, margin_cells=8)
>>> region.maybe_recenter((0.0, 0.0))        # first call always places it
True
>>> region.maybe_recenter((4.0, 0.0))        # within the 32 m margin → no move
False
>>> region.X.shape
(64, 64)
"""

from __future__ import annotations

import numpy as np

__all__ = ["WindRegion"]


class WindRegion:
    """
    A player-centred, grid-snapped 2-D tile of wind cells with hysteresis.

    The tile covers ``cells × cells`` cells of ``cell_m`` meters (256 m square
    at the defaults).  :meth:`maybe_recenter` snaps the origin to follow the
    player, but only when the player drifts more than ``margin_cells`` from the
    tile centre on either axis, and always snapping the origin to ``snap_cells``
    so consecutive tiles overlap exactly on the cell grid (no sub-cell crawl,
    stable sampling, and bit-identical field values at world points shared
    between two placements).

    After any placing recenter, :attr:`X` and :attr:`Y` hold the world-space
    coordinates of every cell **centre**, indexed ``[i, j]`` (``ij`` meshgrid
    order, matching the field's ``[x, y]`` cell layout).

    Parameters
    ----------
    cells : int
        Cells per axis (e.g. 64).  Must be a multiple of ``snap_cells``.
    cell_m : float
        Cell edge in meters (e.g. 4.0).
    snap_cells : int, default 8
        Origin snap granularity in cells (larger = fewer, bigger recenters).
    margin_cells : int, default 8
        Recenter when the player is more than this many cells from the tile
        centre on either axis.

    Example
    -------
    >>> region = WindRegion(cells=64, cell_m=4.0)
    >>> region.maybe_recenter((100.0, -40.0))
    True
    >>> ox, oy = region.origin_m
    >>> # Player is within half a tile of the centre on both axes.
    >>> abs(100.0 - (ox + 64 * 4.0 / 2)) <= 32.0 + 1e-6
    True
    """

    def __init__(
        self,
        cells: int,
        cell_m: float,
        snap_cells: int = 8,
        margin_cells: int = 8,
    ) -> None:
        if cells % snap_cells != 0:
            raise ValueError(
                f"cells ({cells}) must be a multiple of snap_cells ({snap_cells})"
            )
        self.cells = int(cells)
        self.cell_m = float(cell_m)
        self.snap_cells = int(snap_cells)
        self.margin_cells = int(margin_cells)
        # World cell index of cell (0,0)'s corner; None until first recenter.
        self.origin_cell: tuple[int, int] | None = None
        # Cached cell-centre world-coordinate meshes (set on each placement).
        self.X: np.ndarray | None = None
        self.Y: np.ndarray | None = None
        # 1-D cell index ramp 0..cells-1, reused to rebuild the meshes cheaply.
        self._ramp = np.arange(self.cells, dtype=np.float32)

    # ------------------------------------------------------------------

    @property
    def origin_m(self) -> tuple[float, float]:
        """
        World position (meters) of cell ``(0, 0)``'s corner.

        This is the ``u_wind_origin`` value later work packages bind to the GPU
        (texel-(0,0)-corner convention).  Raises ``ValueError`` if
        :meth:`maybe_recenter` has never been called.
        """
        if self.origin_cell is None:
            raise ValueError("WindRegion.maybe_recenter() never called")
        return (self.origin_cell[0] * self.cell_m,
                self.origin_cell[1] * self.cell_m)

    @property
    def size_m(self) -> float:
        """World edge length of the tile in meters (``cells * cell_m``)."""
        return self.cells * self.cell_m

    def _desired_origin(self, player_xy) -> tuple[int, int]:
        """Snapped origin (in cells) that centres the tile on ``player_xy``."""
        out = []
        for c in (player_xy[0], player_xy[1]):
            cell = int(np.floor(float(c) / self.cell_m)) - self.cells // 2
            snapped = int(np.floor(cell / self.snap_cells)) * self.snap_cells
            out.append(snapped)
        return (out[0], out[1])

    def _rebuild_meshes(self) -> None:
        """Recompute the cached cell-centre world meshes from the origin."""
        assert self.origin_cell is not None
        ox, oy = self.origin_cell
        # Cell centres: (origin + index + 0.5) * cell_m.
        xs = (self._ramp + (ox + 0.5)) * self.cell_m
        ys = (self._ramp + (oy + 0.5)) * self.cell_m
        self.X, self.Y = np.meshgrid(xs, ys, indexing="ij")

    def needs_recenter(self, player_xy) -> bool:
        """
        True when the player has drifted past the hysteresis margin (or the
        tile was never placed).  Non-mutating — does not move the origin.
        """
        if self.origin_cell is None:
            return True
        half = self.cells * 0.5
        for axis in range(2):
            centre = (self.origin_cell[axis] + half) * self.cell_m
            if abs(float(player_xy[axis]) - centre) \
                    > self.margin_cells * self.cell_m:
                return True
        return False

    def maybe_recenter(self, player_xy) -> bool:
        """
        Follow the player; return True when the tile moved (and meshes rebuilt).

        Parameters
        ----------
        player_xy : sequence of 2 floats
            Player world XY in meters (any indexable; a ``Vec3``'s ``[0], [1]``
            work — Z is ignored).

        Returns
        -------
        bool
            True when ``origin_cell`` changed (the cached :attr:`X` / :attr:`Y`
            meshes were recomputed), False when the player is still within the
            hysteresis margin.

        Example
        -------
        >>> region = WindRegion(cells=64, cell_m=4.0)
        >>> region.maybe_recenter((0.0, 0.0))
        True
        >>> region.maybe_recenter((1.0, 1.0))   # < 32 m margin
        False
        """
        if self.needs_recenter(player_xy):
            self.origin_cell = self._desired_origin(player_xy)
            self._rebuild_meshes()
            return True
        return False
