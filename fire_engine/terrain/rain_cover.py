"""
terrain/rain_cover.py — Headless top-down rain-cover heightmap.

:class:`RainCoverField` answers one question per 1 m world column around the
player: **what is the world Z of the highest solid voxel above it?**  A roof or
overhang raises that height; an open-to-sky column reports the ground (or a
floor sentinel when no terrain is loaded there).  The GPU rain renderer
(``world/rain_renderer.py``) uploads this as a texture and discards any rain
streak whose world Z is *below* the cover height at its XY — so rain never falls
under a roof (the M6 headline fix).

This is the **headless** half (Hard Rule 1): pure numpy, no panda3d.  The world
component owns an instance, feeds it loaded chunks, marks dirty regions on
``ChunkLoadedEvent`` / ``TerrainEditedEvent``, recenters it as the player moves,
and uploads the array to a texture.

Field layout
------------
* A ``cells × cells`` grid of 1 m square columns (``cells`` ≈ 256 →  a 256 m
  window), centered on the player and snapped to whole cells (committed-origin
  discipline, mirroring ``wind``/``weather``).
* ``height[row, col]`` is the world Z (meters) of the **top face** of the
  highest solid voxel in that column, or :data:`OPEN_SKY_Z` when no solid voxel
  is known there.  Layout matches the weather/wind convention: ``row`` indexes
  world **+Y**, ``col`` indexes world **+X**.
* ``origin_m`` is the min-corner world XY (meters) of texel ``(0, 0)``; the GPU
  maps a world XY to a texel via ``(world_xy - origin_m) / cell_m``.

Vectorisation
-------------
The per-chunk reduction is a single vectorised pass over a chunk's
``(32, 32, 32)`` solidity mask: an ``argmax`` over reversed Z finds the highest
solid voxel per ``(x, y)`` column with **no Python loop over voxels** (Hard Rule
4).  Multiple chunk-Z layers in the same column are folded with ``np.maximum``.

Units: meters, voxels (0.5 m), world Z-up.

Example
-------
    from fire_engine.core import load_config
    from fire_engine.terrain import ChunkManager, RainCoverField
    from fire_engine.core import EventBus

    cfg = load_config()
    cm = ChunkManager(cfg, EventBus())
    field = RainCoverField(cfg)
    field.recenter((0.0, 0.0))               # snap the window under the player
    field.rebuild_all(cm.chunks)             # fold every loaded chunk's top-solid
    h = field.height                          # (cells, cells) float32 world Z (m)
    ox, oy = field.origin_m                   # min-corner world XY of texel (0,0)
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from fire_engine.core.config import Config
from fire_engine.terrain.chunk import Chunk

__all__ = ["RainCoverField", "OPEN_SKY_Z"]

#: World Z (meters) reported for a column with no known solid voxel — far below
#: any terrain, so an open-to-sky column never clips falling rain.  Rain streaks
#: live well above this, so ``instance_z >= cover`` is always true for open sky.
OPEN_SKY_Z: float = -1.0e9


class RainCoverField:
    """
    Top-down cover heightmap of the highest solid voxel per 1 m column.

    Parameters
    ----------
    config : Config
        Reads ``rain_cover_cells`` (window resolution, square), ``rain_cover_cell_m``
        (column edge in meters, 1.0), ``chunk_size``/``voxel_size`` (chunk geometry).

    Attributes
    ----------
    cells : int
        Columns per axis (the window is ``cells × cells``).
    cell_m : float
        Column edge in meters (1.0).
    span_m : float
        Window extent in meters (``cells · cell_m``).
    height : np.ndarray
        ``(cells, cells)`` float32 world Z (meters); ``height[row, col]`` with
        ``row`` → world +Y, ``col`` → world +X.  :data:`OPEN_SKY_Z` where unknown.
    origin_m : tuple[float, float]
        Min-corner world XY (meters) of texel ``(0, 0)`` — the committed origin,
        refreshed only by :meth:`recenter`.

    Units: meters, voxels (0.5 m), world Z-up.
    """

    def __init__(self, config: Config) -> None:
        self.cells: int = int(getattr(config, "rain_cover_cells", 256))
        self.cell_m: float = float(getattr(config, "rain_cover_cell_m", 1.0))
        self.span_m: float = self.cells * self.cell_m
        self._chunk_n: int = int(config.chunk_size)
        self._voxel_m: float = float(config.voxel_size)
        self._chunk_m: float = self._chunk_n * self._voxel_m

        # Window min-corner (world XY, meters).  Starts at the origin; recenter()
        # snaps it under the player.  Committed-origin: callers refresh the GPU
        # origin only in the same step they re-upload the texture.
        self._origin_x: float = 0.0
        self._origin_y: float = 0.0

        # The heightmap.  float32 so it uploads to an R32F texture directly.
        self.height: np.ndarray = np.full(
            (self.cells, self.cells), OPEN_SKY_Z, dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @property
    def origin_m(self) -> tuple[float, float]:
        """Min-corner world XY (meters) of texel ``(0, 0)`` — the committed origin."""
        return (self._origin_x, self._origin_y)

    def recenter(self, center_xy: tuple[float, float]) -> tuple[float, float]:
        """
        Snap the window's min-corner so it is centered on *center_xy*.

        The origin is snapped to a whole ``cell_m`` grid (committed-origin
        discipline) so world XY → texel maps to integer texels and the window
        never half-shifts.  Re-uploading must happen in the same step as the
        caller reads :attr:`origin_m`.

        Parameters
        ----------
        center_xy : tuple[float, float]
            World XY (meters) to center the window on (the player position).

        Returns
        -------
        tuple[float, float]
            The new committed ``origin_m`` (min-corner world XY, meters).
        """
        cx, cy = float(center_xy[0]), float(center_xy[1])
        half = 0.5 * self.span_m
        # Snap the min-corner to the cell grid so texels align to whole meters.
        self._origin_x = np.floor((cx - half) / self.cell_m) * self.cell_m
        self._origin_y = np.floor((cy - half) / self.cell_m) * self.cell_m
        return (self._origin_x, self._origin_y)

    # ------------------------------------------------------------------
    # Per-chunk top-solid reduction (vectorised — Hard Rule 4)
    # ------------------------------------------------------------------

    def _chunk_top_z(self, chunk: Chunk) -> np.ndarray | None:
        """
        World Z (meters) of the top face of the highest solid voxel per column.

        Vectorised ``argmax`` over reversed Z of the chunk's ``(32, 32, 32)``
        solidity mask — one pass, no per-voxel Python loop.

        Returns
        -------
        np.ndarray | None
            ``(32, 32)`` float32 of top-face world Z indexed ``[x, y]``, or
            ``None`` when the chunk is entirely air (nothing to fold).
        """
        solid = chunk.materials > 0                       # (n, n, n) [x, y, z]
        col_any = solid.any(axis=2)                       # (n, n) has any solid?
        if not col_any.any():
            return None
        # Highest solid z-index per column: argmax on the reversed Z axis finds
        # the first solid from the TOP; convert back to a forward z-index.
        n = self._chunk_n
        rev_first = np.argmax(solid[:, :, ::-1], axis=2)  # (n, n) from top
        top_z_idx = (n - 1) - rev_first                   # (n, n) forward index
        # World Z of the voxel's TOP face = origin_z + (z_idx + 1) * voxel_size.
        origin_z = chunk.coord[2] * self._chunk_m
        top_z = origin_z + (top_z_idx.astype(np.float32) + 1.0) * self._voxel_m
        # Columns with no solid stay at OPEN_SKY_Z (folded out by np.maximum).
        return np.where(col_any, top_z, np.float32(OPEN_SKY_Z)).astype(np.float32)

    def _fold_chunk(self, chunk: Chunk) -> None:
        """
        Fold one chunk's per-column top-Z into the heightmap window (vectorised).

        Maps the chunk's ``(32, 32)`` column footprint to the window's texel
        grid by world XY and applies an in-place ``np.maximum`` over the
        overlapping texels — so a higher chunk (a roof) wins over a lower one
        (the floor) in the same column.  No-op when the chunk is air or its
        footprint lies entirely outside the window.
        """
        top_z = self._chunk_top_z(chunk)
        if top_z is None:
            return
        n = self._chunk_n
        # Chunk column (x, y) world-min corner → window texel index.  Column
        # (x, y) covers world X∈[ox+x·vs, …); its center maps to a texel.  We
        # bin by the column's center to the nearest texel (cell_m grid).
        ox = chunk.coord[0] * self._chunk_m
        oy = chunk.coord[1] * self._chunk_m
        xi = np.arange(n)
        yi = np.arange(n)
        col_cx = ox + (xi.astype(np.float64) + 0.5) * self._voxel_m   # (n,) world X
        col_cy = oy + (yi.astype(np.float64) + 0.5) * self._voxel_m   # (n,) world Y
        tx = np.floor((col_cx - self._origin_x) / self.cell_m).astype(np.int64)  # col
        ty = np.floor((col_cy - self._origin_y) / self.cell_m).astype(np.int64)  # row
        # In-window mask per axis.
        mx = (tx >= 0) & (tx < self.cells)
        my = (ty >= 0) & (ty < self.cells)
        if not (mx.any() and my.any()):
            return
        # Restrict to in-window columns; build the (rows, cols) destination.
        gx_idx = np.where(mx)[0]            # chunk-local x indices in window
        gy_idx = np.where(my)[0]            # chunk-local y indices in window
        dst_cols = tx[gx_idx]               # window col per kept x  (→ world +X)
        dst_rows = ty[gy_idx]               # window row per kept y  (→ world +Y)
        # top_z is [x, y]; the window is [row=Y, col=X].  Select the kept block
        # and transpose to [y, x] so it lines up with [row, col].
        block = top_z[np.ix_(gx_idx, gy_idx)].T            # (len(y), len(x))
        # Scatter-max into the window (multiple chunk-Z layers fold here too).
        np.maximum.at(self.height, (dst_rows[:, None], dst_cols[None, :]), block)

    # ------------------------------------------------------------------
    # Public rebuild API (driven by the world component)
    # ------------------------------------------------------------------

    def rebuild_all(self, chunks: Mapping[tuple[int, int, int], Chunk]) -> None:
        """
        Recompute the whole window from scratch over *chunks*.

        Clears the heightmap to :data:`OPEN_SKY_Z`, then folds every chunk whose
        XY footprint overlaps the window.  Use after a recenter or for a full
        cold rebuild; for incremental edits prefer :meth:`rebuild_region`.

        Parameters
        ----------
        chunks : Mapping[coord, Chunk]
            Loaded chunks (e.g. ``ChunkManager.chunks``).
        """
        self.height.fill(OPEN_SKY_Z)
        for coord in self._chunks_in_window(chunks):
            self._fold_chunk(chunks[coord])

    def rebuild_columns(
        self,
        chunks: Mapping[tuple[int, int, int], Chunk],
        chunk_columns: Iterable[tuple[int, int]],
    ) -> None:
        """
        Rebuild only the window texels under the given chunk **columns** (cx, cy).

        A chunk column is the full Z stack ``(cx, cy, *)`` of loaded chunks; its
        window footprint is cleared to :data:`OPEN_SKY_Z` then re-folded from all
        loaded Z layers, so removing a roof (a ``TerrainEditedEvent``) correctly
        *lowers* the cover height there.  This is the incremental path the
        component amortises a budget of columns over.

        Parameters
        ----------
        chunks : Mapping[coord, Chunk]
            Loaded chunks.
        chunk_columns : Iterable[tuple[int, int]]
            ``(cx, cy)`` chunk-column coords to refresh.
        """
        for cx, cy in chunk_columns:
            self._clear_chunk_column(cx, cy)
            # Re-fold every loaded Z layer of this column.
            for coord, chunk in chunks.items():
                if coord[0] == cx and coord[1] == cy:
                    self._fold_chunk(chunk)

    def _clear_chunk_column(self, cx: int, cy: int) -> None:
        """Reset the window texels covered by chunk column (cx, cy) to OPEN_SKY_Z."""
        ox = cx * self._chunk_m
        oy = cy * self._chunk_m
        # World XY span of the chunk column.
        c0 = int(np.floor((ox - self._origin_x) / self.cell_m))
        c1 = int(np.ceil((ox + self._chunk_m - self._origin_x) / self.cell_m))
        r0 = int(np.floor((oy - self._origin_y) / self.cell_m))
        r1 = int(np.ceil((oy + self._chunk_m - self._origin_y) / self.cell_m))
        c0 = max(c0, 0); c1 = min(c1, self.cells)
        r0 = max(r0, 0); r1 = min(r1, self.cells)
        if c1 > c0 and r1 > r0:
            self.height[r0:r1, c0:c1] = OPEN_SKY_Z

    def _chunks_in_window(
        self, chunks: Mapping[tuple[int, int, int], Chunk]
    ) -> list[tuple[int, int, int]]:
        """Loaded chunk coords whose XY footprint overlaps the current window."""
        out: list[tuple[int, int, int]] = []
        wx0, wy0 = self._origin_x, self._origin_y
        wx1, wy1 = wx0 + self.span_m, wy0 + self.span_m
        for coord in chunks:
            cox = coord[0] * self._chunk_m
            coy = coord[1] * self._chunk_m
            if cox + self._chunk_m <= wx0 or cox >= wx1:
                continue
            if coy + self._chunk_m <= wy0 or coy >= wy1:
                continue
            out.append(coord)
        return out

    @staticmethod
    def chunk_column_of(coord: tuple[int, int, int]) -> tuple[int, int]:
        """The (cx, cy) chunk column a chunk coord belongs to (drops cz)."""
        return (int(coord[0]), int(coord[1]))
