"""
lighting/volume.py — Camera-centered geometry volumes for GPU lighting.

The GPU lighting pipeline lights the world through *radiance cascades*:
camera-centered 3-D textures holding, per cell, what the world looks like to
light (occupancy + albedo + emission).  This module is the headless half: it
owns the **window math** (where each cascade's box sits in the world, when it
recenters as the camera moves) and the **volume assembly** (slicing loaded
chunk material arrays into one contiguous numpy block per cascade, palette-
indexed to albedo/emission).  The panda3d half (`lighting/gpu.py`) only
uploads these arrays and dispatches compute shaders.

Units & conventions
-------------------
- World space in meters, Z-up.  A *cell* is one volume texel; cascade 0 uses
  ``cell_m = voxel_size`` (0.5 m), cascade 1 a coarser multiple (2.0 m).
- ``origin_cell`` is the integer world-cell index of texel ``(0, 0, 0)``:
  texel ``(i, j, k)`` covers world meters
  ``[(origin_cell + (i,j,k)) * cell_m, … + cell_m)``.
- Volume arrays are indexed ``[x, y, z]`` like ``Chunk.materials``.

No panda3d imports.  No per-voxel Python loops — assembly iterates *chunks*
(a few hundred at most) and uses vectorised slice/reshape ops inside.

Example
-------
>>> import numpy as np
>>> from fire_engine.lighting.volume import VolumeWindow, assemble_geometry
>>> from fire_engine.lighting.palette import MaterialPalette
>>> win = VolumeWindow(cells=32, cell_m=0.5)
>>> win.recenter((8.0, 8.0, 8.0))   # first call always (re)places the window
True
>>> class _Chunk:                    # minimal chunk stand-in
...     def __init__(self): self.materials = np.zeros((32, 32, 32), np.uint8)
>>> chunks = {(0, 0, 0): _Chunk()}
>>> vol = assemble_geometry(win, chunks, MaterialPalette(),
...                         chunk_size=32, voxel_size=0.5)
>>> vol.albedo_occ.shape
(32, 32, 32, 4)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fire_engine.lighting.palette import MaterialPalette

__all__ = [
    "VolumeWindow",
    "GeometryVolume",
    "assemble_geometry",
    "EMISSION_SCALE",
]

# Emission is HDR (a torch glow is ~2.0) but stored in uint8 textures; values
# are divided by this scale on pack and multiplied back in the shader.
EMISSION_SCALE: float = 8.0


class VolumeWindow:
    """
    A camera-centered, grid-snapped axis-aligned box of light cells.

    The window covers ``cells³`` cells of ``cell_m`` meters.  ``recenter``
    moves it to follow the camera, but only when the camera has drifted more
    than ``margin_cells`` cells from the window centre, and always snapping
    the origin to ``snap_cells`` so consecutive windows overlap exactly on
    the cell grid (no sub-cell crawl, stable GPU sampling).

    Parameters
    ----------
    cells : int
        Texels per axis (e.g. 96).
    cell_m : float
        Cell edge in meters (e.g. 0.5).  Must be an integer multiple of the
        terrain voxel size for assembly to slice chunk arrays exactly.
    snap_cells : int, default 8
        Origin snap granularity in cells.  Larger = fewer, bigger recenters.
    margin_cells : int, default 8
        Recenter when the camera is more than this many cells from the
        window centre on any axis.

    Example
    -------
    >>> win = VolumeWindow(cells=96, cell_m=0.5)
    >>> win.recenter((0.0, 0.0, 0.0))
    True
    >>> win.recenter((1.0, 0.0, 0.0))   # within margin — no move
    False
    >>> win.world_origin_m  # doctest: +ELLIPSIS
    (-24.0, -24.0, -24.0)
    """

    def __init__(
        self,
        cells: int,
        cell_m: float,
        snap_cells: int = 8,
        margin_cells: int = 8,
    ) -> None:
        if cells % snap_cells != 0:
            raise ValueError(f"cells ({cells}) must be a multiple of "
                             f"snap_cells ({snap_cells})")
        self.cells = int(cells)
        self.cell_m = float(cell_m)
        self.snap_cells = int(snap_cells)
        self.margin_cells = int(margin_cells)
        # World cell index of texel (0,0,0); None until first recenter.
        self.origin_cell: tuple[int, int, int] | None = None

    # ------------------------------------------------------------------

    @property
    def world_origin_m(self) -> tuple[float, float, float]:
        """World position (meters) of the window's min corner.

        Raises ``ValueError`` if ``recenter`` has never been called.
        """
        if self.origin_cell is None:
            raise ValueError("VolumeWindow.recenter() never called")
        return (self.origin_cell[0] * self.cell_m,
                self.origin_cell[1] * self.cell_m,
                self.origin_cell[2] * self.cell_m)

    @property
    def size_m(self) -> float:
        """World edge length of the window box in meters."""
        return self.cells * self.cell_m

    def _desired_origin(self, camera_pos) -> tuple[int, int, int]:
        """Snapped origin that centres the window on ``camera_pos``."""
        out = []
        for c in (camera_pos[0], camera_pos[1], camera_pos[2]):
            cell = int(np.floor(float(c) / self.cell_m)) - self.cells // 2
            snapped = int(np.floor(cell / self.snap_cells)) * self.snap_cells
            out.append(snapped)
        return (out[0], out[1], out[2])

    def recenter(self, camera_pos) -> bool:
        """
        Follow the camera; return True when the window moved.

        Parameters
        ----------
        camera_pos : sequence of 3 floats
            Camera world position in meters (``Vec3`` works — indexable).

        Returns
        -------
        bool
            True when ``origin_cell`` changed (caller must reassemble and
            re-upload the volume), False when the camera is still within the
            hysteresis margin.
        """
        if self.origin_cell is None:
            self.origin_cell = self._desired_origin(camera_pos)
            return True
        half = self.cells * 0.5
        for axis in range(3):
            centre = (self.origin_cell[axis] + half) * self.cell_m
            if abs(float(camera_pos[axis]) - centre) \
                    > self.margin_cells * self.cell_m:
                self.origin_cell = self._desired_origin(camera_pos)
                return True
        return False


@dataclass
class GeometryVolume:
    """
    Packed world-geometry block for one cascade, ready for GPU upload.

    Attributes
    ----------
    albedo_occ : numpy.ndarray
        ``uint8 (N, N, N, 4)`` indexed ``[x, y, z]``: RGB = surface albedo
        (linear, 0–255), A = 255 where the cell contains any solid voxel,
        0 where it is air.
    emission : numpy.ndarray
        ``uint8 (N, N, N, 4)``: RGB = emitted radiance / ``EMISSION_SCALE``
        (clipped to 255), A unused (255).
    origin_cell : tuple[int, int, int]
        World cell index of texel (0,0,0) at assembly time.
    cell_m : float
        Cell edge in meters.
    """

    albedo_occ: np.ndarray
    emission: np.ndarray
    origin_cell: tuple[int, int, int]
    cell_m: float


def assemble_geometry(
    window: VolumeWindow,
    chunks: dict,
    palette: MaterialPalette,
    chunk_size: int,
    voxel_size: float,
) -> GeometryVolume:
    """
    Slice loaded chunks into one contiguous geometry block for ``window``.

    For every chunk intersecting the window, the overlapping sub-array of
    ``chunk.materials`` is (a) downsampled to the window's cell size by
    taking the **max material id** per cell block (so any solid voxel makes
    the cell solid, and grass skin wins over dirt bulk for the bounce
    colour), then (b) palette-indexed to albedo/emission.  Cells outside any
    loaded chunk are air.

    Parameters
    ----------
    window : VolumeWindow
        Placed window (``recenter`` called at least once).
    chunks : dict[tuple[int, int, int], Chunk]
        Loaded chunks (``ChunkManager.chunks``).  Only ``chunk.materials``
        (``uint8 (S, S, S)``) is read.
    palette : MaterialPalette
        Material → albedo/emission lookup.
    chunk_size : int
        Voxels per chunk edge (``config.chunk_size``).
    voxel_size : float
        Meters per voxel (``config.voxel_size``).  ``window.cell_m`` must be
        an integer multiple of it.

    Returns
    -------
    GeometryVolume

    Notes
    -----
    Deterministic: pure function of the chunk materials, window placement and
    palette.  Python iterates **chunks** only (≤ a few hundred); all per-cell
    work is numpy slicing (Hard Rule 4).
    """
    if window.origin_cell is None:
        raise ValueError("VolumeWindow.recenter() never called")
    k = window.cell_m / voxel_size          # voxels per cell edge
    if abs(k - round(k)) > 1e-9 or k < 1:
        raise ValueError(
            f"cell_m ({window.cell_m}) must be an integer multiple of "
            f"voxel_size ({voxel_size})")
    k = int(round(k))
    n = window.cells
    cells_per_chunk = (chunk_size // k)     # window cells per chunk edge
    if cells_per_chunk * k != chunk_size:
        raise ValueError("chunk_size must be divisible by cells-per-voxel")

    ox, oy, oz = window.origin_cell         # window origin, in cells
    materials = np.zeros((n, n, n), dtype=np.uint8)

    # Chunk index range intersecting the window (in chunk coords).
    lo = [int(np.floor(o / cells_per_chunk)) for o in (ox, oy, oz)]
    hi = [int(np.floor((o + n - 1) / cells_per_chunk)) for o in (ox, oy, oz)]

    for ccx in range(lo[0], hi[0] + 1):
        for ccy in range(lo[1], hi[1] + 1):
            for ccz in range(lo[2], hi[2] + 1):
                chunk = chunks.get((ccx, ccy, ccz))
                if chunk is None:
                    continue
                # Chunk extent in window-cell coordinates.
                c0 = (ccx * cells_per_chunk, ccy * cells_per_chunk,
                      ccz * cells_per_chunk)
                # Overlap range in absolute cell coords.
                a = [max(c0[i], (ox, oy, oz)[i]) for i in range(3)]
                b = [min(c0[i] + cells_per_chunk,
                         (ox, oy, oz)[i] + n) for i in range(3)]
                if any(b[i] <= a[i] for i in range(3)):
                    continue
                # Source slice in chunk voxels; dest slice in window cells.
                src = tuple(
                    slice((a[i] - c0[i]) * k, (b[i] - c0[i]) * k)
                    for i in range(3))
                dst = tuple(
                    slice(a[i] - (ox, oy, oz)[i], b[i] - (ox, oy, oz)[i])
                    for i in range(3))
                block = chunk.materials[src]
                if k > 1:
                    s = block.shape
                    block = block.reshape(
                        s[0] // k, k, s[1] // k, k, s[2] // k, k
                    ).max(axis=(1, 3, 5))
                materials[dst] = block

    solid = materials > 0
    albedo_occ = np.empty((n, n, n, 4), dtype=np.uint8)
    albedo_occ[..., :3] = np.clip(
        palette.albedo[materials] * 255.0, 0.0, 255.0).astype(np.uint8)
    albedo_occ[..., 3] = np.where(solid, 255, 0).astype(np.uint8)

    emission = np.empty((n, n, n, 4), dtype=np.uint8)
    emission[..., :3] = np.clip(
        palette.emission[materials] * (255.0 / EMISSION_SCALE),
        0.0, 255.0).astype(np.uint8)
    emission[..., 3] = 255

    return GeometryVolume(
        albedo_occ=albedo_occ,
        emission=emission,
        origin_cell=window.origin_cell,
        cell_m=window.cell_m,
    )
