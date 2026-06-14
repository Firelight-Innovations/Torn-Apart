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

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from fire_engine.lighting.occluders import TreeOccluderSet, splat_tree_occluders
from fire_engine.lighting.palette import MaterialPalette

__all__ = [
    "EMISSION_SCALE",
    "ChunkBlockCache",
    "GeometryOccupancyProvider",
    "GeometryVolume",
    "VolumeWindow",
    "assemble_geometry",
    "pack_volume",
    "window_chunk_span",
]


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
            raise ValueError(f"cells ({cells}) must be a multiple of snap_cells ({snap_cells})")
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
        return (
            self.origin_cell[0] * self.cell_m,
            self.origin_cell[1] * self.cell_m,
            self.origin_cell[2] * self.cell_m,
        )

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

    def needs_recenter(self, camera_pos) -> bool:
        """
        True when the camera has drifted past the hysteresis margin (or the
        window was never placed) — i.e. a reassembly *should* be scheduled.

        Non-mutating: unlike :meth:`recenter`, this does NOT move
        ``origin_cell``.  The async assembly path uses it to decide whether to
        submit a job while leaving the *committed* origin (what the GPU volume
        and shader uniforms currently use) untouched until the new volume is
        actually uploaded.  See ``lighting/gpu.py``.
        """
        if self.origin_cell is None:
            return True
        half = self.cells * 0.5
        for axis in range(3):
            centre = (self.origin_cell[axis] + half) * self.cell_m
            if abs(float(camera_pos[axis]) - centre) > self.margin_cells * self.cell_m:
                return True
        return False

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
            if abs(float(camera_pos[axis]) - centre) > self.margin_cells * self.cell_m:
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
        (linear, 0–255), A = **solid sub-voxel fraction ×255**: the fraction
        of the cell's ``k³`` terrain voxels that are solid, rounded to a byte.
        At cascade 0 (``cell_m == voxel_size``, ``k == 1``) this is exactly
        255 (solid) or 0 (air), identical to a binary occupancy flag.  At the
        coarse cascades it is a partial value, so a hollow room reads air
        (A == 0) in its interior and only its 1-voxel walls read partly-solid
        — the GPU probes no longer treat a hollow box as a solid block.
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


def _downsample_chunk_block(mats: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Downsample a chunk's ``uint8 (S, S, S)`` material array to a per-cell
    mini-block at a ``k`` voxels/cell ratio.

    Returns a ``(material_id, solid_count)`` pair, both ``(S//k, S//k, S//k)``:

    - ``material_id`` (``uint8``) — **max** material id over each ``k³`` block
      (so grass skin wins over dirt bulk for the bounce albedo, and any solid
      voxel keeps a non-zero id for the palette lookup).
    - ``solid_count`` (``uint16``) — number of solid (``> 0``) voxels in each
      ``k³`` block, ``0 .. k³``.  Divided by ``k³`` this is the cell's solid
      sub-voxel fraction; ``×255`` and rounded it becomes the occupancy alpha.

    At ``k == 1`` (cascade 0) ``material_id`` is the array itself and
    ``solid_count`` is 0/1 — i.e. binary occupancy, byte-identical to the old
    any-solid behaviour.  All numpy bulk ops; no per-voxel Python loops.
    """
    if k == 1:
        return mats, (mats > 0).astype(np.uint16)
    s = mats.shape
    folded = mats.reshape(s[0] // k, k, s[1] // k, k, s[2] // k, k)
    material_id = folded.max(axis=(1, 3, 5)).astype(np.uint8)
    solid_count = (folded > 0).sum(axis=(1, 3, 5)).astype(np.uint16)
    return material_id, solid_count


def assemble_geometry(
    window: VolumeWindow,
    chunks: dict,
    palette: MaterialPalette,
    chunk_size: int,
    voxel_size: float,
    cache: ChunkBlockCache | None = None,
    occluders: TreeOccluderSet | None = None,
    trunk_occ: float = 0.0,
    canopy_gain: float = 0.0,
    providers: tuple[GeometryOccupancyProvider, ...] = (),
) -> GeometryVolume:
    """
    Slice loaded chunks into one contiguous geometry block for ``window``.

    For every chunk intersecting the window, the overlapping sub-array of
    ``chunk.materials`` is (a) downsampled to the window's cell size — the
    cell's material id is the **max** id over its ``k³`` voxels (so any solid
    voxel keeps a non-zero id and grass skin wins over dirt bulk for the
    bounce colour), and the cell's occupancy is the **solid sub-voxel
    fraction** (count of solid voxels ÷ ``k³``) — then (b) palette-indexed to
    albedo/emission.  Cells outside any loaded chunk are air.

    The occupancy alpha is therefore ``round(255 × solid_fraction)``, NOT a
    binary any-solid flag: a hollow room (1-voxel walls, air interior)
    downsamples to ``A == 0`` in its interior and partial ``A`` on its walls,
    so the GPU lighting probes inside it read air instead of a solid block.
    At cascade 0 (``cell_m == voxel_size``) the fraction is exactly 0 or 1, so
    the output is byte-identical to the previous binary-occupancy behaviour.

    Parameters
    ----------
    window : VolumeWindow
        Placed window (``recenter`` called at least once).
    chunks : dict[tuple[int, int, int], Chunk | numpy.ndarray]
        Loaded chunks (``ChunkManager.chunks``).  Each value may be a chunk
        object (only ``chunk.materials`` — ``uint8 (S, S, S)`` — is read) or a
        bare ``materials`` ndarray (the form passed by the async assembly
        worker, which snapshots arrays rather than live chunks).
    palette : MaterialPalette
        Material → albedo/emission lookup.
    chunk_size : int
        Voxels per chunk edge (``config.chunk_size``).
    voxel_size : float
        Meters per voxel (``config.voxel_size``).  ``window.cell_m`` must be
        an integer multiple of it.
    cache : ChunkBlockCache, optional
        Per-chunk downsampled-block cache (see :class:`ChunkBlockCache`).  When
        supplied, each chunk's full ``(material_id, solid_count)`` mini-block at
        this ``cell_m`` is reused across reassemblies instead of recomputed — a
        large saving for the coarse far cascade, whose ~33k-chunk window is
        otherwise re-downsampled from scratch on every recenter.  Output is
        byte-identical with and without the cache.
    occluders : TreeOccluderSet, optional
        Static tree/bush occluders splatted onto the assembled block as
        fractional occupancy + bounce albedo (see ``lighting/occluders.py``)
        so the lighting marches see trees.  ``None`` (default) leaves the
        output byte-identical to the chunks-only assembly.
    trunk_occ : float
        Trunk splat opacity (``config.light_tree_trunk_occ``).  Ignored when
        ``occluders`` is ``None``.
    canopy_gain : float
        Multiplier on each instance's leaf-derived per-meter canopy
        extinction (``config.light_tree_canopy_extinction_gain``).  Ignored
        when ``occluders`` is ``None``.
    providers : tuple[GeometryOccupancyProvider, ...], optional
        Non-terrain geometry providers (buildings, future props) splatted into
        the volume after the chunk gather + tree occluders, max-combining
        occupancy (see :class:`GeometryOccupancyProvider`).  Empty (default)
        leaves the output byte-identical to the chunks-only assembly.

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
    k = window.cell_m / voxel_size  # voxels per cell edge
    if abs(k - round(k)) > 1e-9 or k < 1:
        raise ValueError(
            f"cell_m ({window.cell_m}) must be an integer multiple of voxel_size ({voxel_size})"
        )
    k = int(round(k))
    n = window.cells
    cells_per_chunk = chunk_size // k  # window cells per chunk edge
    if cells_per_chunk * k != chunk_size:
        raise ValueError("chunk_size must be divisible by cells-per-voxel")

    ox, oy, oz = window.origin_cell  # window origin, in cells
    materials = np.zeros((n, n, n), dtype=np.uint8)
    # Per-cell solid sub-voxel count (0 .. k³); ÷k³ ×255 → occupancy alpha.
    solid_count = np.zeros((n, n, n), dtype=np.uint16)

    # Chunk index range intersecting the window (in chunk coords).
    lo = [int(np.floor(o / cells_per_chunk)) for o in (ox, oy, oz)]
    hi = [int(np.floor((o + n - 1) / cells_per_chunk)) for o in (ox, oy, oz)]

    for ccx in range(lo[0], hi[0] + 1):
        for ccy in range(lo[1], hi[1] + 1):
            for ccz in range(lo[2], hi[2] + 1):
                coord = (ccx, ccy, ccz)
                chunk = chunks.get(coord)
                if chunk is None:
                    continue
                # Per-chunk mini-blocks downsampled to this cell size.  A cache
                # hit reuses them; a miss computes + stores them.  Only caches
                # the coarse cascades (k > 1): at k == 1 the "block" aliases the
                # live chunk array, and there's no downsample cost to amortise.
                use_cache = cache is not None and k > 1
                blk = cache.get(coord, window.cell_m) if use_cache else None
                if blk is None:
                    # Accept either a chunk object (.materials) or a bare
                    # ndarray snapshot (async assembly worker passes the latter).
                    mats = getattr(chunk, "materials", chunk)
                    blk = _downsample_chunk_block(mats, k)
                    if use_cache:
                        cache.put(coord, window.cell_m, blk)
                chunk_mat, chunk_cnt = blk
                # Chunk extent in window-cell coordinates.
                c0 = (ccx * cells_per_chunk, ccy * cells_per_chunk, ccz * cells_per_chunk)
                # Overlap range in absolute cell coords.
                a = [max(c0[i], (ox, oy, oz)[i]) for i in range(3)]
                b = [min(c0[i] + cells_per_chunk, (ox, oy, oz)[i] + n) for i in range(3)]
                if any(b[i] <= a[i] for i in range(3)):
                    continue
                # Source slice in the chunk's cell mini-block; dest in window.
                src = tuple(slice(a[i] - c0[i], b[i] - c0[i]) for i in range(3))
                dst = tuple(slice(a[i] - (ox, oy, oz)[i], b[i] - (ox, oy, oz)[i]) for i in range(3))
                materials[dst] = chunk_mat[src]
                solid_count[dst] = chunk_cnt[src]

    albedo_occ = np.empty((n, n, n, 4), dtype=np.uint8)
    albedo_occ[..., :3] = np.clip(palette.albedo[materials] * 255.0, 0.0, 255.0).astype(np.uint8)
    # A = solid sub-voxel fraction ×255, rounded.  k³ at k==1 → 0/255 binary.
    occ = np.rint(solid_count.astype(np.float32) * (255.0 / (k**3)))
    albedo_occ[..., 3] = np.clip(occ, 0.0, 255.0).astype(np.uint8)

    # Static tree/bush occluders — splatted after the chunk gather so terrain
    # solids win the max-combine (a tree inside a hill stays hill-solid).
    if occluders is not None and occluders.count:
        splat_tree_occluders(
            albedo_occ, window.origin_cell, window.cell_m, occluders, trunk_occ, canopy_gain
        )

    emission = np.empty((n, n, n, 4), dtype=np.uint8)
    emission[..., :3] = np.clip(
        palette.emission[materials] * (255.0 / EMISSION_SCALE), 0.0, 255.0
    ).astype(np.uint8)
    emission[..., 3] = 255

    # Non-terrain geometry providers (buildings, props) splat last so terrain
    # and tree solids win the max-combine.  Empty → byte-identical output.
    for provider in providers:
        provider.rasterize_occupancy(window.origin_cell, n, window.cell_m, albedo_occ, emission)

    return GeometryVolume(
        albedo_occ=albedo_occ,
        emission=emission,
        origin_cell=window.origin_cell,
        cell_m=window.cell_m,
    )


def window_chunk_span(
    origin_cell: tuple[int, int, int],
    cells: int,
    cell_m: float,
    chunk_size: int,
    voxel_size: float,
) -> list[tuple[int, int, int]]:
    """
    Chunk coordinates whose voxels intersect a window placed at ``origin_cell``.

    Mirrors the lo/hi chunk-range computation inside :func:`assemble_geometry`
    so the async assembly path can snapshot exactly the chunks a reassembly
    will read (a small superset is harmless; missing chunks would leave holes).

    Returns
    -------
    list[tuple[int, int, int]]
        All ``(cx, cy, cz)`` chunk coords overlapping the window box.  Pure /
        deterministic; no chunk lookups (caller filters against loaded chunks).
    """
    k = int(round(cell_m / voxel_size))
    cells_per_chunk = chunk_size // k
    lo = [int(np.floor(o / cells_per_chunk)) for o in origin_cell]
    hi = [int(np.floor((o + cells - 1) / cells_per_chunk)) for o in origin_cell]
    out: list[tuple[int, int, int]] = []
    for ccx in range(lo[0], hi[0] + 1):
        for ccy in range(lo[1], hi[1] + 1):
            for ccz in range(lo[2], hi[2] + 1):
                out.append((ccx, ccy, ccz))
    return out


def pack_volume(arr: np.ndarray) -> bytes:
    """
    Pack a ``uint8 (N, N, N, 4)`` ``[x, y, z]`` block into Panda3D 3-D-texture
    RAM bytes (page-major ``(z, y, x)``, BGRA channel order).

    This is the numpy-heavy half of a volume upload (transpose + channel swap +
    contiguous copy).  Factored out of ``lighting/gpu._upload_volume`` so the
    async assembly worker can run it off the main thread, leaving only the
    cheap ``Texture.set_ram_image(bytes)`` memcpy on the render thread.
    """
    data = np.ascontiguousarray(np.transpose(arr, (2, 1, 0, 3))[..., [2, 1, 0, 3]])
    return data.tobytes()


class ChunkBlockCache:
    """
    Thread-safe LRU cache of per-chunk downsampled geometry mini-blocks.

    Reassembling a coarse cascade re-downsamples every intersecting chunk's
    material array from scratch — at ``light_c2_cell_m`` = 8 m cells the far
    cascade's 512 m window touches tens of thousands of chunk coords, and a
    full recenter re-folds all their ``uint8 (32,32,32)`` arrays each time,
    which on the assembly worker outruns the recenter interval.  This cache
    stores, per ``(chunk coord, cell_m)``, the chunk's
    ``(material_id, solid_count)`` mini-block (the output of
    ``_downsample_chunk_block``) so a recenter that re-reads the same chunk
    copies the block instead of recomputing it.  A 16 m chunk yields a
    ``2×2×2`` block at 8 m cells (``8×8×8`` at 2 m cells), so the entries are
    tiny.

    Thread-safety
    -------------
    The assembly worker thread reads + populates the cache (via
    :func:`assemble_geometry`) while the main thread invalidates edited chunks.
    All access is guarded by a single :class:`threading.Lock`; the blocks are
    small so a coarse lock is fine.  Stored blocks are immutable
    (``WRITEABLE`` cleared); :func:`assemble_geometry` slices but never mutates
    them, so a hit can safely hand out a reference without copying.

    Palette dependency
    ------------------
    The cached mini-blocks are material ids + solid counts — **palette-
    independent** — so a palette change does NOT invalidate them (the palette
    is applied after the cache, on the assembled material array).  The cache
    *is* keyed only by chunk coord + cell size and assumes one terrain
    voxel→chunk geometry per run; terrain edits invalidate per-chunk via
    :meth:`invalidate`.

    Eviction
    --------
    Bounded LRU: at most ``max_entries`` ``(coord, cell_m)`` entries
    (default 4096).  Each entry is a few hundred bytes to a few KB, so the cap
    bounds the cache to single-digit MB.  The least-recently-used entry is
    evicted on overflow.

    Parameters
    ----------
    max_entries : int, default 4096
        Hard cap on stored ``(coord, cell_m)`` mini-blocks.

    Example
    -------
    >>> cache = ChunkBlockCache(max_entries=8192)
    >>> vol = assemble_geometry(win, chunks, palette, 32, 0.5, cache=cache)
    >>> cache.invalidate((cx, cy, cz))   # after a terrain edit in that chunk
    """

    def __init__(self, max_entries: int = 4096) -> None:
        self.max_entries = int(max_entries)
        # key: (coord, cell_m) -> (material_id, solid_count) read-only arrays.
        self._store: OrderedDict[tuple, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._lock = threading.Lock()

    def get(
        self,
        coord: tuple[int, int, int],
        cell_m: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Return the cached ``(material_id, solid_count)`` mini-block for
        ``(coord, cell_m)``, or ``None`` on a miss.  Marks the entry MRU.

        The returned arrays are read-only views into the cache — callers must
        not mutate them (``assemble_geometry`` only slices/copies out of them).
        """
        key = (coord, float(cell_m))
        with self._lock:
            blk = self._store.get(key)
            if blk is not None:
                self._store.move_to_end(key)
            return blk

    def put(
        self,
        coord: tuple[int, int, int],
        cell_m: float,
        block: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """
        Store a chunk's mini-block, evicting the LRU entry past the cap.

        The arrays are frozen read-only (their ``WRITEABLE`` flag is cleared)
        so a later :meth:`get` can hand out references without copying.
        """
        mat, cnt = block
        mat.setflags(write=False)
        cnt.setflags(write=False)
        key = (coord, float(cell_m))
        with self._lock:
            self._store[key] = (mat, cnt)
            self._store.move_to_end(key)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)  # evict LRU

    def invalidate(self, coord: tuple[int, int, int]) -> None:
        """
        Drop every cached mini-block for ``coord`` (all cell sizes).

        Call when a terrain edit changes that chunk's material array so the
        next reassembly recomputes the affected blocks.
        """
        with self._lock:
            for key in [k for k in self._store if k[0] == coord]:
                del self._store[key]

    def clear(self) -> None:
        """Drop all cached blocks (e.g. on world reload)."""
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        """Number of cached ``(coord, cell_m)`` entries."""
        with self._lock:
            return len(self._store)
