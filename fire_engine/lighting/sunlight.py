"""
lighting/sunlight.py — CPU sunlight column pass + box-blur diffusion.

Phase-4 v0: CPU-only, numpy-vectorised.  No panda3d import needed here — the
light values are baked into vertex colours via the mesher hook
(``build_mesh(..., light_sampler=...)``).  A future phase will upload the grid
as a 3-D texture for fragment-shader sampling; the data layout is already
GPU-uploadable as-is.

Algorithm overview
------------------
1. **Occupancy** — for each chunk, call
   :func:`fire_engine.lighting.light_grid.occupancy_from_materials` to produce a
   ``bool (16, 16, 16)`` grid: True = at least one terrain voxel is solid.

2. **Column pass** (per (cx, cy) column, across all loaded Z chunks) —
   Stack the per-chunk occupancy arrays vertically to produce a tall boolean
   column of shape ``(16, 16, total_z_cells)`` (x, y, z-globally).  Starting
   from the **top** (highest Z), do a cumulative-OR sweep downward: once a cell
   (or any cell above it) is occupied the column from that point down is
   "shadowed."  Light = **LIGHT_FULL (255)** for unshadowed cells,
   **LIGHT_AMBIENT (40)** for shadowed cells.

   Implementation: ``np.maximum.accumulate`` on the boolean column from the
   top downward (reversal, accumulate, re-reverse) gives a 1/0 "has-solid-
   at-or-above" flag per cell without any Python loop over cells.

3. **Box-blur** (3×3×3 uniform smoothing) — applied to the whole column stack
   before splitting back into per-chunk arrays.  Implemented as summed slices
   of a padded array (27 neighbours, each accessed by slicing the padded array
   with offsets 0..2 along each axis, summed and divided by 27) — no
   ``scipy``, no per-cell loop.  The float intermediate is then re-mapped so
   that blurred values stay in ``[LIGHT_AMBIENT, LIGHT_FULL]``, preserving the
   semantic floor (ambient) and ceiling (full sun).

4. **Event subscriptions** — ``TerrainEditedEvent`` and ``ChunkLoadedEvent``
   are subscribed on the given ``EventBus``.  When terrain is edited, the
   affected (cx, cy) columns are recomputed and the touched chunks are marked
   ``dirty = True`` on the ``ChunkManager`` so the next ``stream_frame`` remeshes
   them with fresh light.

No per-cell / per-column Python loops. Every operation is a numpy array
expression over the full column stack.

Constants
---------
LIGHT_FULL    = 255  (from light_grid)
LIGHT_AMBIENT = 40   (from light_grid)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from fire_engine.core import (
    ChunkLoadedEvent,
    Config,
    EventBus,
    TerrainEditedEvent,
    get_logger,
)
from fire_engine.lighting.light_grid import (
    LIGHT_AMBIENT,
    LIGHT_FULL,
    LightGrid,
    occupancy_from_materials,
)

_log = get_logger("lighting.sunlight")


# ---------------------------------------------------------------------------
# Chunk-provider protocol (same contract as terrain's chunk_provider).
# ---------------------------------------------------------------------------


class _ChunkProvider(Protocol):
    """Minimal protocol for a chunk container / provider."""

    @property
    def chunks(self) -> dict[tuple[int, int, int], Any]: ...  # coord → Chunk


# ---------------------------------------------------------------------------
# Public sampler factory (plugs into build_mesh's light_sampler argument)
# ---------------------------------------------------------------------------


def make_light_sampler(
    light_grid: LightGrid,
    config: Config,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a ``light_sampler`` callable compatible with the mesher's contract.

    The returned callable maps face-centre world positions to per-face light
    values in ``[0.0, 1.0]``.  It is vectorised: no per-face Python loop.

    Mesher contract (docs/systems/terrain.md §build_mesh signature)
    ---------------------------------------------------------------
    - **Input**: ``float32 (F, 3)`` — face-centre world positions in **meters**,
      one row per exposed face (``F = mesh.face_count``).
    - **Output**: ``float32 (F,)`` — per-face light in **[0.0, 1.0]**
      (0 = black, 1 = full sun).

    Mapping from world position to light value
    ------------------------------------------
    1. Determine the **chunk coord** for each face centre:
       ``chunk_coord = floor(world_pos / chunk_meters)`` per axis.
    2. Determine the **light-cell coord** within that chunk:
       ``cell = floor((world_pos - chunk_origin) / light_cell_meters)``
       clamped to ``[0, 15]``.
    3. Look up ``light_grid.get(chunk_coord)[cell_x, cell_y, cell_z]``.
    4. Divide by 255 to convert ``uint8 → float [0.0, 1.0]``.

    Fallback behaviour
    ------------------
    - If a face's chunk has **no computed light array** (newly loaded, outside
      range, or the chunk was evicted), the face defaults to **full bright
      (1.0)**.  This prevents black flashes on freshly streamed chunks and
      matches the mesher's own default (``light_sampler=None`` → full bright).
    - Face centres that map outside ``[0, 15]`` after clamping (numerical
      precision edge case at chunk boundaries) are clamped without error.

    Parameters
    ----------
    light_grid : LightGrid
        The populated light store.  Must remain alive as long as the sampler
        is in use (the sampler holds a reference to it).
    config : Config
        Engine config for ``chunk_meters`` (16.0 m) and
        ``light_cell_meters`` (1.0 m).

    Returns
    -------
    Callable[[numpy.ndarray], numpy.ndarray]
        ``(float32 (F, 3)) → float32 (F,)``

    Example
    -------
    >>> from fire_engine.core import load_config
    >>> from fire_engine.lighting import LightGrid, make_light_sampler
    >>> lg = LightGrid()
    >>> sampler = make_light_sampler(lg, load_config())
    >>> import numpy as np
    >>> positions = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    >>> sampler(positions)   # no light data → full bright fallback
    array([1.], dtype=float32)
    """
    chunk_m = float(config.chunk_meters)  # 16.0
    cell_m = float(config.light_cell_meters)  # 1.0
    grid_cells = config.chunk_size // config.light_grid_scale  # 16

    def _sampler(face_centers: np.ndarray) -> np.ndarray:
        """
        Sample the light grid at the given face-centre world positions.

        Parameters
        ----------
        face_centers : numpy.ndarray
            ``float32 (F, 3)`` — world positions in meters.

        Returns
        -------
        numpy.ndarray
            ``float32 (F,)`` in ``[0.0, 1.0]``.
        """
        pos = np.asarray(face_centers, dtype=np.float64)  # (F, 3)
        F = pos.shape[0]
        if F == 0:
            return np.empty(0, dtype=np.float32)

        # 1. Chunk coordinates for each face centre.
        chunk_coords_f = np.floor(pos / chunk_m).astype(np.int32)  # (F, 3)

        # 2. World origin of each face's chunk.
        chunk_origins = chunk_coords_f.astype(np.float64) * chunk_m  # (F, 3)

        # 3. Light-cell coords within the chunk, clamped to [0, grid_cells-1].
        cell_coords = np.floor((pos - chunk_origins) / cell_m).astype(np.int32)  # (F, 3)
        cell_coords = np.clip(cell_coords, 0, grid_cells - 1)  # (F, 3)

        # 4. Gather light values — process per unique chunk to amortise dict
        #    lookups (most faces come from the same chunk).
        out = np.ones(F, dtype=np.float32)  # default: full bright

        # Build a unique-chunk index to batch lookups.
        # Encode (cx,cy,cz) as a single int64 key for fast unique.
        # Range of coords: assume ±2048 is sufficient (± 32 768 m = 2 048 chunks).
        cx = chunk_coords_f[:, 0].astype(np.int64)
        cy = chunk_coords_f[:, 1].astype(np.int64)
        cz = chunk_coords_f[:, 2].astype(np.int64)
        # Pack: shift each component to a distinct bit range.
        # 13 bits each → supports ±4096 chunk coords (±65 536 m).
        KEY_SHIFT = 13
        MASK = (1 << KEY_SHIFT) - 1
        packed_keys = (
            ((cx & MASK) << (2 * KEY_SHIFT)) | ((cy & MASK) << KEY_SHIFT) | (cz & MASK)
        )  # (F,)

        unique_keys, inverse = np.unique(packed_keys, return_inverse=True)

        for ui, key in enumerate(unique_keys):
            # Decode chunk coord from packed key (sign-extend via << >>).
            _cz_raw = int(key & MASK)
            _cy_raw = int((key >> KEY_SHIFT) & MASK)
            _cx_raw = int((key >> (2 * KEY_SHIFT)) & MASK)

            # Sign-extend 13-bit values.
            def _sign_ext(v: int) -> int:
                if v >= (1 << (KEY_SHIFT - 1)):
                    return v - (1 << KEY_SHIFT)
                return v

            coord = (_sign_ext(_cx_raw), _sign_ext(_cy_raw), _sign_ext(_cz_raw))

            arr = light_grid.get(coord)
            if arr is None:
                # No light data → leave as full bright (1.0) for this chunk.
                continue

            # Face indices belonging to this chunk.
            mask = inverse == ui  # (F,) bool
            lx = cell_coords[mask, 0]
            ly = cell_coords[mask, 1]
            lz = cell_coords[mask, 2]
            # Fancy index: arr[lx, ly, lz] → uint8 (K,)
            out[mask] = arr[lx, ly, lz].astype(np.float32) / 255.0

        return out

    return _sampler


# ---------------------------------------------------------------------------
# SunlightComputer
# ---------------------------------------------------------------------------


class SunlightComputer:
    """
    CPU sunlight column-pass + box-blur for all loaded chunks.

    Constructs with references to the engine's config, chunk manager (or any
    object with a ``.chunks`` dict mapping coord → Chunk), light grid store,
    and event bus.  Subscribes to :class:`TerrainEditedEvent` and
    :class:`ChunkLoadedEvent` to keep light current without manual calls.

    Light constants
    ---------------
    - ``LIGHT_FULL   = 255`` — no solid voxel at or above this cell in its column.
    - ``LIGHT_AMBIENT = 40`` — at least one solid voxel exists above this cell.

    Column pass
    -----------
    For each unique (cx, cy) column among the loaded chunks, collect all
    loaded chunks in that column sorted by cz (ascending).  Stack their 16×16
    occupancy layers into a tall ``(16, 16, T)`` boolean column (where T is the
    total number of light layers across all chunks in the column, i.e.
    ``num_chunks * 16``).

    Sunlight comes from +Z (top).  Sweep from the **highest** Z downward using
    a cumulative-OR: once any layer in the column is occupied, all layers from
    there down are shadowed.

    ``np.maximum.accumulate`` on the reversed column (top→bottom becomes index
    0→T-1, flip back after) provides the cumulative-OR in O(T) numpy time, no
    Python loop over layers.

    Box blur
    --------
    After the column pass, apply a 3×3×3 uniform box filter for penumbra.
    Implementation: build a padded array ``(18, 18, T+2)`` (pad 1 on each side),
    sum 27 slices with offsets ``(i, j, k)`` for i,j,k in {0,1,2}, divide by 27.
    The float result is re-mapped to ``uint8`` in ``[LIGHT_AMBIENT, LIGHT_FULL]``
    using linear interpolation between the ambient floor and the full ceiling:
    ``out = LIGHT_AMBIENT + round(fraction * (LIGHT_FULL - LIGHT_AMBIENT))``.
    This preserves the semantic floor (a fully-shadowed region never goes darker
    than ambient) and ceiling (a fully-lit region stays at full sun).

    Event subscriptions
    -------------------
    - **TerrainEditedEvent** → for each affected chunk coord, extract the
      ``(cx, cy)`` column and call ``recompute_column(cx, cy)``.  All chunks
      in that column are marked ``dirty = True`` on the chunk manager so the
      next ``stream_frame`` remeshes them with new light.
    - **ChunkLoadedEvent** → call ``recompute_column(cx, cy)`` for the newly
      loaded chunk's column so its light is ready before the first mesh.

    Parameters
    ----------
    config : Config
        Engine config.
    chunk_provider : object
        Any object with a ``.chunks`` attribute (a dict mapping
        ``tuple[int,int,int] → Chunk``).  ``ChunkManager`` satisfies this.
    light_grid : LightGrid
        The light store to populate.
    bus : EventBus
        Event bus to subscribe on.

    Example
    -------
    >>> from fire_engine.core import load_config, EventBus
    >>> from fire_engine.core.rng import set_world_seed
    >>> from fire_engine.world.terrain import ChunkManager
    >>> from fire_engine.lighting import LightGrid, SunlightComputer
    >>> set_world_seed(1337)
    >>> cfg = load_config()
    >>> bus = EventBus()
    >>> cm = ChunkManager(cfg, bus)
    >>> lg = LightGrid()
    >>> sc = SunlightComputer(cfg, cm, lg, bus)
    >>> sc.recompute_all_loaded()
    """

    def __init__(
        self,
        config: Config,
        chunk_provider: Any,
        light_grid: LightGrid,
        bus: EventBus,
    ) -> None:
        self._config = config
        self._provider = chunk_provider
        self._grid = light_grid
        self._bus = bus

        # Derived constants.
        self._grid_cells: int = config.chunk_size // config.light_grid_scale  # 16
        self._chunk_m: float = float(config.chunk_meters)  # 16.0

        # Subscribe to events.
        bus.subscribe(TerrainEditedEvent, self._on_terrain_edited)
        bus.subscribe(ChunkLoadedEvent, self._on_chunk_loaded)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_terrain_edited(self, event: TerrainEditedEvent) -> None:
        """
        React to a brush edit: recompute light for every affected column.

        ``TerrainEditedEvent.chunk_coords`` may be a single coord tuple or a
        frozenset of tuples (apply_brush emits one event per touched chunk, each
        with a single coord as a tuple).

        Marks all chunks in each affected column ``dirty = True`` so
        ``ChunkManager.stream_frame`` remeshes them.
        """
        coords: Any = event.chunk_coords
        # chunk_coords may be a single tuple or a frozenset / set of tuples.
        if isinstance(coords, tuple) and len(coords) == 3 and isinstance(coords[0], int):
            columns = {(coords[0], coords[1])}
        else:
            columns = {(c[0], c[1]) for c in coords}

        for cx, cy in columns:
            self._recompute_column_and_mark_dirty(cx, cy)

    def _on_chunk_loaded(self, event: ChunkLoadedEvent) -> None:
        """
        React to a newly loaded chunk: compute its column's light.

        Ensures new chunks have light data before the first mesh build (the
        ChunkManager's stream_frame publishes ChunkLoadedEvent *after* meshing,
        but subsequent dirty-remesh passes will use the freshly computed light).
        """
        cx, cy, _ = event.coord
        self._recompute_column_and_mark_dirty(cx, cy)

    # ------------------------------------------------------------------
    # Public recompute API
    # ------------------------------------------------------------------

    def recompute_column(self, cx: int, cy: int) -> None:
        """
        Recompute sunlight for all loaded chunks in the (cx, cy) column.

        Updates ``self._grid`` with new ``uint8 (16, 16, 16)`` arrays.  Does
        NOT mark chunks dirty — use ``_recompute_column_and_mark_dirty`` when
        the result needs to trigger remeshing.

        Parameters
        ----------
        cx : int
            X chunk coordinate of the column.
        cy : int
            Y chunk coordinate of the column.
        """
        chunks_dict = self._provider.chunks
        g = self._grid_cells  # 16

        # Collect all loaded chunks in this column, sorted by cz ascending.
        column_chunks = sorted(
            [
                (coord[2], coord, chunk)
                for coord, chunk in chunks_dict.items()
                if coord[0] == cx and coord[1] == cy
            ],
            key=lambda t: t[0],
        )

        if not column_chunks:
            return

        # Build per-chunk occupancy.
        chunk_occ = []
        for _, _coord, chunk in column_chunks:
            occ = occupancy_from_materials(chunk.materials)  # (16,16,16) bool
            chunk_occ.append(occ)

        num_chunks = len(chunk_occ)
        T = num_chunks * g  # total light layers in column

        # Stack occupancy: (16, 16, T) with z increasing from bottom to top.
        # chunk_occ[0] is the lowest chunk (smallest cz).
        # occupancy array axes: [x, y, z] with z=0 at chunk bottom, z=15 at top.
        # Stack along axis 2: shape (16, 16, T).
        occ_stack = np.concatenate(chunk_occ, axis=2).astype(np.uint8)  # (16,16,T)

        # ------------------------------------
        # Column pass: cumulative-OR downward.
        # ------------------------------------
        # Sunlight comes from +Z (top).  Shadowed = has_solid_at_or_above.
        # Reverse Z so index 0 = top (highest Z) for accumulate, then flip back.
        occ_rev = occ_stack[:, :, ::-1]  # (16,16,T) with index 0 = topmost layer
        # cumulative-OR: once True, stays True going down.
        shadow_rev = np.maximum.accumulate(occ_rev, axis=2)  # (16,16,T) bool/uint8
        shadow = shadow_rev[:, :, ::-1]  # (16,16,T) index 0 = bottommost layer

        # Map: shadowed → LIGHT_AMBIENT, unshadowed → LIGHT_FULL.
        light_float = np.where(shadow, float(LIGHT_AMBIENT), float(LIGHT_FULL))

        # ------------------------------------
        # Box blur: 3×3×3 uniform filter.
        # ------------------------------------
        # Build padded array: (18, 18, T+2) with edge replicate (clamp).
        padded = np.pad(
            light_float,
            pad_width=1,
            mode="edge",
        )  # (18, 18, T+2)

        # Sum 27 slices with offsets i,j,k ∈ {0,1,2} — no scipy, no per-cell loop.
        blurred = np.zeros_like(light_float)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    blurred += padded[i : i + 16, j : j + 16, k : k + T]
        blurred /= 27.0
        # Note: the 27-iteration loop is over the *constant* 3×3×3 neighbourhood
        # (27 iterations total, not per-cell) — this is O(27 * N) where N is the
        # array size, equivalent to a scipy uniform_filter.  Hard Rule 4 bans
        # loops "over 32³ elements"; this loop iterates 27 times regardless of
        # scene size.

        # Re-map blurred float to uint8 in [LIGHT_AMBIENT, LIGHT_FULL].
        # Clamp first to handle numerical edge cases, then quantise.
        blurred_clamped = np.clip(blurred, float(LIGHT_AMBIENT), float(LIGHT_FULL))
        light_uint8 = blurred_clamped.round().astype(np.uint8)  # (16, 16, T)

        # ------------------------------------
        # Split back into per-chunk arrays.
        # ------------------------------------
        for chunk_idx, (_cz, coord, _) in enumerate(column_chunks):
            z_start = chunk_idx * g
            z_end = z_start + g
            arr = np.ascontiguousarray(light_uint8[:, :, z_start:z_end])  # (16,16,16)
            self._grid.set(coord, arr)

    def _recompute_column_and_mark_dirty(self, cx: int, cy: int) -> None:
        """
        Recompute a column and mark all its chunks dirty for remeshing.

        Called by event handlers so that edited/loaded columns trigger a remesh
        with the new light values on the next ``stream_frame``.

        Parameters
        ----------
        cx : int
        cy : int
        """
        self.recompute_column(cx, cy)
        chunks_dict = self._provider.chunks
        for coord, chunk in chunks_dict.items():
            if coord[0] == cx and coord[1] == cy:
                chunk.dirty = True

    def recompute_all_loaded(self) -> None:
        """
        Recompute sunlight for every currently loaded (cx, cy) column.

        Call once after initial chunk loading (before the first frame) to seed
        the light grid.  Subsequent updates are driven by events.

        The orchestrator (``world/app.py``) should call this once per boot after
        the initial set of chunks has been streamed, or after loading a save.
        It is also safe to call every N frames for full correctness at the cost
        of CPU time (not required for v0).
        """
        chunks_dict = self._provider.chunks
        columns: set[tuple[int, int]] = {(coord[0], coord[1]) for coord in chunks_dict}
        for cx, cy in columns:
            self.recompute_column(cx, cy)
