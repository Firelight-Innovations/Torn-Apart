"""
Thread-safe LRU cache of per-chunk downsampled geometry mini-blocks.

Extracted from ``fire_engine.lighting.volume`` to keep that module under the
500-line limit.  Re-exported from ``volume.py`` so all historical import paths
(``from fire_engine.lighting.volume import ChunkBlockCache``, etc.) remain valid.

Docs: docs/systems/lighting.md
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

import numpy as np

__all__ = ["ChunkBlockCache"]


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

    Docs: docs/systems/lighting._impl.md
    """

    def __init__(self, max_entries: int = 4096) -> None:
        self.max_entries = int(max_entries)
        # key: (coord, cell_m) -> (material_id, solid_count) read-only arrays.
        self._store: OrderedDict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = OrderedDict()
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

        Docs: docs/systems/lighting._impl.md
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

        Docs: docs/systems/lighting._impl.md
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

        Docs: docs/systems/lighting._impl.md
        """
        with self._lock:
            for key in [k for k in self._store if k[0] == coord]:
                del self._store[key]

    def clear(self) -> None:
        """Drop all cached blocks (e.g. on world reload).

        Docs: docs/systems/lighting._impl.md
        """
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        """Number of cached ``(coord, cell_m)`` entries."""
        with self._lock:
            return len(self._store)
