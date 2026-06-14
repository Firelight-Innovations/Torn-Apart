"""
lighting/light_grid.py — Per-chunk light storage and occupancy downsampling.

The light grid is a coarser voxel grid that stores the computed sunlight (and
later point-light) values for the world.  Each light cell covers a
``light_grid_scale × light_grid_scale × light_grid_scale`` cube of terrain
voxels.  With the locked defaults (``light_grid_scale = 2``, ``voxel_size =
0.5 m``) each light cell is **1 m × 1 m × 1 m** and each chunk has a **16 ×
16 × 16** light array.

Light values are stored as ``uint8`` in the range **[0, 255]** where:
- **255** = full sunlight (no obstruction above)
- **40**  = ambient (shadowed — at or below the first occupied cell in the
            column)

These two constants are the *only* light levels in Phase-4 v0 (sunlight only).
Later phases will introduce point-light flood fill producing intermediate values.

No panda3d imports.  This file is fully headless-testable.
"""

from __future__ import annotations

import numpy as np

# Light-value constants (Phase-4 v0 only two levels; blur produces intermediates).
LIGHT_FULL: int = 255  # full sunlight — no occupancy at or above this cell
LIGHT_AMBIENT: int = 40  # ambient — at or below the first solid cell in column


def occupancy_from_materials(materials: np.ndarray) -> np.ndarray:
    """
    Downsample a 32³ uint8 material array to a 16³ bool occupancy grid.

    A light cell (1 m cube) covers a 2×2×2 block of terrain voxels.  The cell
    is occupied (True) if **any** of its 8 voxels is solid (material != 0).
    Vectorised via reshape + max — no Python loops over cells.

    Parameters
    ----------
    materials : numpy.ndarray
        ``uint8`` shape ``(32, 32, 32)`` indexed ``[x, y, z]``, where 0 = air
        and ≥ 1 = solid.  Corresponds to ``Chunk.materials``.

    Returns
    -------
    numpy.ndarray
        ``bool`` shape ``(16, 16, 16)`` indexed ``[cx, cy, cz]``.
        ``True`` where at least one of the 8 constituent terrain voxels is solid.

    Notes
    -----
    Reshape strategy: ``(32, 32, 32)`` → ``(16, 2, 16, 2, 16, 2)`` with axes
    ordered ``(cx, dx, cy, dy, cz, dz)`` where ``d*`` are the 2-voxel sub-axes.
    Taking the max over axes ``(1, 3, 5)`` collapses each 2×2×2 block.

    Example
    -------
    >>> import numpy as np
    >>> mat = np.zeros((32, 32, 32), dtype=np.uint8)
    >>> mat[0, 0, 0] = 1   # one solid voxel in the first light cell
    >>> occ = occupancy_from_materials(mat)
    >>> occ.shape
    (16, 16, 16)
    >>> occ[0, 0, 0]       # first light cell is occupied
    True
    >>> occ[1, 0, 0]       # neighbouring cell is empty
    False
    """
    n = materials.shape[0]
    s = n // 2  # light grid edge = 16 when n = 32
    # Reshape to (s, 2, s, 2, s, 2): axes = (cx, dx, cy, dy, cz, dz)
    # max over the three sub-axes (1, 3, 5) collapses each 2×2×2 block.
    return materials.reshape(s, 2, s, 2, s, 2).max(axis=(1, 3, 5)) > 0


class LightGrid:
    """
    Per-chunk light array store.

    Holds one ``uint8 (16, 16, 16)`` light array per loaded chunk coord,
    computed by :class:`fire_engine.lighting.sunlight.SunlightComputer`.
    Also tracks which chunks have *valid* (freshly computed) light vs.
    *dirty* (needs recompute / remesh).

    Light arrays are indexed ``[cx, cy, cz]`` (light-cell coords local to the
    chunk, 0..15).  Values are in ``[0, 255]``:
    - 255 → full sunlight
    - 40  → ambient (shadowed)
    - intermediate values → penumbra (after box blur)

    Parameters
    ----------
    None — the store starts empty; ``SunlightComputer`` populates it.

    Example
    -------
    >>> import numpy as np
    >>> lg = LightGrid()
    >>> arr = np.full((16, 16, 16), 255, dtype=np.uint8)
    >>> lg.set((0, 0, 0), arr)
    >>> lg.get((0, 0, 0)).shape
    (16, 16, 16)
    >>> lg.has_valid((0, 0, 0))
    True
    """

    def __init__(self) -> None:
        # coord → uint8 (16,16,16) light array
        self._arrays: dict[tuple[int, int, int], np.ndarray] = {}
        # coords whose light has been computed (not yet invalidated)
        self._valid: set[tuple[int, int, int]] = set()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, coord: tuple[int, int, int]) -> np.ndarray | None:
        """
        Return the light array for ``coord``, or ``None`` if not computed.

        Parameters
        ----------
        coord : tuple[int, int, int]
            Chunk coordinate (cx, cy, cz).

        Returns
        -------
        numpy.ndarray or None
            ``uint8 (16, 16, 16)`` light values in ``[0, 255]``, or ``None``
            when this chunk has no computed light (caller should default to
            full bright).
        """
        return self._arrays.get(coord)

    def has_valid(self, coord: tuple[int, int, int]) -> bool:
        """
        Return True when ``coord`` has a current (non-dirty) light array.

        Parameters
        ----------
        coord : tuple[int, int, int]
        """
        return coord in self._valid

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def set(self, coord: tuple[int, int, int], arr: np.ndarray) -> None:
        """
        Store a computed light array for ``coord`` and mark it valid.

        Parameters
        ----------
        coord : tuple[int, int, int]
            Chunk coordinate.
        arr : numpy.ndarray
            ``uint8 (16, 16, 16)`` light values, freshly computed by the
            sunlight pass.  Stored by reference (the array should not be
            mutated by the caller after this call).
        """
        self._arrays[coord] = arr
        self._valid.add(coord)

    def invalidate(self, coord: tuple[int, int, int]) -> None:
        """
        Mark ``coord`` as needing recomputation (light is stale).

        The array is kept in memory so the sampler can still return values
        during the recompute, but ``has_valid`` returns ``False``.

        Parameters
        ----------
        coord : tuple[int, int, int]
        """
        self._valid.discard(coord)

    def remove(self, coord: tuple[int, int, int]) -> None:
        """
        Remove all light data for ``coord`` (called when a chunk is unloaded).

        Parameters
        ----------
        coord : tuple[int, int, int]
        """
        self._arrays.pop(coord, None)
        self._valid.discard(coord)

    def loaded_coords(self) -> list[tuple[int, int, int]]:
        """
        Return all coords that currently have a light array (valid or not).

        Returns
        -------
        list[tuple[int, int, int]]
        """
        return list(self._arrays.keys())
