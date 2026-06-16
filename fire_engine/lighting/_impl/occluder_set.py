"""
Registry of dynamic shadow-caster AABBs for the GPU lighting pipeline.

Extracted from ``fire_engine.lighting.lights`` to keep that module under the
one-public-class structural limit.  Re-exported from ``lights.py`` so all
historical import paths remain valid.

Docs: docs/systems/lighting.md
"""

from __future__ import annotations

import numpy as np

__all__ = ["MAX_OCCLUDERS", "OccluderSet"]

#: Maximum dynamic occluder boxes uploaded to the GPU (mirrors the fixed
#: uniform array length in ``lighting/glsl.py``).
MAX_OCCLUDERS: int = 16


class OccluderSet:
    """
    Registry of dynamic shadow-caster AABBs for the GPU lighting pipeline.

    Anything that should cast voxel-style shadows but is NOT part of the
    terrain voxel field (dev cubes, props, NPCs) sets its world bounding box
    here once per frame.  ``set_boxes`` only bumps ``version`` when the
    packed data actually changed (beyond ~1 cm), so static objects cost
    nothing and the pipeline re-injects only when something moved.

    Limits: the GPU uniform arrays hold :data:`MAX_OCCLUDERS` (16) boxes;
    extra boxes are dropped (warn-once) — dev tooling scale, not gameplay.

    Example
    -------
    >>> occ = OccluderSet()
    >>> occ.set_boxes([((0.0, 0.0, 8.0), (1.0, 1.0, 9.0))])
    True
    >>> mins, maxs, count = occ.pack()
    >>> count
    1

    Docs: docs/systems/lighting._impl.md
    """

    def __init__(self) -> None:
        self.version: int = 0
        self._mins = np.zeros((MAX_OCCLUDERS, 3), dtype=np.float32)
        self._maxs = np.zeros((MAX_OCCLUDERS, 3), dtype=np.float32)
        self._count: int = 0
        self._warned: bool = False

    @property
    def count(self) -> int:
        """Number of active occluder boxes.

        Docs: docs/systems/lighting._impl.md
        """
        return self._count

    def set_boxes(
        self,
        boxes: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
    ) -> bool:
        """
        Replace the full occluder list with ``boxes`` (world-space AABBs).

        Parameters
        ----------
        boxes : list of ((min_x, min_y, min_z), (max_x, max_y, max_z))
            World meters.  Call every frame with the current boxes; change
            detection is internal.

        Returns
        -------
        bool — True when the set changed (``version`` was bumped).

        Docs: docs/systems/lighting._impl.md
        """
        if len(boxes) > MAX_OCCLUDERS and not self._warned:
            self._warned = True
            import logging

            logging.getLogger("fire_engine.lighting.lights").warning(
                "OccluderSet: %d boxes > max %d — extras dropped", len(boxes), MAX_OCCLUDERS
            )
        boxes = boxes[:MAX_OCCLUDERS]
        n = len(boxes)
        mins = np.zeros((MAX_OCCLUDERS, 3), dtype=np.float32)
        maxs = np.zeros((MAX_OCCLUDERS, 3), dtype=np.float32)
        if n:
            mins[:n] = np.asarray([b[0] for b in boxes], dtype=np.float32)
            maxs[:n] = np.asarray([b[1] for b in boxes], dtype=np.float32)
        if (
            n == self._count
            and np.allclose(mins, self._mins, atol=0.01)
            and np.allclose(maxs, self._maxs, atol=0.01)
        ):
            return False
        self._mins, self._maxs, self._count = mins, maxs, n
        self.version += 1
        return True

    def pack(self) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Packed GPU arrays: ``(mins (16, 3) float32, maxs (16, 3), count)``.

        Rows past ``count`` are zero.  Layout mirrors ``u_box_min`` /
        ``u_box_max`` / ``u_num_boxes`` in ``lighting/glsl.py``.

        Docs: docs/systems/lighting._impl.md
        """
        return self._mins, self._maxs, self._count
