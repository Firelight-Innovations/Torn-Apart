"""
terrain/lod/streamer.py — Async chunk-streaming driver over ``TerrainLodPool``.

:class:`LodStreamer` is the off-main-thread counterpart of
``ChunkManager.stream_frame`` (Hard Rule 12: the main thread orchestrates only).
Each frame it drains finished meshes from a :class:`TerrainLodPool` into the
``ChunkManager``'s existing ``pending_meshes`` dict (so the render upload loop is
nearly unchanged), then submits a bounded batch of fresh mesh jobs — dirty loaded
chunks first, then the nearest missing desired chunks — each carrying an immutable
snapshot of the chunk + its neighbours.

It owns the **staleness authority**: ``self._node_seq`` records the latest ``seq``
submitted per coord, so when results return out of order (the pool drains in
completion order, not submit order) only the newest result for a coord is kept and
in-flight stale results are discarded.  This means a brush re-dirty during flight
re-submits with a fresh ``seq`` and wins.

No panda3d import — fully headless-testable.  The synchronous
``ChunkManager.stream_frame`` path remains intact for the baked-light / editor case.

Docs: docs/systems/world.terrain.lod.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fire_engine.core import ChunkLoadedEvent
from fire_engine.world.terrain.lod.pool import TerrainLodPool
from fire_engine.world.terrain.lod.types import LodJob

if TYPE_CHECKING:
    from fire_engine.core.config import Config
    from fire_engine.core.math3d import Vec3
    from fire_engine.world.terrain.chunk_manager import ChunkManager

__all__ = ["LodStreamer"]


class LodStreamer:
    """
    Async streaming driver: submit mesh jobs off-thread, drain into ``pending_meshes``.

    Mirrors ``ChunkManager.stream_frame``'s contract (load/mesh near the camera,
    unload far, publish ``ChunkLoadedEvent`` / ``ChunkUnloadedEvent``), but the
    meshing runs on a :class:`TerrainLodPool` worker pool instead of the main
    thread.  Produced meshes land in ``chunk_manager.pending_meshes`` exactly as
    the synchronous path leaves them, so the render upload loop is unchanged.

    Staleness discipline
    --------------------
    ``self._node_seq[coord]`` holds the latest ``seq`` submitted for ``coord``.
    A drained :class:`~fire_engine.world.terrain.lod.types.LodResult` is kept only
    if its coord is still loaded **and** ``result.seq == self._node_seq[coord]``;
    otherwise it is discarded (the chunk was unloaded or re-submitted with a newer
    snapshot while the result was in flight).  Newest wins, order-independent.

    Parameters
    ----------
    chunk_manager : ChunkManager
        The live chunk store (provides ``chunks``, ``pending_meshes``,
        ``desired_set``, ``camera_chunk``, ``get_or_create``,
        ``_neighbor_materials`` / ``_neighbor_solids``, ``_unload_far``, ``bus``).
    pool : TerrainLodPool
        Started worker pool that meshes submitted :class:`LodJob`\\ s off-thread.
    config : Config
        Engine config; reads ``lod_submit_per_frame``, ``mesh_style``,
        ``chunk_size``, ``voxel_size``, ``facet_shade_strength``.

    Example
    -------
    >>> from fire_engine.core import EventBus, load_config
    >>> from fire_engine.core.rng import set_world_seed
    >>> from fire_engine.world.terrain.chunk_manager import ChunkManager
    >>> from fire_engine.world.terrain.lod import LodStreamer, TerrainLodPool
    >>> from fire_engine.core.math3d import Vec3
    >>> set_world_seed(1337)
    >>> cfg = load_config()
    >>> cm = ChunkManager(cfg, EventBus())
    >>> pool = TerrainLodPool(cfg.lod_worker_threads); pool.start()
    >>> streamer = LodStreamer(cm, pool, cfg)
    >>> streamer.stream_frame(Vec3(0, 0, 20))   # submits jobs; later frames drain them
    >>> pool.stop()

    Docs: docs/systems/world.terrain.lod.md
    """

    def __init__(
        self,
        chunk_manager: ChunkManager,
        pool: TerrainLodPool,
        config: Config,
    ) -> None:
        self._cm = chunk_manager
        self._pool = pool
        self._config = config
        self._seq = 0
        # Latest submitted seq per coord — the staleness authority.
        self._node_seq: dict[tuple[int, int, int], int] = {}

    def stream_frame(self, camera_pos: Vec3) -> None:
        """
        Stream one frame asynchronously (the off-thread analogue of
        ``ChunkManager.stream_frame``).

        1. Drain finished meshes from the pool into ``pending_meshes`` (newest
           result per still-loaded coord wins; stale/unloaded discarded).
        2. Reset ``unloaded_this_frame``; compute the desired set + camera chunk.
        3. Submit up to ``config.lod_submit_per_frame`` jobs — dirty loaded
           chunks first (nearest-first), then the nearest missing desired chunks
           (publishing ``ChunkLoadedEvent`` for each newly created chunk) —
           clearing each submitted chunk's ``dirty`` flag so it isn't resubmitted
           next frame (a later brush re-dirties it → fresh ``seq`` → newest wins).
        4. Unload far chunks via ``ChunkManager._unload_far`` and prune
           ``_node_seq`` of any coord no longer loaded.

        Parameters
        ----------
        camera_pos : Vec3
            Current camera position (world meters).

        Docs: docs/systems/world.terrain.lod.md
        """
        cm = self._cm
        self._drain()

        cm.unloaded_this_frame = []
        desired = cm.desired_set(camera_pos)
        ccx, ccy, ccz = cm.camera_chunk(camera_pos)

        def dist2(coord: tuple[int, int, int]) -> int:
            return (coord[0] - ccx) ** 2 + (coord[1] - ccy) ** 2 + (coord[2] - ccz) ** 2

        budget = int(self._config.lod_submit_per_frame)

        # 1. Dirty loaded chunks first (nearest-first) — brush edits / relights.
        dirty = [c for c, ch in cm.chunks.items() if ch.dirty]
        dirty.sort(key=dist2)
        for coord in dirty:
            if budget <= 0:
                break
            self._submit(coord)
            budget -= 1

        # 2. Nearest missing desired chunks with the remaining budget.
        if budget > 0:
            missing = [c for c in desired if c not in cm.chunks]
            missing.sort(key=dist2)
            for coord in missing:
                if budget <= 0:
                    break
                cm.get_or_create(coord)
                cm.bus.publish(ChunkLoadedEvent(coord=coord))
                self._submit(coord)
                budget -= 1

        # 3. Unload far + prune the staleness map of unloaded coords.
        cm._unload_far((ccx, ccy, ccz))
        self._node_seq = {c: s for c, s in self._node_seq.items() if c in cm.chunks}

    def _drain(self) -> None:
        """Pop finished results; keep the newest per still-loaded coord.

        A result is kept only if its coord is still in ``chunks`` AND its ``seq``
        equals the latest submitted for that coord (``_node_seq``); otherwise the
        chunk was unloaded or re-submitted mid-flight and the result is dropped.
        Kept meshes go to ``pending_meshes`` (render uploads them).
        Docs: docs/systems/world.terrain.lod.md
        """
        cm = self._cm
        for result in self._pool.drain_results():
            coord = result.coord
            if coord in cm.chunks and result.seq == self._node_seq.get(coord):
                cm.pending_meshes[coord] = result.mesh

    def _submit(self, coord: tuple[int, int, int]) -> None:
        """Build + submit a job for ``coord`` and clear the chunk's ``dirty`` flag.

        Docs: docs/systems/world.terrain.lod.md
        """
        job = self._make_job(coord)
        self._pool.submit(job)
        self._cm.chunks[coord].dirty = False

    def _make_job(self, coord: tuple[int, int, int]) -> LodJob:
        """
        Snapshot ``coord`` (materials + neighbours) into an immutable :class:`LodJob`.

        Bumps ``self._seq``, records it as the latest for ``coord`` in
        ``_node_seq`` (the staleness authority), copies the chunk's ``materials``,
        and copies every non-``str`` neighbour array from
        ``ChunkManager._neighbor_materials`` (faceted, default) or
        ``_neighbor_solids`` (``mesh_style == "blocky"``) so the worker reads an
        immutable snapshot with no cross-thread race (Hard Rule 12).

        Parameters
        ----------
        coord : tuple[int, int, int]
            The chunk coordinate to mesh.  Must already be in ``chunks``.

        Returns
        -------
        LodJob
            The immutable hand-off snapshot, tagged with the bumped ``seq``.

        Docs: docs/systems/world.terrain.lod.md
        """
        cm = self._cm
        config = self._config
        self._seq += 1
        self._node_seq[coord] = self._seq

        materials = cm.chunks[coord].materials.copy()
        mesh_style = str(getattr(config, "mesh_style", "faceted"))
        raw: dict[Any, Any]
        if mesh_style == "blocky":
            raw = cm._neighbor_solids(coord)
        else:
            raw = cm._neighbor_materials(coord)
        neighbors = {k: (v if isinstance(v, str) else v.copy()) for k, v in raw.items()}

        return LodJob(
            coord=coord,
            materials=materials,
            neighbors=neighbors,
            chunk_size=int(config.chunk_size),
            voxel_size=float(config.voxel_size),
            shade_strength=float(getattr(config, "facet_shade_strength", 0.25)),
            mesh_style=mesh_style,
            seq=self._seq,
        )
