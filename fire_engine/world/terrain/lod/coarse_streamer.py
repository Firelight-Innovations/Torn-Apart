"""
terrain/lod/coarse_streamer.py — Async coarse-node streaming over ``TerrainLodPool``.

:class:`CoarseLodStreamer` is the distant-horizon counterpart of
:class:`~fire_engine.world.terrain.lod.streamer.LodStreamer`: each frame it plans
the desired coarse nodes (ranks ``1..lod_max_rank``) for the camera via
:func:`~fire_engine.world.terrain.lod.desired.desired_node_set`, submits a bounded
batch of :class:`~fire_engine.world.terrain.lod.types.LodJob`\\ s (``rank > 0``)
to a :class:`~fire_engine.world.terrain.lod.pool.TerrainLodPool`, and drains the
finished coarse meshes into the ``ChunkManager``'s ``pending_coarse_meshes``
channel for the render upload loop (Hard Rule 12: the main thread orchestrates
only — the gather + downsample + mesh of a coarse node runs off-thread).

Each coarse node is keyed by its full
:class:`~fire_engine.world.terrain.lod.node.LodNode` key ``(rank, nx, ny, nz)``.
The streamer owns the **staleness authority** the same way ``LodStreamer`` does:
``self._node_seq[key]`` records the latest ``seq`` submitted for that node, so an
out-of-order drain keeps only the newest mesh per node.  When the camera moves,
nodes that leave the desired set are recorded in
``ChunkManager.unloaded_coarse_this_frame`` so the render layer detaches them the
same frame the L0 chunks they covered begin streaming (the hard band cut — no
double-draw; pop is acceptable for P2, crossfade is P3).

Submission order is **coarsest-far-first, then refine inward** (highest rank
first, nearest node within a rank first) so the distant silhouette fills before
the mid-field detail — the opposite end of the scale from ``LodStreamer`` which
loads the nearest editable chunks first.

The off-thread coarse path uses a **separate** ``TerrainLodPool`` from the near
``LodStreamer`` so coarse (heavy) jobs never steal near results and the two
drains stay independent; coarse results (``rank > 0``) only ever land in
``pending_coarse_meshes``.

No panda3d import — fully headless-testable (Hard Rule 1).

Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-render
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fire_engine.world.terrain.generation import generate_chunk
from fire_engine.world.terrain.lod.coarse_assembly import assemble_coarse_materials
from fire_engine.world.terrain.lod.desired import desired_node_set
from fire_engine.world.terrain.lod.node import LodNode
from fire_engine.world.terrain.lod.types import LodJob

if TYPE_CHECKING:
    from fire_engine.core.config import Config
    from fire_engine.core.math3d import Vec3
    from fire_engine.world.terrain.chunk_manager import ChunkManager
    from fire_engine.world.terrain.lod.pool import TerrainLodPool

__all__ = ["CoarseLodStreamer"]

# Inclusive Z chunk band relative to the camera chunk (mirrors ChunkManager).
_Z_MIN: int = -2
_Z_MAX: int = 4


class CoarseLodStreamer:
    """
    Async coarse-node streaming driver: submit downsampled-node jobs off-thread.

    Mirrors :class:`~fire_engine.world.terrain.lod.streamer.LodStreamer`'s
    contract for the coarse ranks: plan the desired nodes, drain finished coarse
    meshes into ``chunk_manager.pending_coarse_meshes``, submit a bounded batch
    of fresh jobs, and record nodes that left the desired set in
    ``chunk_manager.unloaded_coarse_this_frame``.  Coarse meshing (gather the
    ``(2**L)³`` chunk block, downsample, mesh) runs on a
    :class:`~fire_engine.world.terrain.lod.pool.TerrainLodPool`.

    Staleness discipline
    --------------------
    ``self._node_seq[key]`` holds the latest ``seq`` submitted for coarse-node
    ``key = (rank, nx, ny, nz)``.  A drained
    :class:`~fire_engine.world.terrain.lod.types.LodResult` (with ``rank > 0``)
    is kept only if its node is still desired **and**
    ``result.seq == self._node_seq[key]``; otherwise it is dropped (the node
    left the desired set or was re-submitted mid-flight).  Newest wins,
    order-independent.

    Parameters
    ----------
    chunk_manager : ChunkManager
        The live chunk store.  Provides ``chunks`` (live materials),
        ``camera_chunk``, ``config``, and the coarse channels
        ``pending_coarse_meshes`` / ``unloaded_coarse_this_frame`` (the manager
        creates them lazily; this streamer ensures they exist).
    pool : TerrainLodPool
        Started worker pool that meshes submitted coarse :class:`LodJob`\\ s.
    config : Config
        Engine config; reads ``lod_max_rank``, ``lod_far_radius_chunks``,
        ``lod_near_radius_chunks``, the band radii, ``lod_downsample_mode``,
        ``lod_coarse_submit_per_frame``, ``chunk_size``, ``voxel_size``,
        ``chunk_meters``, ``mesh_style``, ``facet_shade_strength``.

    Example
    -------
    >>> from fire_engine.core import EventBus, load_config
    >>> from fire_engine.core.rng import set_world_seed
    >>> from fire_engine.core.math3d import Vec3
    >>> from fire_engine.world.terrain.chunk_manager import ChunkManager
    >>> from fire_engine.world.terrain.lod import CoarseLodStreamer, TerrainLodPool
    >>> set_world_seed(1337)
    >>> cfg = load_config()
    >>> cm = ChunkManager(cfg, EventBus())
    >>> pool = TerrainLodPool(cfg.lod_worker_threads); pool.start()
    >>> cs = CoarseLodStreamer(cm, pool, cfg)
    >>> cs.stream_frame(Vec3(0, 0, 20))   # submits coarse jobs; later frames drain them
    >>> pool.stop()

    Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-render
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
        # Latest submitted seq per coarse-node key — the staleness authority and
        # (with pending_coarse_meshes) the set of nodes with a render presence.
        self._node_seq: dict[tuple[int, int, int, int], int] = {}
        self._ensure_channels()

    def _ensure_channels(self) -> None:
        """Lazily create the coarse channels on the chunk manager.

        Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-render
        """
        cm = self._cm
        if not hasattr(cm, "pending_coarse_meshes"):
            cm.pending_coarse_meshes = {}
        if not hasattr(cm, "unloaded_coarse_this_frame"):
            cm.unloaded_coarse_this_frame = []

    def stream_frame(self, camera_pos: Vec3) -> None:
        """
        Stream one coarse frame asynchronously (off-thread node meshing).

        1. Drain finished coarse meshes from the pool into
           ``pending_coarse_meshes`` (newest result per still-desired node wins;
           stale/undesired dropped).
        2. Plan the desired coarse nodes for the camera chunk via
           :func:`desired_node_set`; record nodes that LEFT the desired set in
           ``unloaded_coarse_this_frame`` (the render layer detaches them) and
           prune ``_node_seq`` of them.
        3. Submit up to ``config.lod_coarse_submit_per_frame`` fresh coarse jobs
           — coarsest rank first, nearest node within a rank first — skipping
           nodes already in flight at the latest seq (so a node isn't
           re-submitted every frame until its result returns).

        Parameters
        ----------
        camera_pos : Vec3
            Current camera position (world meters).

        Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-render
        """
        cm = self._cm
        config = self._config
        self._ensure_channels()
        cm.unloaded_coarse_this_frame = []

        max_rank = int(getattr(config, "lod_max_rank", 0))
        self._drain()

        if max_rank <= 0:
            # Coarse disabled: every previously-desired node must unload.
            self._retire(set())
            return

        ccx, ccy, ccz = cm.camera_chunk(camera_pos)
        plan = desired_node_set((ccx, ccy, ccz), config, (_Z_MIN, _Z_MAX), max_rank)
        desired = {key for nodes in plan.coarse_nodes.values() for key in nodes}
        self._retire(desired)

        # Submit coarsest-far-first, nearest-within-rank-first.
        budget = int(getattr(config, "lod_coarse_submit_per_frame", 4))
        if budget <= 0:
            return

        def dist2(key: tuple[int, int, int, int]) -> int:
            node = LodNode(*key)
            ox, oy, oz = node.chunk_origin()
            half = node.factor // 2
            return (ox + half - ccx) ** 2 + (oy + half - ccy) ** 2 + (oz + half - ccz) ** 2

        for rank in range(max_rank, 0, -1):
            if budget <= 0:
                break
            keys = sorted(plan.coarse_nodes.get(rank, set()), key=dist2)
            for key in keys:
                if budget <= 0:
                    break
                # Skip nodes already meshed (in pending) or in flight at latest seq.
                if key in cm.pending_coarse_meshes:
                    continue
                if key in self._node_seq:
                    continue  # an unconsumed in-flight submit owns this node
                self._submit(LodNode(*key))
                budget -= 1

    def _retire(self, desired: set[tuple[int, int, int, int]]) -> None:
        """Record + drop coarse nodes no longer desired (the hard band cut).

        Any node with a render presence — submitted/in-flight (``_node_seq``) or
        a delivered mesh (``pending_coarse_meshes``) — that is NOT in ``desired``
        is appended to ``unloaded_coarse_this_frame`` (render detaches it),
        removed from ``pending_coarse_meshes``, and pruned from ``_node_seq``.
        Because the retire is keyed on render presence (``_node_seq`` /
        ``pending_coarse_meshes``), which both shrink as nodes are retired, a
        re-call with the same ``desired`` reports nothing — each node is retired
        exactly once, not every frame (e.g. when coarse is disabled
        mid-session).  Un-meshed planned nodes never produced a Geom, so there is
        nothing to detach for them.
        Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-render
        """
        cm = self._cm
        tracked = set(self._node_seq) | set(cm.pending_coarse_meshes)
        gone = tracked - desired
        for key in gone:
            cm.unloaded_coarse_this_frame.append(key)
            cm.pending_coarse_meshes.pop(key, None)
            self._node_seq.pop(key, None)

    def _drain(self) -> None:
        """Pop finished coarse results; keep the newest per still-desired node.

        A coarse result (``rank > 0``) is kept only if its node key is still in
        ``_node_seq`` AND its ``seq`` equals the latest submitted for that node;
        otherwise the node was retired or re-submitted mid-flight and the result
        is dropped.  Once a result lands, ``_node_seq`` no longer guards
        re-submission (the mesh is in ``pending_coarse_meshes``), so the entry is
        cleared — a later re-desire re-submits with a fresh seq.
        Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-render
        """
        cm = self._cm
        for result in self._pool.drain_results():
            if result.rank <= 0:
                continue  # not ours (defensive; coarse pool only meshes rank>0)
            key = (result.rank, result.coord[0], result.coord[1], result.coord[2])
            if self._node_seq.get(key) == result.seq:
                cm.pending_coarse_meshes[key] = result.mesh
                # Mesh delivered — stop guarding re-submission for this node.
                self._node_seq.pop(key, None)

    def _submit(self, node: LodNode) -> None:
        """Build + submit a coarse job for ``node`` (records the latest seq).

        Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-render
        """
        job = self._make_job(node)
        self._pool.submit(job)

    def _materials_for(self, coord: tuple[int, int, int]) -> np.ndarray:
        """L0 chunk materials for ``coord`` — live if loaded, else generated.

        Loaded chunks contribute their (possibly brush-edited) live materials;
        unloaded chunks contribute the deterministic ``generate_chunk`` baseline,
        so a coarse node is byte-identical regardless of load order (the same
        determinism the near faceted path relies on).
        Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-render
        """
        cm = self._cm
        chunk = cm.chunks.get(coord)
        if chunk is not None:
            return chunk.materials.copy()
        return generate_chunk(coord, cm.config)

    def _make_job(self, node: LodNode) -> LodJob:
        """
        Assemble + snapshot ``node`` into an immutable coarse :class:`LodJob`.

        Bumps ``self._seq``, records it as the latest for ``node.key`` in
        ``_node_seq``, gathers the node's ``(2**rank)³`` chunk block via
        :func:`~fire_engine.world.terrain.lod.coarse_assembly.assemble_coarse_materials`
        (loaded-or-generated materials), and packages the downsampled coarse
        block as a ``rank > 0`` :class:`LodJob` at the node coord with the scaled
        coarse voxel size.  Coarse jobs carry no neighbours in P2 (open coarse
        borders — hard band cuts; seam-stitching is P3).

        Parameters
        ----------
        node : LodNode
            The coarse node to mesh (rank ≥ 1).

        Returns
        -------
        LodJob
            The immutable hand-off snapshot tagged with the bumped ``seq`` and
            ``rank = node.rank``.

        Docs: docs/systems/world.terrain.lod.md#coarse-ranks-p2-render
        """
        config = self._config
        self._seq += 1
        self._node_seq[node.key] = self._seq

        chunk_size = int(config.chunk_size)
        base_vs = float(config.voxel_size)
        mode = str(getattr(config, "lod_downsample_mode", "any"))
        coarse_materials = assemble_coarse_materials(
            node, self._materials_for, chunk_size=chunk_size, mode=mode
        )
        mesh_style = str(getattr(config, "mesh_style", "faceted"))
        return LodJob(
            coord=(node.nx, node.ny, node.nz),
            materials=coarse_materials,
            neighbors={},
            chunk_size=chunk_size,
            voxel_size=base_vs * node.factor,
            shade_strength=float(getattr(config, "facet_shade_strength", 0.25)),
            mesh_style=mesh_style,
            seq=self._seq,
            rank=node.rank,
        )
