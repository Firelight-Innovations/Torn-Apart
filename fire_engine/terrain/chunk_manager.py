"""
terrain/chunk_manager.py — Chunk streaming, provider, and Saveable.

The ``ChunkManager`` owns the dictionary of loaded chunks and drives streaming
by camera proximity.  It is the ``chunk_provider`` for brush edits and
raycasts, and it implements the ``Saveable`` protocol (``save_key = "terrain"``)
so brush edits survive a save/load round-trip as deltas.

Responsibilities
----------------
- ``desired_set(camera_pos)`` — PURE FUNCTION: which chunk coords should be
  loaded for a camera position (XY radius + a fixed Z band).  Independently
  testable, no side effects.
- ``stream_frame(camera_pos)`` — load/generate/mesh **at most 2 chunks per
  frame** (nearest-first), unload chunks beyond ``radius + 1`` (hysteresis so
  chunks don't thrash at the boundary).  Publishes ``ChunkLoadedEvent`` /
  ``ChunkUnloadedEvent``.  Stores produced ``MeshArrays`` in ``pending_meshes``
  for the World layer to upload via ``world/geometry_bridge.py``.
- ``get_or_create(coord)`` — the provider used by brush/raycast.
- ``Saveable`` — ``get_delta()`` returns ``{coord: materials}`` for edited
  chunks only; ``apply_delta(delta)`` overlays saved materials onto freshly
  generated chunks and marks them ``edited`` (re-save) + ``dirty`` (remesh).

Handoff to world/
-----------------
The manager produces ``MeshArrays`` (pure numpy) and records them in
``pending_meshes`` / ``unloaded_this_frame``.  It NEVER imports panda3d or
touches the scene graph.  The World layer drains ``pending_meshes`` each frame,
calls ``world/geometry_bridge.to_geom`` on each, and uploads the Geom.  This
keeps terrain fully headless-testable (Hard Rule 1).
"""

from __future__ import annotations

import numpy as np

from fire_engine.core import (
    ChunkLoadedEvent,
    ChunkUnloadedEvent,
    Config,
    EventBus,
    for_domain,  # noqa: F401  (kept for parity / future per-chunk rng use)
    get_logger,
)
from fire_engine.core.math3d import Vec3
from fire_engine.terrain.chunk import Chunk
from fire_engine.terrain.generation import generate_chunk
from fire_engine.terrain.meshing import (
    MeshArrays,
    WORLD_FLOOR_SOLID,
    build_mesh,
)
from fire_engine.terrain.surface_nets import (
    NEIGHBOR_OFFSETS_26,
    build_mesh_faceted,
)

_log = get_logger("terrain.chunk_manager")

# Z band of streamed chunks relative to the camera chunk.
_Z_MIN: int = -2
_Z_MAX: int = 4
# Lowest streamed Z overall acts as the world floor (pad solid below it).
_WORLD_FLOOR_CZ: int = _Z_MIN

_MAX_LOADS_PER_FRAME: int = 2

_FACE_DIRS: tuple[tuple[int, int, int], ...] = (
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
)


class ChunkManager:
    """
    Streaming chunk store + terrain ``Saveable``.

    Parameters
    ----------
    config : Config
        Engine config (chunk size, voxel size, view distance).
    event_bus : EventBus
        Bus on which load/unload events are published.

    Attributes
    ----------
    save_key : str
        ``"terrain"`` — the Saveable registration key.
    chunks : dict[tuple[int,int,int], Chunk]
        Currently loaded chunks.
    pending_meshes : dict[tuple[int,int,int], MeshArrays]
        Meshes produced this/recent frames awaiting upload by the World layer.
        The World layer pops entries after uploading.
    unloaded_this_frame : list[tuple[int,int,int]]
        Coords unloaded in the last ``stream_frame`` (World removes their Geoms).

    Example
    -------
    >>> from fire_engine.core import load_config, EventBus
    >>> from fire_engine.core.rng import set_world_seed
    >>> set_world_seed(1337)
    >>> cm = ChunkManager(load_config(), EventBus())
    >>> from fire_engine.core.math3d import Vec3
    >>> cm.stream_frame(Vec3(0, 0, 20))   # loads up to 2 chunks
    >>> chunk = cm.get_or_create((0, 0, 0))
    """

    save_key: str = "terrain"

    def __init__(self, config: Config, event_bus: EventBus) -> None:
        self.config = config
        self.bus = event_bus
        self.chunks: dict[tuple[int, int, int], Chunk] = {}
        self.pending_meshes: dict[tuple[int, int, int], MeshArrays] = {}
        self.unloaded_this_frame: list[tuple[int, int, int]] = []
        self._chunk_m = config.chunk_meters
        self._n = int(config.chunk_size)
        self._vs = float(config.voxel_size)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def camera_chunk(self, camera_pos: Vec3) -> tuple[int, int, int]:
        """Chunk coordinate containing ``camera_pos`` (world meters)."""
        p = camera_pos.to_numpy()
        return (
            int(np.floor(p[0] / self._chunk_m)),
            int(np.floor(p[1] / self._chunk_m)),
            int(np.floor(p[2] / self._chunk_m)),
        )

    def desired_set(self, camera_pos: Vec3) -> set[tuple[int, int, int]]:
        """
        PURE FUNCTION: chunk coords that should be loaded for ``camera_pos``.

        Chunks within ``view_distance_chunks`` in the XY plane (Chebyshev/square
        radius about the camera chunk) and Z in ``[-2, +4]`` relative to the
        camera chunk.  No side effects, deterministic.

        Returns
        -------
        set[tuple[int,int,int]]
            The desired-loaded chunk coordinate set.
        """
        ccx, ccy, ccz = self.camera_chunk(camera_pos)
        r = int(self.config.view_distance_chunks)
        out: set[tuple[int, int, int]] = set()
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(_Z_MIN, _Z_MAX + 1):
                    out.add((ccx + dx, ccy + dy, ccz + dz))
        return out

    # ------------------------------------------------------------------
    # Provider (for brush / raycast)
    # ------------------------------------------------------------------

    def get_or_create(self, coord: tuple[int, int, int]) -> Chunk:
        """
        Return the chunk at ``coord``, generating it from seed if not loaded.

        This is the ``chunk_provider`` contract used by ``apply_brush`` and
        ``raycast_voxel``.  The chunk is added to ``self.chunks`` but NOT
        meshed here (meshing is the streaming budget's job).
        """
        chunk = self.chunks.get(coord)
        if chunk is None:
            materials = generate_chunk(coord, self.config)
            chunk = Chunk(
                coord,
                materials,
                chunk_size=self._n,
                voxel_size=self._vs,
            )
            self.chunks[coord] = chunk
        return chunk

    # alias so the manager itself is directly callable as a provider
    def __call__(self, coord: tuple[int, int, int]) -> Chunk:
        return self.get_or_create(coord)

    # ------------------------------------------------------------------
    # Meshing
    # ------------------------------------------------------------------

    def _neighbor_solids(self, coord: tuple[int, int, int]) -> dict:
        """
        Build the ``neighbor_solids`` dict for meshing ``coord``.

        Uses already-loaded neighbour chunks; absent neighbours are omitted
        (mesher pads air) **except** the −Z direction when this chunk sits at
        the world-floor Z band, where the sentinel forces a solid pad.
        """
        cx, cy, cz = coord
        out: dict = {}
        for d in _FACE_DIRS:
            ncoord = (cx + d[0], cy + d[1], cz + d[2])
            nb = self.chunks.get(ncoord)
            if nb is not None:
                out[d] = nb.is_solid_mask()
            elif d == (0, 0, -1) and cz <= _WORLD_FLOOR_CZ:
                out[d] = WORLD_FLOOR_SOLID
        return out

    def _neighbor_materials(self, coord: tuple[int, int, int]) -> dict:
        """
        Build the ``neighbor_materials`` dict (all 26 offsets) for faceted
        meshing of ``coord``.

        Loaded neighbours contribute their live (possibly brush-edited)
        materials.  Unloaded neighbours contribute the deterministic
        ``generate_chunk`` baseline **without** being inserted into
        ``self.chunks`` — generation is a pure sub-millisecond function, and
        baseline data is byte-identical to what that neighbour would hold if
        loaded, so chunk borders are seam-correct regardless of load order.
        (Edited neighbours are the only divergence; ``apply_brush`` marks
        border neighbours dirty so they remesh.)

        No ``WORLD_FLOOR_SOLID`` sentinel is needed here: below-ground
        neighbour chunks generate fully solid, which culls the world-floor
        faces naturally.
        """
        cx, cy, cz = coord
        out: dict = {}
        for off in NEIGHBOR_OFFSETS_26:
            ncoord = (cx + off[0], cy + off[1], cz + off[2])
            nb = self.chunks.get(ncoord)
            if nb is not None:
                out[off] = nb.materials
            else:
                out[off] = generate_chunk(ncoord, self.config)
        return out

    def mesh_chunk(self, coord: tuple[int, int, int], light_sampler=None) -> MeshArrays:
        """
        Build (and store in ``pending_meshes``) the mesh for a loaded chunk.

        Clears the chunk's ``dirty`` flag.  ``light_sampler`` is forwarded to
        the mesher (Phase 4 wires sunlight here).

        The mesher is selected by ``config.mesh_style``:
        ``"faceted"`` (default) → ``build_mesh_faceted`` (flat-shaded surface
        nets, per-face materials); ``"blocky"`` → the classic culled-face
        cube mesher ``build_mesh``.
        """
        chunk = self.get_or_create(coord)
        if getattr(self.config, "mesh_style", "faceted") == "blocky":
            mesh = build_mesh(chunk, self._neighbor_solids(coord), light_sampler)
        else:
            mesh = build_mesh_faceted(
                chunk,
                self._neighbor_materials(coord),
                light_sampler,
                shade_strength=float(
                    getattr(self.config, "facet_shade_strength", 0.25)
                ),
            )
        self.pending_meshes[coord] = mesh
        chunk.dirty = False
        return mesh

    def remesh_edited(self, coords, light_sampler=None) -> int:
        """
        Remesh brush-edited chunks NOW, bypassing the streaming budget.

        ``stream_frame``'s 2-chunk budget is sized for background world
        loading; a brush edit (explosion crater) routed through it appears
        over several frames — and until a border neighbour remeshes, the
        faces the edit newly exposed in it don't exist yet, so the player
        sees a hole through the world.  Call this right after ``apply_brush``
        with its returned coord set to make the whole edit appear the same
        frame (typical cost: 1–4 chunks ≈ 10–30 ms, an acceptable one-frame
        hitch for a discrete edit).

        Remeshes every still-``dirty`` loaded chunk in ``coords`` plus their
        26-neighbourhood (``apply_brush`` border-flags neighbours as plain
        ``dirty``).  Untouched dirty chunks elsewhere (e.g. an F9 load) stay
        on the budgeted ``stream_frame`` path.

        Parameters
        ----------
        coords : Iterable[tuple[int, int, int]]
            Chunk coords the brush changed (``apply_brush``'s return value).
        light_sampler : Callable | None
            Forwarded to the mesher (None on the GPU lighting backend).

        Returns
        -------
        int
            Number of chunks remeshed (into ``pending_meshes``).
        """
        pending: set[tuple[int, int, int]] = set()
        for c in coords:
            for off in ((0, 0, 0),) + NEIGHBOR_OFFSETS_26:
                n = (c[0] + off[0], c[1] + off[1], c[2] + off[2])
                ch = self.chunks.get(n)
                if ch is not None and ch.dirty:
                    pending.add(n)
        for c in pending:
            self.mesh_chunk(c, light_sampler)
        return len(pending)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream_frame(self, camera_pos: Vec3, light_sampler=None) -> None:
        """
        Stream one frame: load/mesh ≤2 chunks nearest the camera, unload far.

        Parameters
        ----------
        camera_pos : Vec3
            Current camera position (world meters).
        light_sampler : Callable | None
            Optional light sampler forwarded to the mesher (Phase 4).

        Behaviour
        ---------
        - **Re-meshes dirty loaded chunks FIRST** (within the 2-chunk budget):
          dirty means a brush edit or a relight the player is looking at, so
          it must never be starved by world loading.  (With ~1.2k chunks in
          the desired set, loading takes hundreds of frames — if loads ran
          first, a crater would stay invisible for minutes.)
        - Then loads + meshes the desired but not-yet-loaded chunks with the
          remaining budget, nearest-first.
        - Unloads loaded chunks beyond ``view_distance_chunks + 1`` (XY) —
          hysteresis prevents boundary thrash.  Edited chunks are kept in the
          delta via ``get_delta`` regardless (they are still removed from RAM).
        """
        self.unloaded_this_frame = []
        desired = self.desired_set(camera_pos)
        ccx, ccy, ccz = self.camera_chunk(camera_pos)

        def dist2(coord):
            return (coord[0] - ccx) ** 2 + (coord[1] - ccy) ** 2 + (coord[2] - ccz) ** 2

        budget = _MAX_LOADS_PER_FRAME

        # 1. Remesh dirty loaded chunks (brush edits, relights, border
        #    neighbours of edits) — before loading, so edits show promptly.
        dirty = [c for c, ch in self.chunks.items() if ch.dirty]
        dirty.sort(key=dist2)
        for coord in dirty:
            if budget <= 0:
                break
            self.mesh_chunk(coord, light_sampler)
            budget -= 1

        # 2. Load + mesh nearest missing chunks with the remaining budget.
        if budget > 0:
            missing = [c for c in desired if c not in self.chunks]
            missing.sort(key=dist2)
            for coord in missing:
                if budget <= 0:
                    break
                self.get_or_create(coord)
                self.mesh_chunk(coord, light_sampler)
                self.bus.publish(ChunkLoadedEvent(coord=coord))
                budget -= 1

        # 3. Unload chunks beyond radius + 1 (hysteresis).
        r = int(self.config.view_distance_chunks) + 1
        to_unload = []
        for coord in self.chunks:
            dx = abs(coord[0] - ccx)
            dy = abs(coord[1] - ccy)
            dz_low = coord[2] - ccz < _Z_MIN - 1
            dz_high = coord[2] - ccz > _Z_MAX + 1
            if dx > r or dy > r or dz_low or dz_high:
                to_unload.append(coord)
        for coord in to_unload:
            del self.chunks[coord]
            self.pending_meshes.pop(coord, None)
            self.unloaded_this_frame.append(coord)
            self.bus.publish(ChunkUnloadedEvent(coord=coord))

    # ------------------------------------------------------------------
    # Baseline reset (revert edits before re-applying a save delta)
    # ------------------------------------------------------------------

    def reset_to_baseline(self) -> None:
        """
        Revert every edited loaded chunk to its procedural (seed) baseline.

        Regenerates ``materials`` from ``generate_chunk(coord, config)`` for each
        loaded chunk whose ``edited`` flag is set, clears ``edited``, marks the
        chunk ``dirty`` (so the next ``stream_frame`` remeshes it), and drops any
        stale entry in ``pending_meshes`` so the World layer re-uploads the fresh
        geometry.

        Why this exists
        ---------------
        ``apply_delta`` only touches chunks present in the saved delta.  After a
        save, the player may dig *more* craters; loading the save must undo those
        extra craters too.  ``reset_to_baseline()`` wipes ALL edits back to the
        deterministic baseline first, then ``SaveManager.load`` re-applies only
        the saved craters via ``apply_delta``.  The canonical F9 flow is therefore:

            cm.reset_to_baseline()
            sm.load("saves/quick.ta")   # apply_delta re-adds saved craters

        Both passes mark touched chunks ``dirty``; subsequent ``stream_frame``
        calls remesh them (light recomputes via the SunlightComputer's event
        subscriptions) and the App re-uploads their Geoms.

        Notes
        -----
        Only *loaded* chunks are reset.  Unloaded chunks already hold no edits in
        RAM — they regenerate from seed (baseline) on their next
        ``get_or_create``, so they need no explicit reset.

        No window / GPU required; headless-testable.
        """
        for coord, chunk in self.chunks.items():
            if not chunk.edited:
                continue
            chunk.materials[...] = generate_chunk(coord, self.config)
            chunk.edited = False
            chunk.dirty = True
            # Drop stale mesh so the world re-uploads after the remesh.
            self.pending_meshes.pop(coord, None)

    # ------------------------------------------------------------------
    # Saveable protocol
    # ------------------------------------------------------------------

    def get_delta(self) -> dict:
        """
        Return the save delta: ``{coord_tuple: materials_uint8_array}``.

        Only chunks with ``edited == True`` (deviating from their procedural
        baseline) are included.  Values are copies of the ``uint8 (32,32,32)``
        material arrays — plain numpy, no live object references, no pickle
        (Hard Rule 3).

        Returns
        -------
        dict[tuple[int,int,int], numpy.ndarray]
        """
        return {
            coord: chunk.materials.copy()
            for coord, chunk in self.chunks.items()
            if chunk.edited
        }

    def apply_delta(self, delta: dict) -> None:
        """
        Overlay saved chunk materials after baseline regeneration.

        For each ``coord -> materials`` entry: ensure the chunk exists (generate
        baseline from seed if needed), replace its materials with the saved
        array, and mark it ``edited`` (so it re-saves) and ``dirty`` (so it
        remeshes on the next ``stream_frame``).

        Parameters
        ----------
        delta : dict[tuple[int,int,int], numpy.ndarray]
            As produced by :meth:`get_delta`.
        """
        for coord, materials in delta.items():
            coord_t = (int(coord[0]), int(coord[1]), int(coord[2]))
            chunk = self.chunks.get(coord_t)
            if chunk is None:
                chunk = Chunk(
                    coord_t,
                    chunk_size=self._n,
                    voxel_size=self._vs,
                )
                self.chunks[coord_t] = chunk
            chunk.materials[...] = np.asarray(materials, dtype=np.uint8)
            chunk.edited = True
            chunk.dirty = True
            # Drop any stale mesh so the world re-uploads after remesh.
            self.pending_meshes.pop(coord_t, None)
