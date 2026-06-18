"""
render/_impl/app_terrain.py — Terrain render-integration helpers for App.

Extracted from render/app.py to keep that module under 500 lines (C0302).
Functions take the App instance as their first argument (``self_obj``) and are
called from the class as ``_func(self, ...)``.  Not part of the public API.

Docs: docs/systems/render.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fire_engine.render.app import App


def setup_terrain_rendering(
    self_obj: App,
    ground_texture: Any = None,
    material_textures: Any = None,
) -> None:
    """
    Configure the terrain render state once at boot.

    Call after injecting ``self_obj.chunk_manager``.  Stores the per-material
    texture map (used by the mesh upload path to texture grass and dirt
    faces separately), applies the optional fallback ground texture to
    ``terrain_root``, and turns Panda3D lighting OFF so the default
    fixed-function pipeline renders **texture × vertex colour**.  The
    mesher has already baked sunlight into the vertex colours (greyscale ×
    light), so adding a Panda3D light would double-light the scene.

    Parameters
    ----------
    self_obj : App — the App instance
    ground_texture : panda3d.core.Texture | None
        Fallback texture applied at the ``terrain_root`` node level.  It
        covers blocky-mesher geometry (no ``face_materials``) and any
        material id missing from ``material_textures``.  If None, no
        node-level texture is set.
    material_textures : dict[int, panda3d.core.Texture] | None
        Material id → texture map for the faceted mesher's per-material
        Geom split (``{MATERIAL_DIRT: dirt_tex, MATERIAL_GRASS:
        grass_tex}``).  Forwarded to ``geometry_bridge.to_geom_node`` on
        every chunk upload.  Geom-level texture states compose over the
        node-level fallback, so they win where both exist.

    Notes
    -----
    - ``set_light_off()`` ensures no ambient/directional light multiplies the
      vertex colours; the baked-light look is preserved exactly.
    - The geom vertex format (geometry_bridge.make_vertex_format) includes a
      C4 colour column, so vertex colours are active by default — no
      ``set_color_off`` is issued.

    Docs: docs/systems/render.md
    """
    self_obj.material_textures = material_textures
    if ground_texture is not None:
        self_obj.terrain_root.set_texture(ground_texture)
    # Baked light lives in vertex colours — disable scene lighting so the
    # pipeline renders texture × vertex-colour (no extra light term).
    self_obj.terrain_root.set_light_off()


def stream_and_upload_terrain(self_obj: App) -> None:
    """
    Drive chunk streaming and sync produced meshes to the scene graph.

    Per frame (when a ``chunk_manager`` is injected):
      1. Stream around the camera.  If an async ``lod_streamer`` is wired,
         ``lod_streamer.stream_frame(camera_pos)`` drains finished off-thread
         meshes into ``pending_meshes`` and submits fresh jobs (Hard Rule 12);
         otherwise the synchronous ``cm.stream_frame(camera_pos, light_sampler)``
         meshes ≤2 chunks on the main thread (baked-light / editor path).  Both
         populate ``pending_meshes`` and ``unloaded_this_frame``.  When a
         ``coarse_streamer`` is also wired, it streams the distant horizon of
         coarse LOD nodes into ``pending_coarse_meshes`` /
         ``unloaded_coarse_this_frame`` (P2).
      2. Drain ``pending_meshes`` **nearest-first**, uploading at most
         ``config.lod_max_uploads_per_frame`` this frame (leftovers wait for the
         next frame, capping per-frame GPU upload cost): convert each
         ``MeshArrays`` to a GeomNode (bulk-write Geom) and parent it under
         ``terrain_root``.  Mesh positions are absolute world meters, so the
         NodePath is placed at the origin (no offset).  Any existing NodePath
         for that coord is detached first (remesh replaces stale geometry).
      3. Drain ``unloaded_this_frame``: detach + forget those coords' Geoms.
      4. Upload + retire coarse-node Geoms the same way (see
         :func:`_upload_coarse_terrain`).  A coarse node and the L0 chunks it
         covers are mutually exclusive in the planner (``desired_node_set``), so
         they never both draw — the hard band cut for P2 (pop is acceptable;
         crossfade is P3).

    All scene-graph writes are bulk Geom uploads (Hard Rule 7); no per-vertex
    Python loops (those live in the headless mesher / geometry_bridge).

    Docs: docs/systems/render.md
    """
    cm = self_obj.chunk_manager
    if cm is None:
        return

    # Lazy import: terrain → world is an allowed downward dependency, but we
    # import here to keep the module importable when panda3d-only tooling
    # constructs a bare App.
    from fire_engine.render.bridges.geometry_bridge import to_geom_node

    pos = self_obj.camera_go.transform.position

    # 1. Stream around the camera.  Async streamer (off-thread meshing) when
    #    wired; else the synchronous main-thread path (light_sampler may be
    #    None → full-bright).
    if getattr(self_obj, "lod_streamer", None) is not None:
        self_obj.lod_streamer.stream_frame(pos)
    else:
        cm.stream_frame(pos, self_obj.light_sampler)
    if getattr(self_obj, "coarse_streamer", None) is not None:
        self_obj.coarse_streamer.stream_frame(pos)

    # 2. Upload freshly produced meshes, nearest-first, capped per frame.
    #    Leftovers stay in pending_meshes for a later frame.
    ccx, ccy, ccz = (int(v) for v in cm.camera_chunk(pos))

    def _dist2(coord: tuple[int, int, int]) -> int:
        return (coord[0] - ccx) ** 2 + (coord[1] - ccy) ** 2 + (coord[2] - ccz) ** 2

    max_uploads = int(cm.config.lod_max_uploads_per_frame)
    ready = sorted(cm.pending_meshes.keys(), key=_dist2)[:max_uploads]
    for coord in ready:
        mesh = cm.pending_meshes.pop(coord)
        # Replace any stale NodePath for this coord (remesh after a brush edit).
        old = self_obj._chunk_nodes.pop(coord, None)
        if old is not None:
            old.remove_node()
        geom_node = to_geom_node(
            mesh,
            name=f"chunk_{coord[0]}_{coord[1]}_{coord[2]}",
            material_textures=getattr(self_obj, "material_textures", None),
        )
        np_node = self_obj.terrain_root.attach_new_node(geom_node)
        # Positions are absolute world meters — no per-chunk offset.
        self_obj._chunk_nodes[coord] = np_node

    # 3. Remove Geoms for chunks unloaded this frame.
    for coord in cm.unloaded_this_frame:
        node = self_obj._chunk_nodes.pop(coord, None)
        if node is not None:
            node.remove_node()

    # 4. Coarse LOD horizon (P2): upload + retire coarse-node Geoms.
    _upload_coarse_terrain(self_obj, to_geom_node, (ccx, ccy, ccz))


def _upload_coarse_terrain(
    self_obj: App,
    to_geom_node: Any,
    camera_chunk: tuple[int, int, int],
) -> None:
    """
    Upload + retire coarse LOD node Geoms (the distant horizon — P2).

    Drains ``chunk_manager.pending_coarse_meshes`` **nearest-first**, uploading
    at most ``config.lod_coarse_uploads_per_frame`` coarse-node Geoms this frame
    (leftovers wait — a separate budget from the near
    ``lod_max_uploads_per_frame`` so coarse uploads never starve near streaming).
    Each coarse mesh's positions are absolute world meters (the ``_CoarseChunk``
    shim lands the node at the correct world position), so the Geom parents under
    ``terrain_root`` at the origin exactly like a near chunk — the same
    ``set_light_off`` + per-material texture state applies.  Stale Geoms for a
    re-meshed node are detached first; nodes in
    ``unloaded_coarse_this_frame`` (left the desired set — the hard band cut) are
    detached and forgotten.

    No-op when no ``coarse_streamer`` is wired (the coarse channels are absent).

    Parameters
    ----------
    self_obj : App — the App instance (owns ``_coarse_nodes`` + ``terrain_root``).
    to_geom_node : callable
        ``geometry_bridge.to_geom_node`` (passed in so the lazy import is shared).
    camera_chunk : tuple[int, int, int]
        Camera chunk coord, for the nearest-first upload ordering.

    Docs: docs/systems/render.md
    """
    cm = self_obj.chunk_manager
    pending = getattr(cm, "pending_coarse_meshes", None)
    if pending is None:
        return
    coarse_nodes: dict[tuple[int, int, int, int], Any] = self_obj._coarse_nodes
    ccx, ccy, ccz = camera_chunk

    def _node_dist2(key: tuple[int, int, int, int]) -> int:
        rank, nx, ny, nz = key
        k = 1 << rank
        # Node-block centre in chunk coords, distance to the camera chunk.
        cx, cy, cz = nx * k + k // 2, ny * k + k // 2, nz * k + k // 2
        return (cx - ccx) ** 2 + (cy - ccy) ** 2 + (cz - ccz) ** 2

    max_uploads = int(getattr(cm.config, "lod_coarse_uploads_per_frame", 4))
    ready = sorted(pending.keys(), key=_node_dist2)[:max_uploads]
    for key in ready:
        mesh = pending.pop(key)
        old = coarse_nodes.pop(key, None)
        if old is not None:
            old.remove_node()
        if mesh.is_empty:
            continue  # all-air coarse node: nothing to draw (still retired below)
        rank, nx, ny, nz = key
        geom_node = to_geom_node(
            mesh,
            name=f"coarse_L{rank}_{nx}_{ny}_{nz}",
            material_textures=getattr(self_obj, "material_textures", None),
        )
        coarse_nodes[key] = self_obj.terrain_root.attach_new_node(geom_node)

    # Retire coarse nodes that left the desired set (hard band cut).
    for key in getattr(cm, "unloaded_coarse_this_frame", []):
        node = coarse_nodes.pop(key, None)
        if node is not None:
            node.remove_node()
