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
      1. ``stream_frame(camera_pos, light_sampler)`` — loads/meshes ≤2 chunks
         near the camera and remeshes dirty (edited/relit) chunks, populating
         ``pending_meshes`` and ``unloaded_this_frame``.
      2. Drain ``pending_meshes``: convert each ``MeshArrays`` to a GeomNode
         (bulk-write Geom) and parent it under ``terrain_root``.  Mesh
         positions are absolute world meters, so the NodePath is placed at the
         origin (no offset).  Any existing NodePath for that coord is detached
         first (remesh replaces stale geometry).
      3. Drain ``unloaded_this_frame``: detach + forget those coords' Geoms.

    All scene-graph writes are bulk Geom uploads (Hard Rule 7); no per-vertex
    Python loops (those live in the headless mesher / geometry_bridge).
    """
    cm = self_obj.chunk_manager
    if cm is None:
        return

    # Lazy import: terrain → world is an allowed downward dependency, but we
    # import here to keep the module importable when panda3d-only tooling
    # constructs a bare App.
    from fire_engine.render.bridges.geometry_bridge import to_geom_node

    # 1. Stream around the camera (light_sampler may be None → full-bright).
    cm.stream_frame(self_obj.camera_go.transform.position, self_obj.light_sampler)

    # 2. Upload freshly produced meshes.  Copy keys first: we mutate the dict.
    for coord in list(cm.pending_meshes.keys()):
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
