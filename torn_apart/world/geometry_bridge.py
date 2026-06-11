"""
world/geometry_bridge.py — Convert terrain ``MeshArrays`` to a Panda3D ``Geom``.

This is the terrain → render handoff.  The terrain layer produces pure-numpy
``MeshArrays`` (positions/normals/uvs/colors/indices); this module — the only
terrain-geometry file allowed to import panda3d (it lives in ``world/``) — turns
those arrays into a renderable ``Geom``.

Performance contract (DEVELOPMENT_PLAN.md "Known Traps")
--------------------------------------------------------
Each vertex array and the index array is written with **one bulk memoryview /
``modify_array`` write** — never a ``GeomVertexWriter`` per-vertex loop (those
blow the per-frame budget once chunks stream).  We lay out a single interleaved
vertex buffer (V3N3T2C4 format) and ``memoryview``-copy the whole thing once,
then copy the index buffer once.
"""

from __future__ import annotations

import numpy as np
from panda3d.core import (  # type: ignore[import]
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexArrayFormat,
    GeomVertexData,
    GeomVertexFormat,
    GeomEnums,
    RenderState,
    TextureAttrib,
    TextureStage,
)

__all__ = ["to_geom", "to_geom_node", "make_vertex_format",
           "make_material_state"]

# Texture stages for the GPU-lighting terrain shader: sort order maps to
# p3d_Texture0 (albedo), p3d_Texture1 (normal map), p3d_Texture2 (emission).
_STAGE_ALBEDO = TextureStage("ta_albedo")
_STAGE_ALBEDO.set_sort(0)
_STAGE_NORMAL = TextureStage("ta_normal")
_STAGE_NORMAL.set_sort(1)
_STAGE_EMISSION = TextureStage("ta_emission")
_STAGE_EMISSION.set_sort(2)


def make_material_state(entry) -> RenderState:
    """
    Build the per-material Geom RenderState from a texture entry.

    Parameters
    ----------
    entry : panda3d.core.Texture | tuple | None
        Either a single albedo ``Texture`` (legacy fixed-function path) or an
        ``(albedo, normal, emission)`` triple for the GPU terrain shader
        (``p3d_Texture0/1/2`` by stage sort order).  ``None`` → empty state
        (inherits any node-level fallback texture).

    Returns
    -------
    panda3d.core.RenderState
    """
    if entry is None:
        return RenderState.make_empty()
    if isinstance(entry, tuple):
        attrib = TextureAttrib.make()
        for stage, tex in zip(
                (_STAGE_ALBEDO, _STAGE_NORMAL, _STAGE_EMISSION), entry):
            if tex is not None:
                attrib = attrib.add_on_stage(stage, tex)
        return RenderState.make(attrib)
    return RenderState.make(TextureAttrib.make(entry))


def make_vertex_format() -> GeomVertexFormat:
    """
    Build the interleaved vertex format used by terrain geometry.

    Layout (one interleaved array, all float32):
        vertex   : 3 floats (world position, meters)
        normal   : 3 floats (flat per-face normal)
        texcoord : 2 floats (planar UV)
        color    : 4 floats (RGBA in [0,1], baked light)

    Returns
    -------
    panda3d.core.GeomVertexFormat
        A registered interleaved format (V3 N3 T2 C4).
    """
    arr = GeomVertexArrayFormat()
    arr.add_column("vertex", 3, GeomEnums.NT_float32, GeomEnums.C_point)
    arr.add_column("normal", 3, GeomEnums.NT_float32, GeomEnums.C_normal)
    arr.add_column("texcoord", 2, GeomEnums.NT_float32, GeomEnums.C_texcoord)
    arr.add_column("color", 4, GeomEnums.NT_float32, GeomEnums.C_color)
    fmt = GeomVertexFormat()
    fmt.add_array(arr)
    return GeomVertexFormat.register_format(fmt)


def to_geom(mesh) -> Geom:
    """
    Build a Panda3D ``Geom`` from terrain ``MeshArrays`` with bulk writes only.

    Parameters
    ----------
    mesh : MeshArrays
        From ``terrain.build_mesh``.  Expects:
        - ``positions`` ``float32 (N, 3)`` world meters
        - ``normals``   ``float32 (N, 3)``
        - ``uvs``       ``float32 (N, 2)``
        - ``colors``    ``float32 (N, 4)`` RGBA in [0,1]
        - ``indices``   ``uint32 (M,)``

    Returns
    -------
    panda3d.core.Geom
        A triangle ``Geom`` ready to attach to a ``GeomNode``.  For an empty
        mesh (no faces) an empty Geom with zero primitives is returned.

    Notes
    -----
    The interleaved vertex buffer is assembled in numpy (one contiguous
    ``(N, 12)`` float32 block) and copied into the ``GeomVertexData`` via a
    single ``memoryview`` assignment on ``modify_array(0)``.  The index buffer
    is copied once via ``modify_handle().copy_data_from``.  No per-vertex Python
    loops — satisfies the bulk-write rule (CLAUDE.md Hard Rule 7).
    """
    fmt = make_vertex_format()
    n_verts = int(mesh.positions.shape[0])

    vdata = GeomVertexData("terrain_chunk", fmt, Geom.UH_static)
    vdata.set_num_rows(n_verts)

    if n_verts > 0:
        # Interleave into one (N, 12) float32 block: [px,py,pz, nx,ny,nz, u,v, r,g,b,a]
        interleaved = np.empty((n_verts, 12), dtype=np.float32)
        interleaved[:, 0:3] = mesh.positions
        interleaved[:, 3:6] = mesh.normals
        interleaved[:, 6:8] = mesh.uvs
        interleaved[:, 8:12] = mesh.colors
        interleaved = np.ascontiguousarray(interleaved)

        # ONE bulk write of the whole vertex buffer.
        varray = vdata.modify_array(0)
        view = memoryview(varray).cast("B")
        view[:] = memoryview(interleaved).cast("B")

    prim = GeomTriangles(Geom.UH_static)
    if n_verts > 0 and mesh.indices.shape[0] > 0:
        prim.set_index_type(GeomEnums.NT_uint32)
        idx = np.ascontiguousarray(mesh.indices, dtype=np.uint32)
        iarray = prim.modify_vertices()
        iarray.set_num_rows(int(idx.shape[0]))
        # ONE bulk write of the whole index buffer via the array's buffer protocol.
        iview = memoryview(iarray).cast("B")
        iview[:] = memoryview(idx).cast("B")

    geom = Geom(vdata)
    geom.add_primitive(prim)
    return geom


def _face_indices(face_count: int, verts_per_face: int) -> np.ndarray:
    """
    Rebuild the ``uint32`` triangle index array for ``face_count`` contiguous
    faces of ``verts_per_face`` vertices each.

    Valid because both meshers emit faces as contiguous, non-shared vertex
    runs: 4 verts/face → quad pattern ``(0,1,2)(0,2,3)``; 6 verts/face →
    two independent triangles, indices are simply sequential.
    """
    if verts_per_face == 6:
        return np.arange(face_count * 6, dtype=np.uint32)
    base = (np.arange(face_count, dtype=np.uint32) * verts_per_face)[:, None]
    quad = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)[None, :]
    return (base + quad).reshape(-1)


def to_geom_node(
    mesh,
    name: str = "terrain_chunk",
    material_textures: dict | None = None,
) -> GeomNode:
    """
    Wrap a chunk mesh in a named ``GeomNode``, optionally split per material.

    Parameters
    ----------
    mesh : MeshArrays
        Terrain mesh arrays.  When ``mesh.face_materials`` is set (faceted
        mesher) AND ``material_textures`` is given, the mesh is split into
        one Geom per material id, each added with a ``RenderState`` carrying
        that material's texture — so grass faces render with the grass
        texture and dirt faces with the dirt texture inside a single node.
        Otherwise one untextured Geom is added (the caller's node-level
        ``set_texture`` applies, as before).
    name : str, default "terrain_chunk"
        Node name (use the chunk coord string for debugging).
    material_textures : dict[int, Texture | tuple] | None
        Material id → albedo ``Texture`` (legacy) or an ``(albedo, normal,
        emission)`` triple for the GPU terrain shader (see
        :func:`make_material_state`).  Ids missing from the dict fall back
        to an empty RenderState (inherits the node-level texture, if any).

    Returns
    -------
    panda3d.core.GeomNode
        A GeomNode with the chunk's geometry, ready to parent under a NodePath.

    Notes
    -----
    The split is pure-numpy boolean selection on whole faces (vertex runs are
    contiguous per face) followed by the same bulk-write :func:`to_geom` path
    — no per-vertex Python loops (Hard Rule 7).  Geom-level RenderStates
    compose *over* NodePath states, so per-material textures win over any
    node-level fallback texture.
    """
    node = GeomNode(name)
    face_mats = getattr(mesh, "face_materials", None)
    if face_mats is None or material_textures is None or face_mats.size == 0:
        node.add_geom(to_geom(mesh))
        return node

    from torn_apart.terrain.meshing import MeshArrays  # numpy-only dataclass

    vpf = int(mesh.verts_per_face)
    for mat in np.unique(face_mats):
        sel = face_mats == mat                      # (F,) whole-face select
        vsel = np.repeat(sel, vpf)                  # (N,) vertex select
        count = int(sel.sum())
        sub = MeshArrays(
            positions=mesh.positions[vsel],
            normals=mesh.normals[vsel],
            uvs=mesh.uvs[vsel],
            colors=mesh.colors[vsel],
            indices=_face_indices(count, vpf),
            face_materials=face_mats[sel],
            verts_per_face=vpf,
        )
        node.add_geom(to_geom(sub),
                      make_material_state(material_textures.get(int(mat))))
    return node
