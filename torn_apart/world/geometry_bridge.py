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
)

__all__ = ["to_geom", "to_geom_node", "make_vertex_format"]


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


def to_geom_node(mesh, name: str = "terrain_chunk") -> GeomNode:
    """
    Convenience: wrap :func:`to_geom` output in a named ``GeomNode``.

    Parameters
    ----------
    mesh : MeshArrays
        Terrain mesh arrays.
    name : str, default "terrain_chunk"
        Node name (use the chunk coord string for debugging).

    Returns
    -------
    panda3d.core.GeomNode
        A GeomNode with the chunk's geometry, ready to parent under a NodePath.
    """
    node = GeomNode(name)
    node.add_geom(to_geom(mesh))
    return node
