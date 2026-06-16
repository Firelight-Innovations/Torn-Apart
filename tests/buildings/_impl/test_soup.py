"""
tests/buildings/_impl/test_soup.py — the shared triangle-soup accumulator.

Covers the MeshArrays contract, slab counts, the sloped-prism primitive, and
winding auto-flip to a supplied outward normal.

Headless (numpy only — buildings/ never imports panda3d).
"""

import numpy as np

from fire_engine.buildings._impl.soup import Soup

_SQUARE = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float)


def _contract(mesh):
    assert mesh.positions.dtype == np.float32
    assert mesh.normals.dtype == np.float32
    assert mesh.face_materials is None
    assert np.all(mesh.colors == 1.0)
    nl = np.linalg.norm(mesh.normals, axis=1)
    assert np.allclose(nl, 1.0, atol=1e-5)


def test_empty_soup_builds_empty_mesh():
    mesh = Soup().build()
    assert mesh.is_empty
    assert mesh.positions.shape == (0, 3)


def test_slab_counts_and_contract():
    soup = Soup()
    soup.add_slab(_SQUARE, 0.0, 0.2)
    mesh = soup.build()
    _contract(mesh)
    # 2 top + 2 bottom tris + 4 side quads (8 tris) = 12 tris.
    assert mesh.tri_count == 12


def test_single_material_leaves_face_materials_none():
    soup = Soup()
    soup.add_slab(_SQUARE, 0.0, 0.2)  # default material 0
    assert soup.build().face_materials is None


def test_mixed_materials_emit_per_face_uint8_ids():
    soup = Soup()
    soup.add_slab(_SQUARE, 0.0, 0.2, material=0)  # 12 faces
    soup.add_slab(_SQUARE, 0.2, 0.4, material=2)  # 12 faces
    mesh = soup.build()
    assert mesh.face_materials is not None
    assert mesh.face_materials.dtype == np.uint8
    assert mesh.face_materials.shape[0] == mesh.tri_count
    assert set(mesh.face_materials.tolist()) == {0, 2}


def test_prism_top_bottom_and_sides():
    # A flat square "panel" given 0.5 m drop: 2 top + 2 bottom + 4 side quads.
    top = np.array([[0, 0, 5.0], [4, 0, 5.0], [4, 4, 5.0], [0, 4, 5.0]], dtype=float)
    soup = Soup()
    soup.add_prism(top, 0.5)
    mesh = soup.build()
    _contract(mesh)
    assert mesh.tri_count == 12
    assert np.isclose(mesh.positions[:, 2].max(), 5.0)
    assert np.isclose(mesh.positions[:, 2].min(), 4.5)


def test_prism_zero_drop_is_single_sheet():
    top = np.array([[0, 0, 5.0], [4, 0, 5.0], [4, 4, 5.0], [0, 4, 5.0]], dtype=float)
    soup = Soup()
    soup.add_prism(top, 0.0)
    # Just the 2 top triangles, no bottom or sides.
    assert soup.build().tri_count == 2


def test_sloped_prism_normal_points_up():
    # Tilted panel (rises in +y): the top-face normal must have +z.
    top = np.array([[0, 0, 5.0], [4, 0, 5.0], [4, 4, 6.0], [0, 4, 6.0]], dtype=float)
    soup = Soup()
    soup.add_prism(top, 0.3)
    mesh = soup.build()
    assert np.any(mesh.normals[:, 2] > 0.5)  # an up-facing slope exists


def test_add_quads_flips_winding_to_normal():
    # Quad listed clockwise; outward normal +z. After flip, geometric normal
    # agrees with the supplied one.
    corners = np.array([[[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]], dtype=float)
    soup = Soup()
    soup.add_quads(corners, np.array([[0.0, 0.0, 1.0]]))
    mesh = soup.build()
    tri = mesh.positions[:3]
    geo = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    assert geo[2] > 0.0  # winding now matches +z
