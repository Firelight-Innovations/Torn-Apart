"""
tests/test_buildings_meshing.py — building meshing: exact face counts for
straight/arc/opening walls and slabs, dtypes, unit flat normals, opening
emptiness, determinism, and a whole-building smoke test.

Headless (numpy only — fire_engine/buildings/ never imports panda3d).
"""

import numpy as np
import pytest

from fire_engine.buildings import Building, BuildingDefaults, OpeningKind
from fire_engine.buildings.model import Opening, Wall
from fire_engine.buildings.meshing import mesh_building, mesh_slab, mesh_wall
from fire_engine.core.config import Config
from fire_engine.core.math3d import Quat, Vec3

_CFG = Config()
_QPQ = _CFG.building_arc_segments_per_quarter


def _tris(mesh) -> int:
    return mesh.positions.shape[0] // 3


def _assert_contract(mesh):
    assert mesh.positions.dtype == np.float32
    assert mesh.normals.dtype == np.float32
    assert mesh.uvs.dtype == np.float32
    assert mesh.colors.dtype == np.float32
    assert mesh.indices.dtype == np.uint32
    assert mesh.positions.shape[0] == mesh.normals.shape[0]
    assert mesh.positions.shape[0] == mesh.uvs.shape[0]
    assert mesh.positions.shape[0] == mesh.indices.shape[0]
    assert np.all(mesh.colors == 1.0)               # flat white
    assert mesh.face_materials is None
    # flat unit normals
    nl = np.linalg.norm(mesh.normals, axis=1)
    assert np.allclose(nl, 1.0, atol=1e-5)


class TestWallCounts:
    def test_straight_wall_no_openings(self):
        # front + back + top cap + 2 end caps = 5 quads = 10 tris = 30 verts.
        w = Wall(id=1, a=(0, 0), b=(4, 0), thickness_m=0.3)
        mesh = mesh_wall(w, 0.2, 3.0, _QPQ)
        _assert_contract(mesh)
        assert _tris(mesh) == 10
        assert np.isclose(mesh.positions[:, 2].min(), 0.2)
        assert np.isclose(mesh.positions[:, 2].max(), 3.0)

    def test_arc_wall_panels_track_tessellation(self):
        w = Wall(id=1, a=(0, 0), b=(4, 0), bulge=0.5, thickness_m=0.3)
        p = w.tessellate(_QPQ).shape[0]
        # front + back + top = 3*(p-1) panels, + 2 end caps.
        expected_quads = 3 * (p - 1) + 2
        mesh = mesh_wall(w, 0.2, 3.0, _QPQ)
        _assert_contract(mesh)
        assert _tris(mesh) == 2 * expected_quads

    def test_window_adds_reveals_and_removes_hole_cell(self):
        # 3 s-cells x 3 z-cells = 9 panels - 1 hole = 8 per face (16),
        # + 3 top + 2 ends + 4 reveals (2 jambs, head, sill) = 25 quads.
        w = Wall(id=1, a=(0, 0), b=(4, 0), thickness_m=0.3, openings=[
            Opening(id=9, kind=OpeningKind.WINDOW, offset_m=1.0, width_m=1.2,
                    sill_m=1.0, head_m=2.2)])
        mesh = mesh_wall(w, 0.2, 3.0, _QPQ)
        _assert_contract(mesh)
        assert _tris(mesh) == 2 * 25

    def test_door_has_no_sill_reveal(self):
        # door (sill=0): 3 s-cells x 2 z-cells = 6 - 1 hole = 5 per face (10),
        # + 3 top + 2 ends + 3 reveals (2 jambs, head, NO sill) = 18 quads.
        w = Wall(id=1, a=(0, 0), b=(4, 0), thickness_m=0.3, openings=[
            Opening(id=8, kind=OpeningKind.DOOR, offset_m=1.0, width_m=0.9,
                    sill_m=0.0, head_m=2.0)])
        mesh = mesh_wall(w, 0.2, 3.0, _QPQ)
        _assert_contract(mesh)
        assert _tris(mesh) == 2 * 18

    def test_no_vertex_strictly_inside_opening_box(self):
        # Wall along +x; faces at y=±0.15. Opening x∈(1,2.2), z∈(1.2,2.4).
        w = Wall(id=1, a=(0, 0), b=(4, 0), thickness_m=0.3, openings=[
            Opening(id=9, kind=OpeningKind.WINDOW, offset_m=1.0, width_m=1.2,
                    sill_m=1.0, head_m=2.2)])
        p = mesh_wall(w, 0.2, 3.0, _QPQ).positions
        e = 1e-4
        inside = ((p[:, 0] > 1.0 + e) & (p[:, 0] < 2.2 - e) &
                  (p[:, 1] > -0.15 + e) & (p[:, 1] < 0.15 - e) &
                  (p[:, 2] > 1.2 + e) & (p[:, 2] < 2.4 - e))
        assert not np.any(inside)


class TestSlab:
    def test_square_slab_counts(self):
        # 2 top tris + 2 bottom tris + 4 side quads (8 tris) = 12 tris.
        poly = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float)
        mesh = mesh_slab(poly, 0.0, 0.2)
        _assert_contract(mesh)
        assert _tris(mesh) == 12

    def test_slab_top_and_bottom_face_outward(self):
        poly = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float)
        mesh = mesh_slab(poly, 0.0, 0.2)
        # Some normal points up, some down (top/bottom), some sideways.
        nz = mesh.normals[:, 2]
        assert np.any(nz > 0.99)
        assert np.any(nz < -0.99)
        assert np.any(np.abs(nz) < 1e-5)


def _demo_building() -> Building:
    d = BuildingDefaults.from_config(_CFG)
    b = Building(name="demo", position=Vec3(0, 0, 8.0),
                 rotation=Quat.identity(), defaults=d)
    s0 = b.add_storey()
    south = s0.add_wall((0, 0), (8, 0))
    s0.add_wall((8, 0), (8, 6))
    s0.add_wall((8, 6), (0, 6))
    s0.add_wall((0, 6), (0, 0))
    s0.add_opening(south.id, OpeningKind.DOOR, offset_m=3.5, width_m=0.9,
                   head_m=2.0)
    s0.add_opening(south.id, OpeningKind.WINDOW, offset_m=1.0, width_m=1.2,
                   sill_m=1.0, head_m=2.2)
    b.set_foundation()
    b.set_roof()
    return b


class TestWholeBuilding:
    def test_building_meshes_nonempty_and_valid(self):
        mesh = mesh_building(_demo_building(), _CFG)
        _assert_contract(mesh)
        assert _tris(mesh) > 0
        # Foundation reaches below local z=0; roof above the single storey.
        assert mesh.positions[:, 2].min() < 0.0
        assert mesh.positions[:, 2].max() > 3.0

    def test_meshing_is_deterministic(self):
        a = mesh_building(_demo_building(), _CFG)
        b = mesh_building(_demo_building(), _CFG)
        assert np.array_equal(a.positions, b.positions)
        assert np.array_equal(a.normals, b.normals)
        assert np.array_equal(a.uvs, b.uvs)
        assert np.array_equal(a.indices, b.indices)

    def test_positions_are_building_local_not_world(self):
        # Building origin at z=8 in world, but local mesh straddles z=0.
        b = _demo_building()
        mesh = mesh_building(b, _CFG)
        assert mesh.positions[:, 2].min() < 1.0   # local, not offset by +8
