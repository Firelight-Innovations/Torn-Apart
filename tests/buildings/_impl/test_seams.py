"""
tests/buildings/_impl/test_seams.py — corner-filler seam geometry.

Verifies that shared wall corners produce filler polygons that span the butt
joint, that lone wall ends produce none, that the post height tracks the
shortest incident wall band, and that the result is deterministic.

Headless (numpy only — buildings/ never imports panda3d).
"""

import numpy as np

from fire_engine.buildings import Building, BuildingDefaults
from fire_engine.buildings._impl.seams import _wall_end_offsets, corner_filler_polys
from fire_engine.core.config import Config
from fire_engine.core.math3d import Quat, Vec3

_CFG = Config()
_QPQ = _CFG.building_arc_segments_per_quarter
_SNAP = _CFG.building_snap_eps_m


def _box(thickness: float = 0.4) -> Building:
    d = BuildingDefaults.from_config(_CFG)
    b = Building(name="box", position=Vec3(0, 0, 0), rotation=Quat.identity(), defaults=d)
    s0 = b.add_storey()
    for a, c in [((0, 0), (8, 0)), ((8, 0), (8, 6)), ((8, 6), (0, 6)), ((0, 6), (0, 0))]:
        s0.add_wall(a, c, thickness_m=thickness)
    return b


def test_wall_end_offsets_straddle_centerline():
    b = _box(0.4)
    wall = b.storeys[0].walls[0]  # (0,0)->(8,0) along +x
    a, a_off, bend, _b_off = _wall_end_offsets(wall, _QPQ)
    assert np.allclose(a, [0, 0]) and np.allclose(bend, [8, 0])
    # Offsets sit ±0.2 (t/2) off the centerline, on the y axis for an +x wall.
    assert np.allclose(sorted(a_off[:, 1]), [-0.2, 0.2], atol=1e-9)
    assert np.allclose(a_off[:, 0], 0.0, atol=1e-9)


def test_closed_box_has_four_fillers():
    fillers = corner_filler_polys(_box().storeys[0], _QPQ, _SNAP)
    assert len(fillers) == 4
    for hull, band in fillers:
        assert hull.shape[1] == 2 and hull.shape[0] >= 3
        assert band > 0.0


def test_lone_wall_has_no_filler():
    d = BuildingDefaults.from_config(_CFG)
    b = Building(name="lone", position=Vec3(0, 0, 0), rotation=Quat.identity(), defaults=d)
    s0 = b.add_storey()
    s0.add_wall((0, 0), (8, 0), thickness_m=0.4)  # two free ends, no junction
    assert corner_filler_polys(s0, _QPQ, _SNAP) == []


def test_filler_band_tracks_shortest_incident_wall():
    d = BuildingDefaults.from_config(_CFG)
    b = Building(name="mix", position=Vec3(0, 0, 0), rotation=Quat.identity(), defaults=d)
    s0 = b.add_storey()  # default storey 3.0 m, slab 0.2 m → full band 2.8 m
    s0.add_wall((0, 0), (8, 0), thickness_m=0.4)  # full-height
    s0.add_wall((0, 0), (0, 6), thickness_m=0.4, height_m=1.1)  # half-wall
    fillers = corner_filler_polys(s0, _QPQ, _SNAP)
    assert len(fillers) == 1
    _, band = fillers[0]
    assert np.isclose(band, 1.1)  # shortest incident band, not the 2.8 m wall


def test_corner_fillers_deterministic():
    f1 = corner_filler_polys(_box().storeys[0], _QPQ, _SNAP)
    f2 = corner_filler_polys(_box().storeys[0], _QPQ, _SNAP)
    assert len(f1) == len(f2)
    for (h1, b1), (h2, b2) in zip(f1, f2, strict=True):
        assert np.array_equal(h1, h2) and b1 == b2
