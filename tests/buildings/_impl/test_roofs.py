"""
tests/buildings/_impl/test_roofs.py — pitched-roof geometry (gable/hip/shed).

Verifies the flat roof is unchanged, each pitched kind raises a ridge above the
wall top, shed slopes monotonically across the span, generation is
deterministic, and rotating the ridge direction rotates the geometry.

Headless (numpy only — buildings/ never imports panda3d).
"""

import math

import numpy as np

from fire_engine.buildings._impl.roofs import add_roof
from fire_engine.buildings._impl.soup import Soup
from fire_engine.buildings.enums import RoofKind
from fire_engine.buildings.types import RoofSlab

_RECT = np.array([[0, 0], [10, 0], [10, 6], [0, 6]], dtype=float)
_TOP = 3.0


def _mesh(kind, **kw):
    roof = RoofSlab(polygon=_RECT, thickness_m=0.2, kind=kind, **kw)
    soup = Soup()
    add_roof(soup, roof, _TOP, qpq=8)
    return soup.build()


def test_flat_roof_is_a_slab_at_top():
    mesh = _mesh(RoofKind.FLAT)
    assert mesh.tri_count == 12  # box slab
    assert np.isclose(mesh.positions[:, 2].min(), _TOP)
    assert np.isclose(mesh.positions[:, 2].max(), _TOP + 0.2)


def test_pitched_kinds_raise_a_ridge():
    for kind in (RoofKind.SHED, RoofKind.GABLE, RoofKind.HIP):
        mesh = _mesh(kind, pitch_deg=35.0)
        assert not mesh.is_empty
        assert mesh.positions[:, 2].max() > _TOP + 0.5, kind


def test_gable_ridge_height_matches_pitch():
    # Ridge over a 6 m span at 30°: rise = 3 m * tan(30) = 1.732 m.
    mesh = _mesh(RoofKind.GABLE, pitch_deg=30.0)
    expected = _TOP + 3.0 * math.tan(math.radians(30.0))
    assert np.isclose(mesh.positions[:, 2].max(), expected, atol=1e-6)


def test_shed_low_and_high_eaves_differ():
    mesh = _mesh(RoofKind.SHED, pitch_deg=25.0)
    full = 6.0 * math.tan(math.radians(25.0))
    assert np.isclose(mesh.positions[:, 2].max() - _TOP, full, atol=1e-6)
    # The low-eave *top surface* sits at the wall top (the prism underside dips
    # thickness below it).
    assert np.any(np.isclose(mesh.positions[:, 2], _TOP, atol=1e-6))


def test_roof_is_deterministic():
    a = _mesh(RoofKind.HIP, pitch_deg=40.0, overhang_m=0.5)
    b = _mesh(RoofKind.HIP, pitch_deg=40.0, overhang_m=0.5)
    assert np.array_equal(a.positions, b.positions)
    assert np.array_equal(a.normals, b.normals)


def test_ridge_direction_rotates_geometry():
    # Square footprint so the span (and ridge height) is rotation-invariant;
    # only the plan orientation changes.
    sq = np.array([[0, 0], [6, 0], [6, 6], [0, 6]], dtype=float)

    def mesh(theta):
        roof = RoofSlab(polygon=sq, thickness_m=0.2, kind=RoofKind.GABLE, ridge_dir_rad=theta)
        soup = Soup()
        add_roof(soup, roof, _TOP, qpq=8)
        return soup.build()

    straight = mesh(0.0)
    turned = mesh(math.pi / 2)
    assert np.isclose(straight.positions[:, 2].max(), turned.positions[:, 2].max(), atol=1e-6)
    assert not np.allclose(straight.positions[:, 0], turned.positions[:, 0])
