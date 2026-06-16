"""
tests/world/terrain/test_types.py — SphereBrush, BoxBrush, CylinderBrush correctness.
Headless: no panda3d imports.
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.core.math3d import Vec3
from fire_engine.world.terrain.types import BoxBrush, Brush, CylinderBrush, SphereBrush


class TestSphereBrush:
    def test_frozen_dataclass(self):
        b = SphereBrush(radius_m=2.5)
        assert b.radius_m == pytest.approx(2.5)
        with pytest.raises((AttributeError, TypeError)):
            b.radius_m = 99.0  # type: ignore[misc]

    def test_aabb_symmetric(self):
        b = SphereBrush(radius_m=3.0)
        center = np.array([0.0, 0.0, 0.0])
        lo, hi = b.aabb(center)
        np.testing.assert_allclose(lo, [-3.0, -3.0, -3.0])
        np.testing.assert_allclose(hi, [3.0, 3.0, 3.0])

    def test_aabb_offset_center(self):
        b = SphereBrush(radius_m=1.0)
        center = np.array([5.0, 10.0, -2.0])
        lo, hi = b.aabb(center)
        np.testing.assert_allclose(lo, [4.0, 9.0, -3.0])
        np.testing.assert_allclose(hi, [6.0, 11.0, -1.0])

    def test_mask_center_voxel_inside(self):
        b = SphereBrush(radius_m=2.0)
        center = np.array([0.0, 0.0, 0.0])
        X = np.array([0.0])
        Y = np.array([0.0])
        Z = np.array([0.0])
        mask = b.mask(X, Y, Z, center)
        assert mask[0] is np.True_

    def test_mask_voxel_outside(self):
        b = SphereBrush(radius_m=1.0)
        center = np.array([0.0, 0.0, 0.0])
        X = np.array([2.0])
        Y = np.array([0.0])
        Z = np.array([0.0])
        mask = b.mask(X, Y, Z, center)
        assert mask[0] is np.False_

    def test_mask_boundary_on_sphere(self):
        """A point exactly on the sphere surface (r^2 == r^2) is included."""
        b = SphereBrush(radius_m=1.0)
        center = np.array([0.0, 0.0, 0.0])
        # distance == radius exactly
        X = np.array([1.0])
        Y = np.array([0.0])
        Z = np.array([0.0])
        mask = b.mask(X, Y, Z, center)
        assert mask[0] is np.True_

    def test_mask_vectorised_grid(self):
        """Vectorised grid: all points within radius 1 of origin."""
        b = SphereBrush(radius_m=1.0)
        center = np.array([0.0, 0.0, 0.0])
        coords = np.array([-1.0, 0.0, 1.0])
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
        mask = b.mask(X.ravel(), Y.ravel(), Z.ravel(), center)
        # Only points where x^2+y^2+z^2 <= 1 are True
        expected = (X.ravel() ** 2 + Y.ravel() ** 2 + Z.ravel() ** 2) <= 1.0
        np.testing.assert_array_equal(mask, expected)


class TestBoxBrush:
    def test_frozen_dataclass(self):
        b = BoxBrush(half_extents_m=Vec3(1.0, 2.0, 3.0))
        assert b.half_extents_m.x == pytest.approx(1.0)
        with pytest.raises((AttributeError, TypeError)):
            b.half_extents_m = Vec3(0.0, 0.0, 0.0)  # type: ignore[misc]

    def test_aabb_matches_half_extents(self):
        b = BoxBrush(half_extents_m=Vec3(1.0, 2.0, 3.0))
        center = np.array([0.0, 0.0, 0.0])
        lo, hi = b.aabb(center)
        np.testing.assert_allclose(lo, [-1.0, -2.0, -3.0])
        np.testing.assert_allclose(hi, [1.0, 2.0, 3.0])

    def test_mask_center_inside(self):
        b = BoxBrush(half_extents_m=Vec3(2.0, 2.0, 2.0))
        center = np.array([0.0, 0.0, 0.0])
        mask = b.mask(np.array([0.0]), np.array([0.0]), np.array([0.0]), center)
        assert mask[0] is np.True_

    def test_mask_corner_on_boundary(self):
        """Point at exactly the half-extent corner is inside (<=)."""
        b = BoxBrush(half_extents_m=Vec3(1.0, 1.0, 1.0))
        center = np.array([0.0, 0.0, 0.0])
        mask = b.mask(np.array([1.0]), np.array([1.0]), np.array([1.0]), center)
        assert mask[0] is np.True_

    def test_mask_outside(self):
        b = BoxBrush(half_extents_m=Vec3(1.0, 1.0, 1.0))
        center = np.array([0.0, 0.0, 0.0])
        mask = b.mask(np.array([1.5]), np.array([0.0]), np.array([0.0]), center)
        assert mask[0] is np.False_

    def test_mask_anisotropic(self):
        """Non-uniform half-extents: wide in X, narrow in Y."""
        b = BoxBrush(half_extents_m=Vec3(5.0, 0.5, 5.0))
        center = np.array([0.0, 0.0, 0.0])
        inside = b.mask(np.array([4.0]), np.array([0.4]), np.array([0.0]), center)
        outside = b.mask(np.array([4.0]), np.array([1.0]), np.array([0.0]), center)
        assert inside[0] is np.True_
        assert outside[0] is np.False_


class TestCylinderBrush:
    def test_frozen_dataclass(self):
        b = CylinderBrush(radius_m=1.5, height_m=4.0)
        assert b.radius_m == pytest.approx(1.5)
        assert b.height_m == pytest.approx(4.0)
        with pytest.raises((AttributeError, TypeError)):
            b.radius_m = 0.0  # type: ignore[misc]

    def test_aabb_shape(self):
        b = CylinderBrush(radius_m=2.0, height_m=6.0)
        center = np.array([0.0, 0.0, 0.0])
        lo, hi = b.aabb(center)
        np.testing.assert_allclose(lo, [-2.0, -2.0, -3.0])
        np.testing.assert_allclose(hi, [2.0, 2.0, 3.0])

    def test_mask_center_inside(self):
        b = CylinderBrush(radius_m=1.0, height_m=2.0)
        center = np.array([0.0, 0.0, 0.0])
        mask = b.mask(np.array([0.0]), np.array([0.0]), np.array([0.0]), center)
        assert mask[0] is np.True_

    def test_mask_radially_outside(self):
        b = CylinderBrush(radius_m=1.0, height_m=4.0)
        center = np.array([0.0, 0.0, 0.0])
        mask = b.mask(np.array([1.5]), np.array([0.0]), np.array([0.0]), center)
        assert mask[0] is np.False_

    def test_mask_vertically_outside(self):
        b = CylinderBrush(radius_m=5.0, height_m=2.0)
        center = np.array([0.0, 0.0, 0.0])
        mask = b.mask(np.array([0.0]), np.array([0.0]), np.array([1.5]), center)
        assert mask[0] is np.False_

    def test_mask_on_radial_boundary(self):
        """Point at exactly radius distance is inside (<= comparison)."""
        b = CylinderBrush(radius_m=1.0, height_m=4.0)
        center = np.array([0.0, 0.0, 0.0])
        mask = b.mask(np.array([1.0]), np.array([0.0]), np.array([0.0]), center)
        assert mask[0] is np.True_

    def test_mask_on_vertical_boundary(self):
        """Point at exactly half_height is inside (<= comparison)."""
        b = CylinderBrush(radius_m=5.0, height_m=2.0)
        center = np.array([0.0, 0.0, 0.0])
        mask = b.mask(np.array([0.0]), np.array([0.0]), np.array([1.0]), center)
        assert mask[0] is np.True_


class TestBrushTypeAlias:
    def test_sphere_is_brush(self):
        """SphereBrush is part of the Brush union type."""
        b: Brush = SphereBrush(1.0)
        assert isinstance(b, SphereBrush)

    def test_box_is_brush(self):
        b: Brush = BoxBrush(Vec3(1.0, 1.0, 1.0))
        assert isinstance(b, BoxBrush)

    def test_cylinder_is_brush(self):
        b: Brush = CylinderBrush(1.0, 2.0)
        assert isinstance(b, CylinderBrush)
