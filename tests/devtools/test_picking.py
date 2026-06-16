"""
tests/devtools/test_picking.py — tests for fire_engine/devtools/picking.py.

Covers ray_aabb, Selectable.world_aabb, and pick(). All headless; no panda3d imports.
Uses a minimal duck-typed stand-in for GameObject/Transform so no render/ imports
are needed at test setup time.
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.core.math3d import Vec3
from fire_engine.devtools.picking import Selectable, pick, ray_aabb

# ---------------------------------------------------------------------------
# Minimal duck-typed stand-ins (no panda3d)
# ---------------------------------------------------------------------------


_UNIT_SCALE = Vec3(1, 1, 1)


class _FakeTransform:
    def __init__(self, pos: Vec3, scale: Vec3 | None = None) -> None:
        self.position = pos
        self.local_scale = scale if scale is not None else _UNIT_SCALE


class _FakeGO:
    def __init__(self, name: str, pos: Vec3, scale: Vec3 | None = None) -> None:
        self.name = name
        self.transform = _FakeTransform(pos, scale)


# ---------------------------------------------------------------------------
# ray_aabb
# ---------------------------------------------------------------------------


class TestRayAabb:
    def test_hit_along_y_axis(self):
        bmin = np.array([-1.0, -1.0, -1.0])
        bmax = np.array([1.0, 1.0, 1.0])
        t = ray_aabb(Vec3(0, -5, 0), Vec3(0, 1, 0), bmin, bmax)
        assert t is not None
        assert abs(t - 4.0) < 1e-6

    def test_miss_pointing_away(self):
        bmin = np.array([-1.0, -1.0, -1.0])
        bmax = np.array([1.0, 1.0, 1.0])
        assert ray_aabb(Vec3(0, -5, 0), Vec3(0, -1, 0), bmin, bmax) is None

    def test_miss_parallel_offset(self):
        bmin = np.array([-1.0, -1.0, -1.0])
        bmax = np.array([1.0, 1.0, 1.0])
        assert ray_aabb(Vec3(5, -5, 0), Vec3(0, 1, 0), bmin, bmax) is None

    def test_origin_inside_box_returns_zero(self):
        bmin = np.array([-1.0, -1.0, -1.0])
        bmax = np.array([1.0, 1.0, 1.0])
        t = ray_aabb(Vec3(0, 0, 0), Vec3(0, 1, 0), bmin, bmax)
        assert t == 0.0

    def test_tangent_ray_hits_at_corner(self):
        bmin = np.array([0.0, 0.0, 0.0])
        bmax = np.array([2.0, 2.0, 2.0])
        # Ray along Y=0,Z=0 (the edge of the box)
        t = ray_aabb(Vec3(0, -5, 0), Vec3(0, 1, 0), bmin, bmax)
        # Hits at the near face (y=0 slab) from y=-5; t=5
        assert t is not None
        assert t == pytest.approx(5.0, abs=1e-6)

    def test_deterministic(self):
        bmin = np.array([-1.0, -1.0, -1.0])
        bmax = np.array([1.0, 1.0, 1.0])
        t1 = ray_aabb(Vec3(0, -5, 0), Vec3(0, 1, 0), bmin, bmax)
        t2 = ray_aabb(Vec3(0, -5, 0), Vec3(0, 1, 0), bmin, bmax)
        assert t1 == t2


# ---------------------------------------------------------------------------
# Selectable.world_aabb
# ---------------------------------------------------------------------------


class TestSelectableWorldAabb:
    def test_unit_extents_at_origin(self):
        go = _FakeGO("box", Vec3(0, 0, 0))
        sel = Selectable(go, Vec3(0.5, 0.5, 0.5))  # type: ignore[arg-type]
        bmin, bmax = sel.world_aabb()
        assert np.allclose(bmin, [-0.5, -0.5, -0.5])
        assert np.allclose(bmax, [0.5, 0.5, 0.5])

    def test_follows_position(self):
        go = _FakeGO("box", Vec3(10, 0, 0))
        sel = Selectable(go, Vec3(0.5, 0.5, 0.5))  # type: ignore[arg-type]
        bmin, bmax = sel.world_aabb()
        # center (10,0,0), half-extents 0.5*scale(1,1,1) = 0.5 each axis
        assert np.allclose(bmin, [9.5, -0.5, -0.5])
        assert np.allclose(bmax, [10.5, 0.5, 0.5])

    def test_scales_half_extents(self):
        go = _FakeGO("box", Vec3(10, 0, 0), scale=Vec3(2, 2, 2))
        sel = Selectable(go, Vec3(0.5, 0.5, 0.5))  # type: ignore[arg-type]
        bmin, bmax = sel.world_aabb()
        assert np.allclose(bmin, [9.0, -1.0, -1.0])
        assert np.allclose(bmax, [11.0, 1.0, 1.0])

    def test_negative_scale_uses_abs(self):
        go = _FakeGO("box", Vec3(0, 0, 0), scale=Vec3(-2, 1, 1))
        sel = Selectable(go, Vec3(1.0, 1.0, 1.0))  # type: ignore[arg-type]
        bmin, bmax = sel.world_aabb()
        # abs(-2)*1.0 = 2.0 on X
        assert np.allclose(bmin, [-2.0, -1.0, -1.0])
        assert np.allclose(bmax, [2.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# pick function
# ---------------------------------------------------------------------------


class TestPick:
    def test_returns_nearest_object(self):
        near = _FakeGO("near", Vec3(0, 5, 0))
        far = _FakeGO("far", Vec3(0, 20, 0))
        sels = [
            Selectable(far, Vec3(1, 1, 1)),  # type: ignore[arg-type]
            Selectable(near, Vec3(1, 1, 1)),  # type: ignore[arg-type]
        ]
        hit = pick(Vec3(0, 0, 0), Vec3(0, 1, 0), sels)
        assert hit is near

    def test_returns_none_on_full_miss(self):
        go = _FakeGO("box", Vec3(0, 5, 0))
        sels = [Selectable(go, Vec3(1, 1, 1))]  # type: ignore[arg-type]
        assert pick(Vec3(50, 0, 0), Vec3(0, 1, 0), sels) is None

    def test_empty_selectables_returns_none(self):
        assert pick(Vec3(0, 0, 0), Vec3(0, 1, 0), []) is None

    def test_single_hit(self):
        go = _FakeGO("only", Vec3(0, 5, 0))
        sels = [Selectable(go, Vec3(1, 1, 1))]  # type: ignore[arg-type]
        hit = pick(Vec3(0, 0, 0), Vec3(0, 1, 0), sels)
        assert hit is go
