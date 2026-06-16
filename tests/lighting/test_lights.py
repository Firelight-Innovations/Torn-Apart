"""
tests/lighting/test_lights.py — SpotLight packing + OccluderSet registry.

Headless (no panda3d).  The pack layouts here are the CPU↔GPU contract with
``lighting/glsl.py`` (INJECT light loop + boxVis), so the assertions pin
exact row/column semantics.

Relocated from tests/test_dynamic_lights.py.
"""

from __future__ import annotations

import math

import numpy as np

from fire_engine.lighting.lights import (
    LIGHT_TYPE_AREA,
    LIGHT_TYPE_POINT,
    LIGHT_TYPE_SPOT,
    MAX_OCCLUDERS,
    AreaLight,
    LightSet,
    OccluderSet,
    PointLight,
    SpotLight,
)


class TestSpotLight:
    def test_pack_layout(self):
        ls = LightSet()
        ls.add(
            SpotLight(
                position=(1.0, 2.0, 3.0),
                direction=(0.0, 0.0, -2.0),
                color=(1.0, 0.5, 0.25),
                intensity=10.0,
                radius=30.0,
                cone_deg=60.0,
            )
        )
        arr, count = ls.pack(max_lights=4)
        assert count == 1
        row = arr[0]
        assert tuple(row[0:3]) == (1.0, 2.0, 3.0)
        assert row[3] == 30.0
        assert np.allclose(row[4:7], (10.0, 5.0, 2.5))
        assert row[7] == LIGHT_TYPE_SPOT
        # Direction normalised at pack time.
        assert np.allclose(row[8:11], (0.0, 0.0, -1.0))
        assert math.isclose(float(row[11]), math.cos(math.radians(30.0)), rel_tol=1e-6)

    def test_zero_direction_defaults_down(self):
        ls = LightSet()
        ls.add(
            SpotLight(
                position=(0, 0, 0),
                direction=(0.0, 0.0, 0.0),
                color=(1, 1, 1),
                intensity=1.0,
                radius=5.0,
            )
        )
        arr, _ = ls.pack(4)
        assert np.allclose(arr[0, 8:11], (0.0, 0.0, -1.0))

    def test_types_distinct(self):
        ls = LightSet()
        ls.add(PointLight((0, 0, 0), (1, 1, 1), 1.0, 5.0))
        ls.add(AreaLight((0, 0, 0), (1, 1, 1), (1, 1, 1), 1.0, 5.0))
        ls.add(SpotLight((0, 0, 0), (0, 1, 0), (1, 1, 1), 1.0, 5.0))
        arr, count = ls.pack(8)
        assert count == 3
        assert set(arr[:3, 7].tolist()) == {LIGHT_TYPE_POINT, LIGHT_TYPE_AREA, LIGHT_TYPE_SPOT}

    def test_notify_changed_bumps_version(self):
        ls = LightSet()
        lid = ls.add(SpotLight((0, 0, 0), (0, 1, 0), (1, 1, 1), 1.0, 5.0))
        v = ls.version
        light = ls.get(lid)
        light.position = (1.0, 0.0, 0.0)
        ls.notify_changed()
        assert ls.version == v + 1
        arr, _ = ls.pack(4)
        assert tuple(arr[0, 0:3]) == (1.0, 0.0, 0.0)  # pack reads live state

    def test_get_unknown_returns_none(self):
        assert LightSet().get(42) is None


class TestOccluderSet:
    def test_pack_layout(self):
        occ = OccluderSet()
        changed = occ.set_boxes([((0.0, 1.0, 2.0), (3.0, 4.0, 5.0))])
        assert changed
        mins, maxs, count = occ.pack()
        assert count == 1
        assert mins.shape == (MAX_OCCLUDERS, 3)
        assert tuple(mins[0]) == (0.0, 1.0, 2.0)
        assert tuple(maxs[0]) == (3.0, 4.0, 5.0)
        assert (mins[1:] == 0).all()

    def test_change_detection(self):
        occ = OccluderSet()
        box = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        assert occ.set_boxes([box])
        v = occ.version
        # Same boxes (and sub-cm jitter) → no version bump.
        assert not occ.set_boxes([box])
        assert not occ.set_boxes([((0.001, 0.0, 0.0), (1.0, 1.0, 1.0))])
        assert occ.version == v
        # A real move bumps.
        assert occ.set_boxes([((0.5, 0.0, 0.0), (1.5, 1.0, 1.0))])
        assert occ.version == v + 1

    def test_empty_and_clear(self):
        occ = OccluderSet()
        assert not occ.set_boxes([])  # empty → empty: unchanged
        occ.set_boxes([((0, 0, 0), (1, 1, 1))])
        assert occ.set_boxes([])  # removing a box = change
        assert occ.count == 0

    def test_overflow_drops_extras(self):
        occ = OccluderSet()
        boxes = [
            ((float(i), 0.0, 0.0), (float(i) + 1.0, 1.0, 1.0)) for i in range(MAX_OCCLUDERS + 5)
        ]
        occ.set_boxes(boxes)
        _, _, count = occ.pack()
        assert count == MAX_OCCLUDERS
