"""
tests/lighting/_impl/test_occluder_set.py — Headless tests for
fire_engine.lighting._impl.occluder_set.

Covers:
- MAX_OCCLUDERS constant.
- OccluderSet: set_boxes, pack layout, change detection, overflow, clear.

No panda3d imports.
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.lighting._impl.occluder_set import MAX_OCCLUDERS, OccluderSet


class TestConstants:
    def test_max_occluders_is_positive_int(self):
        assert isinstance(MAX_OCCLUDERS, int)
        assert MAX_OCCLUDERS > 0

    def test_max_occluders_value_is_16(self):
        """Pin the documented value (mirrors the GPU uniform array length)."""
        assert MAX_OCCLUDERS == 16


class TestOccluderSetInit:
    def test_initial_version_is_zero(self):
        assert OccluderSet().version == 0

    def test_initial_count_is_zero(self):
        assert OccluderSet().count == 0

    def test_pack_empty_count(self):
        _mins, _maxs, count = OccluderSet().pack()
        assert count == 0

    def test_pack_shapes(self):
        mins, maxs, _count = OccluderSet().pack()
        assert mins.shape == (MAX_OCCLUDERS, 3)
        assert maxs.shape == (MAX_OCCLUDERS, 3)
        assert mins.dtype == np.float32
        assert maxs.dtype == np.float32


class TestSetBoxes:
    def test_first_set_returns_true(self):
        occ = OccluderSet()
        changed = occ.set_boxes([((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])
        assert changed is True

    def test_count_updated(self):
        occ = OccluderSet()
        occ.set_boxes([((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])
        assert occ.count == 1

    def test_version_bumped_on_first_set(self):
        occ = OccluderSet()
        v = occ.version
        occ.set_boxes([((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])
        assert occ.version == v + 1

    def test_same_boxes_no_version_bump(self):
        occ = OccluderSet()
        box = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        occ.set_boxes([box])
        v = occ.version
        changed = occ.set_boxes([box])
        assert not changed
        assert occ.version == v

    def test_sub_cm_jitter_no_version_bump(self):
        occ = OccluderSet()
        occ.set_boxes([((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])
        v = occ.version
        changed = occ.set_boxes([((0.001, 0.0, 0.0), (1.0, 1.0, 1.0))])
        assert not changed
        assert occ.version == v

    def test_real_move_bumps_version(self):
        occ = OccluderSet()
        occ.set_boxes([((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])
        v = occ.version
        occ.set_boxes([((0.5, 0.0, 0.0), (1.5, 1.0, 1.0))])
        assert occ.version == v + 1

    def test_empty_to_empty_no_change(self):
        occ = OccluderSet()
        changed = occ.set_boxes([])
        assert not changed

    def test_remove_box_is_a_change(self):
        occ = OccluderSet()
        occ.set_boxes([((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])
        changed = occ.set_boxes([])
        assert changed
        assert occ.count == 0


class TestPack:
    def test_pack_layout(self):
        occ = OccluderSet()
        occ.set_boxes([((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))])
        mins, maxs, count = occ.pack()
        assert count == 1
        np.testing.assert_allclose(mins[0], (1.0, 2.0, 3.0))
        np.testing.assert_allclose(maxs[0], (4.0, 5.0, 6.0))
        # Rows past count must be zero.
        assert (mins[1:] == 0).all()
        assert (maxs[1:] == 0).all()

    def test_overflow_drops_extras(self):
        occ = OccluderSet()
        boxes = [
            ((float(i), 0.0, 0.0), (float(i) + 1.0, 1.0, 1.0)) for i in range(MAX_OCCLUDERS + 5)
        ]
        occ.set_boxes(boxes)
        _, _, count = occ.pack()
        assert count == MAX_OCCLUDERS

    def test_multiple_boxes_pack_correctly(self):
        occ = OccluderSet()
        boxes = [((float(i), 0.0, 0.0), (float(i) + 1.0, 1.0, 1.0)) for i in range(3)]
        occ.set_boxes(boxes)
        mins, maxs, count = occ.pack()
        assert count == 3
        for i in range(3):
            assert mins[i, 0] == pytest.approx(float(i))
            assert maxs[i, 0] == pytest.approx(float(i) + 1.0)
