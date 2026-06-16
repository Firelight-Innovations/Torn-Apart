"""
tests/buildings/test_enums.py — correctness tests for buildings/enums.py.

Verifies enum members, values, and identity behaviour.
Headless (no panda3d).
"""

from __future__ import annotations

from fire_engine.buildings.enums import OpeningKind, WallKind


class TestWallKind:
    def test_segment_member_exists(self):
        assert WallKind.SEGMENT is WallKind.SEGMENT

    def test_arc_member_exists(self):
        assert WallKind.ARC is WallKind.ARC

    def test_segment_value(self):
        assert WallKind.SEGMENT.value == "segment"

    def test_arc_value(self):
        assert WallKind.ARC.value == "arc"

    def test_only_two_members(self):
        assert len(list(WallKind)) == 2

    def test_members_are_distinct(self):
        assert WallKind.SEGMENT is not WallKind.ARC

    def test_round_trip_via_value(self):
        for member in WallKind:
            assert WallKind(member.value) is member

    def test_is_enum(self):
        import enum

        assert isinstance(WallKind.SEGMENT, WallKind)
        assert issubclass(WallKind, enum.Enum)


class TestOpeningKind:
    def test_window_member_exists(self):
        assert OpeningKind.WINDOW is OpeningKind.WINDOW

    def test_door_member_exists(self):
        assert OpeningKind.DOOR is OpeningKind.DOOR

    def test_window_value(self):
        assert OpeningKind.WINDOW.value == "window"

    def test_door_value(self):
        assert OpeningKind.DOOR.value == "door"

    def test_only_two_members(self):
        assert len(list(OpeningKind)) == 2

    def test_members_are_distinct(self):
        assert OpeningKind.WINDOW is not OpeningKind.DOOR

    def test_round_trip_via_value(self):
        for member in OpeningKind:
            assert OpeningKind(member.value) is member

    def test_is_enum(self):
        import enum

        assert isinstance(OpeningKind.WINDOW, OpeningKind)
        assert issubclass(OpeningKind, enum.Enum)


class TestRoofKind:
    def test_has_four_members(self):
        from fire_engine.buildings.enums import RoofKind

        assert {k.value for k in RoofKind} == {"flat", "shed", "gable", "hip"}

    def test_round_trip_via_value(self):
        from fire_engine.buildings.enums import RoofKind

        for member in RoofKind:
            assert RoofKind(member.value) is member

    def test_default_is_flat(self):
        from fire_engine.buildings.enums import RoofKind

        assert RoofKind.FLAT.value == "flat"


class TestCrossEnumDistinction:
    def test_segment_and_window_are_different_types(self):
        # Enum members from different enums must not compare equal.
        assert WallKind.SEGMENT != OpeningKind.WINDOW
        assert WallKind.ARC != OpeningKind.DOOR

    def test_enums_exported_from_module(self):
        import fire_engine.buildings.enums as m

        assert hasattr(m, "WallKind")
        assert hasattr(m, "OpeningKind")
        assert hasattr(m, "RoofKind")
