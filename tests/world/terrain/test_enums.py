"""
tests/world/terrain/test_enums.py — BrushMode enum correctness.
Headless: no panda3d imports.
"""

from __future__ import annotations

from fire_engine.world.terrain.enums import BrushMode


class TestBrushMode:
    def test_members_exist(self):
        """ADD and REMOVE must be present."""
        assert BrushMode.ADD is not None
        assert BrushMode.REMOVE is not None

    def test_values(self):
        """String values match the documented constants."""
        assert BrushMode.ADD.value == "add"
        assert BrushMode.REMOVE.value == "remove"

    def test_from_value(self):
        """Enum can be constructed by value string (BrushMode('add'))."""
        assert BrushMode("add") is BrushMode.ADD
        assert BrushMode("remove") is BrushMode.REMOVE

    def test_distinct_members(self):
        """ADD and REMOVE are different enum members."""
        assert BrushMode.ADD is not BrushMode.REMOVE

    def test_exhaustive_members(self):
        """Exactly two members — ensures no silent extras that break switches."""
        members = list(BrushMode)
        assert len(members) == 2
        assert BrushMode.ADD in members
        assert BrushMode.REMOVE in members

    def test_is_enum(self):
        """BrushMode is a proper Enum subclass."""
        import enum

        assert issubclass(BrushMode, enum.Enum)

    def test_name_strings(self):
        assert BrushMode.ADD.name == "ADD"
        assert BrushMode.REMOVE.name == "REMOVE"
