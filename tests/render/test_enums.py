"""
tests/render/test_enums.py — Headless tests for render/enums.py (Space enum).

No panda3d imports; pure-Python enumeration assertions.
"""

from __future__ import annotations

from fire_engine.render.enums import Space


class TestSpaceEnum:
    """Tests for the Space reference-frame selector enum."""

    def test_members_exist(self) -> None:
        assert hasattr(Space, "SELF")
        assert hasattr(Space, "WORLD")

    def test_exhaustive_member_count(self) -> None:
        """Space must have exactly 2 members — no silent additions."""
        assert len(Space) == 2

    def test_self_and_world_are_distinct(self) -> None:
        assert Space.SELF is not Space.WORLD
        assert Space.SELF != Space.WORLD

    def test_members_are_space_instances(self) -> None:
        assert isinstance(Space.SELF, Space)
        assert isinstance(Space.WORLD, Space)

    def test_values_are_integers(self) -> None:
        """auto() produces int values."""
        assert isinstance(Space.SELF.value, int)
        assert isinstance(Space.WORLD.value, int)

    def test_unique_values(self) -> None:
        assert Space.SELF.value != Space.WORLD.value

    def test_roundtrip_by_value(self) -> None:
        assert Space(Space.SELF.value) is Space.SELF
        assert Space(Space.WORLD.value) is Space.WORLD

    def test_names(self) -> None:
        assert Space.SELF.name == "SELF"
        assert Space.WORLD.name == "WORLD"
