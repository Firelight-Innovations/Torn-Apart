"""
tests/core/test_events.py — Mirror test for fire_engine/core/events.py.

Covers:
- All event dataclasses are frozen (immutable)
- Field names and types match documented attributes
- Instances compare equal with same field values
- __all__ exports all expected event classes
- Instances can be used as dict keys / in sets (hashable via frozen dataclass)
"""

from __future__ import annotations

import dataclasses

import pytest

from fire_engine.core.events import (
    BuildingChangedEvent,
    ChunkLoadedEvent,
    ChunkUnloadedEvent,
    GameDayTickEvent,
    LightningStrikeEvent,
    TerrainEditedEvent,
    ThunderEvent,
    WeatherChangedEvent,
)


class TestChunkLoadedEvent:
    def test_construction(self):
        ev = ChunkLoadedEvent(coord=(1, 2, 3))
        assert ev.coord == (1, 2, 3)

    def test_frozen(self):
        ev = ChunkLoadedEvent(coord=(0, 0, 0))
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.coord = (1, 1, 1)  # type: ignore[misc]

    def test_equality(self):
        assert ChunkLoadedEvent(coord=(1, 2, 3)) == ChunkLoadedEvent(coord=(1, 2, 3))
        assert ChunkLoadedEvent(coord=(0, 0, 0)) != ChunkLoadedEvent(coord=(1, 0, 0))

    def test_hashable(self):
        """Frozen dataclasses must be hashable for use in sets."""
        s = {ChunkLoadedEvent(coord=(0, 0, 0)), ChunkLoadedEvent(coord=(1, 0, 0))}
        assert len(s) == 2


class TestChunkUnloadedEvent:
    def test_construction(self):
        ev = ChunkUnloadedEvent(coord=(5, 6, 7))
        assert ev.coord == (5, 6, 7)

    def test_frozen(self):
        ev = ChunkUnloadedEvent(coord=(0, 0, 0))
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.coord = (1, 1, 1)  # type: ignore[misc]

    def test_equality(self):
        assert ChunkUnloadedEvent(coord=(1, 2, 3)) == ChunkUnloadedEvent(coord=(1, 2, 3))


class TestTerrainEditedEvent:
    def test_construction(self):
        class FakeBrush:
            pass

        brush = FakeBrush()
        ev = TerrainEditedEvent(chunk_coords=(0, 0, 0), brush=brush)
        assert ev.chunk_coords == (0, 0, 0)
        assert ev.brush is brush

    def test_frozen(self):
        ev = TerrainEditedEvent(chunk_coords=(0,), brush=object())
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.chunk_coords = (1,)  # type: ignore[misc]


class TestWeatherChangedEvent:
    def test_construction(self):
        ev = WeatherChangedEvent(previous="clear", current="rain", day=3)
        assert ev.previous == "clear"
        assert ev.current == "rain"
        assert ev.day == 3

    def test_frozen(self):
        ev = WeatherChangedEvent(previous="clear", current="rain", day=0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.day = 99  # type: ignore[misc]

    def test_equality(self):
        a = WeatherChangedEvent(previous="clear", current="rain", day=1)
        b = WeatherChangedEvent(previous="clear", current="rain", day=1)
        assert a == b

    def test_hashable(self):
        s = {
            WeatherChangedEvent(previous="clear", current="rain", day=1),
            WeatherChangedEvent(previous="rain", current="clear", day=2),
        }
        assert len(s) == 2


class TestLightningStrikeEvent:
    def test_construction(self):
        ev = LightningStrikeEvent(
            pos=(10.0, 20.0, 220.0),
            ground_pos=(10.0, 20.0, 8.0),
            seed=42,
            time_abs=3600.0,
            cell_id=7,
            intensity=0.9,
        )
        assert ev.pos == (10.0, 20.0, 220.0)
        assert ev.ground_pos == (10.0, 20.0, 8.0)
        assert ev.seed == 42
        assert ev.time_abs == 3600.0
        assert ev.cell_id == 7
        assert ev.intensity == pytest.approx(0.9)

    def test_frozen(self):
        ev = LightningStrikeEvent(
            pos=(0.0, 0.0, 0.0),
            ground_pos=(0.0, 0.0, 0.0),
            seed=1,
            time_abs=0.0,
            cell_id=0,
            intensity=1.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.seed = 99  # type: ignore[misc]


class TestThunderEvent:
    def test_construction(self):
        ev = ThunderEvent(
            pos=(10.0, 20.0, 220.0),
            distance_m=343.0,
            delay_s=1.0,
            time_abs=3600.0,
            intensity=0.8,
        )
        assert ev.distance_m == pytest.approx(343.0)
        assert ev.delay_s == pytest.approx(1.0)
        assert ev.intensity == pytest.approx(0.8)

    def test_delay_equals_distance_over_speed_of_sound(self):
        """delay_s should equal distance_m / 343 (caller's responsibility — pin the formula)."""
        distance = 686.0  # 2 * 343
        delay = distance / 343.0
        ev = ThunderEvent(
            pos=(0.0, 0.0, 0.0),
            distance_m=distance,
            delay_s=delay,
            time_abs=0.0,
            intensity=1.0,
        )
        assert ev.delay_s == pytest.approx(2.0)


class TestGameDayTickEvent:
    def test_construction(self):
        ev = GameDayTickEvent(day=5)
        assert ev.day == 5

    def test_frozen(self):
        ev = GameDayTickEvent(day=0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.day = 1  # type: ignore[misc]

    def test_hashable(self):
        s = {GameDayTickEvent(day=0), GameDayTickEvent(day=1), GameDayTickEvent(day=0)}
        assert len(s) == 2


class TestBuildingChangedEvent:
    def test_construction_added(self):
        ev = BuildingChangedEvent(
            building_id=1,
            change="added",
            bounds_min=(0.0, 0.0, 0.0),
            bounds_max=(10.0, 10.0, 5.0),
        )
        assert ev.building_id == 1
        assert ev.change == "added"
        assert ev.bounds_min == (0.0, 0.0, 0.0)
        assert ev.bounds_max == (10.0, 10.0, 5.0)

    def test_valid_change_strings(self):
        """Each documented change kind must construct without error."""
        for kind in ("added", "modified", "removed"):
            ev = BuildingChangedEvent(
                building_id=0,
                change=kind,
                bounds_min=(0.0, 0.0, 0.0),
                bounds_max=(1.0, 1.0, 1.0),
            )
            assert ev.change == kind

    def test_frozen(self):
        ev = BuildingChangedEvent(
            building_id=0,
            change="added",
            bounds_min=(0.0, 0.0, 0.0),
            bounds_max=(1.0, 1.0, 1.0),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.change = "removed"  # type: ignore[misc]


class TestAllExports:
    """Verify __all__ contains every expected event class."""

    def test_all_exports(self):
        import fire_engine.core.events as mod

        expected = {
            "BuildingChangedEvent",
            "ChunkLoadedEvent",
            "ChunkUnloadedEvent",
            "GameDayTickEvent",
            "LightningStrikeEvent",
            "TerrainEditedEvent",
            "ThunderEvent",
            "WeatherChangedEvent",
        }
        assert expected == set(mod.__all__)

    def test_all_exports_importable(self):
        import fire_engine.core.events as mod

        for name in mod.__all__:
            assert hasattr(mod, name), f"__all__ entry {name!r} not found in module"
