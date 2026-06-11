"""
tests/test_event_bus.py — Tests for core/event_bus.py.

Covers:
  - subscribe/publish delivers to all subscribers
  - unsubscribe stops delivery
  - publish_deferred not delivered until drain()
  - drain() dispatches in FIFO order
  - publish_deferred during drain goes to next drain, not current
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from fire_engine.core.event_bus import (
    EventBus,
    ChunkLoadedEvent,
    ChunkUnloadedEvent,
    TerrainEditedEvent,
    GameDayTickEvent,
)


# ---------------------------------------------------------------------------
# Helper event type for testing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _TestEvent:
    value: int


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSubscribePublish:
    def test_single_subscriber_receives_event(self):
        bus = EventBus()
        received = []
        bus.subscribe(_TestEvent, received.append)
        bus.publish(_TestEvent(value=42))
        assert received == [_TestEvent(value=42)]

    def test_multiple_subscribers_all_receive(self):
        bus = EventBus()
        a, b = [], []
        bus.subscribe(_TestEvent, a.append)
        bus.subscribe(_TestEvent, b.append)
        bus.publish(_TestEvent(value=7))
        assert a == [_TestEvent(value=7)]
        assert b == [_TestEvent(value=7)]

    def test_no_subscriber_no_error(self):
        bus = EventBus()
        bus.publish(_TestEvent(value=0))  # should not raise

    def test_publish_does_not_queue(self):
        """Synchronous publish must deliver immediately, not deferred."""
        bus = EventBus()
        received = []
        bus.subscribe(_TestEvent, received.append)
        bus.publish(_TestEvent(value=1))
        # delivery happens before drain — no drain called
        assert len(received) == 1

    def test_different_event_types_do_not_cross(self):
        bus = EventBus()
        chunk_received = []
        day_received   = []
        bus.subscribe(ChunkLoadedEvent, chunk_received.append)
        bus.subscribe(GameDayTickEvent, day_received.append)
        bus.publish(ChunkLoadedEvent(coord=(0, 0, 0)))
        assert len(chunk_received) == 1
        assert len(day_received)   == 0


class TestUnsubscribe:
    def test_unsubscribe_stops_delivery(self):
        bus = EventBus()
        received = []
        bus.subscribe(_TestEvent, received.append)
        bus.unsubscribe(_TestEvent, received.append)
        bus.publish(_TestEvent(value=5))
        assert received == []

    def test_unsubscribe_nonexistent_is_noop(self):
        bus = EventBus()
        bus.unsubscribe(_TestEvent, lambda e: None)  # should not raise

    def test_unsubscribe_only_removes_one(self):
        """If the same handler is subscribed twice, only one is removed."""
        bus = EventBus()
        received = []
        bus.subscribe(_TestEvent, received.append)
        bus.subscribe(_TestEvent, received.append)  # duplicate
        bus.unsubscribe(_TestEvent, received.append)
        bus.publish(_TestEvent(value=1))
        assert len(received) == 1  # one still registered


class TestPublishDeferred:
    def test_deferred_not_delivered_before_drain(self):
        bus = EventBus()
        received = []
        bus.subscribe(_TestEvent, received.append)
        bus.publish_deferred(_TestEvent(value=10))
        assert received == []  # not delivered yet

    def test_deferred_delivered_on_drain(self):
        bus = EventBus()
        received = []
        bus.subscribe(_TestEvent, received.append)
        bus.publish_deferred(_TestEvent(value=10))
        bus.drain()
        assert received == [_TestEvent(value=10)]

    def test_multiple_deferred_all_delivered(self):
        bus = EventBus()
        received = []
        bus.subscribe(_TestEvent, received.append)
        bus.publish_deferred(_TestEvent(value=1))
        bus.publish_deferred(_TestEvent(value=2))
        bus.publish_deferred(_TestEvent(value=3))
        bus.drain()
        assert received == [_TestEvent(1), _TestEvent(2), _TestEvent(3)]

    def test_drain_fifo_order(self):
        """Events must be dispatched in the order they were enqueued."""
        bus = EventBus()
        order = []
        bus.subscribe(_TestEvent, lambda e: order.append(e.value))
        for i in range(10):
            bus.publish_deferred(_TestEvent(value=i))
        bus.drain()
        assert order == list(range(10))

    def test_drain_clears_queue(self):
        """After drain(), a second drain() delivers nothing."""
        bus = EventBus()
        received = []
        bus.subscribe(_TestEvent, received.append)
        bus.publish_deferred(_TestEvent(value=99))
        bus.drain()
        bus.drain()
        assert len(received) == 1  # not delivered twice


class TestDeferredDuringDrain:
    """
    Handlers that call publish_deferred during a drain() must have their
    events deferred to the *next* drain, not the current one.
    """

    def test_deferred_during_drain_goes_to_next_drain(self):
        bus = EventBus()
        first_pass = []
        second_pass = []

        def handler_first(evt: _TestEvent) -> None:
            first_pass.append(evt.value)
            if evt.value == 1:
                # Enqueue another event during drain
                bus.publish_deferred(_TestEvent(value=2))

        bus.subscribe(_TestEvent, handler_first)

        bus.publish_deferred(_TestEvent(value=1))
        bus.drain()          # dispatches value=1; value=2 should NOT be dispatched now

        assert first_pass == [1], "Only the original event in first drain"
        assert second_pass == [], "value=2 not yet delivered"

        # Now add a subscriber for the second pass
        bus.subscribe(_TestEvent, second_pass.append)
        bus.drain()          # now value=2 should dispatch

        # first_pass gets value=2 too (still subscribed), second_pass gets it
        assert 2 in first_pass
        assert _TestEvent(value=2) in second_pass


# ---------------------------------------------------------------------------
# Engine event type tests (smoke checks)
# ---------------------------------------------------------------------------

class TestEngineEventTypes:
    def test_chunk_loaded_event_frozen(self):
        evt = ChunkLoadedEvent(coord=(1, 2, 3))
        with pytest.raises((AttributeError, TypeError)):
            evt.coord = (0, 0, 0)  # type: ignore

    def test_terrain_edited_event_frozen(self):
        evt = TerrainEditedEvent(chunk_coords=(0, 0, 0), brush=None)
        with pytest.raises((AttributeError, TypeError)):
            evt.brush = "new"  # type: ignore

    def test_game_day_tick_event(self):
        bus = EventBus()
        days = []
        bus.subscribe(GameDayTickEvent, lambda e: days.append(e.day))
        bus.publish(GameDayTickEvent(day=5))
        assert days == [5]
