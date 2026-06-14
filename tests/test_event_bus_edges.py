"""
tests/test_event_bus_edges.py — Edge-case / error-path characterisation tests
for core/event_bus.py.  Golden-master style: pin CURRENT behaviour; do NOT
fix bugs.

Does NOT duplicate cases already in test_event_bus.py.  See that file for
basic subscribe/publish/drain correctness.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass

import pytest

from fire_engine.core.event_bus import (
    BuildingChangedEvent,
    ChunkUnloadedEvent,
    EventBus,
    GameDayTickEvent,
    LightningStrikeEvent,
    ThunderEvent,
    WeatherChangedEvent,
)

# ---------------------------------------------------------------------------
# Private test event types (not shared with test_event_bus.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _EvA:
    n: int


@dataclass(frozen=True)
class _EvB:
    n: int


# ---------------------------------------------------------------------------
# Handler-exception policy during publish()
# ---------------------------------------------------------------------------


class TestHandlerExceptionDuringPublish:
    """Pin: an exception in a handler propagates to the publish() caller
    AND aborts remaining handlers in the same dispatch (no swallowing)."""

    def test_exception_propagates_to_publisher(self):
        """A raising handler must let the exception escape publish()."""
        bus = EventBus()

        def boom(evt):
            raise RuntimeError("boom")

        bus.subscribe(_EvA, boom)

        with pytest.raises(RuntimeError, match="boom"):
            bus.publish(_EvA(n=1))

    def test_subsequent_handlers_skipped_after_exception(self):
        """If handler-1 raises, handler-2 registered after it must NOT run."""
        bus = EventBus()
        called = []

        def handler_raise(evt):
            raise ValueError("first handler raises")

        def handler_after(evt):
            called.append("after")

        bus.subscribe(_EvA, handler_raise)
        bus.subscribe(_EvA, handler_after)

        with pytest.raises(ValueError):
            bus.publish(_EvA(n=1))

        # Pin current behaviour: second handler was NOT called because the
        # exception aborted the dispatch loop.
        assert called == [], (
            "SUSPECTED BUG: handler_after was unexpectedly called after "
            "handler_raise raised — exception should abort subsequent handlers"
        )

    def test_handler_before_raiser_did_run(self):
        """Handlers registered BEFORE the raiser must have already been called."""
        bus = EventBus()
        called = []

        def first(evt):
            called.append("first")

        def raiser(evt):
            raise RuntimeError("second raises")

        bus.subscribe(_EvA, first)
        bus.subscribe(_EvA, raiser)

        with pytest.raises(RuntimeError):
            bus.publish(_EvA(n=99))

        assert called == ["first"]


# ---------------------------------------------------------------------------
# Handler-exception policy during drain()
# ---------------------------------------------------------------------------


class TestHandlerExceptionDuringDrain:
    """Pin: an exception in a deferred handler propagates to the drain() caller
    and aborts further processing of the current sweep."""

    def test_exception_from_drain_propagates(self):
        bus = EventBus()

        def boom(evt):
            raise RuntimeError("drain boom")

        bus.subscribe(_EvA, boom)
        bus.publish_deferred(_EvA(n=1))

        with pytest.raises(RuntimeError, match="drain boom"):
            bus.drain()

    def test_remaining_events_skipped_after_drain_exception(self):
        """An exception mid-drain aborts the rest of the current snapshot."""
        bus = EventBus()
        delivered = []

        def first(evt):
            if evt.n == 1:
                raise RuntimeError("stops here")
            delivered.append(evt.n)

        bus.subscribe(_EvA, first)
        bus.publish_deferred(_EvA(n=1))
        bus.publish_deferred(_EvA(n=2))  # should NOT be delivered this drain

        with pytest.raises(RuntimeError):
            bus.drain()

        # Pin: event n=2 was skipped because the exception aborted the sweep.
        assert delivered == [], (
            "SUSPECTED BUG: event n=2 was delivered despite drain aborting on n=1"
        )


# ---------------------------------------------------------------------------
# Double-subscribe
# ---------------------------------------------------------------------------


class TestDoubleSubscribe:
    """Pin: subscribing the same handler twice fires it twice per publish."""

    def test_double_subscribe_fires_twice(self):
        bus = EventBus()
        calls = []
        h = calls.append
        bus.subscribe(_EvA, h)
        bus.subscribe(_EvA, h)
        bus.publish(_EvA(n=5))
        assert len(calls) == 2, (
            f"Expected handler to fire twice after double-subscribe; got {len(calls)}"
        )

    def test_unsubscribe_once_after_double_subscribe_fires_once(self):
        """After one unsubscribe the handler should still fire once."""
        bus = EventBus()
        calls = []
        h = calls.append
        bus.subscribe(_EvA, h)
        bus.subscribe(_EvA, h)
        bus.unsubscribe(_EvA, h)  # removes only the first occurrence
        bus.publish(_EvA(n=7))
        assert len(calls) == 1, f"Expected exactly one call after one unsubscribe; got {len(calls)}"

    def test_unsubscribe_twice_after_double_subscribe_fires_zero(self):
        """Two unsubscribes remove both registrations."""
        bus = EventBus()
        calls = []
        h = calls.append
        bus.subscribe(_EvA, h)
        bus.subscribe(_EvA, h)
        bus.unsubscribe(_EvA, h)
        bus.unsubscribe(_EvA, h)
        bus.publish(_EvA(n=3))
        assert calls == []


# ---------------------------------------------------------------------------
# Unsubscribe of never-subscribed handler
# ---------------------------------------------------------------------------


class TestUnsubscribeNeverSubscribed:
    """Pin: unsubscribing a handler that was never registered is a no-op."""

    def test_unsubscribe_unknown_handler_no_error(self):
        bus = EventBus()
        # Should not raise any exception
        bus.unsubscribe(_EvA, lambda e: None)

    def test_unsubscribe_unknown_type_no_error(self):
        """Unsubscribing from a type that has never been subscribed to."""
        bus = EventBus()
        bus.subscribe(_EvA, lambda e: None)
        # _EvB has no subscribers at all
        bus.unsubscribe(_EvB, lambda e: None)


# ---------------------------------------------------------------------------
# Unsubscribe DURING publish (mid-dispatch mutation safety)
# ---------------------------------------------------------------------------


class TestUnsubscribeDuringPublish:
    """Pin: publish() snapshots the handler list before iterating, so
    unsubscribe during dispatch does NOT affect the current sweep."""

    def test_self_unsubscribe_during_publish_does_not_abort_current_dispatch(self):
        """A handler that unsubscribes itself should still complete this call."""
        bus = EventBus()
        calls = []

        def self_removing(evt):
            bus.unsubscribe(_EvA, self_removing)
            calls.append("self_removing")

        def other(evt):
            calls.append("other")

        bus.subscribe(_EvA, self_removing)
        bus.subscribe(_EvA, other)

        bus.publish(_EvA(n=1))

        # Both handlers run this time because the list was snapshotted
        assert calls == ["self_removing", "other"]

    def test_self_unsubscribe_takes_effect_on_next_publish(self):
        """After the self-removing publish, the handler must be gone."""
        bus = EventBus()
        calls = []

        def self_removing(evt):
            bus.unsubscribe(_EvA, self_removing)
            calls.append("self_removing")

        bus.subscribe(_EvA, self_removing)
        bus.publish(_EvA(n=1))  # unsubscribes here
        bus.publish(_EvA(n=2))  # should NOT fire self_removing again

        assert calls == ["self_removing"]

    def test_handler_unsubscribes_sibling_during_publish(self):
        """A handler can unsubscribe a later-registered sibling; sibling still
        runs in the current dispatch because the list was snapshotted."""
        bus = EventBus()
        calls = []

        def sibling(evt):
            calls.append("sibling")

        def remover(evt):
            bus.unsubscribe(_EvA, sibling)
            calls.append("remover")

        bus.subscribe(_EvA, remover)
        bus.subscribe(_EvA, sibling)

        bus.publish(_EvA(n=1))

        # Pin: sibling still runs this dispatch (snapshot)
        assert "sibling" in calls
        assert "remover" in calls

        # Second publish: sibling must be gone
        calls.clear()
        bus.publish(_EvA(n=2))
        assert "sibling" not in calls
        assert "remover" in calls  # remover itself was never removed


# ---------------------------------------------------------------------------
# publish_deferred during drain — ordering across two drains
# ---------------------------------------------------------------------------


class TestDeferredOrderingAcrossDrains:
    """Extend test_event_bus.py's idea: verify precise ordering across two
    consecutive drains when handlers enqueue new events during draining."""

    def test_chained_deferred_ordering(self):
        """Events queued during drain-1 appear in drain-2 in FIFO order."""
        bus = EventBus()
        order = []

        def handler(evt):
            order.append(("recv", evt.n))
            if evt.n == 10:
                bus.publish_deferred(_EvA(n=20))
                bus.publish_deferred(_EvA(n=30))

        bus.subscribe(_EvA, handler)

        bus.publish_deferred(_EvA(n=10))
        bus.drain()  # delivers 10; enqueues 20, 30

        assert order == [("recv", 10)]

        bus.drain()  # delivers 20 then 30
        assert order == [("recv", 10), ("recv", 20), ("recv", 30)]

    def test_two_events_in_drain1_each_enqueue_drain2_fifo(self):
        """Two events in drain-1 each enqueue one event; drain-2 order is stable."""
        bus = EventBus()
        order = []

        def handler(evt):
            order.append(evt.n)
            if evt.n in (1, 2):
                bus.publish_deferred(_EvA(n=evt.n * 10))

        bus.subscribe(_EvA, handler)
        bus.publish_deferred(_EvA(n=1))
        bus.publish_deferred(_EvA(n=2))
        bus.drain()  # delivers 1, 2; enqueues 10, 20

        assert order == [1, 2]

        bus.drain()  # delivers 10, 20
        assert order == [1, 2, 10, 20]


# ---------------------------------------------------------------------------
# drain() on empty queue
# ---------------------------------------------------------------------------


class TestDrainEmpty:
    def test_drain_empty_is_noop(self):
        bus = EventBus()
        calls = []
        bus.subscribe(_EvA, calls.append)
        bus.drain()  # queue is empty — should not raise
        assert calls == []

    def test_drain_returns_none(self):
        """drain() has no documented return value; pin that it returns None."""
        bus = EventBus()
        result = bus.drain()
        assert result is None

    def test_repeated_drain_empty_no_error(self):
        bus = EventBus()
        for _ in range(5):
            bus.drain()


# ---------------------------------------------------------------------------
# Cross-type isolation (subscribe to A, publish B)
# ---------------------------------------------------------------------------


class TestCrossTypeIsolation:
    def test_handler_for_A_does_not_receive_B(self):
        bus = EventBus()
        received = []
        bus.subscribe(_EvA, received.append)
        bus.publish(_EvB(n=99))
        assert received == []

    def test_handler_for_A_does_not_receive_B_via_deferred(self):
        bus = EventBus()
        received = []
        bus.subscribe(_EvA, received.append)
        bus.publish_deferred(_EvB(n=42))
        bus.drain()
        assert received == []


# ---------------------------------------------------------------------------
# Event dataclass invariants — FrozenInstanceError on mutation
# ---------------------------------------------------------------------------


class TestEventDataclassFrozen:
    """Each event must raise FrozenInstanceError on attribute assignment.
    These are the event types NOT already tested in test_event_bus.py."""

    def test_chunk_unloaded_event_frozen(self):
        evt = ChunkUnloadedEvent(coord=(1, 2, 3))
        with pytest.raises(FrozenInstanceError):
            evt.coord = (0, 0, 0)  # type: ignore

    def test_weather_changed_event_frozen(self):
        evt = WeatherChangedEvent(previous="clear", current="rain", day=0)
        with pytest.raises(FrozenInstanceError):
            evt.current = "snow"  # type: ignore

    def test_game_day_tick_event_frozen(self):
        evt = GameDayTickEvent(day=3)
        with pytest.raises(FrozenInstanceError):
            evt.day = 99  # type: ignore

    def test_lightning_strike_event_frozen(self):
        evt = LightningStrikeEvent(
            pos=(0.0, 0.0, 100.0),
            ground_pos=(0.0, 0.0, 0.0),
            seed=42,
            time_abs=1234.5,
            cell_id=7,
            intensity=0.8,
        )
        with pytest.raises(FrozenInstanceError):
            evt.intensity = 0.0  # type: ignore

    def test_thunder_event_frozen(self):
        evt = ThunderEvent(
            pos=(10.0, 20.0, 100.0),
            distance_m=500.0,
            delay_s=1.46,
            time_abs=9000.0,
            intensity=0.5,
        )
        with pytest.raises(FrozenInstanceError):
            evt.delay_s = 0.0  # type: ignore

    def test_building_changed_event_frozen(self):
        evt = BuildingChangedEvent(
            building_id=1,
            change="added",
            bounds_min=(0.0, 0.0, 0.0),
            bounds_max=(10.0, 10.0, 5.0),
        )
        with pytest.raises(FrozenInstanceError):
            evt.change = "removed"  # type: ignore


# ---------------------------------------------------------------------------
# Event dataclass field names and value equality
# ---------------------------------------------------------------------------


class TestEventDataclassFields:
    """Pin field names, types, and value-equality for all engine event types."""

    def test_weather_changed_event_fields(self):
        evt = WeatherChangedEvent(previous="clear", current="rain", day=2)
        assert evt.previous == "clear"
        assert evt.current == "rain"
        assert evt.day == 2

    def test_weather_changed_event_equality(self):
        a = WeatherChangedEvent(previous="fog", current="clear", day=0)
        b = WeatherChangedEvent(previous="fog", current="clear", day=0)
        assert a == b

    def test_weather_changed_event_inequality(self):
        a = WeatherChangedEvent(previous="clear", current="rain", day=0)
        b = WeatherChangedEvent(previous="clear", current="snow", day=0)
        assert a != b

    def test_lightning_strike_event_fields(self):
        evt = LightningStrikeEvent(
            pos=(1.0, 2.0, 3.0),
            ground_pos=(1.0, 2.0, 0.0),
            seed=99,
            time_abs=500.0,
            cell_id=12,
            intensity=0.6,
        )
        assert evt.pos == (1.0, 2.0, 3.0)
        assert evt.ground_pos == (1.0, 2.0, 0.0)
        assert evt.seed == 99
        assert evt.time_abs == 500.0
        assert evt.cell_id == 12
        assert evt.intensity == pytest.approx(0.6)

    def test_lightning_strike_event_equality(self):
        kwargs = dict(
            pos=(0.0, 0.0, 100.0),
            ground_pos=(0.0, 0.0, 0.0),
            seed=7,
            time_abs=1000.0,
            cell_id=3,
            intensity=1.0,
        )
        assert LightningStrikeEvent(**kwargs) == LightningStrikeEvent(**kwargs)

    def test_thunder_event_fields(self):
        evt = ThunderEvent(
            pos=(5.0, 6.0, 90.0),
            distance_m=343.0,
            delay_s=1.0,
            time_abs=2048.0,
            intensity=0.9,
        )
        assert evt.pos == (5.0, 6.0, 90.0)
        assert evt.distance_m == pytest.approx(343.0)
        assert evt.delay_s == pytest.approx(1.0)
        assert evt.time_abs == pytest.approx(2048.0)
        assert evt.intensity == pytest.approx(0.9)

    def test_thunder_event_equality(self):
        kwargs = dict(
            pos=(0.0, 0.0, 100.0),
            distance_m=200.0,
            delay_s=0.58,
            time_abs=3600.0,
            intensity=0.7,
        )
        assert ThunderEvent(**kwargs) == ThunderEvent(**kwargs)

    def test_building_changed_event_fields(self):
        evt = BuildingChangedEvent(
            building_id=42,
            change="modified",
            bounds_min=(-5.0, -5.0, 0.0),
            bounds_max=(5.0, 5.0, 8.0),
        )
        assert evt.building_id == 42
        assert evt.change == "modified"
        assert evt.bounds_min == (-5.0, -5.0, 0.0)
        assert evt.bounds_max == (5.0, 5.0, 8.0)

    def test_building_changed_event_equality(self):
        kwargs = dict(
            building_id=1,
            change="removed",
            bounds_min=(0.0, 0.0, 0.0),
            bounds_max=(4.0, 4.0, 3.0),
        )
        assert BuildingChangedEvent(**kwargs) == BuildingChangedEvent(**kwargs)

    def test_chunk_unloaded_event_fields(self):
        evt = ChunkUnloadedEvent(coord=(7, 8, 9))
        assert evt.coord == (7, 8, 9)

    def test_chunk_unloaded_event_equality(self):
        assert ChunkUnloadedEvent(coord=(1, 2, 3)) == ChunkUnloadedEvent(coord=(1, 2, 3))
        assert ChunkUnloadedEvent(coord=(1, 2, 3)) != ChunkUnloadedEvent(coord=(9, 9, 9))
