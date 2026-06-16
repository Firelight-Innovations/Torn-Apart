"""
core/event_bus.py — Synchronous/deferred event bus for the Torn Apart engine.

The Event Bus is used for **upward and sideways state-change notifications**
only. It must never be used for per-frame data plumbing or hot-path work
(see ARCHITECTURE.md §4, communication rules #3).

Dispatch modes
--------------
publish(event)          — synchronous: all handlers called immediately.
publish_deferred(event) — queued: handlers run on the next drain() call.
drain()                 — dispatches all queued events in FIFO order.
                          If a handler calls publish_deferred during drain(),
                          those events go into the *next* drain's queue
                          (the queue is snapshotted at the start of drain).

Events
------
All event types are frozen dataclasses named ``*Event``. The engine-level
events are defined in :mod:`fire_engine.core.events` and re-exported here
for backwards compatibility:

    ChunkLoadedEvent(coord)       — terrain chunk finished generating + meshing
    ChunkUnloadedEvent(coord)     — terrain chunk evicted from memory
    TerrainEditedEvent(chunk_coords, brush) — brush edit completed
    GameDayTickEvent(day)         — one in-game day has elapsed
    WeatherChangedEvent(previous, current, day) — discrete weather state changed
    BuildingChangedEvent(building_id, change, bounds_min, bounds_max)
                                  — a building was added/modified/removed
    LightningStrikeEvent(...)     — a thunderstorm cell fired a lightning strike (M7)
    ThunderEvent(...)             — delayed thunder crack for a strike (M7)

Example
-------
    from fire_engine.core.event_bus import EventBus, ChunkLoadedEvent

    bus = EventBus()

    def on_loaded(evt: ChunkLoadedEvent) -> None:
        print(f"Chunk {evt.coord} is ready")

    bus.subscribe(ChunkLoadedEvent, on_loaded)
    bus.publish(ChunkLoadedEvent(coord=(0, 0, 0)))  # → prints immediately
    bus.unsubscribe(ChunkLoadedEvent, on_loaded)
"""

from __future__ import annotations

import collections
import contextlib
from collections.abc import Callable
from typing import Any

# Re-export all engine event types from the grouping module so that
# `from fire_engine.core.event_bus import ChunkLoadedEvent` (and all similar
# import paths) continue to resolve unchanged.
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

__all__ = [
    "BuildingChangedEvent",
    "ChunkLoadedEvent",
    "ChunkUnloadedEvent",
    "EventBus",
    "GameDayTickEvent",
    "LightningStrikeEvent",
    "TerrainEditedEvent",
    "ThunderEvent",
    "WeatherChangedEvent",
]

# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

Handler = Callable[[Any], None]


class EventBus:
    """
    Central publish-subscribe bus. Fully synchronous; no threads.

    Subscription
    ------------
    subscribe(event_type, handler)   — register handler for event_type.
    unsubscribe(event_type, handler) — de-register (no-op if not subscribed).

    Dispatch
    --------
    publish(event)            — immediate synchronous delivery to all handlers
                                currently registered for type(event).
    publish_deferred(event)   — enqueue; delivered on next drain().
    drain()                   — FIFO dispatch of the deferred queue.
                                Snapshots the queue before dispatching so that
                                any publish_deferred calls made *during* drain
                                go to the *following* drain, not the current one.

    Notes
    -----
    - Handler registration order is preserved (insertion-ordered dict of lists).
    - If the same handler is registered twice for the same type it will be
      called twice; callers are responsible for avoiding duplicate subscriptions.
    - Exceptions raised by handlers propagate immediately (no swallowing).
    """

    def __init__(self) -> None:
        # event_type → list of handlers (ordered)
        self._handlers: dict[type, list[Handler]] = collections.defaultdict(list)
        # deferred queue
        self._deferred: collections.deque[Any] = collections.deque()

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def subscribe(self, event_type: type, handler: Handler) -> None:
        """
        Register *handler* to be called whenever an event of *event_type* is
        published (either immediately or via drain).

        Parameters
        ----------
        event_type : type
            The exact event class (e.g. ``ChunkLoadedEvent``).
        handler    : Callable[[event], None]
            Called with the event object as its sole argument.
        """
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: type, handler: Handler) -> None:
        """
        Remove *handler* from the subscriber list for *event_type*.

        No-op if the handler is not currently subscribed.

        Parameters
        ----------
        event_type : type
        handler    : Callable[[event], None]
        """
        handlers = self._handlers.get(event_type)
        if handlers is not None:
            with contextlib.suppress(ValueError):
                handlers.remove(handler)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, event: Any) -> None:
        """
        Publish *event* synchronously. All handlers registered for
        ``type(event)`` are called immediately, in subscription order.

        Parameters
        ----------
        event : frozen dataclass instance — the event to deliver.

        Example
        -------
        >>> bus.publish(ChunkLoadedEvent(coord=(1, 2, 0)))
        """
        handlers = self._handlers.get(type(event))
        if handlers:
            for handler in list(handlers):  # copy so unsubscribe mid-dispatch is safe
                handler(event)

    def publish_deferred(self, event: Any) -> None:
        """
        Enqueue *event* for delivery on the next call to drain().

        Use for simulation events that should be processed once per tick
        (not in the middle of the current frame's logic).

        Parameters
        ----------
        event : frozen dataclass instance
        """
        self._deferred.append(event)

    def drain(self) -> None:
        """
        Dispatch all currently queued deferred events in FIFO order.

        The queue is **snapshotted** at the start of drain so that any
        ``publish_deferred`` calls made by handlers during draining are
        deferred to the *next* drain, not the current one.

        Example
        -------
        >>> bus.publish_deferred(ChunkLoadedEvent(coord=(0, 0, 1)))
        >>> bus.drain()   # handler called now
        """
        # Snapshot: swap out the current queue for a fresh one
        to_dispatch = self._deferred
        self._deferred = collections.deque()

        for event in to_dispatch:
            handlers = self._handlers.get(type(event))
            if handlers:
                for handler in list(handlers):
                    handler(event)
