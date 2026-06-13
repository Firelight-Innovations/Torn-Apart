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
events exported from this module are:

    ChunkLoadedEvent(coord)       — terrain chunk finished generating + meshing
    ChunkUnloadedEvent(coord)     — terrain chunk evicted from memory
    TerrainEditedEvent(chunk_coords, brush) — brush edit completed
    GameDayTickEvent(day)         — one in-game day has elapsed
    WeatherChangedEvent(previous, current, day) — discrete weather state changed
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
from dataclasses import dataclass
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Engine-level event types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkLoadedEvent:
    """
    Published when a terrain chunk has been generated, meshed, and is ready.

    Attributes
    ----------
    coord : tuple[int, int, int]
        Chunk grid coordinate (cx, cy, cz). Each unit = 16 meters.
    """
    coord: tuple[int, int, int]


@dataclass(frozen=True)
class ChunkUnloadedEvent:
    """
    Published when a terrain chunk has been evicted from memory.

    Attributes
    ----------
    coord : tuple[int, int, int]
        Chunk grid coordinate (cx, cy, cz).
    """
    coord: tuple[int, int, int]


@dataclass(frozen=True)
class TerrainEditedEvent:
    """
    Published after a brush edit has been applied to terrain voxel data.

    Attributes
    ----------
    chunk_coords : tuple
        The set (or single) chunk coordinates affected by this edit.
        May be a single (cx, cy, cz) or a frozenset of coordinates.
    brush : object
        The brush object used (SphereBrush, BoxBrush, etc.).
        Typed loosely as ``object`` to avoid an import cycle with terrain/.
    """
    chunk_coords: tuple
    brush: object


@dataclass(frozen=True)
class WeatherChangedEvent:
    """
    Published by the WeatherSystem (``fire_engine.sky``) whenever the discrete
    weather state changes — at most once per 2-game-hour segment boundary,
    or when a dev override is applied/cleared.  Published via
    ``bus.publish_deferred`` (state-change notification; never per-frame).

    Subscribers: ambience/audio, AI behaviour (seek shelter), render layer
    (particle systems), gameplay (crop growth, fire spread).

    Attributes
    ----------
    previous : str
        The ``WeatherType.value`` string of the outgoing state
        (e.g. ``"clear"``, ``"rain"``).
    current : str
        The ``WeatherType.value`` string of the incoming state.
    day : int
        In-game day number on which the change occurred.
    """
    previous: str
    current: str
    day: int


@dataclass(frozen=True)
class LightningStrikeEvent:
    """
    Published by the WeatherSystem (M7) when a thunderstorm cell fires a strike.

    One event per scheduled strike (deterministic Poisson schedule per active
    THUNDERSTORM cell — see ``fire_engine.weather.lightning``).  Carried via
    ``bus.publish_deferred`` (state-change notification, never per-frame data
    plumbing).  The render half (``world/lightning_renderer.py``) subscribes,
    generates the bolt geometry from ``seed`` (deterministic), animates the
    flash, adds a transient scene light, and re-publishes a
    :class:`ThunderEvent` for the delayed audio crack.

    All randomness for the bolt geometry is keyed off ``seed`` so the same
    strike renders byte-identically on every machine / after a save-load.

    Attributes
    ----------
    pos : tuple[float, float, float]
        Strike origin (cloud-base) world XYZ, meters — the top of the bolt.
    ground_pos : tuple[float, float, float]
        Where the bolt terminates on the ground / roof, world XYZ, meters.
    seed : int
        Bolt RNG seed — deterministic channel geometry
        (``for_domain("weather", "bolt", seed, ...)``).
    time_abs : float
        Absolute game time of the strike, seconds (day·86400 + time-of-day).
    cell_id : int
        Source :class:`~fire_engine.weather.StormCell` id, hashed to an int
        (the cell's stable string id digested deterministically).
    intensity : float
        0–1 strike brightness / scale (peak cell intensity at strike time).
    """
    pos: tuple[float, float, float]
    ground_pos: tuple[float, float, float]
    seed: int
    time_abs: float
    cell_id: int
    intensity: float


@dataclass(frozen=True)
class ThunderEvent:
    """
    Published by the render half (M7) after a :class:`LightningStrikeEvent`,
    carrying the delayed audio crack for the strike.

    The thunder is heard ``delay_s`` after the flash (``distance_m / 343`` — the
    speed of sound), so a distant storm rumbles seconds after its lightning and
    a close one cracks almost immediately.  Audio subscribers schedule a rumble
    that long after the event arrives.  Via ``bus.publish_deferred``.

    Attributes
    ----------
    pos : tuple[float, float, float]
        Strike origin world XYZ, meters (same as the strike's ``pos``).
    distance_m : float
        Distance from the listener (camera) to the strike, meters.
    delay_s : float
        Audio delay before the crack is heard, seconds (``distance_m / 343``).
    time_abs : float
        Absolute game time of the originating flash, seconds.
    intensity : float
        0–1 strike intensity (drives the rumble loudness / low-pass roll-off).
    """
    pos: tuple[float, float, float]
    distance_m: float
    delay_s: float
    time_abs: float
    intensity: float


@dataclass(frozen=True)
class GameDayTickEvent:
    """
    Published by the Clock once per in-game day elapsed.

    Subscribers: world-map AI tier (statistical outcomes), Economy, Politics.

    Attributes
    ----------
    day : int
        The new day number (starts at 0; increments each in-game day).
    """
    day: int


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
            try:
                handlers.remove(handler)
            except ValueError:
                pass

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
