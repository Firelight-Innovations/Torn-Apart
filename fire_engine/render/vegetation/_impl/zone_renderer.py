"""
render/vegetation/_impl/zone_renderer — shared lifecycle helpers for zone-instanced
vegetation renderer components (GrassRendererComponent, FloraRendererComponent,
TreeRendererComponent).

All helpers take the owning component instance as their first positional argument
(``self_obj``) and operate on its attributes in place.  They are **free functions,
not a base class**: the codebase convention is delegation, not inheritance.

Attribute contract (every component that calls these must declare the attributes
as class-level annotations and set them in ``__init__``):

    base          — app object with ``._config`` and ``.terrain_root``
    sky_system    — object with ``.state`` (wind/rain data) or None
    zone_store    — ZoneStore (has ``.version``)
    chunk_provider — object with ``.chunks``
    lighting_pipeline — GpuLightingPipeline or None
    bus           — EventBus or None
    _root         — NodePath or None (the per-renderer root node)
    _time_s       — float accumulator for u_time_s

Docs: docs/systems/render.vegetation.md
"""

from __future__ import annotations

from typing import Any

from panda3d.core import BoundingBox, GeomNode, LPoint3, LVecBase2f

from fire_engine.core import (
    ChunkLoadedEvent,
    TerrainEditedEvent,
)

# ---------------------------------------------------------------------------
# Weather → sway mapping constants (canonical source; re-exported for
# grass_renderer, flora_renderer, tree_renderer to import from here).
# ---------------------------------------------------------------------------

# Tip displacement in meters / oscillation in rad/s.
# wind_speed spans 2.5–12 m/s; rain_intensity is 0–1.
_SWAY_BASE_MIN_M: float = 0.02
_SWAY_BASE_WIND_M: float = 0.16
_SWAY_GUST_MIN_M: float = 0.03
_SWAY_GUST_WIND_M: float = 0.18
_SWAY_GUST_RAIN_M: float = 0.12
_GUST_FREQ_MIN: float = 1.2  # rad/s
_GUST_FREQ_PER_WIND: float = 0.25  # rad/s per m/s of wind
_GUST_FREQ_RAIN: float = 1.8  # extra rad/s at full rain
_WIND_SPEED_MAX: float = 12.0  # normalisation ceiling (storm wind, m/s)

__all__ = [
    "_GUST_FREQ_MIN",
    "_GUST_FREQ_PER_WIND",
    "_GUST_FREQ_RAIN",
    "_SWAY_BASE_MIN_M",
    "_SWAY_BASE_WIND_M",
    "_SWAY_GUST_MIN_M",
    "_SWAY_GUST_RAIN_M",
    "_SWAY_GUST_WIND_M",
    "_WIND_SPEED_MAX",
    "init_zone_renderer",
    "on_chunk_loaded",
    "on_terrain_edited",
    "set_volume_bounds",
    "subscribe_terrain_events",
    "sync_sway_uniforms",
    "unsubscribe_terrain_events",
]


# ---------------------------------------------------------------------------
# Shared __init__ — set the six common zone-renderer attributes
# ---------------------------------------------------------------------------


def init_zone_renderer(
    self_obj: Any,
    base: Any,
    sky_system: Any,
    zone_store: Any,
    chunk_provider: Any,
    lighting_pipeline: Any,
    bus: Any,
) -> None:
    """
    Assign the six dependency attributes shared by every zone-instanced
    vegetation renderer (``GrassRendererComponent``, ``FloraRendererComponent``,
    ``TreeRendererComponent``).

    Call at the TOP of each ``__init__``, BEFORE component-specific
    attribute assignments, so the shared block is always in a delegate
    and pylint never sees duplicate lines:

    ::

        def __init__(self, base=None, sky_system=None, ...):
            super().__init__()
            init_zone_renderer(self, base, sky_system, zone_store,
                               chunk_provider, lighting_pipeline, bus)
            # component-specific attrs follow ...
            self._root = None
            ...

    Parameters
    ----------
    self_obj : zone-renderer component
        The instance to initialise.
    base, sky_system, zone_store, chunk_provider, lighting_pipeline, bus
        Forwarded from the ``__init__`` keyword arguments.

    Docs: docs/systems/render.vegetation._impl.md
    """
    self_obj.base = base
    self_obj.sky_system = sky_system
    self_obj.zone_store = zone_store
    self_obj.chunk_provider = chunk_provider
    self_obj.lighting_pipeline = lighting_pipeline
    self_obj.bus = bus


# ---------------------------------------------------------------------------
# Shared volume bounding-box helper (shader-positioned instances)
# ---------------------------------------------------------------------------


def set_volume_bounds(geom_node: GeomNode, vol: Any, pad: float) -> None:
    """
    Set an explicit ``BoundingBox`` on *geom_node* padded by *pad* metres
    beyond the volume's ``min_corner`` / ``max_corner``, then call
    ``set_final(True)``.

    Instances are positioned entirely in the vertex shader, so Panda3D would
    cull the node by the base Geom's origin bounds.  This call replaces those
    bounds with the volume's real spatial extent plus a generous margin for
    sway / carry reach, and stops further recomputation.

    Parameters
    ----------
    geom_node : GeomNode
        The node whose bounds to replace.
    vol : ZoneVolume
        Source of ``min_corner`` / ``max_corner`` (3-tuples, metres).
    pad : float
        Extra margin in metres added on every side (e.g. blade reach +
        sway travel).

    Docs: docs/systems/render.vegetation._impl.md
    """
    geom_node.set_bounds(
        BoundingBox(
            LPoint3(vol.min_corner[0] - pad, vol.min_corner[1] - pad, vol.min_corner[2] - pad),
            LPoint3(vol.max_corner[0] + pad, vol.max_corner[1] + pad, vol.max_corner[2] + pad),
        )
    )
    geom_node.set_final(True)


# ---------------------------------------------------------------------------
# Shared sway-uniform sync (u_wind_dir / u_sway_* / u_gust_freq / u_time_s)
# ---------------------------------------------------------------------------


def sync_sway_uniforms(self_obj: Any, dt: float) -> None:
    """
    Advance ``self_obj._time_s`` by *dt* and push the four scalar sway
    uniforms + ``u_time_s`` onto ``self_obj._root``.

    Call site: every zone-renderer ``late_update``, after the
    rebuild-or-re-bake check and *before* any per-volume work.

    The block reads ``self_obj.sky_system.state`` for live weather; falls
    back to minimum-sway constants when ``sky_system`` is None or ``state``
    is not yet available.

    Parameters
    ----------
    self_obj : zone-renderer component
        Must have ``._root``, ``._time_s``, and ``.sky_system`` attributes.
    dt : float
        Frame delta in seconds.

    Docs: docs/systems/render.vegetation._impl.md
    """
    self_obj._time_s += dt
    root = self_obj._root
    if root is None:
        return
    st = getattr(self_obj.sky_system, "state", None) if self_obj.sky_system is not None else None
    if st is not None:
        wind = float(st.wind_speed)
        rain = float(st.rain_intensity)
        wn = max(0.0, min(wind / _WIND_SPEED_MAX, 1.0))
        root.set_shader_input(
            "u_wind_dir", LVecBase2f(float(st.wind_dir[0]), float(st.wind_dir[1]))
        )
        root.set_shader_input("u_sway_base", _SWAY_BASE_MIN_M + _SWAY_BASE_WIND_M * wn)
        root.set_shader_input(
            "u_sway_gust", _SWAY_GUST_MIN_M + _SWAY_GUST_WIND_M * wn + _SWAY_GUST_RAIN_M * rain
        )
        root.set_shader_input(
            "u_gust_freq", _GUST_FREQ_MIN + _GUST_FREQ_PER_WIND * wind + _GUST_FREQ_RAIN * rain
        )
    root.set_shader_input("u_time_s", self_obj._time_s)


# ---------------------------------------------------------------------------
# Terrain-event handlers (identical across all zone renderers)
# ---------------------------------------------------------------------------


def subscribe_terrain_events(self_obj: Any) -> None:
    """
    Subscribe ``self_obj`` to ``TerrainEditedEvent`` and
    ``ChunkLoadedEvent`` on ``self_obj.bus``.

    Safe to call when ``bus`` is None (no-op).

    Call site: end of every zone-renderer ``start()``, after the initial
    ``_build_volumes()`` call.

    Docs: docs/systems/render.vegetation._impl.md
    """
    if self_obj.bus is not None:
        self_obj.bus.subscribe(TerrainEditedEvent, self_obj._on_terrain_edited)
        self_obj.bus.subscribe(ChunkLoadedEvent, self_obj._on_chunk_loaded)


def unsubscribe_terrain_events(self_obj: Any) -> None:
    """
    Unsubscribe ``self_obj`` from ``TerrainEditedEvent`` and
    ``ChunkLoadedEvent`` on ``self_obj.bus``.

    Safe to call when ``bus`` is None or when not currently subscribed
    (the bus silently ignores unsubscribing a handler that was never added).

    Call site: every zone-renderer ``on_destroy``.

    Docs: docs/systems/render.vegetation._impl.md
    """
    if self_obj.bus is not None:
        self_obj.bus.unsubscribe(TerrainEditedEvent, self_obj._on_terrain_edited)
        self_obj.bus.unsubscribe(ChunkLoadedEvent, self_obj._on_chunk_loaded)


def on_terrain_edited(self_obj: Any, event: TerrainEditedEvent) -> None:
    """
    Normalise the event's ``chunk_coords`` to a sequence and delegate to
    ``self_obj._mark_dirty_for_coords``.

    Call site: assigned as ``self._on_terrain_edited`` in each zone renderer,
    or called directly from the method body as a one-liner delegate.

    Docs: docs/systems/render.vegetation._impl.md
    """
    coords: Any = event.chunk_coords
    if isinstance(coords, tuple) and len(coords) == 3 and isinstance(coords[0], int):
        coords = (coords,)
    self_obj._mark_dirty_for_coords(coords)


def on_chunk_loaded(self_obj: Any, event: ChunkLoadedEvent) -> None:
    """
    Forward a single loaded chunk coordinate to
    ``self_obj._mark_dirty_for_coords``.

    Call site: assigned as ``self._on_chunk_loaded`` in each zone renderer,
    or called directly from the method body as a one-liner delegate.

    Docs: docs/systems/render.vegetation._impl.md
    """
    self_obj._mark_dirty_for_coords((event.coord,))
