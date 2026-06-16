"""
render/overlay/_overlay_weather.py — Weather Control panel helpers (M8 spatial storm API).

Extracted from devtools_overlay.py; called as free functions taking the overlay
instance as first argument (C0302 fat-class split pattern).

Docs: docs/systems/render.overlay.md
"""

from __future__ import annotations

import contextlib
import math
from typing import TYPE_CHECKING, Any

from fire_engine.core.math3d import Vec3
from fire_engine.devtools import Button, Field, FieldKind, Section
from fire_engine.world.terrain import raycast_voxel

if TYPE_CHECKING:
    from fire_engine.render.overlay.devtools_overlay import DevOverlay

_TERRAIN_RAY_MAX_M = 200.0  # how far a dev click probes for a terrain chunk


def _fmt(value: object) -> str:
    """Compact display string for a scalar field value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


# ---------------------------------------------------------------------------
# Camera / clock helpers (used by several weather functions)
# ---------------------------------------------------------------------------


def camera_xy(self_obj: DevOverlay) -> tuple[float, float]:
    """Player/camera world XY (meters) — the summon + readout reference."""
    p = self_obj._app.camera_go.transform.position
    return (float(p.x), float(p.y))


def time_abs(self_obj: DevOverlay) -> float:
    """Absolute game seconds from the clock (day·86400 + time-of-day)."""
    clk = self_obj._app._clock
    day = int(getattr(clk, "game_day", 0))
    tod = float(getattr(clk, "game_time_of_day", 0.0))
    return day * 86400.0 + tod


# ---------------------------------------------------------------------------
# Panel builder
# ---------------------------------------------------------------------------


def build_weather_control(self_obj: DevOverlay) -> tuple[list[Section], list[Button]]:
    """
    Build the "Weather Control" panel: summon buttons + a live read-out of
    the local weather class and the nearest cell's kind / distance / bearing
    / ETA.  Every engine access is guarded so a weather-API shift degrades to
    blanks rather than crashing the overlay.
    """
    sections = [
        Section(
            "Local",
            [
                Field("class", FieldKind.LABEL, lambda: wx_local_class(self_obj)),
                Field(
                    "humidity",
                    FieldKind.LABEL,
                    lambda: (
                        _fmt(getattr(wx_local_sample(self_obj), "humidity", None))
                        if wx_local_sample(self_obj)
                        else "?"
                    ),
                ),
                Field(
                    "wetness",
                    FieldKind.LABEL,
                    lambda: (
                        _fmt(getattr(wx_local_sample(self_obj), "wetness", None))
                        if wx_local_sample(self_obj)
                        else "?"
                    ),
                ),
            ],
        ),
        Section(
            "Nearest cell",
            [
                Field(
                    "kind",
                    FieldKind.LABEL,
                    lambda: (
                        str(getattr(n[0].kind, "value", "?"))
                        if (n := wx_nearest(self_obj))
                        else "(none)"
                    ),
                ),
                Field(
                    "distance",
                    FieldKind.LABEL,
                    lambda: f"{n[1]:.0f} m" if (n := wx_nearest(self_obj)) else "-",
                ),
                Field(
                    "bearing",
                    FieldKind.LABEL,
                    lambda: f"{n[2]:.0f} deg" if (n := wx_nearest(self_obj)) else "-",
                ),
                Field("ETA", FieldKind.LABEL, lambda: wx_near_eta(self_obj)),
            ],
        ),
    ]
    buttons = [
        Button("Summon Rainstorm", lambda: summon(self_obj, "summon_rainstorm")),
        Button("Summon Thunderstorm", lambda: summon(self_obj, "summon_thunderstorm")),
        Button("Summon Fog Bank", lambda: summon(self_obj, "summon_fog_bank")),
        Button("Clear Skies", lambda: clear_skies(self_obj)),
    ]
    return sections, buttons


# ---------------------------------------------------------------------------
# Local weather read-outs
# ---------------------------------------------------------------------------


def wx_local_class(self_obj: DevOverlay) -> str:
    """Read-out: local weather class from the WeatherSystem (guarded)."""
    try:
        return str(getattr(self_obj._weather.current, "value", "?"))
    except Exception:
        return "?"


def wx_local_sample(self_obj: DevOverlay) -> Any:
    """Return the local WeatherSample at the camera position (guarded)."""
    try:
        return self_obj._weather.sample_local(camera_xy(self_obj), time_abs(self_obj))
    except Exception:
        return None


def wx_nearest(self_obj: DevOverlay) -> tuple[Any, float, float, float] | None:
    """(cell, dist_m, bearing_deg, eta_s) for the nearest active cell."""
    try:
        w = self_obj._weather
        cells = list(w.cells)
        if not cells:
            return None
        t = time_abs(self_obj)
        px, py = camera_xy(self_obj)
        import numpy as _np

        cell = cells[0]  # already nearest-first
        c = cell.center(t, w.synoptic)
        dx, dy = float(c[0]) - px, float(c[1]) - py
        dist = float(_np.hypot(dx, dy))
        bearing = (math.degrees(math.atan2(dx, dy))) % 360.0  # 0=+Y(N)
        eta = float(w.cell_eta_s(cell, t, (px, py)))
        return cell, dist, bearing, eta
    except Exception:
        return None


def wx_near_eta(self_obj: DevOverlay) -> str:
    """ETA string for the nearest weather cell (guarded)."""
    n = wx_nearest(self_obj)
    if not n:
        return "-"
    eta = n[3]
    if not math.isfinite(eta):
        return "receding"
    return f"{eta / 60.0:.1f} min"


# ---------------------------------------------------------------------------
# Summon / clear actions
# ---------------------------------------------------------------------------


def summon(self_obj: DevOverlay, method_name: str) -> None:
    """Call a WeatherSystem summon wrapper aimed at the camera."""
    w = self_obj._weather
    if w is None:
        return
    with contextlib.suppress(Exception):
        getattr(w, method_name)(time_abs=time_abs(self_obj), player_pos=camera_xy(self_obj))


def clear_skies(self_obj: DevOverlay) -> None:
    """Clear summoned cells + suppress the current natural weather."""
    with contextlib.suppress(Exception):
        self_obj._weather.clear_all()


def summon_cell_at_camera(self_obj: DevOverlay) -> None:
    """Debug key (K): stamp a synthetic thunderstorm right at the camera."""
    w = self_obj._weather
    if w is None:
        return
    try:
        from fire_engine.world.weather import CellKind

        w.summon_cell(
            CellKind.THUNDERSTORM,
            time_abs=time_abs(self_obj),
            player_pos=camera_xy(self_obj),
            upwind_m=0.0,
        )
    except Exception:
        pass


def raycast_ground(self_obj: DevOverlay, origin: Vec3, direction: Vec3) -> Vec3 | None:
    """World point where a ray hits terrain, or ``None`` (voxel raycast)."""
    cm = getattr(self_obj._app, "chunk_manager", None)
    if cm is None:
        return None
    hit = raycast_voxel(origin, direction, cm.get_or_create, max_distance_m=_TERRAIN_RAY_MAX_M)
    if hit is None:
        return None
    result: Vec3 | None = getattr(hit, "world_point", None) or getattr(hit, "point", None)
    return result


def fire_lightning_at_crosshair(self_obj: DevOverlay) -> None:
    """
    Debug key (L): publish a :class:`LightningStrikeEvent` at the crosshair.

    Resolves the world point under the camera crosshair (terrain raycast,
    falling back to a point 60 m ahead) and publishes the event on the bus
    per the M7 contract.  The import resolves at boot once M7's event is
    merged into ``core/event_bus`` — this file is excluded from the headless
    suite, so it never needs the event to exist at test time.
    """
    bus = getattr(self_obj._app, "_event_bus", None) or getattr(self_obj._app, "event_bus", None)
    if bus is None:
        return
    try:
        from fire_engine.core.event_bus import LightningStrikeEvent
    except Exception:
        return

    cam_tf = self_obj._app.camera_go.transform
    ground = cam_tf.position + cam_tf.forward * 60.0
    # _cursor_ray is on the overlay instance
    ray = self_obj._cursor_ray()
    if ray is not None:
        hit_pt = raycast_ground(self_obj, *ray)
        if hit_pt is not None:
            ground = hit_pt
    t = time_abs(self_obj)
    pos = (float(ground.x), float(ground.y), float(cam_tf.position.z))
    ground_pos = (float(ground.x), float(ground.y), float(ground.z))
    try:
        ev = LightningStrikeEvent(
            pos=pos,
            ground_pos=ground_pos,
            seed=int(t) & 0x7FFFFFFF,
            time_abs=t,
            cell_id=-1,
            intensity=1.0,
        )
        publish = getattr(bus, "publish", None) or getattr(bus, "publish_deferred", None)
        if publish is not None:
            publish(ev)
    except Exception:
        pass


def toggle_rain_cover_overlay(self_obj: DevOverlay) -> None:
    """
    Debug key (J): toggle a translucent quad visualising the rain-cover
    window (``RainCoverField`` — where rain is blocked by roofs/overhangs).

    Draws a flat card spanning the cover field's footprint at the field
    origin; a second press removes it.  Best-effort: no-op if the rain
    component / cover field is not wired in.
    """
    if self_obj._rain_cover_np is not None:
        self_obj._rain_cover_np.remove_node()
        self_obj._rain_cover_np = None
        return
    cover = rain_cover_field(self_obj)
    if cover is None:
        return
    try:
        from panda3d.core import CardMaker

        ox, oy = cover.origin_m
        span = float(cover.cells) * float(cover.cell_m)
        cm = CardMaker("rain_cover_overlay")
        cm.set_frame(0.0, span, 0.0, span)
        node = self_obj._base.render.attach_new_node(cm.generate())
        node.set_pos(float(ox), float(oy), 0.05)  # just above ground
        node.set_p(-90)  # lay flat (XY plane)
        node.set_transparency(True)
        node.set_color(0.2, 0.55, 1.0, 0.28)
        node.set_light_off()
        node.set_two_sided(True)
        self_obj._rain_cover_np = node
    except Exception:
        self_obj._rain_cover_np = None


def rain_cover_field(self_obj: DevOverlay) -> Any:
    """Locate the headless ``RainCoverField`` owned by the rain component."""
    rain_go = getattr(self_obj._app, "rain_go", None)
    if rain_go is None:
        return None
    try:
        from fire_engine.render.sky.rain_renderer import RainRendererComponent

        comp = rain_go.get_component(RainRendererComponent)
        return getattr(comp, "_cover", None) if comp is not None else None
    except Exception:
        return None
