"""
render/sky/_impl/lightning_bolt — bolt geometry, envelope, and flash helpers for
LightningRendererComponent.

Extracted from ``lightning_renderer.LightningRendererComponent`` to satisfy the
500-line limit (C0302): ``upload_bolt``, ``advance_bolt``, bolt_envelope,
``bolt_sky_flash``, ``add_flash_light``, ``refresh_cover``, ``cover_z``.

``lightning_renderer`` imports FROM this module; this module does NOT import
lightning_renderer (no circular dependency).  The ``TYPE_CHECKING`` guard is
used only for the ``LightningRendererComponent`` annotation.

Docs: docs/systems/render.sky.md
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from panda3d.core import (
    BoundingBox,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexWriter,
    LPoint3,
)

from fire_engine.core.rng import for_domain

if TYPE_CHECKING:
    from fire_engine.render.sky.lightning_renderer import LightningRendererComponent, _Bolt

__all__ = [
    "add_flash_light",
    "advance_bolt",
    "bolt_envelope",
    "bolt_sky_flash",
    "cover_z",
    "refresh_cover",
    "upload_bolt",
]

# Envelope phase durations (seconds, real time).
_LEADER_S: float = 0.16
_RETURN_S: float = 0.10
_AFTERGLOW_S: float = 0.45
_RESTRIKE_GAP_S: float = 0.09

# HDR brightness of each phase (multiplies per-segment brightness).
_LEADER_FLASH: float = 1.2
_RETURN_FLASH: float = 6.0
_RESTRIKE_FLASH: float = 3.0

# Sky/cloud flash-pulse peak.
_SKY_FLASH_PEAK: float = 0.9

# Transient scene-light tuning.
_LIGHT_COLOR: tuple[float, float, float] = (0.80, 0.86, 1.0)
_LIGHT_INTENSITY: float = 40.0
_LIGHT_RADIUS_M: float = 260.0
_LIGHT_TTL_S: float = 0.30

# Ribbon look.
_WIDTH_SCALE_M: float = 0.35


def upload_bolt(
    self_obj: LightningRendererComponent,
    bolt: _Bolt,
    geom: Any,
    intensity: float,
    seed: int,
) -> None:
    """Build the ribbon quad soup for a bolt geometry and ignite the node.

    Extracted from ``LightningRendererComponent._upload_bolt``
    (lightning_renderer.py).

    Docs: docs/systems/render.sky._impl.md
    """
    n = len(geom)
    vdata = GeomVertexData("bolt", self_obj._fmt, Geom.UH_dynamic)
    vdata.set_num_rows(n * 4)
    vw = GeomVertexWriter(vdata, "vertex")
    ow = GeomVertexWriter(vdata, "a_other")
    rw = GeomVertexWriter(vdata, "a_ribbon")

    a = geom.a
    b = geom.b
    width = geom.width
    bright = geom.brightness

    # alongT for each segment = its start-point fraction down the channel
    # (top = 0, ground = 1), driving the top-down reveal.  Use the segment
    # start Z relative to the overall bolt Z span.
    z_top = float(max(a[:, 2].max(), b[:, 2].max()))
    z_bot = float(min(a[:, 2].min(), b[:, 2].min()))
    z_span = max(z_top - z_bot, 1e-3)
    along = (z_top - a[:, 2]) / z_span  # (N,) 0 at top → 1 at bottom

    tris = GeomTriangles(Geom.UH_dynamic)
    for i in range(n):
        ax, ay, az = float(a[i, 0]), float(a[i, 1]), float(a[i, 2])
        bx, by, bz = float(b[i, 0]), float(b[i, 1]), float(b[i, 2])
        w = float(width[i])
        br = float(bright[i])
        t0 = float(along[i])
        # alongT for the b-end uses b's own depth so the ribbon reveals
        # smoothly along its length.
        t1 = float((z_top - b[i, 2]) / z_span)
        # 4 verts: (a,side-1)(a,side+1)(b,side+1)(b,side-1).
        for px, py, pz, ox, oy, oz, side, t in (
            (ax, ay, az, bx, by, bz, -1.0, t0),
            (ax, ay, az, bx, by, bz, +1.0, t0),
            (bx, by, bz, ax, ay, az, +1.0, t1),
            (bx, by, bz, ax, ay, az, -1.0, t1),
        ):
            vw.add_data3(px, py, pz)
            ow.add_data3(ox, oy, oz)
            rw.add_data4(side, t, w, br)
        base = i * 4
        tris.add_vertices(base + 0, base + 1, base + 2)
        tris.add_vertices(base + 0, base + 2, base + 3)

    geom_obj = Geom(vdata)
    geom_obj.add_primitive(tris)
    gn: GeomNode = bolt.node.node()
    gn.remove_all_geoms()
    gn.add_geom(geom_obj)
    big = 1.0e9
    gn.set_bounds(BoundingBox(LPoint3(-big, -big, -big), LPoint3(big, big, big)))
    gn.set_final(True)

    bolt.active = True
    bolt.age_s = 0.0
    bolt.intensity = float(intensity)
    bolt.life_s = _LEADER_S + _RETURN_S + _AFTERGLOW_S
    bolt.channel_len = 1.0
    # One or two seeded restrikes during the afterglow.
    rng = for_domain("weather", "bolt", int(seed), "restrike")
    n_re = int(rng.integers(1, 3))  # 1 or 2
    t_re = _LEADER_S + _RETURN_S
    bolt.restrikes = []
    for _ in range(n_re):
        t_re += _RESTRIKE_GAP_S * float(rng.uniform(1.0, 2.2))
        if t_re < bolt.life_s:
            bolt.restrikes.append(t_re)
    bolt.node.show()
    bolt.node.set_shader_input("u_width_scale", _WIDTH_SCALE_M * (0.7 + 0.6 * intensity))


def advance_bolt(bolt: _Bolt, dt: float) -> None:
    """Step one bolt's reveal + flash envelope; retire it at end of life.

    Extracted from ``LightningRendererComponent._advance_bolt``
    (lightning_renderer.py).

    Docs: docs/systems/render.sky._impl.md
    """
    bolt.age_s += dt
    if bolt.age_s >= bolt.life_s:
        bolt.active = False
        bolt.node.hide()
        bolt.node.set_shader_input("u_flash", 0.0)
        return

    reveal, flash = bolt_envelope(bolt)
    bolt.node.set_shader_input("u_reveal", float(reveal))
    bolt.node.set_shader_input("u_flash", float(flash * (0.5 + bolt.intensity)))


def bolt_envelope(bolt: _Bolt) -> tuple[float, float]:
    """(reveal 0..1, flash HDR) for a bolt at its current age.

    Extracted from ``LightningRendererComponent._envelope``
    (lightning_renderer.py).

    Docs: docs/systems/render.sky._impl.md
    """
    t = bolt.age_s
    if t < _LEADER_S:
        # Leader: reveal the channel top-down, flickering.
        reveal = t / _LEADER_S
        flicker = 0.6 + 0.4 * abs(math.sin(t * 90.0))
        return reveal, _LEADER_FLASH * flicker
    reveal = 1.0
    tr = t - _LEADER_S
    if tr < _RETURN_S:
        # Return stroke: full channel, bright.
        return reveal, _RETURN_FLASH
    # Afterglow: exponential decay, with seeded restrike spikes.
    glow_t = tr - _RETURN_S
    flash = _RETURN_FLASH * math.exp(-glow_t * 6.0) * 0.5
    for rt in bolt.restrikes:
        d = abs(t - rt)
        if d < 0.04:
            flash = max(flash, _RESTRIKE_FLASH * (1.0 - d / 0.04))
    return reveal, flash


def bolt_sky_flash(bolt: _Bolt) -> float:
    """The sky/cloud flash-pulse contribution of one bolt this frame.

    Extracted from ``LightningRendererComponent._bolt_sky_flash``
    (lightning_renderer.py).

    Docs: docs/systems/render.sky._impl.md
    """
    _, flash = bolt_envelope(bolt)
    # Normalise the bolt flash (peak ~_RETURN_FLASH) to the sky pulse range,
    # scaled by the strike intensity.
    return min(_SKY_FLASH_PEAK, _SKY_FLASH_PEAK * (flash / _RETURN_FLASH) * (0.5 + bolt.intensity))


def add_flash_light(
    self_obj: LightningRendererComponent,
    pos: tuple[float, float, float],
    intensity: float,
) -> None:
    """Register a short-lived PointLight at the strike (fades via ttl_s).

    Extracted from ``LightningRendererComponent._add_flash_light``
    (lightning_renderer.py).

    Docs: docs/systems/render.sky._impl.md
    """
    lights = getattr(self_obj.lighting_pipeline, "lights", None)
    if lights is None:
        return
    from fire_engine.lighting.lights import PointLight

    # Lift the light a little above the strike point so it isn't buried.
    lit_pos = (float(pos[0]), float(pos[1]), float(pos[2]) + 6.0)
    lights.add(
        PointLight(
            position=lit_pos,
            color=_LIGHT_COLOR,
            intensity=_LIGHT_INTENSITY * (0.6 + 0.6 * intensity),
            radius=_LIGHT_RADIUS_M,
            ttl_s=_LIGHT_TTL_S,
        )
    )


def refresh_cover(self_obj: LightningRendererComponent) -> None:
    """Recenter + rebuild the cover heightmap when the player roams far.

    Extracted from ``LightningRendererComponent._refresh_cover``
    (lightning_renderer.py).

    Docs: docs/systems/render.sky._impl.md
    """
    cover = self_obj._cover
    if cover is None:
        return
    cam = _camera_pos(self_obj)
    ox, oy = cover.origin_m
    cx_center = ox + 0.5 * cover.span_m
    cy_center = oy + 0.5 * cover.span_m
    if (
        not self_obj._cover_committed
        or abs(cam[0] - cx_center) > self_obj._recenter_threshold_m
        or abs(cam[1] - cy_center) > self_obj._recenter_threshold_m
    ):
        chunks = (
            getattr(self_obj.chunk_provider, "chunks", {})
            if self_obj.chunk_provider is not None
            else {}
        )
        cover.recenter((cam[0], cam[1]))
        cover.rebuild_all(chunks)
        self_obj._cover_committed = True


def cover_z(
    self_obj: LightningRendererComponent,
    x: float,
    y: float,
) -> float | None:
    """World Z (m) of the cover at world XY, or None if outside / unknown.

    Extracted from ``LightningRendererComponent._cover_z``
    (lightning_renderer.py).

    Docs: docs/systems/render.sky._impl.md
    """
    from fire_engine.world.terrain.rain_cover import OPEN_SKY_Z

    cover = self_obj._cover
    if cover is None or not self_obj._cover_committed:
        return None
    ox, oy = cover.origin_m
    col = math.floor((x - ox) / cover.cell_m)
    row = math.floor((y - oy) / cover.cell_m)
    if 0 <= col < cover.cells and 0 <= row < cover.cells:
        z = float(cover.height[row, col])
        if z > OPEN_SKY_Z * 0.5:  # a real solid voxel (not the sentinel)
            return z
    return None


def _camera_pos(
    self_obj: LightningRendererComponent,
) -> tuple[float, float, float]:
    """World-space camera position in meters."""
    go = getattr(self_obj.base, "camera_go", None)
    if go is not None:
        p = go.transform.position
        return float(p.x), float(p.y), float(p.z)
    cp = self_obj.base.camera.get_pos(self_obj.base.render)
    return float(cp.x), float(cp.y), float(cp.z)
