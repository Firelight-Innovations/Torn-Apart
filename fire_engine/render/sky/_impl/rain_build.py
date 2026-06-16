"""
render/sky/_impl/rain_build — build and update helpers for RainRendererComponent.

Extracted from ``rain_renderer.RainRendererComponent`` to satisfy the 500-line
limit (C0302): ``build_particles``, ``build_cylinders``, ``update_cylinders``.

``rain_renderer`` imports FROM this module; this module does NOT import
rain_renderer (no circular dependency).  The ``TYPE_CHECKING`` guard is used
only for the ``RainRendererComponent`` annotation in function signatures.

Docs: docs/systems/render.sky.md
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from panda3d.core import (
    ColorBlendAttrib,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LVecBase2f,
    LVecBase3f,
    SamplerState,
    Shader,
    Texture,
    TransparencyAttrib,
)

from fire_engine.core.rng import for_domain
from fire_engine.render._impl.quad import build_unit_quad, setup_additive_instanced_node
from fire_engine.render.sky import rain_shaders

if TYPE_CHECKING:
    from fire_engine.render.sky.rain_renderer import RainRendererComponent

__all__ = [
    "build_cylinders",
    "build_particles",
    "update_cylinders",
]

# Streak geometry tuning (mirrored from rain_renderer constants).
_RAIN_BOX_M: float = 36.0
_RAIN_SIZE_M: float = 0.035
_RAIN_LENGTH_M: float = 0.7
_RAIN_FALL_MPS: float = 18.0
_RAIN_TINT: tuple[float, float, float] = (0.62, 0.70, 0.85)

# Cylinder mode geometry.
_CYL_LAYERS: tuple[tuple[float, float], ...] = ((4.0, 1.6), (7.0, 1.15), (11.0, 0.85))
_CYL_HEIGHT_M: float = 14.0
_CYL_SEGMENTS: int = 32
_CYL_TEX_U_M: float = 3.0
_CYL_TEX_V_M: float = 12.0
_CYL_BASE_SCROLL: float = 1.4
_CYL_MAX_TILT_DEG: float = 14.0
_RAIN_HIDE_THRESHOLD: float = 0.05


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _rain_hash_seed() -> int:
    """Deterministic rain-streak instance-chain seed via
    ``for_domain("rain", "particles")``.  Bounded to ``[0, 2**31)`` (Panda3D
    passes shader-input ints as signed)."""
    return int(for_domain("rain", "particles").integers(0, 2**31))


def _build_cylinder(radius_m: float, height_m: float, segments: int) -> GeomNode:
    """One open vertical cylinder for a cylinder-mode rain layer.

    UVs tile the rain-streak texture ~world-scaled; V is mirrored (bottom = high
    v) so a DECREASING per-frame scroll translates the pattern downward — the
    same convention the old sky_renderer cylinder used.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, segments + 1)
    x = (radius_m * np.cos(theta)).astype(np.float32)
    y = (radius_m * np.sin(theta)).astype(np.float32)
    u_tiles = (2.0 * np.pi * radius_m) / _CYL_TEX_U_M
    v_tiles = height_m / _CYL_TEX_V_M
    u = np.linspace(0.0, u_tiles, segments + 1).astype(np.float32)

    n = segments + 1
    fmt = GeomVertexFormat.get_v3t2()
    vdata = GeomVertexData("rain_cyl", fmt, Geom.UH_static)
    vdata.set_num_rows(2 * n)
    vw = GeomVertexWriter(vdata, "vertex")
    tw = GeomVertexWriter(vdata, "texcoord")
    for k in range(n):
        vw.add_data3(float(x[k]), float(y[k]), -0.5 * height_m)
        tw.add_data2(float(u[k]), float(v_tiles))
    for k in range(n):
        vw.add_data3(float(x[k]), float(y[k]), 0.5 * height_m)
        tw.add_data2(float(u[k]), 0.0)
    tris = GeomTriangles(Geom.UH_static)
    for j in range(segments):
        b0, b1 = j, j + 1
        t0, t1 = j + n, j + 1 + n
        tris.add_vertices(b0, b1, t1)
        tris.add_vertices(b0, t1, t0)
    geom = Geom(vdata)
    geom.add_primitive(tris)
    node = GeomNode("rain_cyl_layer")
    node.add_geom(geom)
    return node


def _rain_streak_texture() -> Texture:
    """The ``rain_streak`` procedural texture, repeat-wrapped + linear-filtered."""
    from fire_engine.procedural import get as get_procedural
    from fire_engine.render.bridges.texture_bridge import to_panda_texture

    rgba = get_procedural("rain_streak")
    tex = to_panda_texture(rgba)
    tex.set_wrap_u(Texture.WM_repeat)
    tex.set_wrap_v(Texture.WM_repeat)
    tex.set_minfilter(SamplerState.FT_linear)
    tex.set_magfilter(SamplerState.FT_linear)
    return tex


def build_particles(self_obj: RainRendererComponent, cfg: Any) -> None:
    """Build the GPU-instanced rain particle node.

    Extracted from ``RainRendererComponent._build_particles`` (rain_renderer.py).
    Disables ``self_obj`` (sets ``self_obj.enabled = False``) when
    ``gfx_rain_particles <= 0``; stores the node in ``self_obj._particle_node``.

    Docs: docs/systems/render.sky._impl.md
    """
    count = int(getattr(cfg, "gfx_rain_particles", 0))
    if count <= 0:
        from fire_engine.core import get_logger

        get_logger("world.rain").info(
            "RainRendererComponent: gfx_rain_particles <= 0 — nothing to draw"
        )
        self_obj.enabled = False
        return
    shader = Shader.make(
        Shader.SL_GLSL,
        vertex=rain_shaders.RAIN_PARTICLE_VERTEX,
        fragment=rain_shaders.RAIN_PARTICLE_FRAGMENT,
    )
    geom_node = GeomNode("rain_particles")
    geom_node.add_geom(build_unit_quad("rain_quad"))
    # Parent under terrain_root so wind/fog/camera + the weather-map contract
    # (bound on render, inherited by terrain_root) all arrive automatically.
    node = self_obj.base.terrain_root.attach_new_node(geom_node)
    node.set_shader(shader)
    node.set_instance_count(count)
    node.set_shader_input("u_hash_seed", _rain_hash_seed())
    node.set_shader_input("u_rain_box_m", _RAIN_BOX_M)
    node.set_shader_input("u_rain_size_m", _RAIN_SIZE_M)
    node.set_shader_input("u_rain_length_m", _RAIN_LENGTH_M)
    node.set_shader_input("u_rain_fall_mps", _RAIN_FALL_MPS)
    node.set_shader_input("u_rain_intensity", 0.0)
    node.set_shader_input("u_rain_occlusion", 1.0 if self_obj._occlusion else 0.0)
    node.set_shader_input("u_rain_tint", LVecBase3f(*_RAIN_TINT))
    # u_time_s is the shared animation clock.  Grass binds it on ITS own node
    # (grass_root), so it is NOT inherited on terrain_root — bind + refresh
    # our own copy each frame (the mote_renderer split; the camera u_cam_pos
    # IS inherited from terrain_root, so it needs no rebind).
    node.set_shader_input("u_time_s", 0.0)

    # Additive glow; infinite bounding box so the shader-positioned instances
    # are never culled by the base quad's origin bounds.
    setup_additive_instanced_node(node, geom_node)
    self_obj._particle_node = node


def build_cylinders(self_obj: RainRendererComponent, cfg: Any) -> None:
    """Build the cheap nested-cylinder rain layers.

    Extracted from ``RainRendererComponent._build_cylinders`` (rain_renderer.py).
    Appends to ``self_obj._cyl_layers`` and stores the root in
    ``self_obj._cyl_root``.

    Docs: docs/systems/render.sky._impl.md
    """
    rain_tex = _rain_streak_texture()
    shader = Shader.make(
        Shader.SL_GLSL,
        vertex=rain_shaders.RAIN_CYLINDER_VERTEX,
        fragment=rain_shaders.RAIN_CYLINDER_FRAGMENT,
    )
    # Parent under terrain_root for the inherited weather-map contract.
    root = self_obj.base.terrain_root.attach_new_node("rain_cyl_root")
    for radius_m, scroll_mult in _CYL_LAYERS:
        node = _build_cylinder(radius_m, _CYL_HEIGHT_M, _CYL_SEGMENTS)
        layer = root.attach_new_node(node)
        layer.set_shader(shader)
        layer.set_shader_input("u_rain_tex", rain_tex)
        layer.set_shader_input("u_rain_alpha", 0.0)
        layer.set_shader_input("u_rain_intensity", 0.0)
        layer.set_shader_input("u_rain_occlusion", 1.0 if self_obj._occlusion else 0.0)
        layer.set_shader_input("u_uv_scroll", LVecBase2f(0.0, 0.0))
        layer.set_two_sided(True)
        layer.set_light_off()
        layer.set_depth_write(False)
        layer.set_transparency(TransparencyAttrib.M_none)
        layer.set_attrib(
            ColorBlendAttrib.make(
                ColorBlendAttrib.M_add, ColorBlendAttrib.O_one, ColorBlendAttrib.O_one
            )
        )
        self_obj._cyl_layers.append((layer, scroll_mult))
        self_obj._cyl_scroll.append(0.0)
    root.hide()
    self_obj._cyl_root = root
    self_obj._cyl_visible = False


def update_cylinders(
    self_obj: RainRendererComponent,
    cam: tuple[float, float, float],
    dt: float,
) -> None:
    """Advance cylinder-mode rain: visibility, position, tilt, scroll.

    Extracted from ``RainRendererComponent._update_cylinders``
    (rain_renderer.py).
    Invariant: ``self_obj._cyl_root`` is not None (caller guarantees this in
    cylinder mode after build).

    Docs: docs/systems/render.sky._impl.md
    """
    assert self_obj._cyl_root is not None
    st = getattr(self_obj.sky_system, "state", None)
    ri = float(getattr(st, "rain_intensity", 0.0)) if st is not None else 0.0
    if ri < _RAIN_HIDE_THRESHOLD:
        if self_obj._cyl_visible:
            self_obj._cyl_root.hide()
            self_obj._cyl_visible = False
        return
    if not self_obj._cyl_visible:
        self_obj._cyl_root.show()
        self_obj._cyl_visible = True

    self_obj._cyl_root.set_pos(cam[0], cam[1], cam[2])
    # Slight wind tilt from the SkyState wind direction.
    wx, wy = getattr(st, "wind_dir", (0.0, 1.0))
    heading = math.degrees(math.atan2(-float(wx), float(wy)))
    tilt = min(_CYL_MAX_TILT_DEG, float(getattr(st, "wind_speed", 0.0)) * 1.1)
    self_obj._cyl_root.set_hpr(heading, tilt, 0.0)

    rate = _CYL_BASE_SCROLL * (0.5 + 1.5 * ri)
    alpha = _clamp01(0.12 + 0.38 * ri)
    for i, (layer, mult) in enumerate(self_obj._cyl_layers):
        self_obj._cyl_scroll[i] = (self_obj._cyl_scroll[i] - rate * mult * dt) % 1.0
        layer.set_shader_input("u_uv_scroll", LVecBase2f(0.0, self_obj._cyl_scroll[i]))
        layer.set_shader_input("u_rain_alpha", alpha)
