"""
Injection and GI-gather compute-shader dispatch for GpuLightingPipeline.

Extracted from ``fire_engine.lighting.gpu`` to keep that module under the
500-line limit.  All functions receive the pipeline instance as their first
argument and operate on its private attributes.

Docs: docs/systems/lighting.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from panda3d.core import LVecBase3f, LVecBase4f, ShaderAttrib

from fire_engine.lighting import glsl

if TYPE_CHECKING:
    from fire_engine.lighting.gpu import GpuLightingPipeline

__all__ = ["inject_and_gather"]


def inject_and_gather(
    pipeline: GpuLightingPipeline,
    sun: Any,
    packed: Any,
    count: int,
    box_min: Any,
    box_max: Any,
    n_boxes: int,
    engine: Any,
    gsg: Any,
) -> None:
    """Run injection + GI gather for every cascade that needs it.

    Parameters
    ----------
    pipeline : GpuLightingPipeline
        The owning pipeline instance.
    sun : tuple
        ``(sun_dir, sun_rad, moon_dir, moon_rad, sky_amb)`` from
        ``_sky_inputs``.
    packed : numpy.ndarray
        Packed light array from ``LightSet.pack``.
    count : int
        Number of active lights in ``packed``.
    box_min, box_max : list[LVecBase4f]
        Pre-packed occluder AABB corner lists.
    n_boxes : int
        Active box count.
    engine : GraphicsEngine
        Panda3D graphics engine for compute dispatch.
    gsg : GraphicsStateGuardian
        Active GSG.

    Docs: docs/systems/lighting.md
    """
    from fire_engine.lighting.gpu import _groups  # local to avoid circular

    pos_r = [LVecBase4f(*packed[i, 0:4]) for i in range(glsl.MAX_LIGHTS)]
    col_t = [LVecBase4f(*packed[i, 4:8]) for i in range(glsl.MAX_LIGHTS)]
    ext = [LVecBase4f(*packed[i, 8:12]) for i in range(glsl.MAX_LIGHTS)]
    sun_dir, sun_rad, moon_dir, moon_rad, sky_amb = sun
    for casc in pipeline.cascades:
        if not casc.needs_inject:
            continue
        casc.needs_inject = False
        n = casc.inject_np
        n.set_shader_input("u_sun_dir", LVecBase3f(*sun_dir))
        n.set_shader_input("u_sun_radiance", LVecBase3f(*sun_rad))
        n.set_shader_input("u_moon_dir", LVecBase3f(*moon_dir))
        n.set_shader_input("u_moon_radiance", LVecBase3f(*moon_rad))
        n.set_shader_input("u_sky_ambient", LVecBase3f(*sky_amb))
        n.set_shader_input("u_bounce", float(pipeline._config.light_bounce_strength))
        n.set_shader_input("u_num_lights", int(count))
        n.set_shader_input("u_light_pos_r", pos_r)
        n.set_shader_input("u_light_col_t", col_t)
        n.set_shader_input("u_light_ext", ext)
        n.set_shader_input("u_num_boxes", n_boxes)
        n.set_shader_input("u_box_min", box_min)
        n.set_shader_input("u_box_max", box_max)
        n.set_shader_input("u_origin_m", LVecBase3f(*casc.origin_m()))
        n.set_shader_input("u_cell_m", float(casc.cell_m))
        groups = (_groups(casc.cells, 4),) * 3
        engine.dispatch_compute(groups, n.get_attrib(ShaderAttrib), gsg)
        # Gather (ray-marched GI) over the fresh source field.
        for _ in range(pipeline._gi_iters):
            gn = casc.gather_np[casc.ping]  # ping → pong
            gn.set_shader_input("u_sky_ambient", LVecBase3f(*sky_amb))
            gn.set_shader_input("u_origin_m", LVecBase3f(*casc.origin_m()))
            engine.dispatch_compute(groups, gn.get_attrib(ShaderAttrib), gsg)
            casc.ping ^= 1
        # Smooth (air-masked de-noise of the ray component).
        for _ in range(pipeline._gi_smooth):
            sm = casc.smooth_np[casc.ping]  # ping → pong
            engine.dispatch_compute(groups, sm.get_attrib(ShaderAttrib), gsg)
            casc.ping ^= 1
