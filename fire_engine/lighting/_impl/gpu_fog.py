"""
Froxel volumetric-fog compute-shader dispatch for GpuLightingPipeline.

Extracted from ``fire_engine.lighting.gpu`` to keep that module under the
500-line limit.  All functions receive the pipeline instance as their first
argument and operate on its private attributes.

Docs: docs/systems/lighting.md
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from panda3d.core import LVecBase2f, LVecBase3f, ShaderAttrib

# Multiplier turning the weather's subtle exponential fog density into a
# visually volumetric medium (tuned in-game).
_FOG_DENSITY_BOOST = 2.0

if TYPE_CHECKING:
    from fire_engine.lighting.gpu import GpuLightingPipeline

__all__ = ["dispatch_fog", "setup_fog"]


def dispatch_fog(
    pipeline: GpuLightingPipeline,
    camera_pos: Any,
    sun: Any,
    sky_state: Any,
    engine: Any,
    gsg: Any,
) -> None:
    """Fill + integrate the froxel volume for this frame's camera.

    Parameters
    ----------
    pipeline : GpuLightingPipeline
        The owning pipeline instance.
    camera_pos : sequence of 3 floats
        Camera world position (meters).
    sun : tuple
        ``(sun_dir, sun_rad, moon_dir, moon_rad, sky_amb)`` from
        ``_sky_inputs``.
    sky_state : SkyState | None
        Current sky snapshot for fog density.
    engine : GraphicsEngine
        Panda3D graphics engine for compute dispatch.
    gsg : GraphicsStateGuardian
        Active GSG.

    Docs: docs/systems/lighting.md
    """
    from fire_engine.lighting.gpu import _groups  # local to avoid circular

    sun_dir, sun_rad, moon_dir, moon_rad, sky_amb = sun
    cam = pipeline._base.camera
    quat = cam.get_quat(pipeline._base.render)
    fwd = quat.get_forward()
    right = quat.get_right()
    up = quat.get_up()
    lens = pipeline._base.camLens
    fov = lens.get_fov()  # degrees (h, v)
    tan_h = math.tan(math.radians(float(fov[0]) * 0.5))
    tan_v = math.tan(math.radians(float(fov[1]) * 0.5))

    density = 0.0015
    if sky_state is not None:
        density = float(sky_state.fog_density) * _FOG_DENSITY_BOOST

    c1 = pipeline.cascades[1]
    sn = pipeline._fog_scatter_np
    sn.set_shader_input("u_cam_pos", LVecBase3f(*[float(camera_pos[i]) for i in range(3)]))
    sn.set_shader_input("u_cam_fwd", LVecBase3f(fwd[0], fwd[1], fwd[2]))
    sn.set_shader_input("u_cam_right", LVecBase3f(right[0], right[1], right[2]))
    sn.set_shader_input("u_cam_up", LVecBase3f(up[0], up[1], up[2]))
    sn.set_shader_input("u_tan_half_fov", LVecBase2f(tan_h, tan_v))
    sn.set_shader_input("u_fog_density", float(density))
    sn.set_shader_input("u_sun_dir", LVecBase3f(*sun_dir))
    sn.set_shader_input("u_sun_radiance", LVecBase3f(*sun_rad))
    sn.set_shader_input("u_moon_dir", LVecBase3f(*moon_dir))
    sn.set_shader_input("u_moon_radiance", LVecBase3f(*moon_rad))
    sn.set_shader_input("u_sky_ambient", LVecBase3f(*sky_amb))
    sn.set_shader_input("u_c1_radiance", c1.radiance_current)
    sn.set_shader_input("u_c1_origin_m", LVecBase3f(*c1.origin_m()))
    assert pipeline._box_uniforms is not None
    box_min, box_max, n_boxes = pipeline._box_uniforms
    sn.set_shader_input("u_num_boxes", n_boxes)
    sn.set_shader_input("u_box_min", box_min)
    sn.set_shader_input("u_box_max", box_max)

    w, h, z = pipeline._fog_dim
    engine.dispatch_compute((_groups(w, 8), _groups(h, 8), z), sn.get_attrib(ShaderAttrib), gsg)
    engine.dispatch_compute(
        (_groups(w, 8), _groups(h, 8), 1),
        pipeline._fog_integrate_np.get_attrib(ShaderAttrib),
        gsg,
    )


def setup_fog(pipeline: GpuLightingPipeline) -> None:
    """
    Allocate froxel fog textures and configure the scatter/integrate nodes.

    Called once from ``GpuLightingPipeline.__init__`` when ``fog_enabled``
    is True.  Writes ``pipeline.fog_scatter_tex``,
    ``pipeline.fog_integrated_tex``, ``pipeline._fog_scatter_np``, and
    ``pipeline._fog_integrate_np``.

    Docs: docs/systems/lighting.md
    """
    from panda3d.core import LVecBase3i, SamplerState, Shader, Texture

    from fire_engine.lighting import glsl

    config = pipeline._config
    w, h, z = pipeline._fog_dim

    pipeline.fog_scatter_tex = Texture("fog_scatter")
    pipeline.fog_scatter_tex.setup_3d_texture(w, h, z, Texture.T_float, Texture.F_rgba16)
    pipeline.fog_integrated_tex = Texture("fog_integrated")
    pipeline.fog_integrated_tex.setup_3d_texture(w, h, z, Texture.T_float, Texture.F_rgba16)
    for t in (pipeline.fog_scatter_tex, pipeline.fog_integrated_tex):
        t.set_clear_color((0, 0, 0, 1))
        t.set_keep_ram_image(False)
        t.set_minfilter(SamplerState.FT_linear)
        t.set_magfilter(SamplerState.FT_linear)
        t.set_wrap_u(SamplerState.WM_clamp)
        t.set_wrap_v(SamplerState.WM_clamp)
        t.set_wrap_w(SamplerState.WM_clamp)

    from panda3d.core import NodePath

    pipeline._fog_scatter_np = NodePath("fog_scatter")
    pipeline._fog_scatter_np.set_shader(
        Shader.make_compute(Shader.SL_GLSL, glsl.FOG_SCATTER_COMPUTE)
    )
    sn = pipeline._fog_scatter_np
    sn.set_shader_input("u_froxels", pipeline.fog_scatter_tex)
    sn.set_shader_input("u_froxel_dim", LVecBase3i(w, h, z))
    sn.set_shader_input("u_fog_near", pipeline._fog_near)
    sn.set_shader_input("u_fog_far", pipeline._fog_far)
    sn.set_shader_input("u_ground_z", float(config.ground_height_m))
    sn.set_shader_input("u_anisotropy", float(config.fog_anisotropy))
    c1 = pipeline.cascades[1]
    sn.set_shader_input("u_c1_vis", c1.vis)
    sn.set_shader_input("u_c1_cells", float(c1.cells))
    sn.set_shader_input("u_c1_cell_m", float(c1.cell_m))

    pipeline._fog_integrate_np = NodePath("fog_integrate")
    pipeline._fog_integrate_np.set_shader(
        Shader.make_compute(Shader.SL_GLSL, glsl.FOG_INTEGRATE_COMPUTE)
    )
    fi = pipeline._fog_integrate_np
    fi.set_shader_input("u_froxels", pipeline.fog_scatter_tex)
    fi.set_shader_input("u_integrated", pipeline.fog_integrated_tex)
    fi.set_shader_input("u_froxel_dim", LVecBase3i(w, h, z))
    fi.set_shader_input("u_fog_near", pipeline._fog_near)
    fi.set_shader_input("u_fog_far", pipeline._fog_far)
