"""
Surface-shader input binding for GpuLightingPipeline.

Extracted from ``fire_engine.lighting.gpu`` to keep that module under the
500-line limit.  All functions receive the pipeline instance as their first
argument and operate on its private attributes.

Docs: docs/systems/lighting.md
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from panda3d.core import LVecBase2f, LVecBase3f, LVecBase4f, NodePath

from fire_engine.lighting.lights import MAX_OCCLUDERS
from fire_engine.lighting.volume import EMISSION_SCALE

if TYPE_CHECKING:
    from fire_engine.lighting.gpu import GpuLightingPipeline
    from fire_engine.world.sky.sky_state import SkyState

__all__ = ["bind_surface_inputs", "sky_inputs", "update_surface_inputs"]


def bind_surface_inputs(pipeline: GpuLightingPipeline, node: NodePath) -> None:
    """
    Bind the static lighting samplers/uniforms onto a render NodePath.

    Call once at boot with ``app.render`` (main.py does) so every shader
    that includes ``world/shaders/lit_surface.glsl`` — terrain, foliage,
    future buildings/NPCs — inherits the contract scene-graph-wide.
    Per-frame values are refreshed by :func:`update_surface_inputs`.
    Shaders that don't declare these uniforms simply ignore them.

    Docs: docs/systems/lighting.md
    """
    c0, c1, c2 = pipeline.cascades
    node.set_shader_input("u_c0_geom", c0.geom)
    node.set_shader_input("u_c0_vis", c0.vis)
    node.set_shader_input("u_c0_cells", float(c0.cells))
    node.set_shader_input("u_c0_cell_m", float(c0.cell_m))
    node.set_shader_input("u_c1_geom", c1.geom)
    node.set_shader_input("u_c1_vis", c1.vis)
    node.set_shader_input("u_c1_cells", float(c1.cells))
    node.set_shader_input("u_c1_cell_m", float(c1.cell_m))
    node.set_shader_input("u_c2_geom", c2.geom)
    node.set_shader_input("u_c2_vis", c2.vis)
    node.set_shader_input("u_c2_cells", float(c2.cells))
    node.set_shader_input("u_c2_cell_m", float(c2.cell_m))
    node.set_shader_input("u_quant_m", float(pipeline._config.light_quant_m))
    # Celestial penumbra cone half-angle, as a tangent (the refinement
    # march jitters its rays inside this cone for smooth soft edges).
    node.set_shader_input(
        "u_penumbra_tan",
        float(math.tan(math.radians(pipeline._config.light_penumbra_deg))),
    )
    node.set_shader_input("u_ao_strength", float(pipeline._config.light_ao_strength))
    node.set_shader_input("u_emission_scale", float(EMISSION_SCALE))
    node.set_shader_input("u_fog_near", pipeline._fog_near)
    node.set_shader_input("u_fog_far", pipeline._fog_far)
    node.set_shader_input("u_fog_enabled", 1.0 if pipeline.fog_enabled else 0.0)
    if pipeline.fog_enabled:
        node.set_shader_input("u_fog_integrated", pipeline.fog_integrated_tex)
    else:
        # Bind *something* valid for the sampler.
        node.set_shader_input("u_fog_integrated", pipeline.cascades[0].vis)
    # Radiance/origins are per-frame (ping-pong + window scroll).
    update_surface_inputs(pipeline, node, None)


def update_surface_inputs(
    pipeline: GpuLightingPipeline, node: NodePath, sky_state: SkyState | None
) -> None:
    """
    Refresh the per-frame lighting uniforms on a render NodePath.

    Parameters
    ----------
    pipeline : GpuLightingPipeline
        The owning pipeline instance.
    node : NodePath
        Same node given to :func:`bind_surface_inputs` (``app.render``).
    sky_state : SkyState | None
        For sun/moon direction + radiance uniforms.

    Docs: docs/systems/lighting.md
    """
    c0, c1, c2 = pipeline.cascades
    node.set_shader_input("u_c0_radiance", c0.radiance_current)
    node.set_shader_input("u_c1_radiance", c1.radiance_current)
    node.set_shader_input("u_c2_radiance", c2.radiance_current)
    node.set_shader_input("u_c0_emis", c0.emis)
    # Auto-exposure: the adapted tonemap exposure changes every frame.
    node.set_shader_input("u_exposure", float(pipeline.exposure))
    if c0.window.origin_cell is not None:
        node.set_shader_input("u_c0_origin_m", LVecBase3f(*c0.origin_m()))
    if c1.window.origin_cell is not None:
        node.set_shader_input("u_c1_origin_m", LVecBase3f(*c1.origin_m()))
    if c2.window.origin_cell is not None:
        node.set_shader_input("u_c2_origin_m", LVecBase3f(*c2.origin_m()))
    sun_dir, sun_rad, moon_dir, moon_rad, sky_amb = pipeline._sky_inputs(sky_state)
    node.set_shader_input("u_sun_dir", LVecBase3f(*sun_dir))
    node.set_shader_input("u_sun_radiance", LVecBase3f(*sun_rad))
    node.set_shader_input("u_moon_dir", LVecBase3f(*moon_dir))
    node.set_shader_input("u_moon_radiance", LVecBase3f(*moon_rad))
    node.set_shader_input("u_sky_ambient", LVecBase3f(*sky_amb))
    # Dynamic occluder boxes for the shadow-refinement march (same packed
    # lists the inject pass uses; ``update`` repacks them on version bump).
    if pipeline._box_uniforms is not None:
        box_min, box_max, n_boxes = pipeline._box_uniforms
    else:
        box_min, box_max, n_boxes = [], [], 0
    if not box_min:  # GLSL arrays must always be bound (Panda asserts)
        box_min = [LVecBase4f(0.0)] * MAX_OCCLUDERS
        box_max = [LVecBase4f(0.0)] * MAX_OCCLUDERS
    node.set_shader_input("u_num_boxes", int(n_boxes))
    node.set_shader_input("u_box_min", box_min)
    node.set_shader_input("u_box_max", box_max)
    win = pipeline._base.win
    node.set_shader_input(
        "u_viewport", LVecBase2f(float(win.get_x_size()), float(win.get_y_size()))
    )
    # Radians of view angle per screen pixel — lets the surface shader
    # compute its texel footprint ANALYTICALLY (dist * u_px_rad / cos i)
    # instead of with fwidth().  Screen-space derivatives are evaluated on
    # 2x2 pixel quads; where a quad straddles two facets of the faceted
    # terrain mesh the helper pixels extrapolate the wrong plane and the
    # derivatives explode, which made every facet edge of a crater/cliff
    # sparkle as the camera moved (see world.md gotcha 22).
    node.set_shader_input(
        "u_px_rad",
        float(
            math.radians(pipeline._base.camLens.get_fov()[0]) / max(1.0, float(win.get_x_size()))
        ),
    )
    cam_pos = pipeline._base.camera.get_pos(pipeline._base.render)
    node.set_shader_input("u_cam_pos", LVecBase3f(cam_pos[0], cam_pos[1], cam_pos[2]))


def sky_inputs(sky_state: SkyState | None) -> tuple[Any, ...]:
    """
    Extract ``(sun_dir, sun_radiance, moon_dir, moon_radiance, sky_ambient)``
    from a SkyState, with graceful fallbacks for older SkyState versions
    (radiance derived from sun_color × intensity) and for ``None``.

    Docs: docs/systems/lighting.md
    """
    if sky_state is None:
        return (
            (0.3, 0.2, 0.93),
            (3.0, 2.9, 2.6),
            (0.0, 0.0, -1.0),
            (0.0, 0.0, 0.0),
            (0.35, 0.45, 0.70),
        )
    sun_dir = tuple(
        float(v) for v in (sky_state.sun_dir.x, sky_state.sun_dir.y, sky_state.sun_dir.z)
    )
    moon_dir = tuple(
        float(v) for v in (sky_state.moon_dir.x, sky_state.moon_dir.y, sky_state.moon_dir.z)
    )
    sun_rad = getattr(sky_state, "sun_radiance", None)
    if sun_rad is None:
        s = float(sky_state.sun_intensity) * 3.2
        sun_rad = tuple(c * s for c in sky_state.sun_color)
    moon_rad = getattr(sky_state, "moon_radiance", None)
    if moon_rad is None:
        up = max(moon_dir[2], 0.0)
        full = 1.0 - abs(sky_state.moon_phase - 0.5) * 2.0
        moon_rad = (0.05 * up * full, 0.06 * up * full, 0.09 * up * full)
    sky_amb = getattr(sky_state, "sky_ambient", None)
    if sky_amb is None:
        d = float(sky_state.daylight)
        z = sky_state.zenith_color
        sky_amb = (0.02 + z[0] * 0.55 * d, 0.02 + z[1] * 0.6 * d, 0.03 + z[2] * 0.75 * d)
    return (
        sun_dir,
        tuple(map(float, sun_rad)),
        moon_dir,
        tuple(map(float, moon_rad)),
        tuple(map(float, sky_amb)),
    )
