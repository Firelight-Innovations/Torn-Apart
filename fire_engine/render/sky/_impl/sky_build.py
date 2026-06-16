"""
render/sky/_impl/sky_build — start-time build functions for SkyRendererComponent.

Contains ``build_dome`` and ``build_clouds``, extracted from
``sky_renderer.SkyRendererComponent._build_dome`` / ``._build_clouds`` to satisfy
the 500-line limit (C0302).  Geometry helpers, texture loaders, and constants
live in the sibling ``sky_geom`` module (no circular dependency).

``sky_renderer`` imports FROM this module; this module does NOT import
sky_renderer.  The ``TYPE_CHECKING`` guard is used only for the
``SkyRendererComponent`` annotation in function signatures.

Docs: docs/systems/render.sky.md
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from panda3d.core import (
    ColorBlendAttrib,
    LVecBase2f,
    LVecBase3f,
    SamplerState,
    Shader,
    Texture,
    TransparencyAttrib,
)

from fire_engine.core import get_logger
from fire_engine.core.rng import for_domain
from fire_engine.render.sky import sky_shaders
from fire_engine.render.sky._impl.sky_geom import (
    _DOME_RADIUS_M,
    _DOME_SLICES,
    _DOME_STACKS,
    _VCLOUD_ALT_M,
    _VCLOUD_DETAIL_SIZE,
    _VCLOUD_DETAIL_STR,
    _VCLOUD_DETAIL_TILE_M,
    _VCLOUD_HG,
    _VCLOUD_LIGHT_STEP_M,
    _VCLOUD_SHAPE_SIZE,
    _VCLOUD_SHAPE_TILE_M,
    _VCLOUD_SIGMA,
    _VCLOUD_THICK_M,
    _build_dome_node,
    _fallback_moon,
    _fallback_star_cube,
    _load_or_bake_cloud_noise,
    _sky_texture,
)

if TYPE_CHECKING:
    from fire_engine.render.sky.sky_renderer import SkyRendererComponent

__all__ = ["build_clouds", "build_dome"]

_log = get_logger("world.sky_renderer")


def build_dome(self_obj: SkyRendererComponent, star_count: int) -> None:
    """Build the inverted sky-dome sphere + dome shader + star cubemap.

    Attaches the dome NodePath to ``self_obj.base.render`` and stores it
    in ``self_obj._dome_np``.  Also initialises ``self_obj._fog_tex_bound``.

    Extracted from ``SkyRendererComponent._build_dome`` (sky_renderer.py).

    Docs: docs/systems/render.sky._impl.md
    """
    node = _build_dome_node(_DOME_RADIUS_M, _DOME_STACKS, _DOME_SLICES)
    dome = self_obj.base.render.attach_new_node(node)
    dome.set_bin("background", 10)
    dome.set_depth_write(False)
    dome.set_depth_test(False)
    dome.set_light_off()
    dome.set_color_off()

    shader = Shader.make(
        Shader.SL_GLSL,
        vertex=sky_shaders.SKY_DOME_VERTEX,
        fragment=sky_shaders.SKY_DOME_FRAGMENT,
    )
    dome.set_shader(shader)

    # Night-sky star/galaxy CUBE MAP (no equirect pole distortion) and
    # the tilted celestial axis it rotates about: Polaris elevation ==
    # the world's latitude, seed-derived per world — so the night sky
    # wheels properly across the sky instead of pinwheeling at zenith.
    star_cube: Any = None
    try:
        from fire_engine.procedural import get as get_procedural

        star_cube = get_procedural("night_sky_cube", star_count=star_count)
    except Exception as exc:
        _log.warning("night_sky_cube unavailable (%s) — using fallback", exc)
    if star_cube is None:
        star_cube = _fallback_star_cube(star_count)
    from fire_engine.render.bridges.texture_bridge import to_panda_cubemap

    dome.set_shader_input("u_star_cube", to_panda_cubemap(star_cube))
    lat_rad = math.radians(28.0 + 27.0 * float(for_domain("sky", "celestial_latitude").random()))
    dome.set_shader_input("u_celestial_axis", LVecBase3f(0.0, math.cos(lat_rad), math.sin(lat_rad)))
    # Neutral defaults so the first frame renders with every uniform defined.
    dome.set_shader_input("u_sun_dir", LVecBase3f(0.0, 0.0, 1.0))
    dome.set_shader_input("u_sun_color", LVecBase3f(1.0, 0.95, 0.85))
    dome.set_shader_input("u_sun_intensity", 0.0)
    dome.set_shader_input("u_moon_dir", LVecBase3f(0.0, 0.0, -1.0))
    dome.set_shader_input("u_moon_phase", 0.5)
    dome.set_shader_input("u_zenith_color", LVecBase3f(0.2, 0.35, 0.6))
    dome.set_shader_input("u_horizon_color", LVecBase3f(0.6, 0.7, 0.8))
    dome.set_shader_input("u_star_visibility", 0.0)
    dome.set_shader_input("u_star_rotation", 0.0)
    dome.set_shader_input("u_time", 0.0)
    dome.set_shader_input("u_fog_color", LVecBase3f(0.6, 0.65, 0.7))
    dome.set_shader_input("u_fog_blend", 0.0)
    dome.set_shader_input("u_ss_active", 0.0)
    dome.set_shader_input("u_ss_start", LVecBase3f(0.0, 1.0, 0.3))
    dome.set_shader_input("u_ss_travel", LVecBase3f(1.0, 0.0, 0.0))
    dome.set_shader_input("u_ss_progress", 0.0)

    # Physical-atmosphere additions: procedural moon texture, tonemap
    # exposure (matches the terrain shader), night-floor/weather inputs,
    # and the froxel-fog defaults (a 1x1x1 dummy keeps the sampler3D
    # bound; the GPU pipeline's real texture replaces it in late_update).
    moon_tex = _sky_texture("moon_surface", fallback=_fallback_moon())
    moon_tex.set_minfilter(SamplerState.FT_linear_mipmap_linear)
    moon_tex.set_magfilter(SamplerState.FT_linear)
    dome.set_shader_input("u_moon_tex", moon_tex)
    dome.set_shader_input("u_moon_glow", 0.0)
    dome.set_shader_input("u_daylight", 1.0)
    dome.set_shader_input("u_weather_gray", 0.0)
    cfg = getattr(self_obj.base, "_config", None)
    dome.set_shader_input("u_exposure", float(getattr(cfg, "light_exposure", 0.9)))
    # Config-exposed sky/sun tuning (static — set once; see core/config.py).
    dome.set_shader_input(
        "u_sun_disc_intensity", float(getattr(cfg, "gfx_sun_disc_intensity", 45.0))
    )
    dome.set_shader_input(
        "u_sun_halo_intensity", float(getattr(cfg, "gfx_sun_halo_intensity", 1.8))
    )
    dome.set_shader_input(
        "u_sun_min_brightness", float(getattr(cfg, "gfx_sun_min_brightness", 0.25))
    )
    dome.set_shader_input(
        "u_sky_inscatter_scale", float(getattr(cfg, "gfx_sky_inscatter_scale", 0.9))
    )
    dummy_fog = Texture("dome_fog_dummy")
    dummy_fog.setup_3d_texture(1, 1, 1, Texture.T_float, Texture.F_rgba16)
    dummy_fog.set_clear_color((0.0, 0.0, 0.0, 1.0))
    dome.set_shader_input("u_fog_integrated", dummy_fog)
    dome.set_shader_input("u_fog_enabled", 0.0)
    dome.set_shader_input("u_viewport", LVecBase2f(1280.0, 720.0))
    self_obj._fog_tex_bound = False
    self_obj._dome_np = dome


def build_clouds(self_obj: SkyRendererComponent) -> None:
    """Build the volumetric cloud dome + raymarch shader (static uniforms).

    Reuses the inverted-sphere dome geometry purely to get a per-pixel world
    view direction; the fragment shader analytically intersects + marches the
    cloud slab in world space.  Stores the result in ``self_obj._cloud_np``
    (``None`` when ``gfx_clouds`` is disabled).

    Extracted from ``SkyRendererComponent._build_clouds`` (sky_renderer.py).

    Docs: docs/systems/render.sky._impl.md
    """
    cfg = getattr(self_obj.base, "_config", None)
    if not bool(getattr(cfg, "gfx_clouds", True)):
        self_obj._cloud_np = None
        _log.info("Volumetric clouds disabled (gfx_clouds=false)")
        return

    from fire_engine.render.bridges.texture_bridge import to_panda_texture_3d

    node = _build_dome_node(_DOME_RADIUS_M, _DOME_STACKS, _DOME_SLICES)
    clouds = self_obj.base.render.attach_new_node(node)
    clouds.set_bin("background", 15)  # after dome (10), before terrain
    clouds.set_depth_write(False)
    clouds.set_depth_test(False)
    clouds.set_light_off()
    clouds.set_color_off()
    # Premultiplied OVER: out = src.rgb + dst.rgb · src.a, with src.a =
    # transmittance — a bright sun bleeds through thin cloud, thick occludes.
    clouds.set_transparency(TransparencyAttrib.M_none)
    clouds.set_attrib(
        ColorBlendAttrib.make(
            ColorBlendAttrib.M_add, ColorBlendAttrib.O_one, ColorBlendAttrib.O_incoming_alpha
        )
    )

    shader = Shader.make(
        Shader.SL_GLSL,
        vertex=sky_shaders.CLOUD_VOLUMETRIC_VERTEX,
        fragment=sky_shaders.CLOUD_VOLUMETRIC_FRAGMENT,
    )
    clouds.set_shader(shader)

    # Baked, tileable density volumes (disk-cached — deterministic per seed).
    seed = int(getattr(cfg, "world_seed", 0))
    shape_arr, detail_arr = _load_or_bake_cloud_noise(seed, _VCLOUD_SHAPE_SIZE, _VCLOUD_DETAIL_SIZE)
    clouds.set_shader_input("u_shape", to_panda_texture_3d(shape_arr))
    clouds.set_shader_input("u_detail", to_panda_texture_3d(detail_arr))

    # Static uniforms.
    clouds.set_shader_input("u_altitude", _VCLOUD_ALT_M)
    clouds.set_shader_input("u_thickness", _VCLOUD_THICK_M)
    clouds.set_shader_input("u_max_dist", float(getattr(cfg, "gfx_cloud_max_dist_m", 2400.0)))
    clouds.set_shader_input("u_shape_scale", 1.0 / _VCLOUD_SHAPE_TILE_M)
    clouds.set_shader_input("u_detail_scale", 1.0 / _VCLOUD_DETAIL_TILE_M)
    clouds.set_shader_input("u_detail_strength", _VCLOUD_DETAIL_STR)
    clouds.set_shader_input("u_sigma", _VCLOUD_SIGMA)
    clouds.set_shader_input("u_hg", _VCLOUD_HG)
    clouds.set_shader_input("u_light_step_m", _VCLOUD_LIGHT_STEP_M)
    clouds.set_shader_input("u_steps", int(getattr(cfg, "gfx_cloud_steps", 48)))
    clouds.set_shader_input("u_light_steps", int(getattr(cfg, "gfx_cloud_light_steps", 6)))
    clouds.set_shader_input("u_exposure", float(getattr(cfg, "light_exposure", 0.9)))

    # Per-frame defaults (overwritten in _update_clouds).
    clouds.set_shader_input("u_cam_pos", LVecBase3f(0.0, 0.0, 0.0))
    clouds.set_shader_input("u_sun_dir", LVecBase3f(0.0, 0.0, 1.0))
    clouds.set_shader_input("u_moon_dir", LVecBase3f(0.0, 0.0, -1.0))
    clouds.set_shader_input("u_sun_radiance", LVecBase3f(3.0, 2.9, 2.6))
    clouds.set_shader_input("u_moon_radiance", LVecBase3f(0.06, 0.07, 0.10))
    clouds.set_shader_input("u_sky_ambient", LVecBase3f(0.4, 0.5, 0.7))
    clouds.set_shader_input("u_coverage", 0.5)
    clouds.set_shader_input("u_cloud_density", 1.0)
    clouds.set_shader_input("u_wind", LVecBase2f(0.0, 0.0))
    clouds.set_shader_input("u_time", 0.0)

    # M4 weather-map contract defaults: the WeatherMapComponent binds the
    # real texture + origin + enable on ``render`` (inherited here), but a
    # dummy 1x1 sampler and disabled state keep the shader valid even when
    # that component is absent / the feature is off (pre-M4 flat-ambient
    # look).  A bound sampler2D is required (an unbound one is UB).
    dummy_wmap = Texture("weather_map_dummy")
    dummy_wmap.setup_2d_texture(1, 1, Texture.T_half_float, Texture.F_rgba16)
    dummy_wmap.set_clear_color((0.0, 0.0, 0.0, 0.0))
    clouds.set_shader_input("u_weather_map", dummy_wmap)
    clouds.set_shader_input("u_wmap_origin", LVecBase2f(0.0, 0.0))
    clouds.set_shader_input("u_wmap_cell_m", 1.0)
    clouds.set_shader_input("u_wmap_cells", 1.0)
    clouds.set_shader_input("u_weather_map_enabled", 0)
    clouds.set_shader_input("u_weather_ambient", LVecBase2f(0.0, 0.0))
    clouds.set_shader_input("u_virga_enabled", 0)

    # M9 WMO cloud genera: layered high/mid/low altitude bands derived
    # in-shader from the weather map (no new texture data).  All static —
    # config tunables pushed once here; the band selection is per-step in
    # the shader from the existing coverage/density/precip channels.  Gated
    # by gfx_cloud_genera (requires gfx_weather_map; off ⇒ single slab, the
    # pre-M9 look — the shader's u_cloud_genera_enabled==0 path).
    genera_on = bool(getattr(cfg, "gfx_cloud_genera", False)) and bool(
        getattr(cfg, "gfx_weather_map", False)
    )
    clouds.set_shader_input("u_cloud_genera_enabled", 1 if genera_on else 0)
    clouds.set_shader_input(
        "u_genera_high_alt", float(getattr(cfg, "cloud_genera_high_alt_m", 1400.0))
    )
    clouds.set_shader_input(
        "u_genera_high_thick", float(getattr(cfg, "cloud_genera_high_thick_m", 120.0))
    )
    clouds.set_shader_input(
        "u_genera_mid_alt", float(getattr(cfg, "cloud_genera_mid_alt_m", 850.0))
    )
    clouds.set_shader_input(
        "u_genera_mid_thick", float(getattr(cfg, "cloud_genera_mid_thick_m", 220.0))
    )
    clouds.set_shader_input(
        "u_genera_high_floor", float(getattr(cfg, "cloud_genera_high_cov_floor", 0.06))
    )
    clouds.set_shader_input(
        "u_genera_high_cov_w", float(getattr(cfg, "cloud_genera_high_cov_weight", 0.35))
    )
    clouds.set_shader_input(
        "u_genera_high_density", float(getattr(cfg, "cloud_genera_high_density", 0.30))
    )
    clouds.set_shader_input(
        "u_genera_mid_cov_w", float(getattr(cfg, "cloud_genera_mid_cov_weight", 0.60))
    )
    clouds.set_shader_input(
        "u_genera_high_detail", float(getattr(cfg, "cloud_genera_high_detail_scale", 0.45))
    )
    clouds.set_shader_input(
        "u_genera_mid_detail", float(getattr(cfg, "cloud_genera_mid_detail_scale", 0.85))
    )
    self_obj._cloud_np = clouds
