"""
render/sky/_impl/sky_update — per-frame update helpers for SkyRendererComponent.

Extracted from ``sky_renderer.SkyRendererComponent`` to satisfy the 500-line
limit (C0302): ``update_dome``, ``update_shooting_star``, ``update_clouds``,
``update_fog_and_light``.

``sky_renderer`` imports FROM this module; this module does NOT import
sky_renderer (no circular dependency).  The ``TYPE_CHECKING`` guard is used
only for the ``SkyRendererComponent`` annotation in function signatures.

Docs: docs/systems/render.sky.md
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from panda3d.core import (
    LVecBase2f,
    LVecBase3f,
    LVecBase4f,
)

from fire_engine.core.rng import for_domain

if TYPE_CHECKING:
    from fire_engine.render.sky.sky_renderer import SkyRendererComponent

__all__ = [
    "update_clouds",
    "update_dome",
    "update_fog_and_light",
    "update_shooting_star",
]

# Shooting-star schedule constants (mirrored from sky_renderer constants).
_SS_SLOT_GAME_S: float = 1800.0
_SS_DURATION_REAL_S: float = 1.2
_SS_SPAWN_P: float = 0.5
_SS_MIN_STAR_VIS: float = 0.5

_GAME_SECONDS_PER_DAY: float = 86400.0


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _fog_blend(st: Any) -> float:
    """
    Map ``fog_density`` (1/m, ~0.0008 clear … 0.025 heavy) to a 0-1 factor
    for blending the horizon band / clear colour toward the fog colour.
    """
    return _clamp01((float(st.fog_density) - 0.0008) / (0.020 - 0.0008))


def update_dome(
    self_obj: SkyRendererComponent,
    st: Any,
    cx: float,
    cy: float,
    cz: float,
) -> None:
    """Follow the camera (translation only) and push the dome uniforms.

    Extracted from ``SkyRendererComponent._update_dome`` (sky_renderer.py).
    Invariant: ``self_obj._dome_np`` is not None (caller guarantees this).

    Docs: docs/systems/render.sky._impl.md
    """
    dome = self_obj._dome_np
    assert dome is not None
    dome.set_pos(cx, cy, cz)  # NEVER parented under the camera: world-oriented

    sun = st.sun_dir
    moon = st.moon_dir
    dome.set_shader_input("u_sun_dir", LVecBase3f(float(sun.x), float(sun.y), float(sun.z)))
    dome.set_shader_input("u_sun_color", LVecBase3f(*st.sun_color))
    dome.set_shader_input("u_sun_intensity", float(st.sun_intensity))
    dome.set_shader_input("u_moon_dir", LVecBase3f(float(moon.x), float(moon.y), float(moon.z)))
    dome.set_shader_input("u_moon_phase", float(st.moon_phase))
    dome.set_shader_input("u_zenith_color", LVecBase3f(*st.zenith_color))
    dome.set_shader_input("u_horizon_color", LVecBase3f(*st.horizon_color))
    dome.set_shader_input("u_star_visibility", float(st.star_visibility))
    dome.set_shader_input("u_time", float(self_obj._time_s))
    dome.set_shader_input("u_fog_color", LVecBase3f(*st.fog_color))
    # Legacy horizon fog band only on the CPU backend; the froxel fog
    # owns atmosphere depth under external (GPU volumetric) lighting.
    dome.set_shader_input("u_fog_blend", 0.0 if self_obj.external_lighting else _fog_blend(st))

    # Physical-atmosphere per-frame inputs.
    dome.set_shader_input("u_daylight", float(st.daylight))
    gray = min(1.0, 1.6 * float(st.cloud_coverage) * float(st.cloud_density))
    dome.set_shader_input("u_weather_gray", gray)
    illum = 0.5 * (1.0 - math.cos(2.0 * math.pi * float(st.moon_phase)))
    dome.set_shader_input("u_moon_glow", float(illum))

    # Froxel-fog composite: bind the pipeline's integrated texture once
    # it exists (the pipeline is created after the sky GameObject).
    pipeline = getattr(self_obj.base, "lighting_pipeline", None)
    if self_obj.external_lighting and pipeline is not None:
        # Auto-exposure: the dome uses the COMPRESSED adaptation
        # (pipeline.exposure_sky) — terrain brightens fully in the dark,
        # the night sky deepens only slightly (stars keep their contrast).
        dome.set_shader_input(
            "u_exposure",
            float(getattr(pipeline, "exposure_sky", getattr(pipeline, "exposure", 0.9))),
        )
        if getattr(pipeline, "fog_enabled", False):
            if not self_obj._fog_tex_bound:
                dome.set_shader_input("u_fog_integrated", pipeline.fog_integrated_tex)
                dome.set_shader_input("u_fog_enabled", 1.0)
                self_obj._fog_tex_bound = True
            win = self_obj.base.win
            dome.set_shader_input(
                "u_viewport",
                LVecBase2f(float(win.get_x_size()), float(win.get_y_size())),
            )

    # Slow whole-sky star rotation: one revolution per game day.
    rot = 0.0
    if self_obj.clock is not None:
        rot = (float(self_obj.clock.game_time_of_day) / _GAME_SECONDS_PER_DAY) * 2.0 * math.pi
    dome.set_shader_input("u_star_rotation", rot)


def update_shooting_star(
    self_obj: SkyRendererComponent,
    st: Any,
    dt: float,
) -> None:
    """Animate + deterministically schedule shooting stars.

    Game time is divided into 30-game-minute slots; per slot,
    ``for_domain("sky", "shooting_stars", game_day, slot)`` decides spawn
    (p≈0.5) plus start/travel directions, so every run of the same seed
    shows the same meteors at the same in-game moments.  The streak
    animates over ~1.2 real seconds and only spawns while
    ``star_visibility > 0.5``.

    Extracted from ``SkyRendererComponent._update_shooting_star`` (sky_renderer.py).
    Invariant: ``self_obj._dome_np`` is not None (caller guarantees this).

    Docs: docs/systems/render.sky._impl.md
    """
    dome = self_obj._dome_np
    assert dome is not None
    # Animate the active streak.
    if self_obj._ss_progress >= 0.0:
        self_obj._ss_progress += dt / _SS_DURATION_REAL_S
        if self_obj._ss_progress >= 1.0:
            self_obj._ss_progress = -1.0
            dome.set_shader_input("u_ss_active", 0.0)
        else:
            dome.set_shader_input("u_ss_progress", float(self_obj._ss_progress))

    if self_obj.clock is None:
        return
    slot = int(float(self_obj.clock.game_time_of_day) // _SS_SLOT_GAME_S)
    key = (int(self_obj.clock.game_day), slot)
    if key == self_obj._ss_slot:
        return
    self_obj._ss_slot = key
    if float(st.star_visibility) <= _SS_MIN_STAR_VIS or self_obj._ss_progress >= 0.0:
        return

    rng = for_domain("sky", "shooting_stars", key[0], key[1])
    if float(rng.random()) >= _SS_SPAWN_P:
        return
    # Start direction: random azimuth, elevation 20°–70°.
    az = float(rng.random()) * 2.0 * math.pi
    el = math.radians(20.0 + 50.0 * float(rng.random()))
    s = np.array(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)],
        dtype=np.float64,
    )
    # Travel direction: random vector orthogonalised against the start dir.
    az2 = float(rng.random()) * 2.0 * math.pi
    raw = np.array([math.cos(az2), math.sin(az2), -0.6 * float(rng.random())], dtype=np.float64)
    trav = raw - s * float(np.dot(raw, s))
    norm = float(np.linalg.norm(trav))
    if norm < 1e-6:
        return
    trav /= norm
    dome.set_shader_input("u_ss_start", LVecBase3f(float(s[0]), float(s[1]), float(s[2])))
    dome.set_shader_input("u_ss_travel", LVecBase3f(float(trav[0]), float(trav[1]), float(trav[2])))
    dome.set_shader_input("u_ss_active", 1.0)
    dome.set_shader_input("u_ss_progress", 0.0)
    self_obj._ss_progress = 0.0


def update_clouds(
    self_obj: SkyRendererComponent,
    st: Any,
    cx: float,
    cy: float,
    cz: float,
) -> None:
    """Follow the camera and push the volumetric-cloud per-frame uniforms.

    Extracted from ``SkyRendererComponent._update_clouds`` (sky_renderer.py).
    No-ops when ``self_obj._cloud_np`` is None (clouds disabled).

    Docs: docs/systems/render.sky._impl.md
    """
    clouds = self_obj._cloud_np
    if clouds is None:
        return
    # Camera-follow (translation only — the dome must stay world-oriented so
    # the slab intersection uses true world directions).
    clouds.set_pos(cx, cy, cz)
    clouds.set_shader_input("u_cam_pos", LVecBase3f(cx, cy, cz))

    sun = st.sun_dir
    moon = st.moon_dir
    clouds.set_shader_input("u_sun_dir", LVecBase3f(float(sun.x), float(sun.y), float(sun.z)))
    clouds.set_shader_input("u_moon_dir", LVecBase3f(float(moon.x), float(moon.y), float(moon.z)))
    clouds.set_shader_input("u_sun_radiance", LVecBase3f(*st.sun_radiance))
    clouds.set_shader_input("u_moon_radiance", LVecBase3f(*st.moon_radiance))
    clouds.set_shader_input("u_sky_ambient", LVecBase3f(*st.sky_ambient))
    clouds.set_shader_input("u_coverage", _clamp01(float(st.cloud_coverage)))
    clouds.set_shader_input("u_cloud_density", _clamp01(0.75 + 0.25 * float(st.cloud_density)))
    clouds.set_shader_input("u_wind", LVecBase2f(self_obj._wind_x_m, self_obj._wind_y_m))
    clouds.set_shader_input("u_time", float(self_obj._time_s))

    # Legacy (post-off) path tonemaps inside the cloud shader — keep its
    # exposure synced to the dome's compressed auto-exposure.
    pipeline = getattr(self_obj.base, "lighting_pipeline", None)
    if pipeline is not None:
        clouds.set_shader_input(
            "u_exposure",
            float(getattr(pipeline, "exposure_sky", getattr(pipeline, "exposure", 0.9))),
        )


def update_fog_and_light(self_obj: SkyRendererComponent, st: Any) -> None:
    """Exponential fog + clear colour + global terrain light scale.

    Extracted from ``SkyRendererComponent._update_fog_and_light`` (sky_renderer.py).

    Docs: docs/systems/render.sky._impl.md
    """
    if self_obj.external_lighting:
        # GPU pipeline owns terrain light + fog; keep a plain horizon
        # clear colour so un-domed pixels match the sky.
        hr, hg, hb = st.horizon_color
        self_obj.base.set_background_color(hr, hg, hb, 1.0)
        return
    fr, fg, fb = st.fog_color
    if self_obj._fog is not None:
        self_obj._fog.set_exp_density(float(st.fog_density))
        self_obj._fog.set_color(LVecBase4f(fr, fg, fb, 1.0))

    # Clear colour behind everything: horizon blended toward fog.
    blend = _fog_blend(st)
    hr, hg, hb = st.horizon_color
    self_obj.base.set_background_color(
        hr + (fr - hr) * blend, hg + (fg - hg) * blend, hb + (fb - hb) * blend, 1.0
    )

    # Lighting integration: baked vertex sunlight × global day/night scale.
    if self_obj.terrain_root is not None:
        sr, sg, sb = st.terrain_light_scale
        self_obj.terrain_root.set_color_scale(float(sr), float(sg), float(sb), 1.0)
