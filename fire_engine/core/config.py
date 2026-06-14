"""
core/config.py — Typed, frozen engine configuration.

Loads values from ``config.toml`` (stdlib tomllib, Python 3.11+) and exposes
them as a single frozen ``Config`` dataclass.  All engine code reads config
through this module — no magic numbers.

Flat fields
-----------
    world_seed           : int   — RNG seed for procedural systems (textures,
                                   ambient noise, NPC behaviour) — NOT terrain
    world_size_m         : float — square world footprint side length, meters,
                                   centred on origin (1000 m = 1 km)
    ground_height_m      : float — flat baseline ground surface height (world Z)
    voxel_size           : float — meters per voxel edge (0.5 m)
    chunk_size           : int   — voxels per chunk edge (32 → 16 m cube)
    light_grid_scale     : int   — terrain voxels per light cell (2 → 1 m cells)
    view_distance_chunks : int   — streaming radius in chunks (XY)
    fixed_dt             : float — fixed-update timestep in seconds (0.02 s = 50 Hz)

Debug flags (from [debug] table)
---------------------------------
    show_fps             : bool
    show_chunk_borders   : bool
    show_light_grid      : bool
    debug_wind_ball      : bool  — spawn the dev wind-field physics ball

Sky fields (from [sky] table, flattened like [debug])
------------------------------------------------------
    sky_cloud_altitude_m  : float — cloud layer base altitude (world Z, meters)
    sky_cloud_thickness_m : float — vertical thickness of the cloud layer (meters)
    sky_cloud_cell_m      : float — horizontal size of one cloud cell (meters)
    sky_star_count        : int   — number of stars in the "night_sky" texture

Derived read-only properties
-----------------------------
    chunk_meters         : float — chunk_size * voxel_size  (16.0 m)
    light_cell_meters    : float — voxel_size * light_grid_scale (1.0 m)

Example
-------
    from fire_engine.core.config import load_config

    cfg = load_config("config.toml")
    print(cfg.chunk_meters)   # 16.0
"""

from __future__ import annotations

import tomllib
import warnings
from dataclasses import dataclass, field


# ----------------------------------------------------------------------------
# Graphics-quality presets.  Each maps the heavy/quality-dependent ``gfx_*``
# knobs to a value; aesthetic constants (bloom threshold/knee/strength, cloud
# max distance) are intentionally omitted so they fall back to the dataclass
# default and stay consistent across presets.  ``"high"`` mirrors the Config
# dataclass defaults exactly.  Tune for the target machine via config.toml.
# ----------------------------------------------------------------------------
GRAPHICS_PRESETS: dict[str, dict] = {
    "off": {
        "gfx_post_process": False,
        "gfx_hdr_format": "rgba8",
        "gfx_render_scale": 1.0,
        "gfx_bloom": False,
        "gfx_fxaa": False,
        "gfx_lens_flare": False,
        "gfx_clouds": False,
        "gfx_weather_map": False,
        "gfx_cloud_virga": False,
        "gfx_cloud_genera": False,
        "gfx_god_rays": False,
        "gfx_foliage_shadow_refine": False,
        "gfx_rain_mode": "off",
        "gfx_rain_particles": 0,
        "gfx_rain_occlusion": True,
        "gfx_lightning_bolts": False,
    },
    "low": {
        "gfx_post_process": True,
        "gfx_hdr_format": "rgba16f",
        "gfx_render_scale": 1.0,
        "gfx_bloom": True,
        "gfx_bloom_mips": 3,
        "gfx_fxaa": False,
        "gfx_lens_flare": False,
        "gfx_clouds": True,
        "gfx_cloud_steps": 32,
        "gfx_cloud_light_steps": 4,
        "gfx_cloud_resolution_scale": 0.5,
        "gfx_weather_map": True,
        "gfx_cloud_virga": False,
        "gfx_cloud_genera": False,
        "gfx_god_rays": False,
        "gfx_god_ray_samples": 16,
        "gfx_foliage_shadow_refine": False,
        "gfx_rain_mode": "cylinders",
        "gfx_rain_particles": 0,
        "gfx_rain_occlusion": True,
        "gfx_lightning_bolts": True,
    },
    "medium": {
        "gfx_post_process": True,
        "gfx_hdr_format": "rgba16f",
        "gfx_render_scale": 1.0,
        "gfx_bloom": True,
        "gfx_bloom_mips": 4,
        "gfx_fxaa": True,
        "gfx_lens_flare": True,
        "gfx_clouds": True,
        "gfx_cloud_steps": 48,
        "gfx_cloud_light_steps": 6,
        "gfx_cloud_resolution_scale": 1.0,
        "gfx_weather_map": True,
        "gfx_cloud_virga": True,
        "gfx_cloud_genera": True,
        "gfx_god_rays": True,
        "gfx_god_ray_samples": 24,
        "gfx_foliage_shadow_refine": True,
        "gfx_rain_mode": "particles",
        "gfx_rain_particles": 7_000,
        "gfx_rain_occlusion": True,
        "gfx_lightning_bolts": True,
    },
    "high": {
        "gfx_post_process": True,
        "gfx_hdr_format": "rgba16f",
        "gfx_render_scale": 1.0,
        "gfx_bloom": True,
        "gfx_bloom_mips": 5,
        "gfx_fxaa": True,
        "gfx_lens_flare": True,
        "gfx_clouds": True,
        "gfx_cloud_steps": 96,
        "gfx_cloud_light_steps": 8,
        "gfx_cloud_resolution_scale": 1.0,
        "gfx_weather_map": True,
        "gfx_cloud_virga": True,
        "gfx_cloud_genera": True,
        "gfx_god_rays": True,
        "gfx_god_ray_samples": 32,
        "gfx_foliage_shadow_refine": True,
        "gfx_rain_mode": "particles",
        "gfx_rain_particles": 12_000,
        "gfx_rain_occlusion": True,
        "gfx_lightning_bolts": True,
    },
}


def resolve_graphics_preset(graphics_table: dict | None = None) -> dict:
    """
    Expand a ``[graphics]`` TOML table into flat ``gfx_*`` ``Config`` kwargs.

    The table's ``preset`` key (one of ``off/low/medium/high``, default
    ``"high"``) selects a base set of quality knobs from
    :data:`GRAPHICS_PRESETS`; any other ``gfx_*`` key in the table overrides
    that preset's value.  Unknown / invalid preset names fall back to
    ``"high"`` with a warning (never raises) so a typo can't break startup.

    Deterministic: the same table always yields the same dict.

    Parameters
    ----------
    graphics_table : dict | None
        The raw ``[graphics]`` table from the parsed TOML (or ``None``/empty
        for "no table present" → pure ``"high"`` preset).

    Returns
    -------
    dict
        ``gfx_*`` field → value, including ``gfx_preset`` (the resolved name).

    Example
    -------
    >>> resolve_graphics_preset({"preset": "low"})["gfx_cloud_resolution_scale"]
    0.5
    >>> resolve_graphics_preset({"preset": "low", "gfx_fxaa": True})["gfx_fxaa"]
    True
    """
    table = dict(graphics_table or {})
    requested = str(table.pop("preset", "high")).lower()
    if requested not in GRAPHICS_PRESETS:
        warnings.warn(
            f"unknown graphics preset {requested!r}; falling back to 'high'",
            stacklevel=2,
        )
        requested = "high"
    resolved: dict = dict(GRAPHICS_PRESETS[requested])
    resolved["gfx_preset"] = requested
    # Explicit per-field overrides win over the preset.
    for key, value in table.items():
        resolved[key] = value
    return resolved


@dataclass(frozen=True)
class Config:
    """
    Immutable engine configuration.

    All distance/time values use SI units (meters, seconds) unless noted.
    Instantiate via ``load_config(path)``; do not construct directly in
    production code (defaults are provided for tooling / tests).

    Fields
    ------
    world_seed           : int   — RNG seed for procedural systems (textures,
                                   ambient noise, NPC behaviour).  Terrain is
                                   flat/authored and does NOT use the seed.
    world_size_m         : float — square world footprint side length in meters,
                                   centred on the origin (1000 m = a 1 km × 1 km
                                   area spanning [-500, +500] on X and Y).
    ground_height_m      : float — flat baseline ground surface height (world Z,
                                   meters); solid below it, air above.
    voxel_size           : float — meters per voxel edge (locked at 0.5 m).
    chunk_size           : int   — voxels per chunk edge (locked at 32).
    light_grid_scale     : int   — terrain voxels per light cell edge (2).
    view_distance_chunks : int   — chunk-streaming XY radius in chunks.
    fixed_dt             : float — fixed-update period in seconds (50 Hz = 0.02).
    msaa_samples         : int   — hardware MSAA sample count for the window
                                   framebuffer (0 = off).  Anti-aliases
                                   GEOMETRY edges only (facet silhouettes,
                                   crater rims, the horizon) — surface
                                   interiors stay single-sample, so the
                                   pixel-art texel look is unaffected.
    show_fps             : bool  — overlay FPS counter.
    show_chunk_borders   : bool  — debug overlay for chunk boundaries.
    show_light_grid      : bool  — debug overlay for the light grid.
    debug_wind_ball      : bool  — spawn the dev-only "wind ball": a bright
                                   procedural sphere on the ground near spawn
                                   that is pushed by ``WindField.sample`` each
                                   fixed step (a physics seam proof — it scoots
                                   on gusts, rolls in storms).  Off by default.
    sky_cloud_altitude_m : float — base altitude of the cloud layer (world Z, m).
    sky_cloud_thickness_m: float — vertical thickness of the cloud layer (m).
    sky_cloud_cell_m     : float — horizontal edge of one cloud cell (m); the
                                   renderer fills coverage-fraction of cells.
    sky_star_count       : int   — star count baked into the "night_sky"
                                   procedural texture.
    mesh_style           : str   — terrain mesher: "faceted" (flat-shaded
                                   surface nets — the Daggerfall-ish semi-
                                   smooth look, default) or "blocky" (classic
                                   culled-face cubes).
    facet_shade_strength : float — [0,1] strength of the faceted mesher's
                                   normal-based facet accent shading (0 = off).
    ground_texels_per_m  : float — virtual texels per world meter for the GPU
                                   terrain shader's world-space procedural
                                   ground pattern (non-repeating pixel art);
                                   ~16 → 0.0625 m texels matching the voxel grid.
    lighting_backend     : str   — "gpu" (volumetric radiance cascades, GLSL
                                   compute) or "cpu" (legacy baked-vertex
                                   sunlight column pass).
    light_c0_cells       : int   — cascade-0 texels per axis (96 → 48 m box at
                                   0.5 m cells).
    light_c0_cell_m      : float — cascade-0 cell edge in meters (0.5 = one
                                   terrain voxel).
    light_c1_cells       : int   — cascade-1 texels per axis (96 → 96 m box).
    light_c1_cell_m      : float — cascade-1 cell edge in meters (1.0).
    light_c2_cells       : int   — cascade-2 texels per axis (64 → 256 m box):
                                   the coarse FAR cascade that keeps distant
                                   terrain (and the GI test room) lit with
                                   low-resolution shadows + GI once it leaves
                                   cascade 1, instead of falling back to flat
                                   sky ambient.  Assembled off-thread like the
                                   others.
    light_c2_cell_m      : float — cascade-2 cell edge in meters (4.0).
    light_quant_m        : float — shading sample-grid quantisation in meters
                                   (0.0625 → 8×8×8 visible light pixels per
                                   0.5 m voxel — the pixelated-light look).
                                   This is only the visible sample-snap grid;
                                   the underlying GI *data* resolution is the
                                   cascade-0 cell (``light_c0_cell_m``), so
                                   shrinking this past the cell size yields a
                                   finer-but-smoother grid, not more detail.
    light_gi_rays        : int   — ray-marched GI: sphere directions gathered
                                   per cell (fibonacci spiral; more = smoother
                                   ambient, linearly more inject-time cost).
    light_gi_steps       : int   — max one-cell march steps per GI ray (reach
                                   in meters = steps × the cascade cell size).
    light_gi_iters       : int   — gather iterations per inject; ≥2 lets the
                                   feedback term carry sky→wall→floor bounce
                                   and second-bounce colour bleed.
    light_gi_smooth_passes: int  — air-masked 3³ box-filter passes applied to
                                   the ray-gathered GI component after the
                                   gather (0 disables).  Completes the
                                   gather's 8-phase ray-fan tile (8× the
                                   effective ray count) — removes the blotchy
                                   patch / colour-confetti gather noise.
                                   Contact GI stays voxel-crisp; the filter
                                   never crosses solid cells (no leaks).
    light_penumbra_deg   : float — celestial penumbra cone half-angle in
                                   degrees: the shadow-edge refinement march
                                   jitters its rays inside this cone, so soft
                                   shadow edges widen with occluder distance.
    light_bounce_strength: float — [0,1] albedo-tinted bounce gain (first
                                   bounce at inject + gather feedback).
    light_tree_trunk_occ : float — [0,1] occupancy a tree trunk splats into the
                                  cascade volumes (lighting/occluders.py) —
                                  near-opaque wood.
    light_tree_canopy_extinction_gain : float — multiplier on each tree's
                                  leaf-derived per-METER canopy extinction:
                                  transmittance through X m of crown centre =
                                  exp(-sigma·gain·X), the same at every
                                  cascade cell size.  1.0 = the species' real
                                  leaf density; raise for darker shade, lower
                                  for airier canopies; 0 disables canopy
                                  occlusion.
    light_ao_strength    : float — [0,1] strength of occupancy-based ambient
                                   occlusion at surfaces.
    light_max_point_lights: int  — max simultaneous point/area lights uploaded
                                   to the GPU.
    light_exposure       : float — tonemap exposure multiplier for the HDR
                                   lighting pipeline.
    exposure_adapt_enabled: bool — auto-exposure (eye adaptation) on/off.
    exposure_min / exposure_max : float — clamp range of the adaptation
                                   multiplier (× light_exposure).
    exposure_key         : float — metering key: target multiplier =
                                   key / scene luminance (0.18 ≈ photographic
                                   middle gray; noon open field ≈ 1.0×).
    exposure_tau_dark_s  : float — adaptation time constant entering darkness
                                   in seconds (slow, like real eyes).
    exposure_tau_bright_s: float — adaptation time constant entering bright
                                   light in seconds (fast stop-down).
    fog_enabled          : bool  — volumetric froxel fog + god rays on/off.
    fog_froxels_x/y/z    : int   — froxel grid resolution (screen-aligned X/Y,
                                   exponential depth slices Z).
    fog_far_m            : float — far range of the froxel volume in meters.
    fog_anisotropy       : float — Henyey-Greenstein g for sun scattering
                                   ([0,1); higher = stronger forward god rays).
    grass_density_per_m2 : float — default blade tufts per square meter for
                                   grass volumes lacking a ``density`` param.
    grass_blade_height_m : float — unscaled tuft height in meters (per-blade
                                   jitter scales it 0.7–1.3×).
    grass_fade_start_m   : float — camera distance where blades begin
                                   shrinking away (meters).
    grass_fade_end_m     : float — camera distance where blades are fully
                                   gone (meters).
    grass_max_instances  : int   — hard cap on instances per grass volume.

    Flora fields (from [flora] table, prefix ``flora_``)
    ----------------------------------------------------
    GPU-instanced wildflower sprites (``world/flora_renderer.py``) inside
    ``"flowers"`` zone volumes.  Density is overridable per volume via
    ``params["density"]``.  (Bushes and trees are 3-D meshes now — see the
    [trees] table below.)

    flora_flower_density_per_m2 : float — wildflowers per m² (1.5).
    flora_flower_height_m       : float — flower sprite height (0.45 m).
    flora_flower_fade_start_m   : float — flowers fade like grass (60 m).
    flora_flower_fade_end_m     : float — fully gone (90 m).
    flora_flower_max_instances  : int   — per-volume cap (50 000).

    3-D tree/bush fields (from [trees] table, prefixes ``tree_``/``bush_``)
    -----------------------------------------------------------------------
    Instanced 3-D flora meshes (``world/tree_renderer.py``) inside
    ``"trees"`` / ``"bushes"`` zone volumes, placed CPU-side on a jittered
    grid (``zones/tree_placement.py``).  Near distance draws the variant
    mesh; past the mesh fade window the renderer crossfades to an
    instanced billboard impostor, which itself fades out at the impostor
    window — billboards are LOD only.

    tree_density_per_m2        : float — trees per m² (0.02 = 1 per 50 m²;
                                  ``params["density"]`` overrides per volume).
    tree_min_spacing_m         : float — placement grid floor: no two trunks
                                  closer than ≈0.3× this (3.0 m).
    tree_max_instances         : int   — per-volume cap (2 000).
    tree_mesh_fade_start/end_m : float — 3-D mesh shrink-away window
                                  (110–140 m); the impostor fades IN here.
    tree_impostor_fade_start/end_m : float — impostor shrink-away window
                                  (300–380 m, the old sprite landmark range).
    tree_default_species       : str — species def when a volume names none
                                  ("tree_gnarled_oak").
    bush_density_per_m2        : float — bushes per m² (0.08).
    bush_min_spacing_m         : float — bush spacing floor (1.2 m).
    bush_max_instances         : int   — per-volume cap (5 000).
    bush_mesh_fade_start/end_m : float — bush mesh window (60–80 m).
    bush_impostor_fade_start/end_m : float — bush impostor window (120–150 m).
    bush_default_species       : str — default bush species ("bush_scrub").

    Building fields (from [buildings] table, prefix ``building_``)
    -------------------------------------------------------------
    Free-form floorplan buildings (``fire_engine/buildings/``): per-storey
    2-D plans of segment/arc walls with thickness and parametric openings.
    All distances in meters.

    building_default_storey_height_m  : float — floor-to-floor height when a
                                        storey gets no explicit value (3.0).
    building_default_wall_thickness_m : float — wall thickness when a wall
                                        gets no explicit value (0.3).
    building_slab_thickness_m         : float — floor/ceiling/roof slab
                                        thickness (0.2).
    building_foundation_depth_m       : float — foundation slab depth below
                                        building-local z=0 (0.5).
    building_arc_segments_per_quarter : int   — chords per quarter circle
                                        when tessellating arc walls (8); used
                                        identically by meshing and room
                                        detection so polygons agree.
    building_snap_eps_m               : float — endpoint-snap tolerance for
                                        room auto-detection (0.01 = 1 cm).
    debug_demo_building (in [debug])  : bool  — spawn the feature-showcase
                                        demo house in front of spawn at boot
                                        (on by default — the building-system
                                        evaluation build; set false to hide).

    Wind-field fields (from [wind] table, prefix ``wind_``)
    -------------------------------------------------------
    These drive the spatially-varying wind field (``fire_engine/world/wind/``): a
    64×64-cell × 4 m (256 m) player-centred grid of horizontal wind velocity,
    summed from ~12 seeded spectral gust modes that advect downwind, plus an
    analytic vertical boundary-layer profile.  All distances meters, speeds
    m/s, frequencies rad/s, times seconds.

    wind_time_scale      : float — wind-clock rate in seconds per REAL second
                                   (1.0).  Gust travel/oscillation are an
                                   aesthetic real-time effect, deliberately
                                   independent of the game-clock timescale
                                   (``Clock.game_time_scale``: 60 today, 30
                                   later) — at game-time pacing a 60× clock
                                   would sweep gusts 60× too fast.  Raise for
                                   faster-evolving wind, lower for lazier.
    wind_cells           : int   — grid cells per axis (64 → 256 m region at
                                   4 m cells).
    wind_cell_m          : float — cell edge in meters (4.0).
    wind_snap_cells      : int   — origin snap granularity in cells for the
                                   recenter window (8 → snaps to 32 m).
    wind_margin_cells    : int   — recenter hysteresis: re-snap only when the
                                   player drifts past this many cells from the
                                   region centre (8 → 32 m band).
    wind_gust_modes      : int   — number of spectral Brownian-band gust modes
                                   summed per cell (12).
    wind_gust_wavelen_min/max : float — gust spatial wavelength band in meters
                                   (20–120 m; big slow gusts dominate).
    wind_gust_omega_min/max   : float — intrinsic temporal frequency band in
                                   rad/s (0.15–0.8) — the gust's own pulsing on
                                   top of downwind advection.
    wind_gust_base       : float — base gust amplitude gain (calm air, 0.6).
    wind_gust_storm_gain : float — extra gust amplitude per unit storminess
                                   (1.4): storms gust much harder.
    wind_storm_freq_gain : float — temporal-frequency boost per unit storminess
                                   (0.8): storms are choppier, not just stronger.
    wind_speed_ref       : float — reference mean wind speed (m/s) at which the
                                   gust gain reaches full strength (8.0).
    wind_turb_base       : float — base turbulence channel value, calm (0.2).
    wind_turb_storm_gain : float — turbulence increase per unit storminess (1.0).
    wind_shear           : float — vertical-profile shear exponent (0.18): the
                                   power-law boundary-layer wind shear.
    wind_profile_z_ref   : float — reference height (m) where the vertical
                                   profile reaches 1.0 (10.0).
    wind_profile_floor   : float — minimum profile multiplier at ground level
                                   (0.35): wind never fully stops at z=ground.
    wind_profile_cap     : float — maximum profile multiplier high up (1.6).
    wind_layer_m         : float — vertical band (m) above ground over which
                                   the venturi solver folds terrain occupancy
                                   (8.0; WP2 consumes this).
    wind_venturi_iters   : int   — venturi flux-relaxation iterations (8; WP2).
    wind_venturi_max     : float — clamp on venturi speed-up multiplier (2.2;
                                   WP2).
    wind_deflect_gain    : float — venturi sideways-deflection gain (0.15; WP2).
    wind_updraft_gain    : float — analytic obstacle-updraft gain (0.4; WP2).
    wind_mote_count      : int   — dust/pollen mote instance count (1500; WP4).
    wind_mote_box_m      : float — camera-anchored mote lattice cell size in
                                   meters (24.0; WP4).
    wind_mote_size_m     : float — mote billboard size in meters (0.04; WP4).
    wind_mote_life_s     : float — mote looping lifetime in seconds (6.0; WP4).
    wind_leaf_density_per_m2 : float — leaf-litter instances per m² of a
                                   "trees" zone volume (0.15; WP4).
    wind_leaf_size_m     : float — leaf billboard size in meters (0.12; WP4).
    wind_leaf_max_instances : int — hard cap on leaf instances per volume
                                   (20000; WP4).

    Rain-cover heightmap fields (from [rain] table, prefix ``rain_``)
    -----------------------------------------------------------------
    Drive the top-down cover heightmap (``terrain/rain_cover.py``) that the M6
    volumetric rain renderer samples to discard rain under roofs/overhangs.

    rain_cover_cells     : int   — columns per axis of the player-centred cover
                                   window (256 → a 256 m square at 1 m cells).
    rain_cover_cell_m    : float — column edge in meters (1.0 = light-cell size).
    rain_cover_budget_columns : int — chunk-columns the renderer refolds per
                                   refresh so a full rebuild amortises over
                                   frames (4).

    Graphics-quality fields (from [graphics] table, prefix ``gfx_``)
    ---------------------------------------------------------------
    These drive the HDR post-processing pipeline and volumetric clouds so the
    look can be dialed down (or off) on weak GPUs.  Pick a ``preset``
    (off/low/medium/high) in ``[graphics]``; any explicit ``gfx_*`` key in the
    same table overrides that preset's value (see ``resolve_graphics_preset``).
    The dataclass defaults equal the ``"high"`` preset.

    gfx_preset            : str   — which preset produced these values
                                    (off/low/medium/high; informational).
    gfx_post_process      : bool  — master switch for the offscreen HDR buffer
                                    + post chain.  False ⇒ shaders tonemap
                                    internally (legacy path), no bloom/flare.
    gfx_hdr_format        : str   — scene buffer format: "rgba16f" (float HDR)
                                    or "rgba8" (LDR fallback for GPUs lacking
                                    float render targets).
    gfx_render_scale      : float — internal render resolution scale (1.0 = full;
                                    0.75 = render at 75% then upscale).
    gfx_bloom             : bool  — bloom on/off.
    gfx_bloom_mips        : int   — bloom downsample pyramid depth (more = wider,
                                    softer glow; costs more).
    gfx_bloom_threshold   : float — luminance above which pixels bloom (HDR).
    gfx_bloom_knee        : float — soft-knee width below the threshold.
    gfx_bloom_strength    : float — bloom contribution added back at composite.
    gfx_fxaa              : bool  — cheap post anti-aliasing pass.
    gfx_lens_flare        : bool  — screen-space lens flare when looking near
                                    the (unoccluded) sun.
    gfx_clouds            : bool  — volumetric raymarched clouds on/off.
    gfx_cloud_steps       : int   — primary raymarch sample count (quality).
    gfx_cloud_light_steps : int   — sun light-march steps per sample
                                    (self-shadow quality; dominant cost).
    gfx_cloud_resolution_scale: float — cloud pass resolution (0.5 = half-res,
                                    the biggest perf win on an iGPU).
    gfx_cloud_max_dist_m  : float — far raymarch distance for clouds (meters).
    gfx_weather_map       : bool  — upload the spatial weather-map texture and
                                    sample it in the cloud raymarch (spatial
                                    coverage/density/precip).  Off ⇒ the cloud
                                    shader uses the flat ambient scalars (the
                                    pre-M4 look); the renderer skips the
                                    re-raster + upload entirely.  Master kill
                                    switch for the M4 GPU weather contract.
    gfx_cloud_virga       : bool  — gray rain shafts hanging below storm-cloud
                                    bases (driven by the weather map's precip
                                    channel).  Requires ``gfx_weather_map``;
                                    off ⇒ storm bases still lower/darken but no
                                    virga streaks.
    gfx_cloud_genera      : bool  — render layered WMO cloud genera (M9): a high
                                    cirrus band, a mid alto- band and the low
                                    cumulus/stratus/cumulonimbus deck, each with
                                    a genus-appropriate look, derived in-shader
                                    from the weather map (no extra texture data).
                                    Requires ``gfx_weather_map``; off ⇒ the
                                    single cloud slab (the pre-M9 look).
    gfx_god_rays          : bool  — screen-space crepuscular rays through clouds.
    gfx_god_ray_samples   : int   — radial sample count for god rays.
    gfx_rain_mode         : str   — volumetric rain mode: "off" (no rain),
                                    "cylinders" (cheap camera-following scrolled
                                    shells — the low preset), or "particles"
                                    (GPU-instanced falling streaks — medium+).
                                    Both rendered modes honour the rain-cover
                                    heightmap cull (no rain under a roof) and the
                                    weather-map precip footprint.
    gfx_rain_particles    : int   — instanced rain-streak count in "particles"
                                    mode (per preset: 0 off/cylinders, 7000
                                    medium, 12000 high).
    gfx_rain_occlusion    : bool  — sample the rain-cover heightmap to discard
                                    streaks under cover (all presets, all modes;
                                    false ⇒ rain everywhere, the old look).
    gfx_lightning_bolts   : bool  — render procedural lightning bolts (M7):
                                    camera-facing stepped-leader ribbons, a
                                    two-phase flash, a transient scene light and
                                    a sky/cloud flash pulse, on a strike.  On for
                                    low+ presets, off for "off".  The headless
                                    strike SCHEDULE + ThunderEvents still run
                                    when this is off (audio/gameplay only see no
                                    drawn bolt).
    gfx_foliage_shadow_refine : bool — per-fragment celestial-shadow refinement
                                    march on foliage (grass/flora/trees/
                                    impostors; the lit_surface.glsl ``u_refine``
                                    gate).  Terrain always refines; turning
                                    this off keeps foliage on the cheap
                                    trilinear cascade shadows (iGPU relief).
    gfx_god_ray_strength  : float — god-ray contribution added at composite.
    gfx_lens_flare_strength : float — lens-flare contribution at composite
                                    (lower = subtler flare).
    gfx_lens_flare_threshold: float — HDR luminance the flare isolates as "the
                                    sun" (higher = only the very brightest core
                                    flares, ignoring bright sky).
    gfx_tonemap_hue_preserve: float — [0,1] blend toward the hue-preserving
                                    tonemap.  0 = plain per-channel ACES (bright
                                    sky washes to white); 1 = fully preserve hue
                                    (saturated sky stays coloured as it brightens).
    gfx_sun_disc_intensity: float — HDR gain on the sun disc (how hard it blooms
                                    into a bright blob).
    gfx_sun_halo_intensity: float — HDR gain on the forward-Mie glow haloing the
                                    sun.
    gfx_sun_min_brightness: float — floor on the sun disc/halo transmittance so a
                                    low (sunrise/sunset) sun still reads bright
                                    instead of fading out; hue is preserved.
    gfx_sky_inscatter_scale: float — multiplier on the scattered-sky radiance
                                    (lower = dimmer sky, less low-sun wash-out;
                                    does not touch the sun disc).

    Profiler fields (from [profiler] table, prefix ``profiler_``)
    -------------------------------------------------------------
    Drive the frame profiler (``fire_engine/core/profiler.py``) + its overlay /
    PStats bridge.  Everything is observational — never affects the sim or saves.

    profiler_enabled            : bool  — master switch.  False ⇒ scopes are
                                  no-ops, no ring buffer / overlay / PStats
                                  objects are constructed (truly free).
    profiler_overlay_enabled    : bool  — build the in-game F3 overlay (only
                                  when profiler_enabled).
    profiler_frame_budget_ms    : float — per-frame budget in ms (200 FPS = 5.0).
    profiler_history_frames     : int   — ring-buffer length (percentile span).
    profiler_hitch_abs_ms       : float — absolute hitch threshold floor (ms).
    profiler_hitch_rel_mult     : float — hitch when ms > max(abs, mult × median).
    profiler_hitch_window       : int   — frames the rolling median spans.
    profiler_max_scopes         : int   — preallocated per-scope columns.
    profiler_max_counters       : int   — preallocated per-counter columns.
    profiler_recent_hitches     : int   — recent hitches kept in the snapshot.
    profiler_overlay_graph_frames : int — frames drawn in the overlay graph.
    profiler_overlay_hz         : float — overlay refresh rate (Hz).
    profiler_snapshot_enabled   : bool  — periodically write snapshot JSON.
    profiler_snapshot_path      : str   — snapshot JSON path (AI-agent contract).
    profiler_snapshot_interval_s: float — seconds between snapshot writes.
    profiler_pstats             : bool  — connect to a PStats server at boot.
    """

    world_seed:           int   = 1337
    world_size_m:         float = 1000.0
    ground_height_m:      float = 8.0
    voxel_size:           float = 0.5
    chunk_size:           int   = 32
    light_grid_scale:     int   = 2
    view_distance_chunks: int   = 6
    fixed_dt:             float = 0.02
    msaa_samples:         int   = 4
    show_fps:             bool  = True
    show_chunk_borders:   bool  = False
    show_light_grid:      bool  = False
    debug_wind_ball:      bool  = False
    debug_demo_building:  bool  = True
    sky_cloud_altitude_m:  float = 96.0
    sky_cloud_thickness_m: float = 8.0
    sky_cloud_cell_m:      float = 12.0
    sky_star_count:        int   = 2500
    mesh_style:            str   = "faceted"
    facet_shade_strength:  float = 0.25
    ground_texels_per_m:   float = 16.0
    lighting_backend:      str   = "gpu"
    light_c0_cells:        int   = 96
    light_c0_cell_m:       float = 0.5
    light_c1_cells:        int   = 96
    light_c1_cell_m:       float = 1.0
    light_c2_cells:        int   = 64
    light_c2_cell_m:       float = 4.0
    light_quant_m:         float = 0.0625
    light_gi_rays:         int   = 16
    light_gi_steps:        int   = 24
    light_gi_iters:        int   = 2
    light_gi_smooth_passes: int  = 1
    light_penumbra_deg:    float = 2.5
    light_bounce_strength: float = 0.7
    light_ao_strength:     float = 0.6
    # Static tree/bush occluders (lighting/occluders.py): trunks splat a
    # near-opaque column; canopies are a TRANSLUCENT leaf medium — per-meter
    # extinction derived from the species' real leaf area, scaled by this
    # gain (light through X m of crown = exp(-sigma*gain*X), identical at
    # every cascade cell size).
    light_tree_trunk_occ:  float = 0.85
    light_tree_canopy_extinction_gain: float = 1.0
    light_max_point_lights: int  = 64
    light_exposure:        float = 0.9
    exposure_adapt_enabled: bool = True
    exposure_min:          float = 0.55
    exposure_max:          float = 5.0
    exposure_key:          float = 0.18
    exposure_tau_dark_s:   float = 4.0
    exposure_tau_bright_s: float = 0.7
    fog_enabled:           bool  = True
    fog_froxels_x:         int   = 160
    fog_froxels_y:         int   = 90
    fog_froxels_z:         int   = 64
    fog_far_m:             float = 192.0
    fog_anisotropy:        float = 0.55
    grass_density_per_m2:  float = 12.0
    grass_blade_height_m:  float = 0.6
    grass_fade_start_m:    float = 60.0
    grass_fade_end_m:      float = 90.0
    grass_max_instances:   int   = 200_000
    # --- Flora ([flora] table; consumed by world/flora_renderer.py) ---
    flora_flower_density_per_m2: float = 1.5
    flora_flower_height_m:       float = 0.45
    flora_flower_fade_start_m:   float = 60.0
    flora_flower_fade_end_m:     float = 90.0
    flora_flower_max_instances:  int   = 50_000
    # --- 3-D trees/bushes ([trees] table; world/tree_renderer.py) ---
    tree_density_per_m2:         float = 0.02
    tree_min_spacing_m:          float = 3.0
    tree_max_instances:          int   = 2_000
    tree_mesh_fade_start_m:      float = 110.0
    tree_mesh_fade_end_m:        float = 140.0
    tree_impostor_fade_start_m:  float = 300.0
    tree_impostor_fade_end_m:    float = 380.0
    tree_default_species:        str   = "tree_gnarled_oak"
    bush_density_per_m2:         float = 0.08
    bush_min_spacing_m:          float = 1.2
    bush_max_instances:          int   = 5_000
    bush_mesh_fade_start_m:      float = 60.0
    bush_mesh_fade_end_m:        float = 80.0
    bush_impostor_fade_start_m:  float = 120.0
    bush_impostor_fade_end_m:    float = 150.0
    bush_default_species:        str   = "bush_scrub"
    # --- Buildings ([buildings] table; consumed by fire_engine/buildings/) ---
    building_default_storey_height_m:  float = 3.0
    building_default_wall_thickness_m: float = 0.3
    building_slab_thickness_m:         float = 0.2
    building_foundation_depth_m:       float = 0.5
    building_arc_segments_per_quarter: int   = 8
    building_snap_eps_m:               float = 0.01
    # --- Wind field ([wind] table; consumed by fire_engine/world/wind/) ---
    wind_time_scale:          float = 1.0
    wind_cells:               int   = 64
    wind_cell_m:              float = 4.0
    wind_snap_cells:          int   = 8
    wind_margin_cells:        int   = 8
    wind_gust_modes:          int   = 12
    wind_gust_wavelen_min:    float = 20.0
    wind_gust_wavelen_max:    float = 120.0
    wind_gust_omega_min:      float = 0.15
    wind_gust_omega_max:      float = 0.8
    wind_gust_base:           float = 0.6
    wind_gust_storm_gain:     float = 1.4
    wind_storm_freq_gain:     float = 0.8
    wind_speed_ref:           float = 8.0
    wind_turb_base:           float = 0.2
    wind_turb_storm_gain:     float = 1.0
    wind_shear:               float = 0.18
    wind_profile_z_ref:       float = 10.0
    wind_profile_floor:       float = 0.35
    wind_profile_cap:         float = 1.6
    wind_layer_m:             float = 8.0
    wind_venturi_iters:       int   = 8
    wind_venturi_max:         float = 2.2
    wind_deflect_gain:        float = 0.15
    wind_updraft_gain:        float = 0.4
    wind_mote_count:          int   = 1500
    wind_mote_box_m:          float = 24.0
    wind_mote_size_m:         float = 0.04
    wind_mote_life_s:         float = 6.0
    wind_leaf_density_per_m2: float = 0.15
    wind_leaf_size_m:         float = 0.12
    wind_leaf_max_instances:  int   = 20_000
    # --- Rain cover heightmap ([rain] table; terrain/rain_cover.py + M6 rain) ---
    rain_cover_cells:         int   = 256
    rain_cover_cell_m:        float = 1.0
    rain_cover_budget_columns: int  = 4
    # --- Weather simulation ([weather] table; consumed by fire_engine/world/weather/) ---
    # Synoptic flow: the slow steering current that carries storm cells.
    # Closed-form pure function of (seed, game time) — see weather/synoptic.py.
    weather_synoptic_components:   int   = 4      # vector sinusoids in W(t)
    weather_synoptic_speed_min_ms: float = 1.5    # guaranteed speed band (m/s)
    weather_synoptic_speed_max_ms: float = 11.0
    weather_synoptic_period_min_h: float = 2.5    # sinusoid period band (game h)
    weather_synoptic_period_max_h: float = 14.0
    # Storm cells (M2): spatial showers/storms/cloud-banks/fog-banks that drift
    # on the synoptic flow. Natural spawn schedule is a pure function of
    # (seed, day) — see weather/cells.py. Units: meters, game seconds, m/s.
    weather_domain_m:              float = 6000.0  # half-extent of the spawn square (±m around origin)
    weather_spawn_slots_per_day:   int   = 8       # candidate cell spawn slots per day
    weather_cell_radius_min_m:     float = 350.0   # cell footprint radius band at peak (m)
    weather_cell_radius_max_m:     float = 1200.0
    weather_cell_duration_min_s:   float = 2400.0  # cell lifetime band (game s; 40 min .. 3 h)
    weather_cell_duration_max_s:   float = 10800.0
    weather_storm_wind_max_ms:     float = 9.0     # extra gust a THUNDERSTORM adds at its core (m/s)
    weather_fog_max_density:       float = 0.028   # cap on FOG_BANK fog coefficient (1/m)
    weather_temp_mean_c:           float = 12.0    # daily mean air temperature (°C)
    weather_temp_amp_c:            float = 8.0     # daily temperature swing amplitude (°C)
    # Weather map (M3): a derived raster of the local weather fields around the
    # player — the render + sampling cache (never saved; recomputed each tick).
    weather_map_cells:             int   = 128     # raster resolution (square, N×N texels)
    weather_map_cell_m:            float = 24.0    # raster texel size (m) → 128×24 ≈ 3 km span
    # Ground wetness (M3): closed-form fixed-offset quadrature over the analytic
    # rain history at a point (no integrated state — recompute-free on load).
    weather_wetness_tau_s:         float = 3600.0  # wetness decay time constant (game s)
    weather_wetness_step_s:        float = 600.0   # quadrature step into the past (game s)
    weather_wetness_samples:       int   = 12      # number of past rain samples (window = step·samples)
    # Emergent humidity + fog (M5): fog condenses from conditions, not a state.
    # humidity = base(day) + rain_gain·rain_recent + wetness_gain·wetness; it
    # condenses to fog where it exceeds the temperature-dependent saturation
    # humidity in calm air. All closed-form pure fn of (seed, t, pos). See
    # weather/humidity.py. Units: relative humidity 0–1, °C, m/s, fog 1/m.
    weather_humidity_base_min:     float = 0.35    # seeded per-day calm-air humidity baseline band
    weather_humidity_base_max:     float = 0.65
    weather_humidity_rain_gain:    float = 1.00    # humidity added per unit recent rain (0–1)
    weather_humidity_wetness_gain: float = 0.30    # humidity added per unit ground wetness (0–1)
    weather_humidity_recent_tau_s:    float = 18000.0  # recent-rain decay time constant (game s; 5 h — air holds moisture for hours so evening rain feeds pre-dawn fog)
    weather_humidity_recent_step_s:   float = 1800.0   # recent-rain quadrature step into past (game s)
    weather_humidity_recent_samples:  int   = 12       # recent-rain samples (window = step·samples = 6 h)
    weather_fog_emergent_max:      float = 0.022   # max emergent fog coefficient at full condensation (1/m)
    weather_fog_sat_ref_c:         float = 5.0     # reference temperature for saturation humidity (°C)
    weather_fog_sat_base:          float = 0.63    # saturation humidity at the reference temperature (0–1)
    weather_fog_sat_slope_per_c:   float = 0.011   # saturation humidity rise per °C above the reference
    weather_fog_condense_band:     float = 0.10    # humidity overshoot over saturation for full condensation
    weather_fog_wind_full_ms:      float = 1.0     # full fog at/below this wind speed (m/s)
    weather_fog_wind_none_ms:      float = 3.0     # no emergent fog at/above this wind speed (m/s)
    # WMO cloud genera (M9): the sampled weather (coverage/density/precip +
    # regime) is mapped to layered altitude bands — high cirrus, mid alto-,
    # low strato/cumulus/cumulonimbus — by weather/clouds.py::cloud_layers
    # (headless) and mirrored in-shader by cloud_volumetric.frag (no new
    # texture data; the same coverage/density/precip channels drive the bands).
    # base_altitude / thickness in meters (world Z); cov_weight/density 0–1;
    # detail_scale is a multiplier on the renderer's base noise frequency
    # (cirrus stretched+smooth → low; cumulus billowy → high).
    cloud_genera_high_alt_m:       float = 1400.0  # high (cirrus) band base altitude (m)
    cloud_genera_high_thick_m:     float = 120.0   # high band thickness (m; thin veil)
    cloud_genera_mid_alt_m:        float = 850.0   # mid (alto-) band base altitude (m)
    cloud_genera_mid_thick_m:      float = 220.0   # mid band thickness (m)
    cloud_genera_low_alt_m:        float = 500.0   # low (cumulus/stratus) band base altitude (m)
    cloud_genera_low_thick_m:      float = 400.0   # low band thickness (m; storms deepen it)
    cloud_genera_high_cov_floor:   float = 0.06    # cirrus present even in fair weather (residual floor)
    cloud_genera_high_cov_weight:  float = 0.35    # extra high-band coverage per unit sampled coverage
    cloud_genera_high_density:     float = 0.30    # ice cloud is thin: cap on high-band opacity
    cloud_genera_mid_cov_weight:   float = 0.60    # mid-band coverage per unit sampled coverage
    cloud_genera_high_detail_scale: float = 0.45   # high band: stretched, smooth streaks
    cloud_genera_mid_detail_scale:  float = 0.85   # mid band: moderate lumpiness
    cloud_genera_low_detail_scale:  float = 1.30   # low band: billowy cumulus detail
    # Summon API + gust-front coupling (M8). Summoned cells are spawned UPWIND of
    # the player (so they drift in on the synoptic flow) with per-kind footprint
    # defaults; the gust-front modifier kicks in when a cell's leading edge nears
    # the player. All meters / game seconds / m/s — no magic numbers in code.
    weather_summon_upwind_m:       float = 2500.0  # how far upwind a summoned cell spawns (m)
    weather_summon_rain_radius_m:  float = 700.0   # summoned rainstorm footprint radius (m)
    weather_summon_rain_duration_s: float = 5400.0 # summoned rainstorm lifetime (game s; 90 min)
    weather_summon_rain_intensity: float = 0.85    # summoned rainstorm peak intensity (0–1)
    weather_summon_storm_radius_m: float = 950.0   # summoned thunderstorm footprint radius (m)
    weather_summon_storm_duration_s: float = 6000.0 # summoned thunderstorm lifetime (game s; 100 min)
    weather_summon_storm_intensity: float = 1.0    # summoned thunderstorm peak intensity (0–1)
    weather_summon_fog_radius_m:   float = 600.0   # summoned fog-bank footprint radius (m)
    weather_summon_fog_duration_s: float = 7200.0  # summoned fog-bank lifetime (game s; 2 h)
    weather_summon_fog_intensity:  float = 0.9     # summoned fog-bank peak intensity (0–1)
    weather_gustfront_range_m:     float = 600.0   # register a gust-front modifier when a cell's leading edge is within this range of the player (m)
    weather_gustfront_strength_ms: float = 7.0     # peak added wind speed along the summoned gust front (m/s)
    weather_gustfront_width_m:     float = 80.0    # gust-front band half-width (Gaussian sigma, m)
    # Procedural lightning (M7): a deterministic Poisson strike schedule per
    # active THUNDERSTORM cell (pure fn of seed+cell+time window — recomputes
    # identically after a save/load mid-storm) and a stepped-leader bolt grown
    # through a cheap seeded potential field.  See weather/lightning.py +
    # weather/bolt.py. Units: strikes/min, meters, degrees, counts.
    weather_lightning_strikes_per_min: float = 2.5   # peak strike rate per cell at full intensity (thinned by cell intensity)
    weather_lightning_cloud_base_m:    float = 220.0 # cloud-base height above ground the bolt starts at (m)
    weather_lightning_ground_z_m:      float = 8.0   # fallback ground-plane world Z when no cover heightmap (m; == ground_height_m)
    bolt_step_len_min_m:   float = 5.0     # stepped-leader step length band (m)
    bolt_step_len_max_m:   float = 15.0
    bolt_cone_deg:         float = 38.0    # half-angle of the downward candidate-direction fan (deg)
    bolt_candidates:       int   = 7       # K candidate directions fanned each step
    bolt_softmax_temp:     float = 0.35    # softmax temperature for the seeded direction pick
    bolt_branch_prob:      float = 0.12    # per-step probability a side branch spawns
    bolt_max_steps:        int   = 400     # hard cap on leader steps (the one bounded loop)
    bolt_noise_gain:       float = 0.6     # weight of the seeded value-noise "air resistance" in the score
    bolt_repulsion_gain:   float = 0.45    # weight of repulsion from the existing channel in the score
    bolt_branch_max_depth: int   = 3       # branches stop spawning sub-branches past this depth
    # --- Graphics quality ([graphics] table; defaults == "high" preset) ---
    gfx_preset:                 str   = "high"
    gfx_post_process:           bool  = True
    gfx_hdr_format:             str   = "rgba16f"
    gfx_render_scale:           float = 1.0
    gfx_bloom:                  bool  = True
    gfx_bloom_mips:             int   = 5
    gfx_bloom_threshold:        float = 1.0
    gfx_bloom_knee:             float = 0.5
    gfx_bloom_strength:         float = 0.06
    gfx_fxaa:                   bool  = True
    gfx_lens_flare:             bool  = True
    gfx_clouds:                 bool  = True
    gfx_cloud_steps:            int   = 96
    gfx_cloud_light_steps:      int   = 8
    gfx_cloud_resolution_scale: float = 1.0
    gfx_cloud_max_dist_m:       float = 6000.0
    gfx_weather_map:            bool  = True
    gfx_cloud_virga:            bool  = True
    gfx_cloud_genera:           bool  = True   # M9: layered WMO genera bands (high cirrus / mid alto / low cumulus+cb). Off ⇒ single-slab clouds (pre-M9 look). Requires gfx_weather_map.
    gfx_god_rays:               bool  = True
    gfx_god_ray_samples:        int   = 32
    gfx_foliage_shadow_refine:  bool  = True
    # Volumetric rain (M6): "off" / "cylinders" (cheap scrolled shells) /
    # "particles" (GPU-instanced falling streaks).  Both gated modes honour the
    # rain-cover heightmap cull (no rain under a roof) + the weather-map precip
    # footprint.  Defaults == the "high" preset (particles, 12k).
    gfx_rain_mode:              str   = "particles"
    gfx_rain_particles:         int   = 12_000
    gfx_rain_occlusion:         bool  = True
    # Procedural lightning bolts (M7): the render half of the lightning system
    # (camera-facing stepped-leader ribbons + flash + transient scene light).
    # On for low+ presets, off for "off".  Gates world/lightning_renderer.py.
    gfx_lightning_bolts:        bool  = True
    # Aesthetic tuning — NOT carried by the presets (so they stay consistent
    # across off/low/medium/high); override freely in [graphics] in config.toml.
    gfx_god_ray_strength:       float = 0.4
    gfx_lens_flare_strength:    float = 0.055
    gfx_lens_flare_threshold:   float = 4.0
    gfx_tonemap_hue_preserve:   float = 0.8
    gfx_sun_disc_intensity:     float = 45.0
    gfx_sun_halo_intensity:     float = 1.8
    gfx_sun_min_brightness:     float = 0.25
    gfx_sky_inscatter_scale:    float = 0.9
    # --- Performance profiler ([profiler] table; fire_engine/core/profiler.py) ---
    # The engine-agnostic frame profiler + its render-side overlay / PStats
    # bridge.  Nearly free enabled, truly free disabled (no buffers / overlay /
    # PStats objects are constructed when ``profiler_enabled`` is false).  All
    # thresholds/sizes here so nothing is a magic number.  See
    # docs/systems/profiler.md.
    profiler_enabled:           bool  = False
    profiler_overlay_enabled:   bool  = True
    profiler_frame_budget_ms:   float = 5.0
    profiler_history_frames:    int   = 1024
    profiler_hitch_abs_ms:      float = 8.0
    profiler_hitch_rel_mult:    float = 1.5
    profiler_hitch_window:      int   = 120
    profiler_max_scopes:        int   = 64
    profiler_max_counters:      int   = 32
    profiler_recent_hitches:    int   = 16
    profiler_overlay_graph_frames: int = 240
    profiler_overlay_hz:        float = 8.0
    profiler_snapshot_enabled:  bool  = False
    profiler_snapshot_path:     str   = "profiling/latest.json"
    profiler_snapshot_interval_s: float = 1.0
    profiler_pstats:            bool  = False

    # ------------------------------------------------------------------
    # Derived read-only properties
    # ------------------------------------------------------------------

    @property
    def chunk_meters(self) -> float:
        """
        World-space side length of one chunk in meters.

        ``chunk_size * voxel_size`` — always 16.0 m with the locked defaults.
        """
        return float(self.chunk_size) * self.voxel_size

    @property
    def light_cell_meters(self) -> float:
        """
        World-space side length of one light-grid cell in meters.

        ``voxel_size * light_grid_scale`` — always 1.0 m with the locked defaults.
        """
        return self.voxel_size * float(self.light_grid_scale)


def load_config(path: str = "config.toml") -> Config:
    """
    Load engine configuration from a TOML file, returning a frozen ``Config``.

    The TOML file may have ``[debug]``, ``[sky]``, ``[terrain]``,
    ``[lighting]``, ``[fog]``, ``[grass]``, ``[flora]``, ``[trees]``,
    ``[buildings]``, ``[wind]``, ``[rain]``, ``[weather]``, ``[graphics]`` and
    ``[profiler]`` tables; their keys are flattened into the same ``Config``
    struct.  Any key
    absent from the file falls back to the
    ``Config`` dataclass default.

    ``[graphics]`` is special: its ``preset`` key (off/low/medium/high) is
    expanded into the ``gfx_*`` quality fields via
    :func:`resolve_graphics_preset`, and any explicit ``gfx_*`` key in the
    table overrides the chosen preset.

    If the file does not exist or cannot be read, returns a default ``Config``
    (same as ``Config()``).

    Parameters
    ----------
    path : str, default "config.toml"
        Path to the TOML configuration file.

    Returns
    -------
    Config
        Frozen configuration object.

    Example
    -------
    >>> cfg = load_config("config.toml")
    >>> cfg.world_seed
    1337
    >>> cfg.chunk_meters
    16.0
    """
    try:
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
    except (FileNotFoundError, OSError):
        return Config()

    # Flatten the organisational tables into one top-level dict.  Most tables
    # just carry fully-named Config fields; [graphics] is special — its keys go
    # through preset expansion first (see resolve_graphics_preset).
    _TABLES = ("debug", "sky", "terrain", "lighting", "fog", "grass", "flora",
               "trees", "buildings", "wind", "rain", "weather", "graphics",
               "profiler")
    flat: dict = {k: v for k, v in raw.items() if k not in _TABLES}
    for table in _TABLES:
        if table == "graphics":
            continue
        flat.update(raw.get(table, {}))
    flat.update(resolve_graphics_preset(raw.get("graphics", {})))

    # Build Config by extracting only known fields (ignore unknown keys)
    known_fields = {f.name for f in Config.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs = {k: flat[k] for k in flat if k in known_fields}

    return Config(**kwargs)
