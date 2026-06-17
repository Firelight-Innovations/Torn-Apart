"""
core/config.py — Typed, frozen engine configuration.

Loads values from ``config.toml`` (stdlib tomllib, Python 3.11+) and exposes
them as a single frozen ``Config`` dataclass.  All engine code reads config
through this module — no magic numbers.

Instantiate via ``load_config(path)``; do not construct directly in production
code (all fields have defaults for tooling / tests).  All distance/time values
use SI units (meters, seconds) unless noted.

See ``docs/systems/core.md`` (Config fields) for the full per-field reference
with units, semantics and TOML table origins.

Example
-------
    from fire_engine.core.config import load_config

    cfg = load_config("config.toml")
    print(cfg.chunk_meters)   # 16.0

Docs: docs/systems/core.md
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """
    Immutable engine configuration.

    All distance/time values use SI units (meters, seconds) unless noted.
    Instantiate via ``load_config(path)``; do not construct directly in
    production code (defaults are provided for tooling / tests).

    See ``docs/systems/core.md`` (Config fields) for the per-field reference.

    Docs: docs/systems/core.md
    """

    world_seed: int = 1337
    world_size_m: float = 1000.0
    ground_height_m: float = 8.0
    voxel_size: float = 0.5
    chunk_size: int = 32
    light_grid_scale: int = 2
    view_distance_chunks: int = 6
    fixed_dt: float = 0.02
    msaa_samples: int = 4
    show_fps: bool = True
    show_chunk_borders: bool = False
    show_light_grid: bool = False
    debug_wind_ball: bool = False
    debug_demo_building: bool = True
    sky_cloud_altitude_m: float = 96.0
    sky_cloud_thickness_m: float = 8.0
    sky_cloud_cell_m: float = 12.0
    sky_star_count: int = 2500
    mesh_style: str = "faceted"
    facet_shade_strength: float = 0.25
    ground_texels_per_m: float = 16.0
    # Threaded terrain LOD streaming ([terrain] table; world/terrain/lod/).
    # These tune SCHEDULING only — never mesh output (determinism preserved).
    lod_streaming_enabled: bool = True  # use the off-thread LodStreamer path (vs sync stream_frame)
    lod_worker_threads: int = 4  # TerrainLodPool worker thread count
    lod_submit_per_frame: int = 16  # max chunk mesh jobs submitted per stream_frame
    lod_max_uploads_per_frame: int = 8  # max finished meshes uploaded to the scene graph per frame
    # --- Coarse LOD ranks (P2; world/terrain/lod/desired.py + coarse_streamer.py) ---
    # The distant horizon: ranks 1..lod_max_rank are downsampled coarse nodes
    # meshed off-thread and drawn beyond the near (rank-0) editable radius.
    # Scheduling + extents only — the downsample is deterministic, so mesh output
    # never depends on these (Hard Rule 2 holds). Hard band cuts in P2 (no
    # double-draw, pop at the boundary); crossfade is P3.
    lod_max_rank: int = 3  # highest coarse rank generated (L3 = 8x); 0 = near-only (P1)
    lod_near_radius_chunks: int = 6  # L0 editable/lit/saved radius (chunks; == view_distance)
    lod_far_radius_chunks: int = 32  # outer XY radius (chunks) of the coarse window (512 m)
    lod_band_l1_m: float = 32.0  # distance (m) at/above which a column drops to L1 (2x)
    lod_band_l2_m: float = 96.0  # distance (m) for L1 -> L2 (4x)
    lod_band_l3_m: float = 192.0  # distance (m) for L2 -> L3 (8x)
    lod_downsample_mode: str = "any"  # material reduce: "any" = max-id, "majority" = mode
    lod_coarse_submit_per_frame: int = 4  # max coarse mesh jobs submitted per frame
    lod_coarse_uploads_per_frame: int = 4  # max coarse node Geoms uploaded per frame
    lighting_backend: str = "gpu"
    light_c0_cells: int = 96
    light_c0_cell_m: float = 0.5
    light_c1_cells: int = 96
    light_c1_cell_m: float = 1.0
    light_c2_cells: int = 64
    light_c2_cell_m: float = 4.0
    light_quant_m: float = 0.0625
    light_gi_rays: int = 16
    light_gi_steps: int = 24
    light_gi_iters: int = 2
    light_gi_smooth_passes: int = 1
    light_penumbra_deg: float = 2.5
    light_bounce_strength: float = 0.7
    light_ao_strength: float = 0.6
    # Static tree/bush occluders (lighting/occluders.py): trunks splat a
    # near-opaque column; canopies are a TRANSLUCENT leaf medium — per-meter
    # extinction derived from the species' real leaf area, scaled by this
    # gain (light through X m of crown = exp(-sigma*gain*X), identical at
    # every cascade cell size).
    light_tree_trunk_occ: float = 0.85
    light_tree_canopy_extinction_gain: float = 1.0
    light_max_point_lights: int = 64
    light_exposure: float = 0.9
    exposure_adapt_enabled: bool = True
    exposure_min: float = 0.55
    exposure_max: float = 5.0
    exposure_key: float = 0.18
    exposure_tau_dark_s: float = 4.0
    exposure_tau_bright_s: float = 0.7
    fog_enabled: bool = True
    fog_froxels_x: int = 160
    fog_froxels_y: int = 90
    fog_froxels_z: int = 64
    fog_far_m: float = 192.0
    fog_anisotropy: float = 0.55
    grass_density_per_m2: float = 12.0
    grass_blade_height_m: float = 0.6
    grass_fade_start_m: float = 60.0
    grass_fade_end_m: float = 90.0
    grass_max_instances: int = 200_000
    # --- Flora ([flora] table; consumed by world/flora_renderer.py) ---
    flora_flower_density_per_m2: float = 1.5
    flora_flower_height_m: float = 0.45
    flora_flower_fade_start_m: float = 60.0
    flora_flower_fade_end_m: float = 90.0
    flora_flower_max_instances: int = 50_000
    # --- 3-D trees/bushes ([trees] table; world/tree_renderer.py) ---
    tree_density_per_m2: float = 0.02
    tree_min_spacing_m: float = 3.0
    tree_max_instances: int = 2_000
    tree_mesh_fade_start_m: float = 110.0
    tree_mesh_fade_end_m: float = 140.0
    tree_impostor_fade_start_m: float = 300.0
    tree_impostor_fade_end_m: float = 380.0
    tree_default_species: str = "tree_gnarled_oak"
    bush_density_per_m2: float = 0.08
    bush_min_spacing_m: float = 1.2
    bush_max_instances: int = 5_000
    bush_mesh_fade_start_m: float = 60.0
    bush_mesh_fade_end_m: float = 80.0
    bush_impostor_fade_start_m: float = 120.0
    bush_impostor_fade_end_m: float = 150.0
    bush_default_species: str = "bush_scrub"
    # --- Buildings ([buildings] table; consumed by fire_engine/buildings/) ---
    building_default_storey_height_m: float = 3.0
    building_default_wall_thickness_m: float = 0.3
    building_slab_thickness_m: float = 0.2
    building_foundation_depth_m: float = 0.5
    building_arc_segments_per_quarter: int = 8
    building_snap_eps_m: float = 0.01
    # --- Wind field ([wind] table; consumed by fire_engine/world/wind/) ---
    wind_time_scale: float = 1.0
    wind_cells: int = 64
    wind_cell_m: float = 4.0
    wind_snap_cells: int = 8
    wind_margin_cells: int = 8
    wind_gust_modes: int = 12
    wind_gust_wavelen_min: float = 20.0
    wind_gust_wavelen_max: float = 120.0
    wind_gust_omega_min: float = 0.15
    wind_gust_omega_max: float = 0.8
    wind_gust_base: float = 0.6
    wind_gust_storm_gain: float = 1.4
    wind_storm_freq_gain: float = 0.8
    wind_speed_ref: float = 8.0
    wind_turb_base: float = 0.2
    wind_turb_storm_gain: float = 1.0
    wind_shear: float = 0.18
    wind_profile_z_ref: float = 10.0
    wind_profile_floor: float = 0.35
    wind_profile_cap: float = 1.6
    wind_layer_m: float = 8.0
    wind_venturi_iters: int = 8
    wind_venturi_max: float = 2.2
    wind_deflect_gain: float = 0.15
    wind_updraft_gain: float = 0.4
    wind_mote_count: int = 1500
    wind_mote_box_m: float = 24.0
    wind_mote_size_m: float = 0.04
    wind_mote_life_s: float = 6.0
    wind_leaf_density_per_m2: float = 0.15
    wind_leaf_size_m: float = 0.12
    wind_leaf_max_instances: int = 20_000
    # --- Rain cover heightmap ([rain] table; terrain/rain_cover.py + M6 rain) ---
    rain_cover_cells: int = 256
    rain_cover_cell_m: float = 1.0
    rain_cover_budget_columns: int = 4
    # --- Weather simulation ([weather] table; consumed by fire_engine/world/weather/) ---
    # Synoptic flow: the slow steering current that carries storm cells.
    # Closed-form pure function of (seed, game time) — see weather/synoptic.py.
    weather_synoptic_components: int = 4  # vector sinusoids in W(t)
    weather_synoptic_speed_min_ms: float = 1.5  # guaranteed speed band (m/s)
    weather_synoptic_speed_max_ms: float = 11.0
    weather_synoptic_period_min_h: float = 2.5  # sinusoid period band (game h)
    weather_synoptic_period_max_h: float = 14.0
    # Storm cells (M2): spatial showers/storms/cloud-banks/fog-banks that drift
    # on the synoptic flow. Natural spawn schedule is a pure function of
    # (seed, day) — see weather/cells.py. Units: meters, game seconds, m/s.
    weather_domain_m: float = 6000.0  # half-extent of the spawn square (±m around origin)
    weather_spawn_slots_per_day: int = 8  # candidate cell spawn slots per day
    weather_cell_radius_min_m: float = 350.0  # cell footprint radius band at peak (m)
    weather_cell_radius_max_m: float = 1200.0
    weather_cell_duration_min_s: float = 2400.0  # cell lifetime band (game s; 40 min .. 3 h)
    weather_cell_duration_max_s: float = 10800.0
    weather_storm_wind_max_ms: float = 9.0  # extra gust a THUNDERSTORM adds at its core (m/s)
    weather_fog_max_density: float = 0.028  # cap on FOG_BANK fog coefficient (1/m)
    weather_temp_mean_c: float = 12.0  # daily mean air temperature (°C)
    weather_temp_amp_c: float = 8.0  # daily temperature swing amplitude (°C)
    # Weather map (M3): a derived raster of the local weather fields around the
    # player — the render + sampling cache (never saved; recomputed each tick).
    weather_map_cells: int = 128  # raster resolution (square, N×N texels)
    weather_map_cell_m: float = 24.0  # raster texel size (m) → 128×24 ≈ 3 km span
    # Ground wetness (M3): closed-form fixed-offset quadrature over the analytic
    # rain history at a point (no integrated state — recompute-free on load).
    weather_wetness_tau_s: float = 3600.0  # wetness decay time constant (game s)
    weather_wetness_step_s: float = 600.0  # quadrature step into the past (game s)
    weather_wetness_samples: int = 12  # number of past rain samples (window = step·samples)
    # Emergent humidity + fog (M5): fog condenses from conditions, not a state;
    # all closed-form pure fn of (seed, t, pos). See weather/humidity.py.
    # Units: relative humidity 0–1, °C, m/s, fog 1/m.
    weather_humidity_base_min: float = 0.35  # seeded per-day calm-air humidity baseline band
    weather_humidity_base_max: float = 0.65
    weather_humidity_rain_gain: float = 1.00  # humidity added per unit recent rain (0–1)
    weather_humidity_wetness_gain: float = 0.30  # humidity added per unit ground wetness (0–1)
    # recent-rain decay time constant (game s; 5 h — air holds moisture for hours
    # so evening rain feeds pre-dawn fog)
    weather_humidity_recent_tau_s: float = 18000.0
    weather_humidity_recent_step_s: float = 1800.0  # recent-rain quadrature step into past (game s)
    weather_humidity_recent_samples: int = 12  # recent-rain samples (window = step·samples = 6 h)
    weather_fog_emergent_max: float = (
        0.022  # max emergent fog coefficient at full condensation (1/m)
    )
    weather_fog_sat_ref_c: float = 5.0  # reference temperature for saturation humidity (°C)
    weather_fog_sat_base: float = 0.63  # saturation humidity at the reference temperature (0–1)
    weather_fog_sat_slope_per_c: float = (
        0.011  # saturation humidity rise per °C above the reference
    )
    weather_fog_condense_band: float = (
        0.10  # humidity overshoot over saturation for full condensation
    )
    weather_fog_wind_full_ms: float = 1.0  # full fog at/below this wind speed (m/s)
    weather_fog_wind_none_ms: float = 3.0  # no emergent fog at/above this wind speed (m/s)
    # WMO cloud genera (M9): the sampled weather (coverage/density/precip +
    # regime) is mapped to layered altitude bands — high cirrus, mid alto-,
    # low strato/cumulus/cumulonimbus — by weather/clouds.py::cloud_layers
    # (headless) and mirrored in-shader by cloud_volumetric.frag (no new
    # texture data; the same coverage/density/precip channels drive the bands).
    # base_altitude / thickness in meters (world Z); cov_weight/density 0–1;
    # detail_scale is a multiplier on the renderer's base noise frequency
    # (cirrus stretched+smooth → low; cumulus billowy → high).
    cloud_genera_high_alt_m: float = 1400.0  # high (cirrus) band base altitude (m)
    cloud_genera_high_thick_m: float = 120.0  # high band thickness (m; thin veil)
    cloud_genera_mid_alt_m: float = 850.0  # mid (alto-) band base altitude (m)
    cloud_genera_mid_thick_m: float = 220.0  # mid band thickness (m)
    cloud_genera_low_alt_m: float = 500.0  # low (cumulus/stratus) band base altitude (m)
    cloud_genera_low_thick_m: float = 400.0  # low band thickness (m; storms deepen it)
    cloud_genera_high_cov_floor: float = (
        0.06  # cirrus present even in fair weather (residual floor)
    )
    cloud_genera_high_cov_weight: float = 0.35  # extra high-band coverage per unit sampled coverage
    cloud_genera_high_density: float = 0.30  # ice cloud is thin: cap on high-band opacity
    cloud_genera_mid_cov_weight: float = 0.60  # mid-band coverage per unit sampled coverage
    cloud_genera_high_detail_scale: float = 0.45  # high band: stretched, smooth streaks
    cloud_genera_mid_detail_scale: float = 0.85  # mid band: moderate lumpiness
    cloud_genera_low_detail_scale: float = 1.30  # low band: billowy cumulus detail
    # Summon API + gust-front coupling (M8). Summoned cells are spawned UPWIND of
    # the player (so they drift in on the synoptic flow) with per-kind footprint
    # defaults; the gust-front modifier kicks in when a cell's leading edge nears
    # the player. All meters / game seconds / m/s — no magic numbers in code.
    weather_summon_upwind_m: float = 2500.0  # how far upwind a summoned cell spawns (m)
    weather_summon_rain_radius_m: float = 700.0  # summoned rainstorm footprint radius (m)
    weather_summon_rain_duration_s: float = 5400.0  # summoned rainstorm lifetime (game s; 90 min)
    weather_summon_rain_intensity: float = 0.85  # summoned rainstorm peak intensity (0–1)
    weather_summon_storm_radius_m: float = 950.0  # summoned thunderstorm footprint radius (m)
    weather_summon_storm_duration_s: float = (
        6000.0  # summoned thunderstorm lifetime (game s; 100 min)
    )
    weather_summon_storm_intensity: float = 1.0  # summoned thunderstorm peak intensity (0–1)
    weather_summon_fog_radius_m: float = 600.0  # summoned fog-bank footprint radius (m)
    weather_summon_fog_duration_s: float = 7200.0  # summoned fog-bank lifetime (game s; 2 h)
    weather_summon_fog_intensity: float = 0.9  # summoned fog-bank peak intensity (0–1)
    # register a gust-front modifier when a cell's leading edge is within this
    # range of the player (m)
    weather_gustfront_range_m: float = 600.0
    weather_gustfront_strength_ms: float = (
        7.0  # peak added wind speed along the summoned gust front (m/s)
    )
    weather_gustfront_width_m: float = 80.0  # gust-front band half-width (Gaussian sigma, m)
    # Procedural lightning (M7): a deterministic Poisson strike schedule per
    # active THUNDERSTORM cell (pure fn of seed+cell+time window — recomputes
    # identically after a save/load mid-storm) and a stepped-leader bolt grown
    # through a cheap seeded potential field.  See weather/lightning.py +
    # weather/bolt.py. Units: strikes/min, meters, degrees, counts.
    weather_lightning_strikes_per_min: float = (
        2.5  # peak strike rate per cell at full intensity (thinned by cell intensity)
    )
    weather_lightning_cloud_base_m: float = (
        220.0  # cloud-base height above ground the bolt starts at (m)
    )
    weather_lightning_ground_z_m: float = (
        8.0  # fallback ground-plane world Z when no cover heightmap (m; == ground_height_m)
    )
    bolt_step_len_min_m: float = 5.0  # stepped-leader step length band (m)
    bolt_step_len_max_m: float = 15.0
    bolt_cone_deg: float = 38.0  # half-angle of the downward candidate-direction fan (deg)
    bolt_candidates: int = 7  # K candidate directions fanned each step
    bolt_softmax_temp: float = 0.35  # softmax temperature for the seeded direction pick
    bolt_branch_prob: float = 0.12  # per-step probability a side branch spawns
    bolt_max_steps: int = 400  # hard cap on leader steps (the one bounded loop)
    bolt_noise_gain: float = 0.6  # weight of the seeded value-noise "air resistance" in the score
    bolt_repulsion_gain: float = 0.45  # weight of repulsion from the existing channel in the score
    bolt_branch_max_depth: int = 3  # branches stop spawning sub-branches past this depth
    # --- Graphics quality ([graphics] table; defaults == "high" preset) ---
    gfx_preset: str = "high"
    gfx_post_process: bool = True
    gfx_hdr_format: str = "rgba16f"
    gfx_render_scale: float = 1.0
    gfx_bloom: bool = True
    gfx_bloom_mips: int = 5
    gfx_bloom_threshold: float = 1.0
    gfx_bloom_knee: float = 0.5
    gfx_bloom_strength: float = 0.06
    gfx_fxaa: bool = True
    gfx_lens_flare: bool = True
    gfx_clouds: bool = True
    gfx_cloud_steps: int = 96
    gfx_cloud_light_steps: int = 8
    gfx_cloud_resolution_scale: float = 1.0
    gfx_cloud_max_dist_m: float = 6000.0
    gfx_weather_map: bool = True
    gfx_cloud_virga: bool = True
    # M9: layered WMO genera bands (high cirrus / mid alto / low cumulus+cb).
    # Off => single-slab clouds (pre-M9 look). Requires gfx_weather_map.
    gfx_cloud_genera: bool = True
    gfx_god_rays: bool = True
    gfx_god_ray_samples: int = 32
    gfx_foliage_shadow_refine: bool = True
    # Volumetric rain (M6): "off" / "cylinders" (cheap scrolled shells) /
    # "particles" (GPU-instanced falling streaks).  Both gated modes honour the
    # rain-cover heightmap cull (no rain under a roof) + the weather-map precip
    # footprint.  Defaults == the "high" preset (particles, 12k).
    gfx_rain_mode: str = "particles"
    gfx_rain_particles: int = 12_000
    gfx_rain_occlusion: bool = True
    # Procedural lightning bolts (M7): the render half of the lightning system
    # (camera-facing stepped-leader ribbons + flash + transient scene light).
    # On for low+ presets, off for "off".  Gates world/lightning_renderer.py.
    gfx_lightning_bolts: bool = True
    # Aesthetic tuning — NOT carried by the presets (so they stay consistent
    # across off/low/medium/high); override freely in [graphics] in config.toml.
    gfx_god_ray_strength: float = 0.4
    gfx_lens_flare_strength: float = 0.055
    gfx_lens_flare_threshold: float = 4.0
    gfx_tonemap_hue_preserve: float = 0.8
    gfx_sun_disc_intensity: float = 45.0
    gfx_sun_halo_intensity: float = 1.8
    gfx_sun_min_brightness: float = 0.25
    gfx_sky_inscatter_scale: float = 0.9
    # --- Performance profiler ([profiler] table; fire_engine/core/profiler.py) ---
    # The engine-agnostic frame profiler + its render-side overlay / PStats
    # bridge.  Nearly free enabled, truly free disabled (no buffers / overlay /
    # PStats objects are constructed when ``profiler_enabled`` is false).  All
    # thresholds/sizes here so nothing is a magic number.  See
    # docs/systems/profiler.md.
    profiler_enabled: bool = False
    profiler_overlay_enabled: bool = True
    profiler_frame_budget_ms: float = 5.0
    profiler_history_frames: int = 1024
    profiler_hitch_abs_ms: float = 8.0
    profiler_hitch_rel_mult: float = 1.5
    profiler_hitch_window: int = 120
    profiler_max_scopes: int = 64
    profiler_max_counters: int = 32
    profiler_recent_hitches: int = 16
    profiler_overlay_graph_frames: int = 240
    profiler_overlay_hz: float = 8.0
    profiler_snapshot_enabled: bool = False
    profiler_snapshot_path: str = "profiling/latest.json"
    profiler_snapshot_interval_s: float = 1.0
    profiler_pstats: bool = False

    # ------------------------------------------------------------------
    # Derived read-only properties
    # ------------------------------------------------------------------

    @property
    def chunk_meters(self) -> float:
        """
        World-space side length of one chunk in meters.

        ``chunk_size * voxel_size`` — always 16.0 m with the locked defaults.

        Docs: docs/systems/core.md
        """
        return float(self.chunk_size) * self.voxel_size

    @property
    def light_cell_meters(self) -> float:
        """
        World-space side length of one light-grid cell in meters.

        ``voxel_size * light_grid_scale`` — always 1.0 m with the locked defaults.

        Docs: docs/systems/core.md
        """
        return self.voxel_size * float(self.light_grid_scale)


# ---------------------------------------------------------------------------
# Re-export the loader API from config_loader so that load_config,
# resolve_graphics_preset, and GRAPHICS_PRESETS all keep resolving from
# fire_engine.core.config unchanged (the loader moved to config_loader.py).
#
# The import is placed at the BOTTOM of this file (after Config is fully
# defined) so config_loader.py can safely import Config from this module at
# its own top without triggering a circular-import error.
# ---------------------------------------------------------------------------
from fire_engine.core._impl.config_loader import (  # noqa: E402
    GRAPHICS_PRESETS,
    load_config,
    resolve_graphics_preset,
)

__all__ = [
    "GRAPHICS_PRESETS",
    "Config",
    "load_config",
    "resolve_graphics_preset",
]
