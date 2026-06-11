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
from dataclasses import dataclass, field


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
    show_fps             : bool  — overlay FPS counter.
    show_chunk_borders   : bool  — debug overlay for chunk boundaries.
    show_light_grid      : bool  — debug overlay for the light grid.
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
    light_c1_cells       : int   — cascade-1 texels per axis (96 → 192 m box).
    light_c1_cell_m      : float — cascade-1 cell edge in meters (2.0).
    light_c2_cells       : int   — cascade-2 texels per axis (64 → 512 m box):
                                   the coarse FAR cascade that keeps distant
                                   terrain (and the GI test room) lit with
                                   low-resolution shadows + GI once it leaves
                                   cascade 1, instead of falling back to flat
                                   sky ambient.  Assembled off-thread like the
                                   others.
    light_c2_cell_m      : float — cascade-2 cell edge in meters (8.0).
    light_quant_m        : float — shading sample-grid quantisation in meters
                                   (0.0625 → 8×8×8 visible light pixels per
                                   0.5 m voxel — the pixelated-light look).
                                   This is only the visible sample-snap grid;
                                   the underlying GI *data* resolution is the
                                   cascade-0 cell (``light_c0_cell_m``), so
                                   shrinking this past the cell size yields a
                                   finer-but-smoother grid, not more detail.
    light_prop_iters     : int   — GI flood-fill propagation iterations per
                                   frame per cascade (light "flows" over a few
                                   frames after a change).
    light_bounce_strength: float — [0,1] albedo-tinted bounce gain per
                                   propagation step.
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
    """

    world_seed:           int   = 1337
    world_size_m:         float = 1000.0
    ground_height_m:      float = 8.0
    voxel_size:           float = 0.5
    chunk_size:           int   = 32
    light_grid_scale:     int   = 2
    view_distance_chunks: int   = 6
    fixed_dt:             float = 0.02
    show_fps:             bool  = True
    show_chunk_borders:   bool  = False
    show_light_grid:      bool  = False
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
    light_c1_cell_m:       float = 2.0
    light_c2_cells:        int   = 64
    light_c2_cell_m:       float = 8.0
    light_quant_m:         float = 0.0625
    light_prop_iters:      int   = 2
    light_bounce_strength: float = 0.7
    light_ao_strength:     float = 0.6
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

    The TOML file may have ``[debug]``, ``[sky]`` and ``[terrain]`` tables;
    their keys (``show_fps``, ``show_chunk_borders``, ``show_light_grid``,
    ``sky_cloud_altitude_m``, ``sky_cloud_thickness_m``, ``sky_cloud_cell_m``,
    ``sky_star_count``, ``mesh_style``, ``facet_shade_strength``) are
    flattened into the same ``Config`` struct.  Any key absent from the file
    falls back to the ``Config`` dataclass default.

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

    # Flatten [debug], [sky] and [terrain] tables into top-level dict
    _TABLES = ("debug", "sky", "terrain", "lighting", "fog", "grass")
    flat: dict = {k: v for k, v in raw.items() if k not in _TABLES}
    for table in _TABLES:
        flat.update(raw.get(table, {}))

    # Build Config by extracting only known fields (ignore unknown keys)
    known_fields = {f.name for f in Config.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs = {k: flat[k] for k in flat if k in known_fields}

    return Config(**kwargs)
