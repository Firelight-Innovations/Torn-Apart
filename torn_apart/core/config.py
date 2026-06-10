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

Derived read-only properties
-----------------------------
    chunk_meters         : float — chunk_size * voxel_size  (16.0 m)
    light_cell_meters    : float — voxel_size * light_grid_scale (1.0 m)

Example
-------
    from torn_apart.core.config import load_config

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

    The TOML file may have a ``[debug]`` table; its keys (``show_fps``,
    ``show_chunk_borders``, ``show_light_grid``) are flattened into the same
    ``Config`` struct.  Any key absent from the file falls back to the
    ``Config`` dataclass default.

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

    # Flatten [debug] table into top-level dict
    flat: dict = {k: v for k, v in raw.items() if k != "debug"}
    debug = raw.get("debug", {})
    flat.update(debug)

    # Build Config by extracting only known fields (ignore unknown keys)
    known_fields = {f.name for f in Config.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs = {k: flat[k] for k in flat if k in known_fields}

    return Config(**kwargs)
