"""
TOML loader and graphics-preset resolver for the Torn Apart engine configuration.

Moved to :mod:`fire_engine.core._impl.config_loader` to satisfy the
per-directory module-count limit.  ``load_config`` and
``resolve_graphics_preset`` are re-exported from ``config.py`` so all existing
import paths continue to work unchanged:

    from fire_engine.core.config import load_config          # still valid
    from fire_engine.core.config import resolve_graphics_preset  # still valid

This module imports ``Config`` from ``config`` at the top of the file.  The
circular-import chain is safe because ``config.py`` does the reverse import at
the **bottom** of the file (after ``Config`` is fully defined), so ``Config``
is already in ``config``'s namespace when Python evaluates this module.

Docs: docs/systems/core.md
"""

from __future__ import annotations

import tomllib
import warnings
from typing import Any

from fire_engine.core.config import Config

__all__ = ["GRAPHICS_PRESETS", "load_config", "resolve_graphics_preset"]

# ----------------------------------------------------------------------------
# Graphics-quality presets.  Each maps the heavy/quality-dependent ``gfx_*``
# knobs to a value; aesthetic constants (bloom threshold/knee/strength, cloud
# max distance) are intentionally omitted so they fall back to the dataclass
# default and stay consistent across presets.  ``"high"`` mirrors the Config
# dataclass defaults exactly.  Tune for the target machine via config.toml.
# ----------------------------------------------------------------------------
GRAPHICS_PRESETS: dict[str, dict[str, Any]] = {
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


def resolve_graphics_preset(graphics_table: dict[str, Any] | None = None) -> dict[str, Any]:
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
    resolved: dict[str, Any] = dict(GRAPHICS_PRESETS[requested])
    resolved["gfx_preset"] = requested
    # Explicit per-field overrides win over the preset.
    resolved.update(table)
    return resolved


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
    _TABLES = (
        "debug",
        "sky",
        "terrain",
        "lighting",
        "fog",
        "grass",
        "flora",
        "trees",
        "buildings",
        "wind",
        "rain",
        "weather",
        "graphics",
        "profiler",
    )
    flat: dict[str, Any] = {k: v for k, v in raw.items() if k not in _TABLES}
    for table in _TABLES:
        if table == "graphics":
            continue
        flat.update(raw.get(table, {}))
    flat.update(resolve_graphics_preset(raw.get("graphics", {})))

    # Build Config by extracting only known fields (ignore unknown keys)
    known_fields = {f.name for f in Config.__dataclass_fields__.values()}
    kwargs = {k: flat[k] for k in flat if k in known_fields}

    return Config(**kwargs)
