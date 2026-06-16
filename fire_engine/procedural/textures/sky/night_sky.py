"""
procedural/textures/sky/night_sky.py — "night_sky" equirect star-field texture.

Produces a 1024×512 RGBA equirectangular night-sky texture: a power-law
star field with a structured galaxy band (filaments, dust lanes, warm core),
plus faint large-scale nebula tint patches.  The renderer maps it onto the
sky dome with **+Z pole at v = 1** (array row 0 = zenith) and additively
blends it using the alpha channel (alpha = luminance).

Shared galaxy constants and the cube-map helpers (``cube_face_directions``,
``_dirs_to_face_pixels``, ``_hash3i``, ``_value_noise_3d``,
``_upsample2_faces``, ``_ramp_rgb``) live in
:mod:`fire_engine.procedural.textures.sky._night_sky_helpers` and are
re-exported from this module so all historical import paths remain valid.

The cube-map variant is :class:`NightSkyCubeDef`, defined in
:mod:`fire_engine.procedural.textures.sky.night_sky_cube` and re-exported
here so ``from fire_engine.procedural.textures.night_sky import
NightSkyCubeDef`` continues to work.

Usage
-----
::

    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural import get

    set_world_seed(1337)
    arr = get("night_sky")                      # (512, 1024, 4) uint8
    arr = get("night_sky", star_count=4000)     # denser field
    # Preview: python tools/preview_texture.py night_sky

Docs: docs/systems/procedural.md
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from fire_engine.procedural.defs import register_def
from fire_engine.procedural.textures.base import ProceduralTextureDef, value_noise

# Re-export cube-map symbols so the historical path
# ``from fire_engine.procedural.textures.night_sky import X`` still resolves.
# NightSkyCubeDef is imported BEFORE NightSkyDef is defined — night_sky_cube.py
# imports from _night_sky_helpers.py only (no circular dependency back here).
from fire_engine.procedural.textures.sky._night_sky_helpers import (
    _BAND_STAR_FRACTION,
    _BAND_STAR_SIGMA_RAD,
    _BRIGHT_FRACTION,
    _GALAXY_CORE_SIGMA,
    _GALAXY_HALO_SIGMA,
    _GALAXY_INCLINATION_RAD,
    _GALAXY_RAMP_KEYS,
    _GALAXY_RAMP_RGB,
    _SKY_FLOOR,
    _dirs_to_face_pixels,  # noqa: F401  re-export for historical imports
    _ramp_rgb,
    cube_face_directions,
)
from fire_engine.procedural.textures.sky.night_sky_cube import NightSkyCubeDef

__all__ = ["NightSkyCubeDef", "NightSkyDef", "cube_face_directions"]


def _seamless_u(field: np.ndarray) -> np.ndarray:
    """
    Make a (H, W) noise field wrap seamlessly in U (the W axis).

    Triangular crossfade of *field* with its half-width roll: weight 0 at
    the image centre column, 1 at both edges.  Column 0 then equals
    ``field[:, W/2]`` and column W−1 equals ``field[:, W/2 − 1]`` — adjacent
    samples, so wrapping is continuous.  Reuses the plain (non-tiling)
    ``value_noise`` output rather than reimplementing tileable noise.

    Parameters
    ----------
    field : np.ndarray — (H, W) float field (any range).

    Returns
    -------
    np.ndarray — (H, W) float32, same value range, U-seamless.
    """
    _H, W = field.shape
    rolled = np.roll(field, W // 2, axis=1)
    x = np.arange(W, dtype=np.float32)
    s = np.abs(x - W / 2.0) / (W / 2.0)  # 1 at edges, 0 at centre
    s = s[None, :]
    result: np.ndarray = (field * (1.0 - s) + rolled * s).astype(np.float32)
    return result


@register_def
class NightSkyDef(ProceduralTextureDef):
    """
    Equirectangular night-sky texture: star field + structured galaxy band.

    Registered name
    ---------------
    ``"night_sky"``

    Output
    ------
    ``numpy.ndarray`` of shape ``(512, 1024, 4)``, dtype ``uint8`` by
    default.  Equirect mapping with **+Z pole at v = 1** (array row 0 is
    the zenith).  Wraps seamlessly in U.  Alpha = luminance — the renderer
    additively blends the sky dome with it, scaled by
    ``SkyState.star_visibility``.

    Parameters (via ``get("night_sky", ...)``)
    ------------------------------------------
    - ``width`` (int): output width in pixels.  Default 1024.
    - ``height`` (int): output height in pixels.  Default 512.
    - ``star_count`` (int): number of stars.  Default 2500 — callers should
      pass ``Config.sky_star_count`` so the count stays config-driven.

    Example
    -------
    ::

        from fire_engine.core.config import load_config
        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural import get

        cfg = load_config()
        set_world_seed(cfg.world_seed)
        arr = get("night_sky", star_count=cfg.sky_star_count)
        assert arr.shape == (512, 1024, 4) and arr.dtype == "uint8"

    Docs: docs/systems/procedural.md
    """

    name = "night_sky"

    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 512
    DEFAULT_STAR_COUNT = 2500  # mirrors Config.sky_star_count default

    def generate(self, rng: np.random.Generator, **params: Any) -> np.ndarray:
        """
        Generate the night-sky texture.

        Parameters
        ----------
        rng : numpy.random.Generator
            Seeded generator injected by the registry — sole randomness
            source (galaxy orientation, noise fields, star placement).
        **params : any
            ``width`` / ``height`` / ``star_count`` overrides (see class doc).

        Returns
        -------
        numpy.ndarray — shape ``(H, W, 4)``, dtype ``uint8``, RGBA.
        """
        W = int(params.get("width", self.DEFAULT_WIDTH))
        H = int(params.get("height", self.DEFAULT_HEIGHT))
        n_stars = int(params.get("star_count", self.DEFAULT_STAR_COUNT))

        # --- Per-pixel unit directions (equirect, +Z pole at v=1 / row 0) ---
        u = (np.arange(W, dtype=np.float32) + 0.5) / W  # (W,)
        v = 1.0 - (np.arange(H, dtype=np.float32) + 0.5) / H  # (H,)
        lon = (2.0 * np.pi) * u  # (W,)
        lat = (v - 0.5) * np.pi  # (H,)
        cos_lat = np.cos(lat)[:, None]  # (H, 1)
        sin_lat = np.sin(lat)[:, None]  # (H, 1)
        dx = cos_lat * np.cos(lon)[None, :]  # (H, W)
        dy = cos_lat * np.sin(lon)[None, :]  # (H, W)
        dz = np.broadcast_to(sin_lat, (H, W))  # (H, W)

        # --- Galaxy band geometry (rng draw #1: band azimuth) ---
        band_az = float(rng.uniform(0.0, 2.0 * np.pi))
        sin_i, cos_i = math.sin(_GALAXY_INCLINATION_RAD), math.cos(_GALAXY_INCLINATION_RAD)
        # Band plane normal, tilted 60° from +Z toward a random azimuth.
        nx = sin_i * math.cos(band_az)
        ny = sin_i * math.sin(band_az)
        nz = cos_i
        plane_dist = dx * nx + dy * ny + dz * nz  # (H, W) = sin(angle off plane)
        d2 = plane_dist * plane_dist
        core = np.exp(-d2 / (2.0 * _GALAXY_CORE_SIGMA**2)).astype(np.float32)
        halo = np.exp(-d2 / (2.0 * _GALAXY_HALO_SIGMA**2)).astype(np.float32)

        # --- Filament + dust noise (U-seamless) ---
        filaments = _seamless_u(
            value_noise(
                rng,
                (H, W),
                octaves=5,
                persistence=0.55,
                lacunarity=2.2,
                base_freq=8,
            )
        )
        # Sharpen hard: push midtones way down so bright threads pop.
        filaments = filaments**2.4
        halo_mottle = _seamless_u(
            value_noise(
                rng,
                (H, W),
                octaves=4,
                persistence=0.5,
                lacunarity=2.0,
                base_freq=4,
            )
        )
        dust = _seamless_u(
            value_noise(
                rng,
                (H, W),
                octaves=4,
                persistence=0.5,
                lacunarity=2.0,
                base_freq=9,
            )
        )
        # Threshold dust into lanes; they carve hardest through the core.
        dust_lanes = np.clip((dust - 0.45) / 0.28, 0.0, 1.0)
        dust_lanes = dust_lanes * dust_lanes * (3.0 - 2.0 * dust_lanes)

        galaxy = core * (0.35 + 1.30 * filaments) + 0.45 * halo * (0.25 + 0.75 * halo_mottle)
        galaxy *= 1.0 - 0.90 * dust_lanes * np.sqrt(np.clip(core + 0.4 * halo, 0.0, 1.0))
        # Fine grain so saturated core regions keep visible texture.
        fine = _seamless_u(
            value_noise(
                rng,
                (H, W),
                octaves=3,
                persistence=0.6,
                lacunarity=2.0,
                base_freq=24,
            )
        )
        galaxy *= 0.82 + 0.36 * fine
        galaxy = np.clip(galaxy, 0.0, 1.0)

        rgb = np.empty((H, W, 3), dtype=np.float32)
        rgb[:] = _SKY_FLOOR
        rgb += _ramp_rgb(galaxy, _GALAXY_RAMP_KEYS, _GALAXY_RAMP_RGB) * galaxy[..., None]

        # --- Faint large-scale nebula tint patches (U-seamless) ---
        neb_teal = _seamless_u(
            value_noise(
                rng,
                (H, W),
                octaves=3,
                persistence=0.6,
                lacunarity=2.0,
                base_freq=2,
            )
        )
        neb_violet = _seamless_u(
            value_noise(
                rng,
                (H, W),
                octaves=3,
                persistence=0.6,
                lacunarity=2.0,
                base_freq=3,
            )
        )
        teal_w = np.clip((neb_teal - 0.55) / 0.35, 0.0, 1.0)[..., None]
        violet_w = np.clip((neb_violet - 0.57) / 0.33, 0.0, 1.0)[..., None]
        rgb += np.array([0.025, 0.075, 0.090], dtype=np.float32) * teal_w
        rgb += np.array([0.085, 0.035, 0.110], dtype=np.float32) * violet_w

        # --- Stars ---
        n_band = int(n_stars * _BAND_STAR_FRACTION)
        n_uni = n_stars - n_band

        # Uniform-on-sphere stars (correct equirect density: ∝ cos(latitude)).
        z_u = rng.uniform(-1.0, 1.0, n_uni)
        phi_u = rng.uniform(0.0, 2.0 * np.pi, n_uni)
        lat_u = np.arcsin(z_u)
        lon_u = phi_u

        # Band stars: angle along the band + gaussian offset from its plane.
        # Orthonormal basis of the band plane: e1 ⊥ n in the XY-ish sense,
        # e2 = n × e1; direction = cosβ(cosα·e1 + sinα·e2) + sinβ·n.
        e1 = np.array([-math.sin(band_az), math.cos(band_az), 0.0])
        e2 = np.cross([nx, ny, nz], e1)
        alpha = rng.uniform(0.0, 2.0 * np.pi, n_band)
        beta = np.clip(rng.normal(0.0, _BAND_STAR_SIGMA_RAD, n_band), -0.45, 0.45)
        in_plane = np.cos(alpha)[:, None] * e1[None, :] + np.sin(alpha)[:, None] * e2[None, :]
        d_band = (
            np.cos(beta)[:, None] * in_plane
            + np.sin(beta)[:, None] * np.array([nx, ny, nz])[None, :]
        )
        lat_b = np.arcsin(np.clip(d_band[:, 2], -1.0, 1.0))
        lon_b = np.mod(np.arctan2(d_band[:, 1], d_band[:, 0]), 2.0 * np.pi)

        lat_all = np.concatenate([lat_u, lat_b])
        lon_all = np.concatenate([lon_u, lon_b])

        cols = (lon_all / (2.0 * np.pi) * W).astype(np.int64) % W
        rows = np.clip(((1.0 - (lat_all / np.pi + 0.5)) * H).astype(np.int64), 0, H - 1)

        # Brightness power-law (many dim, few bright) + bright tier.
        bright = rng.random(n_stars) ** 2.5 * 0.55 + 0.08
        bright_mask = rng.random(n_stars) < _BRIGHT_FRACTION
        bright[bright_mask] = 0.75 + rng.random(int(bright_mask.sum())) * 0.45

        # Subtle warm/cool color variation (blue-white ↔ amber).
        temp = rng.normal(0.0, 1.0, n_stars)
        star_rgb = np.stack(
            [
                bright * np.clip(1.0 + 0.10 * temp, 0.75, 1.25),
                bright,
                bright * np.clip(1.0 - 0.12 * temp, 0.70, 1.30),
            ],
            axis=1,
        ).astype(np.float32)  # (N, 3)

        # Splat all stars: np.add.at scatter (duplicate pixels accumulate).
        np.add.at(rgb, (rows, cols), star_rgb)

        # Bright stars: a few shifted-add passes → 1-px cross + faint glow.
        br, bc = rows[bright_mask], cols[bright_mask]
        b_rgb = star_rgb[bright_mask]
        # (row offset, col offset, weight): 4-arm cross, then diagonals,
        # then 2-px arm tips — a fixed-count kernel-offset loop (allowed).
        for dr, dc, wgt in (
            (-1, 0, 0.45),
            (1, 0, 0.45),
            (0, -1, 0.45),
            (0, 1, 0.45),
            (-1, -1, 0.15),
            (-1, 1, 0.15),
            (1, -1, 0.15),
            (1, 1, 0.15),
            (-2, 0, 0.15),
            (2, 0, 0.15),
            (0, -2, 0.15),
            (0, 2, 0.15),
        ):
            rr = np.clip(br + dr, 0, H - 1)
            cc = (bc + dc) % W  # wrap in U
            np.add.at(rgb, (rr, cc), b_rgb * wgt)

        # --- Assemble RGBA: alpha = luminance (additive-blend mask) ---
        np.clip(rgb, 0.0, 1.0, out=rgb)
        lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        rgba = np.empty((H, W, 4), dtype=np.uint8)
        rgba[..., :3] = (rgb * 255.0).astype(np.uint8)
        rgba[..., 3] = (np.clip(lum, 0.0, 1.0) * 255.0).astype(np.uint8)
        return rgba
