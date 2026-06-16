"""
procedural/textures/sky/night_sky_cube.py — "night_sky_cube" cubemap texture.

Cube-map night sky: the SAME galaxy/star model as
:class:`~fire_engine.procedural.textures.sky.night_sky.NightSkyDef` evaluated
per-texel on true 3-D directions for all six faces of an OpenGL cube map.
No pole distortion, hardware-filtered face seams, and the renderer can spin it
about ANY (tilted) celestial axis.

Shared galaxy constants and cube-map helpers (``cube_face_directions``,
``_dirs_to_face_pixels``, ``_hash3i``, ``_value_noise_3d``,
``_upsample2_faces``, ``_ramp_rgb``) live in
:mod:`fire_engine.procedural.textures.sky._night_sky_helpers`.

``NightSkyCubeDef`` is re-exported from
:mod:`fire_engine.procedural.textures.sky.night_sky` so the historical path
``from fire_engine.procedural.textures.night_sky import NightSkyCubeDef``
keeps working.

Docs: docs/systems/procedural.md
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from fire_engine.procedural.defs import register_def
from fire_engine.procedural.textures.base import ProceduralTextureDef
from fire_engine.procedural.textures.sky._night_sky_helpers import (
    _BAND_STAR_FRACTION,
    _BAND_STAR_SIGMA_RAD,
    _GALAXY_CORE_SIGMA,
    _GALAXY_HALO_SIGMA,
    _GALAXY_INCLINATION_RAD,
    _GALAXY_RAMP_KEYS,
    _GALAXY_RAMP_RGB,
    _SKY_FLOOR,
    _dirs_to_face_pixels,
    _ramp_rgb,
    _upsample2_faces,
    _value_noise_3d,
    cube_face_directions,
)

__all__ = ["NightSkyCubeDef"]

#: Fraction of cube stars promoted to the bright tier (subtle cross arms).
_CUBE_BRIGHT_FRACTION: float = 0.03


@register_def
class NightSkyCubeDef(ProceduralTextureDef):
    """
    Cube-map night sky: star field + structured galaxy band, per-face.

    Registered name
    ---------------
    ``"night_sky_cube"``

    Output
    ------
    ``numpy.ndarray`` of shape ``(6, face_size, face_size, 4)``, dtype
    ``uint8`` — the six faces of an OpenGL cube map in GL face order
    (+X, −X, +Y, −Y, +Z, −Z), texel layout per
    :func:`~fire_engine.procedural.textures.sky._night_sky_helpers.cube_face_directions`.
    Alpha = luminance (the dome shader's additive-blend / twinkle mask).
    Bridge to the GPU with ``world.texture_bridge.to_panda_cubemap``.

    Same visual family as ``"night_sky"`` (galaxy core/halo, filaments,
    dust lanes, nebula tints, power-law stars) but evaluated on true 3-D
    directions: no pole pinching, seamless faces, denser/smaller stars
    (default 9000 single-texel stars on 512² faces).

    Parameters (via ``get("night_sky_cube", ...)``)
    -----------------------------------------------
    - ``face_size`` (int): texels per face edge (power of two).  Default 512.
    - ``star_count`` (int): number of stars.  Default 9000 — pass
      ``Config.sky_star_count``.

    Example
    -------
    ::

        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural import get

        set_world_seed(1337)
        faces = get("night_sky_cube")            # (6, 512, 512, 4) uint8

    Docs: docs/systems/procedural.md
    """

    name = "night_sky_cube"

    DEFAULT_FACE_SIZE = 512
    DEFAULT_STAR_COUNT = 9000

    def generate(self, rng: np.random.Generator, **params: Any) -> np.ndarray:
        """
        Generate the six cube-map faces.

        Parameters
        ----------
        rng : numpy.random.Generator
            Seeded generator injected by the registry — sole randomness
            source (galaxy orientation, noise seeds, star placement).
        **params : any
            ``face_size`` / ``star_count`` overrides (see class doc).

        Returns
        -------
        numpy.ndarray — ``(6, face_size, face_size, 4)`` uint8 RGBA.

        Docs: docs/systems/procedural.textures.sky.md
        """
        S = int(params.get("face_size", self.DEFAULT_FACE_SIZE))
        n_stars = int(params.get("star_count", self.DEFAULT_STAR_COUNT))

        dirs = cube_face_directions(S)  # (6, S, S, 3)
        # Noise fields evaluate at half resolution (they are low-frequency)
        # then nearest-upsample — 4× cheaper and keeps the chunky look.
        half = cube_face_directions(S // 2)

        # --- Galaxy band geometry (rng draw #1: band azimuth) -------------
        band_az = float(rng.uniform(0.0, 2.0 * np.pi))
        sin_i = math.sin(_GALAXY_INCLINATION_RAD)
        cos_i = math.cos(_GALAXY_INCLINATION_RAD)
        n = np.array(
            [sin_i * math.cos(band_az), sin_i * math.sin(band_az), cos_i], dtype=np.float32
        )
        plane_dist = dirs @ n  # (6, S, S)
        d2 = plane_dist * plane_dist
        core = np.exp(-d2 / (2.0 * _GALAXY_CORE_SIGMA**2)).astype(np.float32)
        halo = np.exp(-d2 / (2.0 * _GALAXY_HALO_SIGMA**2)).astype(np.float32)

        # --- Direction-space noise fields (seam-free across faces) --------
        nseed = int(rng.integers(0, 2**31 - 1))
        filaments = _upsample2_faces(_value_noise_3d(half, 7.0, 5, nseed + 1)) ** 2.4
        halo_mottle = _upsample2_faces(_value_noise_3d(half, 3.5, 4, nseed + 2, persistence=0.5))
        dust = _upsample2_faces(_value_noise_3d(half, 8.0, 4, nseed + 3, persistence=0.5))
        fine = _upsample2_faces(_value_noise_3d(half, 21.0, 3, nseed + 4, persistence=0.6))
        dust_lanes = np.clip((dust - 0.45) / 0.28, 0.0, 1.0)
        dust_lanes = dust_lanes * dust_lanes * (3.0 - 2.0 * dust_lanes)

        galaxy = core * (0.35 + 1.30 * filaments) + 0.45 * halo * (0.25 + 0.75 * halo_mottle)
        galaxy *= 1.0 - 0.90 * dust_lanes * np.sqrt(np.clip(core + 0.4 * halo, 0.0, 1.0))
        galaxy *= 0.82 + 0.36 * fine
        galaxy = np.clip(galaxy, 0.0, 1.0)

        rgb = np.empty((6, S, S, 3), dtype=np.float32)
        rgb[:] = _SKY_FLOOR
        ramp = _ramp_rgb(galaxy.reshape(6 * S, S), _GALAXY_RAMP_KEYS, _GALAXY_RAMP_RGB).reshape(
            6, S, S, 3
        )
        rgb += ramp * galaxy[..., None]

        # --- Nebula tints ---------------------------------------------------
        neb_teal = _upsample2_faces(_value_noise_3d(half, 1.8, 3, nseed + 5, persistence=0.6))
        neb_violet = _upsample2_faces(_value_noise_3d(half, 2.6, 3, nseed + 6, persistence=0.6))
        rgb += (
            np.array([0.025, 0.075, 0.090], np.float32)
            * np.clip((neb_teal - 0.55) / 0.35, 0.0, 1.0)[..., None]
        )
        rgb += (
            np.array([0.085, 0.035, 0.110], np.float32)
            * np.clip((neb_violet - 0.57) / 0.33, 0.0, 1.0)[..., None]
        )

        # --- Stars: uniform sphere + galaxy-band density boost --------------
        n_band = int(n_stars * _BAND_STAR_FRACTION)
        n_uni = n_stars - n_band
        z_u = rng.uniform(-1.0, 1.0, n_uni)
        phi_u = rng.uniform(0.0, 2.0 * np.pi, n_uni)
        r_u = np.sqrt(np.clip(1.0 - z_u * z_u, 0.0, 1.0))
        d_uni = np.stack([r_u * np.cos(phi_u), r_u * np.sin(phi_u), z_u], axis=1)

        e1 = np.array([-math.sin(band_az), math.cos(band_az), 0.0])
        e2 = np.cross(n, e1)
        alpha = rng.uniform(0.0, 2.0 * np.pi, n_band)
        beta = np.clip(rng.normal(0.0, _BAND_STAR_SIGMA_RAD, n_band), -0.45, 0.45)
        in_plane = np.cos(alpha)[:, None] * e1[None, :] + np.sin(alpha)[:, None] * e2[None, :]
        d_band = np.cos(beta)[:, None] * in_plane + np.sin(beta)[:, None] * n[None, :]
        all_dirs = np.concatenate([d_uni, d_band]).astype(np.float32)

        face, row, col = _dirs_to_face_pixels(all_dirs, S)

        # Brightness power law: many dim points, a few bright; the bright
        # tier gets small cross arms (subtler than the equirect version —
        # stars read SMALLER at the cubemap's higher angular resolution).
        bright = rng.random(n_stars) ** 3.0 * 0.50 + 0.06
        bright_mask = rng.random(n_stars) < _CUBE_BRIGHT_FRACTION
        bright[bright_mask] = 0.70 + rng.random(int(bright_mask.sum())) * 0.45
        temp = rng.normal(0.0, 1.0, n_stars)
        star_rgb = np.stack(
            [
                bright * np.clip(1.0 + 0.10 * temp, 0.75, 1.25),
                bright,
                bright * np.clip(1.0 - 0.12 * temp, 0.70, 1.30),
            ],
            axis=1,
        ).astype(np.float32)

        np.add.at(rgb, (face, row, col), star_rgb)
        bf, br, bc = face[bright_mask], row[bright_mask], col[bright_mask]
        b_rgb = star_rgb[bright_mask]
        for dr, dc, wgt in ((-1, 0, 0.30), (1, 0, 0.30), (0, -1, 0.30), (0, 1, 0.30)):
            rr = np.clip(br + dr, 0, S - 1)
            cc = np.clip(bc + dc, 0, S - 1)
            np.add.at(rgb, (bf, rr, cc), b_rgb * wgt)

        # --- Assemble RGBA: alpha = luminance (twinkle/blend mask) ----------
        np.clip(rgb, 0.0, 1.0, out=rgb)
        lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        rgba = np.empty((6, S, S, 4), dtype=np.uint8)
        rgba[..., :3] = (rgb * 255.0).astype(np.uint8)
        rgba[..., 3] = (np.clip(lum, 0.0, 1.0) * 255.0).astype(np.uint8)
        return rgba
