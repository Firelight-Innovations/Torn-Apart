"""
procedural/textures/night_sky.py — "night_sky" equirect star-field texture.

Produces a 1024×512 RGBA equirectangular night-sky texture: a power-law
star field with a structured galaxy band (filaments, dust lanes, warm core),
plus faint large-scale nebula tint patches.  The renderer maps it onto the
sky dome with **+Z pole at v = 1** (array row 0 = zenith) and additively
blends it using the alpha channel (alpha = luminance).

Generation algorithm (all bulk numpy — no per-pixel Python loops)
-----------------------------------------------------------------
1. **Direction grid** — every pixel's unit view direction is computed from
   its equirect (u, v): longitude φ = 2πu, latitude λ = (v − 0.5)π.
   Galaxy geometry is evaluated against these true 3-D directions, so the
   band's shape is automatically correct in equirect space (it widens in
   pixel space toward the poles instead of pinching).
2. **Galaxy band** — a great circle inclined ``_GALAXY_INCLINATION_RAD``
   (60°) to the equator, with gaussian falloff ``exp(−(d·n)²/2σ²)`` from the
   band plane (normal ``n``, σ = 0.16).  The profile is multiplied by a
   5-octave ``value_noise`` field (filaments) and carved by a second,
   thresholded noise layer subtracted as dust lanes (strongest in the bright
   core).  Color ramp: deep indigo → violet → dusty rose → warm pale core.
3. **Seamless wrap in U** — noise fields go through :func:`_seamless_u`,
   a triangular crossfade of the field with its half-width ``np.roll``:
   column 0 and column W−1 become adjacent samples of the rolled copy, so
   the texture tiles horizontally with no visible seam.  (Star positions
   wrap naturally via modulo.)
4. **Stars** — ``star_count`` stars (default 2500, mirror of
   ``Config.sky_star_count``): 60 % uniform on the sphere (which is the
   *correct* equirect density — pixels near the poles cover less solid
   angle and receive proportionally fewer stars), 40 % gaussian-scattered
   around the galaxy plane for higher density inside the band.  Brightness
   follows a power law; ~5 % are bright stars with subtle warm/cool color
   variation.  All stars are splatted with ``np.add.at`` scatter; bright
   stars get a few shifted-add passes forming a 1-px cross + diagonal glow.
5. **Nebulae** — two very-low-frequency noise layers add faint teal and
   violet tint patches.
6. Alpha = luminance of the final RGB (the renderer's additive-blend mask).

Usage
-----
::

    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural import get

    set_world_seed(1337)
    arr = get("night_sky")                      # (512, 1024, 4) uint8
    arr = get("night_sky", star_count=4000)     # denser field
    # Preview: python tools/preview_texture.py night_sky
"""

from __future__ import annotations

import math

import numpy as np

from fire_engine.procedural.defs import register_def
from fire_engine.procedural.textures.base import ProceduralTextureDef, value_noise

__all__ = ["NightSkyDef", "NightSkyCubeDef", "cube_face_directions"]


# ---------------------------------------------------------------------------
# Tuning constants (documented)
# ---------------------------------------------------------------------------

#: Galaxy band inclination to the equator, radians (~60°): the band's great
#: circle reaches latitude ±60°, sweeping diagonally across the sky dome.
_GALAXY_INCLINATION_RAD: float = math.radians(60.0)

#: Gaussian half-widths of the two-scale band profile, in units of ``d·n``
#: (the sine of the angular distance from the band plane): a bright narrow
#: core (~6°-sigma) inside a wide faint halo (~19°-sigma).
_GALAXY_CORE_SIGMA: float = 0.11
_GALAXY_HALO_SIGMA: float = 0.33

#: Galaxy color ramp keyed on local band intensity (0 = edge, 1 = core):
#: deep indigo → violet → dusty rose-violet → dusty rose → warm pale core.
_GALAXY_RAMP_KEYS = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
_GALAXY_RAMP_RGB = np.array(
    [
        [0.02, 0.02, 0.06],
        [0.10, 0.07, 0.22],
        [0.30, 0.18, 0.30],
        [0.55, 0.38, 0.42],
        [0.88, 0.78, 0.68],
    ]
)

#: Base sky floor (very deep indigo) so empty sky is not pure black.
_SKY_FLOOR = np.array([0.012, 0.015, 0.040])

#: Fraction of stars scattered inside the galaxy band (density boost).
_BAND_STAR_FRACTION: float = 0.40

#: Gaussian sigma (radians) of band-star offsets from the galaxy plane.
_BAND_STAR_SIGMA_RAD: float = 0.13

#: Fraction of stars promoted to the bright tier (get cross/glow arms).
_BRIGHT_FRACTION: float = 0.05


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
    H, W = field.shape
    rolled = np.roll(field, W // 2, axis=1)
    x = np.arange(W, dtype=np.float32)
    s = np.abs(x - W / 2.0) / (W / 2.0)  # 1 at edges, 0 at centre
    s = s[None, :]
    return (field * (1.0 - s) + rolled * s).astype(np.float32)


def _ramp_rgb(t: np.ndarray, keys: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """
    Vectorised keyframe color ramp: map (H, W) *t* in [0, 1] through
    ``(keys, rgb)`` stops via ``np.interp`` per channel → (H, W, 3) float32.
    """
    out = np.empty((*t.shape, 3), dtype=np.float32)
    for ch in range(3):  # fixed 3-iteration loop
        out[..., ch] = np.interp(t, keys, rgb[:, ch])
    return out


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
    """

    name = "night_sky"

    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 512
    DEFAULT_STAR_COUNT = 2500  # mirrors Config.sky_star_count default

    def generate(self, rng: np.random.Generator, **params) -> np.ndarray:
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


# ===========================================================================
# Cubemap night sky — "night_sky_cube"
# ===========================================================================
#
# The equirect texture pinches at the poles and forces the renderer to fake
# the celestial rotation about world +Z.  The cubemap version evaluates the
# SAME galaxy/star model per-texel on true 3-D directions for all six faces
# of an OpenGL cube map: no pole distortion, hardware-filtered face seams,
# and the renderer can spin it about ANY (tilted) celestial axis.

#: Fraction of cube stars promoted to the bright tier (subtle cross arms).
_CUBE_BRIGHT_FRACTION: float = 0.03


def cube_face_directions(size: int) -> np.ndarray:
    """
    Per-texel unit view directions for all 6 faces of a GL cube map.

    Face order and texel orientation follow the OpenGL cube-map spec
    (+X, −X, +Y, −Y, +Z, −Z; ``sc``/``tc`` per the GL selection table), with
    array row 0 = ``tc = −1`` — exactly the layout ``set_ram_image`` feeds to
    ``glTexImage2D``, so ``texture(samplerCube, dir)`` in GLSL looks up the
    texel generated for that direction with no flips anywhere.

    Parameters
    ----------
    size : int — texels per face edge.

    Returns
    -------
    numpy.ndarray — ``float32 (6, size, size, 3)`` unit vectors,
    indexed ``[face, row, col]``.
    """
    t = (np.arange(size, dtype=np.float32) + 0.5) / size * 2.0 - 1.0
    tc, sc = np.meshgrid(t, t, indexing="ij")  # row → tc, col → sc
    one = np.ones_like(sc)
    faces = np.stack(
        [
            np.stack([one, -tc, -sc], axis=-1),  # +X
            np.stack([-one, -tc, sc], axis=-1),  # -X
            np.stack([sc, one, tc], axis=-1),  # +Y
            np.stack([sc, -one, -tc], axis=-1),  # -Y
            np.stack([sc, -tc, one], axis=-1),  # +Z
            np.stack([-sc, -tc, -one], axis=-1),  # -Z
        ],
        axis=0,
    )
    return (faces / np.linalg.norm(faces, axis=-1, keepdims=True)).astype(np.float32)


def _dirs_to_face_pixels(dirs: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map unit directions to (face, row, col) cube-map texels (GL convention).

    The exact inverse of :func:`cube_face_directions` — splatting a point at
    the returned texel puts it where the GPU lookup of that direction lands.

    Parameters
    ----------
    dirs : numpy.ndarray — ``(N, 3)`` unit vectors.
    size : int — texels per face edge.

    Returns
    -------
    (face, row, col) : three ``(N,)`` int64 arrays.
    """
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    ax, ay, az = np.abs(x), np.abs(y), np.abs(z)
    # Major axis selection (GL): x beats y beats z on ties.
    fx = (ax >= ay) & (ax >= az)
    fy = ~fx & (ay >= az)
    fz = ~fx & ~fy
    face = np.where(
        fx, np.where(x >= 0, 0, 1), np.where(fy, np.where(y >= 0, 2, 3), np.where(z >= 0, 4, 5))
    )
    ma = np.where(fx, ax, np.where(fy, ay, az))
    # GL per-face (sc, tc) selection table.
    sc = np.where(face == 0, -z, np.where(face == 1, z, np.where(face == 5, -x, x)))
    tc = np.where(face == 2, z, np.where(face == 3, -z, -y))
    col = np.clip(((sc / ma + 1.0) * 0.5 * size).astype(np.int64), 0, size - 1)
    row = np.clip(((tc / ma + 1.0) * 0.5 * size).astype(np.int64), 0, size - 1)
    return face, row, col


def _hash3i(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, seed: int) -> np.ndarray:
    """Vectorised integer-lattice hash → float32 in [0, 1)."""
    h = (
        ix.astype(np.int64) * 73856093
        ^ iy.astype(np.int64) * 19349663
        ^ iz.astype(np.int64) * 83492791
        ^ np.int64(seed)
    ).astype(np.uint32)
    h ^= h >> np.uint32(15)
    h *= np.uint32(0x2C1B3C6D)
    h ^= h >> np.uint32(12)
    h *= np.uint32(0x297A2D39)
    h ^= h >> np.uint32(15)
    return h.astype(np.float32) / np.float32(4294967296.0)


def _value_noise_3d(
    points: np.ndarray,
    base_freq: float,
    octaves: int,
    seed: int,
    persistence: float = 0.55,
    lacunarity: float = 2.1,
) -> np.ndarray:
    """
    fBm trilinear value noise evaluated AT 3-D points (direction space).

    Unlike the 2-D image-space ``value_noise``, this samples a hash lattice
    in 3-D, so values depend only on the world direction — cube-map faces
    join seamlessly with no per-face seams and no pole artifacts.

    Parameters
    ----------
    points : numpy.ndarray — ``(..., 3)`` sample coordinates (unit dirs).
    base_freq : float — lattice cells across the unit sphere at octave 0.
    octaves : int — fBm octave count.
    seed : int — lattice hash seed (mix per-field constants in).
    persistence, lacunarity : float — standard fBm parameters.

    Returns
    -------
    numpy.ndarray — float32, shape ``points.shape[:-1]``, ~[0, 1].
    """
    out = np.zeros(points.shape[:-1], dtype=np.float32)
    amp, total = 1.0, 0.0
    freq = float(base_freq)
    for o in range(octaves):
        q = points * np.float32(freq)
        i = np.floor(q)
        f = (q - i).astype(np.float32)
        f = f * f * (3.0 - 2.0 * f)
        ix, iy, iz = i[..., 0], i[..., 1], i[..., 2]
        fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
        s = seed + o * 1013
        c000 = _hash3i(ix, iy, iz, s)
        c100 = _hash3i(ix + 1, iy, iz, s)
        c010 = _hash3i(ix, iy + 1, iz, s)
        c110 = _hash3i(ix + 1, iy + 1, iz, s)
        c001 = _hash3i(ix, iy, iz + 1, s)
        c101 = _hash3i(ix + 1, iy, iz + 1, s)
        c011 = _hash3i(ix, iy + 1, iz + 1, s)
        c111 = _hash3i(ix + 1, iy + 1, iz + 1, s)
        x00 = c000 + (c100 - c000) * fx
        x10 = c010 + (c110 - c010) * fx
        x01 = c001 + (c101 - c001) * fx
        x11 = c011 + (c111 - c011) * fx
        y0 = x00 + (x10 - x00) * fy
        y1 = x01 + (x11 - x01) * fy
        out += (y0 + (y1 - y0) * fz) * np.float32(amp)
        total += amp
        amp *= persistence
        freq *= lacunarity
    return out / np.float32(total)


def _upsample2_faces(field: np.ndarray) -> np.ndarray:
    """Nearest 2× upsample of a ``(6, h, w)`` field (chunky-noise look)."""
    return np.repeat(np.repeat(field, 2, axis=1), 2, axis=2)


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
    :func:`cube_face_directions`.  Alpha = luminance (the dome shader's
    additive-blend / twinkle mask).  Bridge to the GPU with
    ``world.texture_bridge.to_panda_cubemap``.

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
    """

    name = "night_sky_cube"

    DEFAULT_FACE_SIZE = 512
    DEFAULT_STAR_COUNT = 9000

    def generate(self, rng: np.random.Generator, **params) -> np.ndarray:
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
