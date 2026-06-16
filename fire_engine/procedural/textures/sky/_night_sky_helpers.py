"""
procedural/textures/sky/_night_sky_helpers.py — shared helpers for night-sky textures.

Private module (underscore prefix) holding the shared constants and helper
functions used by both
:mod:`~fire_engine.procedural.textures.sky.night_sky` (equirect) and
:mod:`~fire_engine.procedural.textures.sky.night_sky_cube` (cubemap):

* Galaxy-band tuning constants (inclination, sigma widths, colour ramp, …).
* :func:`_ramp_rgb` — vectorised keyframe colour ramp.
* :func:`cube_face_directions` — per-texel unit directions for a GL cube map.
* :func:`_dirs_to_face_pixels` — inverse: direction → (face, row, col).
* :func:`_hash3i` / :func:`_value_noise_3d` — 3-D value-noise lattice.
* :func:`_upsample2_faces` — nearest 2× upsample of a ``(6,h,w)`` field.

Import via the public night_sky module, not this private helper.

Docs: docs/systems/procedural.md
"""

from __future__ import annotations

import math

import numpy as np

__all__: list[str] = []

# ---------------------------------------------------------------------------
# Shared tuning constants
# ---------------------------------------------------------------------------

#: Galaxy band inclination to the equator, radians (~60°).
_GALAXY_INCLINATION_RAD: float = math.radians(60.0)

#: Gaussian half-widths of the two-scale band profile, in units of ``d·n``
#: (the sine of the angular distance from the band plane).
_GALAXY_CORE_SIGMA: float = 0.11
_GALAXY_HALO_SIGMA: float = 0.33

#: Galaxy colour ramp keyed on local band intensity (0 = edge, 1 = core).
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

#: Fraction of equirect stars promoted to the bright tier (cross/glow arms).
_BRIGHT_FRACTION: float = 0.05


# ---------------------------------------------------------------------------
# Colour ramp helper (used by both equirect and cubemap generators)
# ---------------------------------------------------------------------------


def _ramp_rgb(t: np.ndarray, keys: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """
    Vectorised keyframe colour ramp.

    Map ``t`` in [0, 1] through ``(keys, rgb)`` stops via ``np.interp`` per
    channel, returning a float32 array of shape ``(*t.shape, 3)``.
    """
    out = np.empty((*t.shape, 3), dtype=np.float32)
    for ch in range(3):  # fixed 3-iteration loop
        out[..., ch] = np.interp(t, keys, rgb[:, ch])
    return out


# ---------------------------------------------------------------------------
# Cube-map geometry helpers
# ---------------------------------------------------------------------------


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

    Docs: docs/systems/procedural.md
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
    normed: np.ndarray = (faces / np.linalg.norm(faces, axis=-1, keepdims=True)).astype(np.float32)
    return normed


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
    face = np.where(
        fx,
        np.where(x >= 0, 0, 1),
        np.where(fy, np.where(y >= 0, 2, 3), np.where(z >= 0, 4, 5)),
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
