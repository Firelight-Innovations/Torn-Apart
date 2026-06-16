"""
sky/cloud_noise.py — deterministic, tileable 3-D noise for volumetric clouds.

Bakes the density-field textures the volumetric cloud raymarch samples
(``world/shaders/cloud_volumetric.frag``).  Pure numpy, no panda3d — runs in
the headless suite; ``world/texture_bridge.to_panda_texture_3d`` does the upload.

Two volumes, both **tileable** (every octave's lattice period divides the
texture, so ``WM_repeat`` wraps seamlessly — no visible cloud seam as the field
scrolls with the wind):

* **shape** (``bake_shape_noise``, default 128³ RGBA): R = a billowy
  Perlin-Worley base (the cloud bulk); G/B/A = increasing-frequency Worley FBM
  octaves the shader uses to erode the base into wisps.
* **detail** (``bake_detail_noise``, default 32³ RGBA): high-frequency Worley
  packed across channels — fine edge detail added on top of the shape.

Determinism: every value comes from ``core.rng.for_domain("sky", ...)``, so the
same world seed always bakes byte-identical volumes (a determinism test guards
this).  The bake is a one-time boot cost (vectorised; well under a second at
128³); drop to 64³ if a target machine's boot time matters.

Channel/axis convention: arrays are ``(N, N, N, 4) uint8`` indexed
``[z, y, x, channel]`` — page-major, matching ``to_panda_texture_3d`` (no
transpose on upload).  Noise is isotropic, so the axis labelling is arbitrary.

Example
-------
    from fire_engine.core.rng import set_world_seed
    from fire_engine.world.sky.cloud_noise import bake_shape_noise

    set_world_seed(1337)
    shape = bake_shape_noise(64)        # (64, 64, 64, 4) uint8, deterministic
"""

from __future__ import annotations

import numpy as np

from fire_engine.core.rng import for_domain

__all__ = ["bake_detail_noise", "bake_shape_noise"]


def _smoothstep(f: np.ndarray) -> np.ndarray:
    return f * f * (3.0 - 2.0 * f)


def _value_octave(size: int, freq: int, rng: np.random.Generator) -> np.ndarray:
    """One tileable value-noise octave: trilinear over an ``freq³`` lattice."""
    grid = rng.random((freq, freq, freq)).astype(np.float32)
    t = (np.arange(size, dtype=np.float32) + 0.5) / size * freq
    i0 = np.floor(t).astype(np.intp) % freq
    i1 = (i0 + 1) % freq
    f = _smoothstep(t - np.floor(t))
    fx = f[:, None, None]
    fy = f[None, :, None]
    fz = f[None, None, :]

    def corner(ix, iy, iz):
        return grid[np.ix_(ix, iy, iz)]

    g000 = corner(i0, i0, i0)
    g100 = corner(i1, i0, i0)
    g010 = corner(i0, i1, i0)
    g110 = corner(i1, i1, i0)
    g001 = corner(i0, i0, i1)
    g101 = corner(i1, i0, i1)
    g011 = corner(i0, i1, i1)
    g111 = corner(i1, i1, i1)
    g00 = g000 + (g100 - g000) * fx
    g10 = g010 + (g110 - g010) * fx
    g01 = g001 + (g101 - g001) * fx
    g11 = g011 + (g111 - g011) * fx
    g0 = g00 + (g10 - g00) * fy
    g1 = g01 + (g11 - g01) * fy
    return g0 + (g1 - g0) * fz


def _worley_octave(size: int, freq: int, rng: np.random.Generator) -> np.ndarray:
    """One tileable INVERTED-Worley octave (1 − dist-to-nearest feature)."""
    feat = rng.random((freq, freq, freq, 3)).astype(np.float32)  # offset in cell
    t = (np.arange(size, dtype=np.float32) + 0.5) / size * freq
    cell = np.floor(t).astype(np.intp)
    sx = t[:, None, None]
    sy = t[None, :, None]
    sz = t[None, None, :]
    mind2 = np.full((size, size, size), 1e9, np.float32)
    for ox in (-1, 0, 1):
        cxw = (cell + ox) % freq
        ccx = (cell + ox).astype(np.float32)
        for oy in (-1, 0, 1):
            cyw = (cell + oy) % freq
            ccy = (cell + oy).astype(np.float32)
            for oz in (-1, 0, 1):
                czw = (cell + oz) % freq
                ccz = (cell + oz).astype(np.float32)
                fo = feat[np.ix_(cxw, cyw, czw)]  # (N,N,N,3)
                fx = fo[..., 0] + ccx[:, None, None]
                fy = fo[..., 1] + ccy[None, :, None]
                fz = fo[..., 2] + ccz[None, None, :]
                d2 = (sx - fx) ** 2 + (sy - fy) ** 2 + (sz - fz) ** 2
                np.minimum(mind2, d2, out=mind2)
    w = np.clip(np.sqrt(mind2), 0.0, 1.0)
    return (1.0 - w).astype(np.float32)


def _worley_fbm(
    size: int, base_freq: int, octaves: int, rng: np.random.Generator, gain: float = 0.5
) -> np.ndarray:
    """Sum of inverted-Worley octaves (freq doubles each octave); → [0,1]."""
    total = np.zeros((size, size, size), np.float32)
    amp = 1.0
    norm = 0.0
    freq = base_freq
    for _ in range(octaves):
        if freq > size // 2:
            break
        total += amp * _worley_octave(size, freq, rng)
        norm += amp
        amp *= gain
        freq *= 2
    return total / max(norm, 1e-6)


def _value_fbm(
    size: int, base_freq: int, octaves: int, rng: np.random.Generator, gain: float = 0.5
) -> np.ndarray:
    total = np.zeros((size, size, size), np.float32)
    amp = 1.0
    norm = 0.0
    freq = base_freq
    for _ in range(octaves):
        if freq > size // 2:
            break
        total += amp * _value_octave(size, freq, rng)
        norm += amp
        amp *= gain
        freq *= 2
    return total / max(norm, 1e-6)


def _to_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)


def bake_shape_noise(size: int = 64) -> np.ndarray:
    """
    Bake the cloud SHAPE volume: ``(size, size, size, 4) uint8``.

    R = Perlin-Worley billowy base (cloud bulk); G/B/A = Worley FBM erosion
    octaves at increasing frequency.  Deterministic via
    ``for_domain("sky", "cloud_shape")``; tileable (``WM_repeat``-safe).

    Parameters
    ----------
    size : int, default 64
        Edge length in voxels (power-of-two recommended).  64³ bakes in ~1.7 s;
        128³ is sharper but ~30 s — cache it if you raise this (the bake is
        deterministic, so a disk cache keyed by seed+size is always valid).

    Returns
    -------
    numpy.ndarray
        ``(size, size, size, 4)`` uint8, channel order RGBA, indexed
        ``[z, y, x, c]``.
    """
    rng = for_domain("sky", "cloud_shape")
    perlin = _value_fbm(size, 4, 4, rng)
    worley_lo = _worley_fbm(size, 4, 3, rng)
    # Perlin-Worley (Guerrilla/Nubis): dilate the perlin base by the low Worley
    # so the bulk is billowy rather than smooth.
    pw = worley_lo + perlin * (1.0 - worley_lo)

    out = np.empty((size, size, size, 4), np.uint8)
    out[..., 0] = _to_u8(np.clip(pw, 0.0, 1.0))
    out[..., 1] = _to_u8(_worley_fbm(size, 8, 3, rng))
    out[..., 2] = _to_u8(_worley_fbm(size, 16, 3, rng))
    out[..., 3] = _to_u8(_worley_fbm(size, 32, 2, rng))
    return out


def bake_detail_noise(size: int = 32) -> np.ndarray:
    """
    Bake the cloud DETAIL volume: ``(size, size, size, 4) uint8``.

    High-frequency Worley FBM packed across R/G/B (A = 255) — the shader uses
    it to erode the shape's edges into fine wisps.  Deterministic via
    ``for_domain("sky", "cloud_detail")``; tileable.

    Parameters
    ----------
    size : int, default 32
        Edge length in voxels.

    Returns
    -------
    numpy.ndarray
        ``(size, size, size, 4)`` uint8, RGBA, indexed ``[z, y, x, c]``.
    """
    rng = for_domain("sky", "cloud_detail")
    out = np.empty((size, size, size, 4), np.uint8)
    out[..., 0] = _to_u8(_worley_fbm(size, 4, 3, rng))
    out[..., 1] = _to_u8(_worley_fbm(size, 8, 2, rng))
    out[..., 2] = _to_u8(_worley_fbm(size, 16, 1, rng))
    out[..., 3] = 255
    return out
