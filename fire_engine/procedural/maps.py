"""
procedural/maps.py — Derive normal / emission maps from procedural textures.

Environment textures are 100 % procedural (no asset files), so their
secondary maps are *derived*, not authored: the normal map comes from a
Sobel gradient of the albedo's luminance (treating brightness as height —
exactly right for the pixel-art ground textures, where lit grains read as
raised), and emission maps default to black unless a def provides one.

Pure numpy, no panda3d, no per-pixel Python loops.

Example
-------
>>> import numpy as np
>>> from fire_engine.procedural.maps import derive_normal_map
>>> rgba = np.zeros((8, 8, 4), dtype=np.uint8); rgba[..., 3] = 255
>>> nm = derive_normal_map(rgba)
>>> nm.shape, nm.dtype
((8, 8, 4), dtype('uint8'))
>>> tuple(nm[0, 0, :3])     # flat input → flat "up" normal (128, 128, 255)
(128, 128, 255)
"""

from __future__ import annotations

import numpy as np

__all__ = ["derive_normal_map", "flat_normal_map", "black_emission_map"]


def derive_normal_map(rgba: np.ndarray, strength: float = 1.4) -> np.ndarray:
    """
    Build a tangent-space normal map from a texture's luminance gradient.

    Luminance is treated as a height field; Sobel X/Y gradients (with wrap
    padding — terrain textures tile) become the tangent-space XY slope and
    Z is reconstructed for unit length.  Encoded in the standard
    ``uint8`` convention: ``rgb = normal * 0.5 + 0.5`` (flat = 128,128,255).

    Parameters
    ----------
    rgba : numpy.ndarray
        ``uint8 (H, W, 4)`` source texture (a ``ProceduralTextureDef``
        output).
    strength : float, default 1.4
        Slope gain — higher = deeper-looking relief.  Pixel-art textures
        read well between 1.0 and 2.0.

    Returns
    -------
    numpy.ndarray
        ``uint8 (H, W, 4)`` normal map, alpha 255.
    """
    lum = (
        rgba[..., :3].astype(np.float32) @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    ) / 255.0
    p = np.pad(lum, 1, mode="wrap")
    # Sobel gradients (X = columns axis 1, Y = rows axis 0).
    gx = (
        (p[0:-2, 2:] + 2.0 * p[1:-1, 2:] + p[2:, 2:])
        - (p[0:-2, 0:-2] + 2.0 * p[1:-1, 0:-2] + p[2:, 0:-2])
    ) * 0.25
    gy = (
        (p[2:, 0:-2] + 2.0 * p[2:, 1:-1] + p[2:, 2:])
        - (p[0:-2, 0:-2] + 2.0 * p[0:-2, 1:-1] + p[0:-2, 2:])
    ) * 0.25
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(nx)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    out = np.empty(rgba.shape, dtype=np.uint8)
    out[..., 0] = np.clip(np.rint((nx * inv * 0.5 + 0.5) * 255.0), 0, 255)
    out[..., 1] = np.clip(np.rint((ny * inv * 0.5 + 0.5) * 255.0), 0, 255)
    out[..., 2] = np.clip(np.rint((nz * inv * 0.5 + 0.5) * 255.0), 0, 255)
    out[..., 3] = 255
    return out


def flat_normal_map(size: int = 4) -> np.ndarray:
    """``uint8 (size, size, 4)`` all-flat normal map (128, 128, 255, 255)."""
    out = np.empty((size, size, 4), dtype=np.uint8)
    out[..., 0] = 128
    out[..., 1] = 128
    out[..., 2] = 255
    out[..., 3] = 255
    return out


def black_emission_map(size: int = 4) -> np.ndarray:
    """``uint8 (size, size, 4)`` all-black (non-emissive) emission map."""
    out = np.zeros((size, size, 4), dtype=np.uint8)
    out[..., 3] = 255
    return out
