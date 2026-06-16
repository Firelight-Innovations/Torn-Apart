"""
procedural/textures/ground_lut.py — posterised palette LUT for the GPU ground.

The GPU terrain shader (``world/shaders/terrain.frag``) colours the ground from
a **world-space procedural noise value** rather than a tiled texture, so the
ground never repeats across the 1 km map.  To keep the look identical to the
hand-tuned baked ground textures (``grass_ground``/``dirt_ground``), the exact
posterised colour ramp those defs use is baked here into a small lookup texture:

    row  = material id   (0 = unused/air, 1 = dirt, 2 = grass, …)
    col  = noise value   (256 buckets over [0, 1))
    rgb  = the palette colour that ``_posterise`` would assign to that value

The shader then computes one world-space noise value per fragment and reads
``lut[material][noise]`` — a single nearest-filtered texel fetch.  Because both
sides go through the *same* palette + thresholds (imported from the texture
defs), the procedural ground and the baked previews are guaranteed to agree.

This module is pure numpy (headless, no panda3d); ``world/texture_bridge`` does
the final upload via :func:`to_field_texture`.

Example
-------
::

    from fire_engine.procedural.textures.ground_lut import build_ground_lut
    from fire_engine.procedural.textures.grass_ground import (
        GRASS_PALETTE, GRASS_THRESHOLDS)
    from fire_engine.procedural.textures.dirt_ground import (
        DIRT_PALETTE, DIRT_THRESHOLDS)

    lut = build_ground_lut({
        1: (DIRT_PALETTE,  DIRT_THRESHOLDS),
        2: (GRASS_PALETTE, GRASS_THRESHOLDS),
    })
    # lut.shape == (3, 256, 4), lut.dtype == uint8

Docs: docs/systems/procedural.textures.md
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

__all__ = ["build_ground_lut"]


def build_ground_lut(
    entries: Mapping[int, tuple[np.ndarray, np.ndarray]],
    levels: int = 256,
) -> np.ndarray:
    """
    Bake a per-material posterised palette into an ``(rows, levels, 4)`` LUT.

    Parameters
    ----------
    entries : Mapping[int, tuple[numpy.ndarray, numpy.ndarray]]
        Maps a **material id** to ``(palette, thresholds)`` where ``palette`` is
        ``(N, 3) uint8`` (dark→light) and ``thresholds`` is ``(N-1,) float`` of
        ascending upper bounds — exactly the constants the texture defs expose
        (e.g. ``GRASS_PALETTE``/``GRASS_THRESHOLDS``).  Material ids index rows
        directly; gaps (e.g. air = 0) are left black/opaque.
    levels : int, optional
        Number of noise buckets (LUT width).  Default 256 (one per byte), so the
        shader can address it with ``(noise * 255 + 0.5) / 256``.

    Returns
    -------
    numpy.ndarray
        Shape ``(max(material_id) + 1, levels, 4)``, dtype ``uint8``.  Channel 3
        (alpha) is always 255.  Row ``m``, column ``v`` holds the palette colour
        ``_posterise`` assigns to noise value ``(v + 0.5) / levels``.

    Notes
    -----
    Uses the same ``np.searchsorted(thresholds, value, side="right")`` rule as
    the texture defs' ``_posterise`` so the ramps match bucket-for-bucket.

    Docs: docs/systems/procedural.textures.md
    """
    if not entries:
        raise ValueError("build_ground_lut requires at least one material entry")

    rows = max(entries) + 1
    lut = np.zeros((rows, levels, 4), dtype=np.uint8)
    lut[..., 3] = 255

    # Bucket centres in [0, 1): matches the shader's (noise*255+0.5)/256 fetch.
    ramp = (np.arange(levels, dtype=np.float32) + 0.5) / float(levels)

    for mat, (palette, thresholds) in entries.items():
        palette = np.asarray(palette, dtype=np.uint8)
        thresholds = np.asarray(thresholds, dtype=np.float32)
        if palette.ndim != 2 or palette.shape[1] != 3:
            raise ValueError(f"palette for material {mat} must be (N, 3); got {palette.shape}")
        if thresholds.shape[0] != palette.shape[0] - 1:
            raise ValueError(
                f"material {mat}: expected {palette.shape[0] - 1} thresholds, "
                f"got {thresholds.shape[0]}"
            )
        idx = np.searchsorted(thresholds, ramp, side="right").astype(np.int32)
        np.clip(idx, 0, len(palette) - 1, out=idx)
        lut[mat, :, :3] = palette[idx]

    return lut
