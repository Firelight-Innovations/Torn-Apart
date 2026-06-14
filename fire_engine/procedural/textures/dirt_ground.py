"""
procedural/textures/dirt_ground.py — "dirt_ground" texture definition.

Produces a 64×64 RGBA ground texture evoking dry, compacted post-apocalyptic
dirt: sandy brown mid-tones, small dark clods/stone clusters, and occasional
lighter dust pockets — all rendered as crisp pixel art via a posterised
6-colour palette.

Generation algorithm
--------------------
Two independent layered ``pixel_noise`` fields drive the colour selection:

``base_noise``
    Low-frequency field (3 octaves, base_freq=3) establishing the broad
    dirt/sand tonal variation (light dust ↔ shadowed earth).

``clod_noise``
    High-frequency field (2 octaves, base_freq=10) adding small dark clods,
    pebbles, and surface irregularities — they appear as few-pixel clusters.

The two fields are combined (``base_noise * 0.65 + clod_noise * 0.35``) then
mapped through a **posterised** ramp: hard threshold buckets each assigned one
of 6 palette colours.  No interpolation — all transitions are abrupt, giving
the pixel-art look.  Alpha is always 255.

This definition is registered as ``"dirt_ground"`` at import time via the
``@register_def`` decorator.

Usage
-----
::

    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural import get
    import numpy as np

    set_world_seed(42)
    arr = get("dirt_ground")     # np.ndarray (64, 64, 4) uint8
    # Pass to world/texture_bridge.py for Panda3D use, or preview with:
    # python tools/preview_texture.py dirt_ground
"""

from __future__ import annotations

import numpy as np

from fire_engine.procedural.defs import register_def
from fire_engine.procedural.textures.base import ProceduralTextureDef, pixel_noise

__all__ = ["DIRT_PALETTE", "DIRT_THRESHOLDS", "DirtGroundDef"]


# ---------------------------------------------------------------------------
# Post-apocalyptic dry dirt palette — 6 colours, dark to light
# ---------------------------------------------------------------------------
#
# Palette designed for baked, cracked dirt in a ruined world:
# near-black soil shadows → dark earthy brown → sandy mid-tones →
# pale dust highlights.  Small dark clods read as index 0/1.

_DIRT_PALETTE = np.array(
    [
        (28, 18, 8),  # 0 — dark clod / deep shadow (near-black soil)
        (55, 38, 18),  # 1 — dark earthy brown (shadow clod edge)
        (88, 62, 32),  # 2 — mid earthy brown (dominant base)
        (128, 95, 52),  # 3 — sandy brown mid-tone
        (162, 128, 75),  # 4 — light sandy tan
        (192, 162, 102),  # 5 — pale dust / light highlight
    ],
    dtype=np.uint8,
)

# 5 thresholds divide [0,1] into 6 buckets.
_THRESHOLDS = np.array([0.10, 0.24, 0.42, 0.60, 0.80], dtype=np.float32)

# Public aliases — single source of truth for the dirt ground colour ramp,
# baked into the GPU terrain shader's world-space palette LUT (see
# procedural.textures.ground_lut.build_ground_lut) so the non-repeating
# procedural ground matches this baked-texture art exactly.
DIRT_PALETTE = _DIRT_PALETTE
DIRT_THRESHOLDS = _THRESHOLDS


def _posterise(field: np.ndarray, palette: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Map a float32 ``(H, W)`` field to a fixed palette via hard thresholds.

    Parameters
    ----------
    field : numpy.ndarray
        Shape ``(H, W)``, float32, values in ``[0, 1]``.
    palette : numpy.ndarray
        Shape ``(N, 3)``, uint8, one RGB colour per bucket.
    thresholds : numpy.ndarray
        Shape ``(N-1,)``, float32, ascending upper bounds for each bucket.
        Bucket 0: field < thresholds[0]; last bucket: field >= thresholds[-1].

    Returns
    -------
    numpy.ndarray
        Shape ``(H, W, 3)``, dtype ``uint8``.
    """
    H, W = field.shape
    indices = np.searchsorted(thresholds, field.ravel(), side="right").astype(np.int32)
    np.clip(indices, 0, len(palette) - 1, out=indices)
    return palette[indices].reshape(H, W, 3)


@register_def
class DirtGroundDef(ProceduralTextureDef):
    """
    Post-apocalyptic dry dirt ground texture.

    Generates a 64×64 RGBA pixel-art dirt texture using two layered
    ``pixel_noise`` fields posterised to a fixed 6-colour palette.
    The high-frequency clod noise produces small dark clusters of a few pixels,
    evoking small stones and soil clods.  All colour transitions are hard-edged
    (no smooth gradients) — typical pixel-art ground aesthetics.

    Registered name
    ---------------
    ``"dirt_ground"``

    Output
    ------
    ``numpy.ndarray`` of shape ``(64, 64, 4)``, dtype ``uint8``.
    Alpha channel is always 255 (fully opaque).

    Colour palette (6 colours)
    --------------------------
    Near-black clod shadows → dark earthy brown → sandy mid-tones →
    pale dust highlights.  All post-apocalyptic dry palette; no reds or
    saturated tones.

    Example
    -------
    ::

        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural import get

        set_world_seed(99)
        arr = get("dirt_ground")
        # arr.shape == (64, 64, 4), arr.dtype == uint8
        # Visualise: python tools/preview_texture.py dirt_ground
    """

    name = "dirt_ground"

    DEFAULT_WIDTH = 64
    DEFAULT_HEIGHT = 64

    def generate(self, rng: np.random.Generator, **params) -> np.ndarray:
        """
        Generate the dry dirt ground texture.

        Parameters
        ----------
        rng : numpy.random.Generator
            Seeded generator injected by the registry.
        **params : any
            Optional overrides:
            - ``width``  (int): output width in pixels.  Default 64.
            - ``height`` (int): output height in pixels.  Default 64.

        Returns
        -------
        numpy.ndarray
            Shape ``(H, W, 4)``, dtype ``uint8``.
        """
        W = int(params.get("width", self.DEFAULT_WIDTH))
        H = int(params.get("height", self.DEFAULT_HEIGHT))
        shape = (H, W)

        # --- Noise field 1: broad soil tonal variation (low-frequency) ---
        base_noise = pixel_noise(
            rng,
            shape=shape,
            octaves=4,
            persistence=0.5,
            lacunarity=2.0,
            base_freq=5,
        )

        # --- Noise field 2: clods / small stones (high-frequency) ---
        clod_noise = pixel_noise(
            rng,
            shape=shape,
            octaves=2,
            persistence=0.5,
            lacunarity=2.0,
            base_freq=10,
        )

        # --- Combine: base tone dominates, clods add dark spots ---
        combined = base_noise * 0.65 + clod_noise * 0.35

        # --- Posterise to palette — hard thresholds, no interpolation ---
        rgb = _posterise(combined, _DIRT_PALETTE, _THRESHOLDS)

        # --- Assemble RGBA ---
        rgba = np.empty((H, W, 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = 255  # fully opaque

        return rgba
