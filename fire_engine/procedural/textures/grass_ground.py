"""
procedural/textures/grass_ground.py — "grass_ground" texture definition.

Produces a 64×64 RGBA ground texture evoking living-but-weathered post-
apocalyptic grass: mid-greens with slight desaturation, light blade highlights,
darker patches, and a small posterised palette (~6–8 colours) that reads as
crisp pixel art.

Generation algorithm
--------------------
Two independent layered ``pixel_noise`` fields drive the colour selection:

``base_noise``
    Low-frequency field (3 octaves, base_freq=4) establishing the broad
    variation between darker shaded patches and lighter grass areas.

``blade_noise``
    Higher-frequency field (2 octaves, base_freq=8) adding fine-scale blade
    highlight variation on top of the base.

The two fields are combined (``base_noise * 0.7 + blade_noise * 0.3``) then
mapped through a **posterised** colour ramp: the combined value is divided into
discrete threshold buckets, each assigned one of 6–8 hard-coded palette colours.
Hard thresholds (not interpolation) ensure the final image has a limited, crisp
pixel-art palette.  Alpha is always 255.

This definition is registered as ``"grass_ground"`` at import time via the
``@register_def`` decorator.

Usage
-----
::

    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural import get
    import numpy as np

    set_world_seed(42)
    arr = get("grass_ground")     # np.ndarray (64, 64, 4) uint8
    # Pass to world/texture_bridge.py for Panda3D use, or preview with:
    # python tools/preview_texture.py grass_ground
"""

from __future__ import annotations

import numpy as np

from fire_engine.procedural.defs import register_def
from fire_engine.procedural.textures.base import ProceduralTextureDef, pixel_noise

__all__ = ["GRASS_PALETTE", "GRASS_THRESHOLDS", "GrassGroundDef"]


# ---------------------------------------------------------------------------
# Post-apocalyptic grass palette — 8 colours, from darkest to lightest
# ---------------------------------------------------------------------------
#
# Palette designed for weathered-but-living grass in a ruined world:
# desaturated mid-greens, slightly muddy undertones, muted highlights.
#
# Index 0 = darkest shadow patch (near black-green)
# Index 7 = lightest blade tip / dried highlight

_GRASS_PALETTE = np.array(
    [
        (30, 45, 22),  # 0 — dark shadow / deep root shade
        (48, 65, 32),  # 1 — dark grass patch
        (62, 82, 40),  # 2 — mid-dark weathered green
        (78, 98, 48),  # 3 — base mid-green (dominant colour)
        (94, 112, 55),  # 4 — lighter mid-green
        (115, 128, 62),  # 5 — light grass, slight desaturation
        (140, 148, 78),  # 6 — pale highlight / dried blade tip
        (162, 155, 90),  # 7 — lightest dried/yellowed blade tip
    ],
    dtype=np.uint8,
)

# Threshold boundaries that divide [0, 1] into 8 buckets.
# Values are upper bounds; the last bucket catches the remainder.
_THRESHOLDS = np.array([0.08, 0.20, 0.34, 0.50, 0.64, 0.78, 0.90], dtype=np.float32)

# Public aliases — the single source of truth for the grass ground colour ramp.
# The GPU terrain shader bakes these into its world-space palette LUT
# (world/terrain_shader.py → procedural.textures.ground_lut.build_ground_lut)
# so the non-repeating procedural ground matches this baked-texture art exactly.
GRASS_PALETTE = _GRASS_PALETTE
GRASS_THRESHOLDS = _THRESHOLDS


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
        Bucket 0: field < thresholds[0]; Bucket k: thresholds[k-1] <= field < thresholds[k];
        Bucket N-1: field >= thresholds[-1].

    Returns
    -------
    numpy.ndarray
        Shape ``(H, W, 3)``, dtype ``uint8``.
    """
    H, W = field.shape
    # Use np.searchsorted to assign each pixel to a bucket index.
    # 'side="right"' means value == threshold goes to the next bucket.
    indices = np.searchsorted(thresholds, field.ravel(), side="right").astype(np.int32)
    # Clamp to valid palette range (0 .. N-1)
    np.clip(indices, 0, len(palette) - 1, out=indices)
    # Index palette — shape (H*W, 3) → reshape to (H, W, 3)
    return palette[indices].reshape(H, W, 3)


@register_def
class GrassGroundDef(ProceduralTextureDef):
    """
    Post-apocalyptic living-but-weathered grass ground texture.

    Generates a 64×64 RGBA pixel-art grass texture using two layered
    ``pixel_noise`` fields posterised to a fixed 8-colour palette.
    The result has clearly visible square texels with crisp hard-edged
    colour transitions — no smooth gradients.

    Registered name
    ---------------
    ``"grass_ground"``

    Output
    ------
    ``numpy.ndarray`` of shape ``(64, 64, 4)``, dtype ``uint8``.
    Alpha channel is always 255 (fully opaque).

    Colour palette (8 colours)
    --------------------------
    Dark shadow patches → mid weathered greens → light dried blade tips.
    All colours are desaturated and slightly muddy to fit a post-apocalyptic
    world where the grass survives but is far from lush.

    Example
    -------
    ::

        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural import get

        set_world_seed(99)
        arr = get("grass_ground")
        # arr.shape == (64, 64, 4), arr.dtype == uint8
        # Visualise: python tools/preview_texture.py grass_ground
    """

    name = "grass_ground"

    DEFAULT_WIDTH = 64
    DEFAULT_HEIGHT = 64

    def generate(self, rng: np.random.Generator, **params) -> np.ndarray:
        """
        Generate the grass ground texture.

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

        # --- Noise field 1: broad patch variation (low-frequency) ---
        base_noise = pixel_noise(
            rng,
            shape=shape,
            octaves=3,
            persistence=0.55,
            lacunarity=2.0,
            base_freq=4,
        )

        # --- Noise field 2: fine blade highlights (higher frequency) ---
        blade_noise = pixel_noise(
            rng,
            shape=shape,
            octaves=2,
            persistence=0.5,
            lacunarity=2.0,
            base_freq=8,
        )

        # --- Combine: broad patches dominate, blades add fine variation ---
        combined = base_noise * 0.70 + blade_noise * 0.30

        # --- Posterise to palette — hard thresholds, no interpolation ---
        rgb = _posterise(combined, _GRASS_PALETTE, _THRESHOLDS)

        # --- Assemble RGBA ---
        rgba = np.empty((H, W, 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = 255  # fully opaque

        return rgba
