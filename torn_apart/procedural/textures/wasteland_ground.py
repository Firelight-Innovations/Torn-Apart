"""
procedural/textures/wasteland_ground.py — "wasteland_ground" texture definition.

Produces a 256×256 RGBA ground texture evoking post-apocalyptic dead earth:
dry cracked dirt, patches of sickly yellow-brown dead grass, and faint
rust-coloured staining — browns, tans, and muted sickly greens.

Generation algorithm
--------------------
Two independent layered value-noise fields are generated (using the shared
``value_noise`` helper from ``procedural.textures.base``):

``base_noise``
    Low-frequency, high-contrast field driving the primary dirt/sand colour
    ramp.  4 octaves, base_freq=3.

``patch_noise``
    Higher-frequency field driving dead-grass patch distribution.  5 octaves,
    base_freq=6.

A colour ramp maps each field to an RGB triple, which are blended together
weighted by ``patch_noise``.  The alpha channel is always 255 (fully opaque).

This definition is registered as ``"wasteland_ground"`` at import time via
the ``@register_def`` decorator.

Usage
-----
::

    from torn_apart.core.rng import set_world_seed
    from torn_apart.procedural import get
    import numpy as np

    set_world_seed(42)
    arr = get("wasteland_ground")     # np.ndarray (256, 256, 4) uint8
    # Pass to world/texture_bridge.py for Panda3D use, or preview with:
    # python tools/preview_texture.py wasteland_ground
"""

from __future__ import annotations

import numpy as np

from torn_apart.procedural.defs import ProceduralDef, register_def
from torn_apart.procedural.textures.base import ProceduralTextureDef, value_noise

__all__ = ["WastelandGroundDef"]


# ---------------------------------------------------------------------------
# Colour ramp helpers — pure array operations, no per-pixel loops
# ---------------------------------------------------------------------------

def _apply_ramp(
    field: np.ndarray,
    stops: list[tuple[float, tuple[int, int, int]]],
) -> np.ndarray:
    """
    Map a float32 ``(H, W)`` field through a colour ramp.

    The ramp is defined as a list of ``(position, (R, G, B))`` stops sorted
    by position.  Between stops, colours are linearly interpolated.

    Parameters
    ----------
    field : numpy.ndarray
        Shape ``(H, W)``, float32, values in ``[0, 1]``.
    stops : list of (float, (int, int, int))
        Colour stop list.  Positions must be in ascending order.

    Returns
    -------
    numpy.ndarray
        Shape ``(H, W, 3)``, dtype ``float32``, values in ``[0.0, 255.0]``.
    """
    H, W = field.shape
    result = np.zeros((H, W, 3), dtype=np.float32)

    # Sort stops just in case
    stops = sorted(stops, key=lambda s: s[0])

    for i in range(len(stops) - 1):
        p0, (r0, g0, b0) = stops[i]
        p1, (r1, g1, b1) = stops[i + 1]
        span = p1 - p0
        if span <= 0.0:
            continue
        # Pixels that fall in [p0, p1]
        t = np.clip((field - p0) / span, 0.0, 1.0)  # (H, W)
        mask = (field >= p0) & (field < p1)           # (H, W) bool

        t3 = t[..., None]  # (H, W, 1) for broadcast
        blend = np.array([r0, g0, b0], dtype=np.float32) * (1.0 - t3) + \
                np.array([r1, g1, b1], dtype=np.float32) * t3
        result[mask] = blend[mask]

    # Last stop: pixels at exactly the last position
    p_last, (r_last, g_last, b_last) = stops[-1]
    last_mask = field >= p_last
    result[last_mask] = [r_last, g_last, b_last]

    return result


@register_def
class WastelandGroundDef(ProceduralTextureDef):
    """
    Post-apocalyptic wasteland ground texture.

    Generates a 256×256 RGBA dirt/dead-grass texture using two layered
    value-noise fields blended through colour ramps.

    Registered name
    ---------------
    ``"wasteland_ground"``

    Output
    ------
    ``numpy.ndarray`` of shape ``(256, 256, 4)``, dtype ``uint8``.
    Alpha channel is always 255 (fully opaque).

    Colour palette
    --------------
    Dirt ramp (base_noise):
      dark brown → sandy tan → light ochre

    Dead-grass ramp (patch_noise, blended in at high patch_noise values):
      pale straw → sickly yellow-green → dull olive

    Example
    -------
    ::

        from torn_apart.core.rng import set_world_seed
        from torn_apart.procedural import get

        set_world_seed(99)
        arr = get("wasteland_ground")
        # arr.shape == (256, 256, 4), arr.dtype == uint8
        # Visualise: python tools/preview_texture.py wasteland_ground
    """

    name = "wasteland_ground"

    # Default output size
    DEFAULT_WIDTH  = 256
    DEFAULT_HEIGHT = 256

    def generate(self, rng: np.random.Generator, **params) -> np.ndarray:
        """
        Generate the wasteland ground texture.

        Parameters
        ----------
        rng : numpy.random.Generator
            Seeded generator injected by the registry.
        **params : any
            Optional overrides:
            - ``width``  (int): output width in pixels.  Default 256.
            - ``height`` (int): output height in pixels.  Default 256.

        Returns
        -------
        numpy.ndarray
            Shape ``(H, W, 4)``, dtype ``uint8``.
        """
        W = int(params.get("width",  self.DEFAULT_WIDTH))
        H = int(params.get("height", self.DEFAULT_HEIGHT))
        shape = (H, W)

        # --- Noise field 1: broad dirt base (low-frequency) ---
        base_noise = value_noise(
            rng,
            shape=shape,
            octaves=4,
            persistence=0.55,
            lacunarity=2.0,
            base_freq=3,
        )

        # --- Noise field 2: dead-grass patch distribution (higher freq) ---
        patch_noise = value_noise(
            rng,
            shape=shape,
            octaves=5,
            persistence=0.5,
            lacunarity=2.1,
            base_freq=6,
        )

        # --- Colour ramps (post-apocalyptic palette) ---
        # Dirt ramp: dark cracked earth → sandy tan → pale ochre
        dirt_ramp: list[tuple[float, tuple[int, int, int]]] = [
            (0.00, ( 42,  28,  18)),   # very dark brown
            (0.25, ( 80,  55,  35)),   # dark earthy brown
            (0.50, (130,  95,  60)),   # medium tan-brown
            (0.75, (170, 135,  85)),   # sandy tan
            (1.00, (195, 165, 110)),   # pale ochre / dry dust
        ]

        # Dead-grass ramp: pale dead straw → sickly yellow-green → dull olive
        grass_ramp: list[tuple[float, tuple[int, int, int]]] = [
            (0.00, (135, 120,  65)),   # pale straw
            (0.30, (120, 115,  50)),   # yellowish dry grass
            (0.60, (105, 110,  45)),   # sickly yellow-green
            (0.85, ( 88,  95,  38)),   # dull muted olive
            (1.00, ( 70,  80,  30)),   # dark dead olive-green
        ]

        # Apply ramps → float32 (H, W, 3) in [0, 255]
        dirt_rgb  = _apply_ramp(base_noise,  dirt_ramp)   # (H, W, 3)
        grass_rgb = _apply_ramp(patch_noise, grass_ramp)  # (H, W, 3)

        # --- Blend: use patch_noise as the grass-blend weight ---
        # A threshold + smooth step for sharper-looking patches.
        # patch_noise > 0.55 gradually blends in dead grass.
        patch_threshold = 0.55
        patch_blend = np.clip(
            (patch_noise - patch_threshold) / (1.0 - patch_threshold),
            0.0, 1.0,
        )  # (H, W) in [0, 1]

        # Smooth step for organic-looking patch edges
        patch_blend = patch_blend * patch_blend * (3.0 - 2.0 * patch_blend)
        patch_blend3 = patch_blend[..., None]  # (H, W, 1) for broadcast

        blended_rgb = dirt_rgb * (1.0 - patch_blend3) + grass_rgb * patch_blend3

        # --- Assemble RGBA ---
        rgba = np.empty((H, W, 4), dtype=np.uint8)
        np.clip(blended_rgb, 0.0, 255.0, out=blended_rgb)
        rgba[..., :3] = blended_rgb.astype(np.uint8)
        rgba[..., 3] = 255  # fully opaque

        return rgba
