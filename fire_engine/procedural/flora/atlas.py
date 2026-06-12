"""
procedural/flora/atlas.py — per-species pixel-art texture atlas helpers.

One tree species = ONE small texture atlas so each variant mesh is a single
draw: the left half is tileable posterised **bark** (alpha 255), the right
half a binary-alpha **leaf mass** the foliage quads cut out.  The mesher's
``uv_rect`` arguments and :class:`AtlasLayout` are the shared contract.

These are plain rng-consuming helpers — species defs call them inside
``generate`` with the rng the registry injected.  They deliberately do NOT
nest ``procedural.get()`` calls (that would entangle cache keys and rng
streams); palettes are species data, not separate registered defs.

All output is ``(H, W, 4) uint8`` RGBA, nearest-filter pixel art, built with
:func:`~fire_engine.procedural.textures.base.pixel_noise` (no per-pixel
Python loops — Hard Rule 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fire_engine.procedural.textures.base import pixel_noise

__all__ = ["AtlasLayout", "bark_texture", "leaf_texture", "compose_atlas"]


@dataclass(frozen=True)
class AtlasLayout:
    """
    The species-atlas UV contract shared by mesher, atlas and renderer.

    Attributes
    ----------
    width, height : int
        Atlas texel size.  Default 64×64 (pixel-art scale).
    bark_rect / leaf_rect : tuple[float, float, float, float]
        ``(u0, v0, u1, v1)`` sub-rects: bark fills the left half (opaque),
        leaves the right half (binary alpha cutout).  Pass these as the
        mesher's ``uv_rect`` arguments.
    """

    width: int = 64
    height: int = 64
    bark_rect: tuple[float, float, float, float] = (0.0, 0.0, 0.5, 1.0)
    leaf_rect: tuple[float, float, float, float] = (0.5, 0.0, 1.0, 1.0)

    @property
    def half_px(self) -> tuple[int, int]:
        """(width, height) in texels of one half-atlas region."""
        return self.width // 2, self.height


def bark_texture(
    rng: np.random.Generator,
    width: int,
    height: int,
    palette: np.ndarray,
    *,
    striation_freq: int = 7,
    streak_px: int = 6,
    shade_side: bool = True,
) -> np.ndarray:
    """
    Posterised vertically-striated bark — fully opaque ``(H, W, 4) uint8``.

    Pixel noise is generated at a vertically-compressed resolution and
    stretched back up, producing the streaky grain of split bark, then
    posterised through *palette* (dark → lit).  When *shade_side*, the left
    half of the tile drops one tier — every prism face gets a built-in
    shadow edge, faking form at pixel-art scale.

    Parameters
    ----------
    rng : numpy.random.Generator
        Deterministic generator (consume the species def's rng).
    width, height : int
        Output texel size (one half-atlas, typically 32×64).
    palette : numpy.ndarray
        ``uint8 (T, 3)`` RGB ramp, shadow tone first.
    striation_freq : int
        Horizontal noise frequency — higher = finer grain.  Default 7.
    streak_px : int
        Vertical stretch factor (texels per noise row).  Default 6.
    shade_side : bool
        Darken the left tile half by one tier.  Default True.
    """
    palette = np.asarray(palette, dtype=np.uint8)
    rows = max(1, height // max(1, streak_px))
    small = pixel_noise(rng, (rows, width), octaves=2,
                        base_freq=striation_freq)
    grain = np.repeat(small, streak_px, axis=0)[:height]
    if grain.shape[0] < height:                       # streak_px remainder
        grain = np.vstack([grain, grain[: height - grain.shape[0]]])

    tier = np.clip((grain * len(palette)), 0, len(palette) - 1).astype(np.intp)
    if shade_side:
        tier[:, : width // 2] = np.maximum(tier[:, : width // 2] - 1, 0)

    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = palette[tier]
    rgba[..., 3] = 255
    return rgba


def leaf_texture(
    rng: np.random.Generator,
    width: int,
    height: int,
    palette: np.ndarray,
    *,
    hole_thresh: float = 0.18,
    clump_freq: int = 5,
    berry_color: tuple[int, int, int] | None = None,
    berry_density: float = 0.0,
) -> np.ndarray:
    """
    Ragged binary-alpha leaf mass — ``(H, W, 4) uint8`` blob cutout.

    An elliptical blob filling the tile, its edge and interior broken by
    crisp pixel noise (more holes toward the rim), shaded with a posterised
    ramp that lightens toward the top — the same Vintage-Story read as the
    old sprite canopies, now wrapped onto 3-D foliage quads.  Optional
    berry speckles for fruiting species.

    Parameters
    ----------
    rng : numpy.random.Generator
        Deterministic generator.
    width, height : int
        Output texel size (one half-atlas).
    palette : numpy.ndarray
        ``uint8 (T, 3)`` RGB foliage ramp, dark under-canopy first.
    hole_thresh : float
        Noise cut for sky holes; higher = scragglier.  Default 0.18.
    clump_freq : int
        Noise base frequency — leaf clump size.  Default 5.
    berry_color : tuple[int, int, int] | None
        RGB berry speckle color (e.g. washed red); ``None`` = no berries.
    berry_density : float
        Per-texel berry probability over opaque texels.  Default 0.
    """
    palette = np.asarray(palette, dtype=np.uint8)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx, cy = (width - 1) * 0.5, (height - 1) * 0.5
    dist = np.sqrt(((xx - cx) / (width * 0.5)) ** 2
                   + ((yy - cy) / (height * 0.5)) ** 2)

    noise = pixel_noise(rng, (height, width), octaves=3,
                        base_freq=clump_freq)
    alpha = (dist < 1.0) & (noise > hole_thresh + dist * 0.3)

    light = 1.0 - yy / max(height - 1, 1)             # row 0 (top) brightest
    tier = np.clip(((noise * 0.6 + light * 0.45) * len(palette)),
                   0, len(palette) - 1).astype(np.intp)

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., :3][alpha] = palette[tier][alpha]
    rgba[..., 3][alpha] = 255

    if berry_color is not None and berry_density > 0.0:
        berries = (rng.random((height, width)) < berry_density) & alpha
        rgba[..., :3][berries] = np.asarray(berry_color, dtype=np.uint8)
    return rgba


def compose_atlas(layout: AtlasLayout, bark_rgba: np.ndarray,
                  leaf_rgba: np.ndarray) -> np.ndarray:
    """
    Assemble the species atlas: bark left half, leaves right half.

    Parameters
    ----------
    layout : AtlasLayout
        Defines the atlas size; both inputs must be ``layout.half_px``.
    bark_rgba / leaf_rgba : numpy.ndarray
        ``(H, W/2, 4) uint8`` from :func:`bark_texture` / :func:`leaf_texture`.

    Returns
    -------
    numpy.ndarray
        ``(layout.height, layout.width, 4) uint8``.
    """
    hw, hh = layout.half_px
    for name, arr in (("bark", bark_rgba), ("leaf", leaf_rgba)):
        if arr.shape != (hh, hw, 4) or arr.dtype != np.uint8:
            raise ValueError(
                f"compose_atlas: {name} must be ({hh}, {hw}, 4) uint8, "
                f"got {arr.shape} {arr.dtype}")
    atlas = np.zeros((layout.height, layout.width, 4), dtype=np.uint8)
    atlas[:, :hw] = bark_rgba
    atlas[:, hw:hw * 2] = leaf_rgba
    return atlas
