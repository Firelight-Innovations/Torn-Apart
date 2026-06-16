"""
procedural/flora/atlas.py — per-species pixel-art texture atlas helpers.

One tree species = ONE small texture atlas so each variant mesh is a single
draw: the left half is tileable posterised **bark** (alpha 255), the right
half a binary-alpha **single leaf** every individual leaf card cuts out
(hundreds per canopy, one texture).  The mesher's ``uv_rect`` arguments and
:class:`AtlasLayout` are the shared contract.

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

__all__ = ["AtlasLayout", "bark_texture", "compose_atlas", "leaf_texture"]


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
    small = pixel_noise(rng, (rows, width), octaves=2, base_freq=striation_freq)
    grain = np.repeat(small, streak_px, axis=0)[:height]
    if grain.shape[0] < height:  # streak_px remainder
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
    ONE pixel-art leaf — ``(H, W, 4) uint8`` binary-alpha cutout.

    A single teardrop leaf (stem at the bottom of the tile, tip at the
    top), drawn for the individual-leaf canopy: the mesher puts this on
    every leaf card, hundreds per tree.  The silhouette is two mirrored
    arcs widest below the middle, its edge nibbled by crisp pixel noise
    (*hole_thresh* — dry species look chewed), shaded with the posterised
    ramp split along a darker **midrib**: the right side sits one tier
    lighter than the left, with a top-lit ramp tip-to-stem.  Optional
    berry speckles for fruiting species ride near the leaf base.

    Parameters
    ----------
    rng : numpy.random.Generator
        Deterministic generator.
    width, height : int
        Output texel size (one half-atlas, typically 32×64).
    palette : numpy.ndarray
        ``uint8 (T, 3)`` RGB foliage ramp, dark first.
    hole_thresh : float
        Edge-raggedness cut; higher = scragglier/dying.  Default 0.18.
    clump_freq : int
        Noise base frequency for the edge nibble + interior mottle.
    berry_color : tuple[int, int, int] | None
        RGB berry speckle color (e.g. washed red); ``None`` = no berries.
    berry_density : float
        Per-texel berry probability over the lower leaf half.  Default 0.
    """
    palette = np.asarray(palette, dtype=np.uint8)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)

    # Leaf axis: stem at the BOTTOM row (v=0 after the upload flip is the
    # card's stem end), tip at the top.  t = 0 stem … 1 tip.
    t = 1.0 - yy / max(height - 1, 1)
    cx = (width - 1) * 0.5

    # Teardrop half-width profile: widest ~35 % up from the stem, tapering
    # to a point at the tip and a narrow stem foot.  Slight per-leaf
    # asymmetry so the species' leaf isn't perfectly mirrored.
    wmax = width * float(rng.uniform(0.36, 0.46))
    bulge = float(rng.uniform(0.30, 0.42))  # widest point (t)
    prof = np.where(
        t < bulge,
        0.18 + 0.82 * (t / max(bulge, 1e-3)) ** 0.7,
        np.clip(1.0 - (t - bulge) / max(1.0 - bulge, 1e-3), 0.0, 1.0) ** 1.15,
    )
    half_w = wmax * prof  # (H, W) via t
    skew = (t - 0.5) * width * float(rng.uniform(-0.08, 0.08))
    dx = xx - (cx + skew)

    noise = pixel_noise(rng, (height, width), octaves=3, base_freq=clump_freq)
    # Edge nibble: noise shrinks the silhouette locally; hole_thresh sets
    # how deep the bites go.
    edge = 1.0 - np.clip(noise - (1.0 - hole_thresh * 1.6), 0.0, 1.0) * 2.2
    alpha = np.abs(dx) <= half_w * np.clip(edge, 0.25, 1.0)

    # Stem foot: a 1–2 px column at the very bottom so the card reads
    # attached when it tilts.
    stem_rows = max(2, height // 12)
    stem = (yy >= height - stem_rows) & (np.abs(xx - cx) <= max(width / 24.0, 0.6))
    alpha = alpha | stem

    # Shading: top-lit tip→stem ramp + one-tier midrib split (right of the
    # midrib lighter), plus noise mottle.
    side = (dx > 0).astype(np.float32)
    tier_f = (t * 0.5 + side * 0.22 + noise * 0.38) * len(palette)
    tier = np.clip(tier_f, 0, len(palette) - 1).astype(np.intp)

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., :3][alpha] = palette[tier][alpha]
    rgba[..., 3][alpha] = 255

    # Midrib: the centreline one tier darker, stem to near-tip.
    rib = (np.abs(dx) <= max(width / 28.0, 0.55)) & alpha & (t < 0.92)
    rib_tier = np.maximum(tier - 1, 0)
    rgba[..., :3][rib] = palette[rib_tier][rib]

    if berry_color is not None and berry_density > 0.0:
        # Berries cluster near the leaf base (they hang at the stem).
        berries = (rng.random((height, width)) < berry_density * 3.0) & alpha & (t < 0.45)
        rgba[..., :3][berries] = np.asarray(berry_color, dtype=np.uint8)
    return rgba


def compose_atlas(layout: AtlasLayout, bark_rgba: np.ndarray, leaf_rgba: np.ndarray) -> np.ndarray:
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
                f"compose_atlas: {name} must be ({hh}, {hw}, 4) uint8, got {arr.shape} {arr.dtype}"
            )
    atlas = np.zeros((layout.height, layout.width, 4), dtype=np.uint8)
    atlas[:, :hw] = bark_rgba
    atlas[:, hw : hw * 2] = leaf_rgba
    return atlas
