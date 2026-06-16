"""
tests/test_flora_atlas.py — Characterisation (golden-master) tests for
fire_engine/procedural/flora/atlas.py.

Covers: AtlasLayout, bark_texture, leaf_texture, compose_atlas.
Fixed-seed, headless-only, numpy assertions, no per-element loops.
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.core.rng import for_domain, set_world_seed
from fire_engine.procedural.flora.atlas import (
    AtlasLayout,
    bark_texture,
    compose_atlas,
    leaf_texture,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

# A minimal 4-colour bark palette (dark → light).
BARK_PALETTE = np.array(
    [
        [40, 25, 10],
        [80, 52, 22],
        [120, 78, 34],
        [160, 108, 52],
    ],
    dtype=np.uint8,
)

# A minimal 3-colour leaf palette.
LEAF_PALETTE = np.array(
    [
        [20, 60, 15],
        [40, 110, 28],
        [70, 155, 45],
    ],
    dtype=np.uint8,
)

BERRY_COLOR = (180, 30, 30)


def _rng(seed: int = 42) -> np.random.Generator:
    """Return a fresh, fixed-seed generator for tests."""
    set_world_seed(seed)
    return for_domain("test", "atlas", seed)


# ---------------------------------------------------------------------------
# AtlasLayout
# ---------------------------------------------------------------------------


class TestAtlasLayout:
    def test_defaults(self):
        layout = AtlasLayout()
        assert layout.width == 64
        assert layout.height == 64
        assert layout.bark_rect == (0.0, 0.0, 0.5, 1.0)
        assert layout.leaf_rect == (0.5, 0.0, 1.0, 1.0)

    def test_half_px_default(self):
        layout = AtlasLayout()
        # half_px = (width // 2, height)
        assert layout.half_px == (32, 64)

    def test_half_px_custom(self):
        layout = AtlasLayout(width=128, height=96)
        assert layout.half_px == (64, 96)

    def test_rects_within_unit_square(self):
        layout = AtlasLayout()
        for r in (layout.bark_rect, layout.leaf_rect):
            u0, v0, u1, v1 = r
            assert 0.0 <= u0 < u1 <= 1.0
            assert 0.0 <= v0 < v1 <= 1.0

    def test_bark_and_leaf_rects_cover_atlas(self):
        layout = AtlasLayout()
        # bark ends where leaf begins
        assert layout.bark_rect[2] == layout.leaf_rect[0]


# ---------------------------------------------------------------------------
# bark_texture
# ---------------------------------------------------------------------------


class TestBarkTexture:
    W, H = 32, 64

    def _bark(self, **kw) -> np.ndarray:
        return bark_texture(_rng(), self.W, self.H, BARK_PALETTE, **kw)

    def test_output_shape(self):
        arr = self._bark()
        assert arr.shape == (self.H, self.W, 4)

    def test_output_dtype(self):
        arr = self._bark()
        assert arr.dtype == np.uint8

    def test_alpha_all_255(self):
        arr = self._bark()
        assert (arr[..., 3] == 255).all(), "bark alpha must be fully opaque"

    def test_colors_from_palette(self):
        """Every RGB triplet in the output must appear in the palette."""
        arr = self._bark()
        rgb_flat = arr[..., :3].reshape(-1, 3)
        unique_rgb = np.unique(rgb_flat, axis=0)
        palette_set = set(map(tuple, BARK_PALETTE.tolist()))
        for row in unique_rgb.tolist():
            assert tuple(row) in palette_set, (
                f"RGB {row} is not in the supplied palette — posterisation broke"
            )

    def test_determinism(self):
        arr1 = self._bark()
        arr2 = self._bark()
        assert np.array_equal(arr1, arr2)

    def test_shade_side_true_darkens_left(self):
        """shade_side=True must make the left half strictly darker (on average)."""
        arr_shaded = bark_texture(_rng(1), self.W, self.H, BARK_PALETTE, shade_side=True)
        arr_unshaded = bark_texture(_rng(1), self.W, self.H, BARK_PALETTE, shade_side=False)
        half = self.W // 2
        # The shaded version's left columns should have a lower mean luminance.
        lum_shaded = arr_shaded[:, :half, :3].mean()
        lum_unshaded = arr_unshaded[:, :half, :3].mean()
        assert lum_shaded < lum_unshaded, (
            "shade_side=True should darken the left half relative to shade_side=False"
        )

    def test_shade_side_false_leaves_right_unchanged(self):
        """The right half should be identical regardless of shade_side."""
        arr_t = bark_texture(_rng(7), self.W, self.H, BARK_PALETTE, shade_side=True)
        arr_f = bark_texture(_rng(7), self.W, self.H, BARK_PALETTE, shade_side=False)
        half = self.W // 2
        # Right half is not touched by shade_side logic.
        assert np.array_equal(arr_t[:, half:], arr_f[:, half:]), (
            "shade_side only affects the left half — right half must be unchanged"
        )

    def test_striation_freq_changes_pattern(self):
        """Different striation_freq must produce different pixel patterns."""
        a = bark_texture(_rng(3), self.W, self.H, BARK_PALETTE, striation_freq=3)
        b = bark_texture(_rng(3), self.W, self.H, BARK_PALETTE, striation_freq=12)
        assert not np.array_equal(a, b), (
            "striation_freq changes the noise frequency — outputs must differ"
        )

    def test_streak_px_changes_pattern(self):
        """Different streak_px must produce different pixel patterns."""
        a = bark_texture(_rng(5), self.W, self.H, BARK_PALETTE, streak_px=2)
        b = bark_texture(_rng(5), self.W, self.H, BARK_PALETTE, streak_px=12)
        assert not np.array_equal(a, b), "streak_px changes vertical stretch — outputs must differ"


# ---------------------------------------------------------------------------
# leaf_texture
# ---------------------------------------------------------------------------


class TestLeafTexture:
    W, H = 32, 64

    def _leaf(self, **kw) -> np.ndarray:
        return leaf_texture(_rng(), self.W, self.H, LEAF_PALETTE, **kw)

    def test_output_shape(self):
        arr = self._leaf()
        assert arr.shape == (self.H, self.W, 4)

    def test_output_dtype(self):
        arr = self._leaf()
        assert arr.dtype == np.uint8

    def test_alpha_binary(self):
        """Alpha channel must be exactly 0 or 255 — no partial transparency."""
        arr = self._leaf()
        a = arr[..., 3]
        assert ((a == 0) | (a == 255)).all(), "leaf alpha must be binary (0 or 255 only)"

    def test_some_opaque_pixels(self):
        """The leaf silhouette must cover at least a few pixels."""
        arr = self._leaf()
        assert (arr[..., 3] == 255).any(), "leaf texture has no opaque pixels"

    def test_some_transparent_pixels(self):
        """There must be transparent pixels (the leaf is not a solid block)."""
        arr = self._leaf()
        assert (arr[..., 3] == 0).any(), "leaf texture has no transparent pixels"

    def test_hole_thresh_direction(self):
        """Higher hole_thresh → more holes → fewer opaque pixels."""
        low = leaf_texture(_rng(10), self.W, self.H, LEAF_PALETTE, hole_thresh=0.05)
        high = leaf_texture(_rng(10), self.W, self.H, LEAF_PALETTE, hole_thresh=0.60)
        opaque_low = int((low[..., 3] == 255).sum())
        opaque_high = int((high[..., 3] == 255).sum())
        assert opaque_high < opaque_low, (
            f"Higher hole_thresh should reduce opaque px: low={opaque_low}, high={opaque_high}"
        )

    def test_berry_density_adds_color(self):
        """berry_density > 0 with a berry_color must produce berry-colored pixels."""
        no_berry = leaf_texture(
            _rng(20), self.W, self.H, LEAF_PALETTE, berry_density=0.0, berry_color=BERRY_COLOR
        )
        with_berry = leaf_texture(
            _rng(20), self.W, self.H, LEAF_PALETTE, berry_density=0.5, berry_color=BERRY_COLOR
        )
        # Pin: any pixel matching the berry RGB in the berry version that
        # doesn't exist in the no-berry version.
        br, bg, bb = BERRY_COLOR
        berry_mask = (
            (with_berry[..., 0] == br)
            & (with_berry[..., 1] == bg)
            & (with_berry[..., 2] == bb)
            & (with_berry[..., 3] == 255)
        )
        no_berry_mask = (
            (no_berry[..., 0] == br)
            & (no_berry[..., 1] == bg)
            & (no_berry[..., 2] == bb)
            & (no_berry[..., 3] == 255)
        )
        assert int(berry_mask.sum()) > int(no_berry_mask.sum()), (
            "berry_density>0 with berry_color must add berry-colored pixels"
        )

    def test_determinism(self):
        arr1 = self._leaf()
        arr2 = self._leaf()
        assert np.array_equal(arr1, arr2)

    def test_opaque_rgb_from_palette(self):
        """All opaque non-berry pixels must use a palette colour."""
        arr = self._leaf()
        opaque = arr[..., 3] == 255
        rgb_opaque = arr[opaque][..., :3]
        palette_set = set(map(tuple, LEAF_PALETTE.tolist()))
        unique_opaque = np.unique(rgb_opaque, axis=0)
        for row in unique_opaque.tolist():
            assert tuple(row) in palette_set, f"Leaf RGB {row} not in supplied palette"

    def test_zero_berry_density_no_speckles(self):
        """berry_density=0 must not add any berry-colored pixels."""
        arr = leaf_texture(
            _rng(30), self.W, self.H, LEAF_PALETTE, berry_density=0.0, berry_color=BERRY_COLOR
        )
        br, bg, bb = BERRY_COLOR
        # berry color shouldn't appear unless it also happens to be a palette color
        # (BERRY_COLOR is NOT in LEAF_PALETTE, so we can check directly)
        berry_present = (arr[..., 0] == br) & (arr[..., 1] == bg) & (arr[..., 2] == bb)
        assert not berry_present.any(), "berry_density=0 must not paint berry speckles"


# ---------------------------------------------------------------------------
# compose_atlas
# ---------------------------------------------------------------------------


class TestComposeAtlas:
    def _make_parts(self, layout: AtlasLayout, seed: int = 0):
        hw, hh = layout.half_px
        rng_b = _rng(seed)
        rng_l = _rng(seed + 1)
        bark = bark_texture(rng_b, hw, hh, BARK_PALETTE)
        leaf = leaf_texture(rng_l, hw, hh, LEAF_PALETTE)
        return bark, leaf

    def test_output_shape(self):
        layout = AtlasLayout()
        bark, leaf = self._make_parts(layout)
        atlas = compose_atlas(layout, bark, leaf)
        assert atlas.shape == (layout.height, layout.width, 4)

    def test_output_dtype(self):
        layout = AtlasLayout()
        bark, leaf = self._make_parts(layout)
        atlas = compose_atlas(layout, bark, leaf)
        assert atlas.dtype == np.uint8

    def test_left_bark_region_alpha_255(self):
        """Left (bark) half of the atlas must be fully opaque."""
        layout = AtlasLayout()
        bark, leaf = self._make_parts(layout)
        atlas = compose_atlas(layout, bark, leaf)
        hw = layout.half_px[0]
        assert (atlas[:, :hw, 3] == 255).all(), (
            "bark region (left half) must have alpha=255 everywhere"
        )

    def test_right_leaf_region_binary_alpha(self):
        """Right (leaf) half of the atlas must have binary alpha."""
        layout = AtlasLayout()
        bark, leaf = self._make_parts(layout)
        atlas = compose_atlas(layout, bark, leaf)
        hw = layout.half_px[0]
        leaf_a = atlas[:, hw:, 3]
        assert ((leaf_a == 0) | (leaf_a == 255)).all(), (
            "leaf region (right half) must have binary alpha only"
        )

    def test_bark_region_matches_source(self):
        """Left half of atlas must exactly match the bark texture input."""
        layout = AtlasLayout()
        bark, leaf = self._make_parts(layout)
        atlas = compose_atlas(layout, bark, leaf)
        hw = layout.half_px[0]
        assert np.array_equal(atlas[:, :hw], bark), (
            "bark region in atlas must equal the bark_texture input exactly"
        )

    def test_leaf_region_matches_source(self):
        """Right half of atlas must exactly match the leaf texture input."""
        layout = AtlasLayout()
        bark, leaf = self._make_parts(layout)
        atlas = compose_atlas(layout, bark, leaf)
        hw = layout.half_px[0]
        assert np.array_equal(atlas[:, hw : hw * 2], leaf), (
            "leaf region in atlas must equal the leaf_texture input exactly"
        )

    def test_wrong_shape_raises(self):
        """compose_atlas must raise ValueError when inputs have wrong shape."""
        layout = AtlasLayout()
        hw, hh = layout.half_px
        bark = np.zeros((hh, hw, 4), dtype=np.uint8)
        bad_leaf = np.zeros((hh + 1, hw, 4), dtype=np.uint8)  # wrong height
        with pytest.raises(ValueError):
            compose_atlas(layout, bark, bad_leaf)

    def test_wrong_dtype_raises(self):
        """compose_atlas must raise ValueError when dtype is not uint8."""
        layout = AtlasLayout()
        hw, hh = layout.half_px
        bark = np.zeros((hh, hw, 4), dtype=np.uint8)
        bad_leaf = np.zeros((hh, hw, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            compose_atlas(layout, bad_leaf, bark)

    def test_custom_layout_shape(self):
        """compose_atlas respects a non-default AtlasLayout size."""
        layout = AtlasLayout(width=128, height=96)
        hw, hh = layout.half_px  # (64, 96)
        rng_b = _rng(99)
        rng_l = _rng(100)
        bark = bark_texture(rng_b, hw, hh, BARK_PALETTE)
        leaf = leaf_texture(rng_l, hw, hh, LEAF_PALETTE)
        atlas = compose_atlas(layout, bark, leaf)
        assert atlas.shape == (96, 128, 4)
        assert atlas.dtype == np.uint8


# ---------------------------------------------------------------------------
# Cross-cutting determinism
# ---------------------------------------------------------------------------


class TestDeterminismAcrossBoard:
    """Two identical seeds must produce byte-identical outputs for all helpers."""

    def test_bark_determinism_twice(self):
        a = bark_texture(_rng(77), 32, 64, BARK_PALETTE)
        b = bark_texture(_rng(77), 32, 64, BARK_PALETTE)
        assert np.array_equal(a, b)

    def test_leaf_determinism_twice(self):
        a = leaf_texture(_rng(88), 32, 64, LEAF_PALETTE)
        b = leaf_texture(_rng(88), 32, 64, LEAF_PALETTE)
        assert np.array_equal(a, b)

    def test_compose_atlas_determinism_twice(self):
        layout = AtlasLayout()
        hw, hh = layout.half_px

        def _make(seed):
            bark = bark_texture(_rng(seed), hw, hh, BARK_PALETTE)
            leaf = leaf_texture(_rng(seed + 1), hw, hh, LEAF_PALETTE)
            return compose_atlas(layout, bark, leaf)

        assert np.array_equal(_make(55), _make(55))

    def test_different_seeds_differ_bark(self):
        a = bark_texture(_rng(1), 32, 64, BARK_PALETTE)
        b = bark_texture(_rng(2), 32, 64, BARK_PALETTE)
        assert not np.array_equal(a, b)

    def test_different_seeds_differ_leaf(self):
        a = leaf_texture(_rng(1), 32, 64, LEAF_PALETTE)
        b = leaf_texture(_rng(2), 32, 64, LEAF_PALETTE)
        assert not np.array_equal(a, b)
