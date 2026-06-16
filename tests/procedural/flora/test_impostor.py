"""
tests/test_flora_impostor.py — characterization (golden-master) tests for
procedural/flora/impostor.py.

Pins CURRENT behaviour of rasterize_impostor and impostor_atlas.
Does NOT fix bugs — suspected anomalies are noted inline.
Headless, fixed seed, numpy assertions only, no per-element Python loops.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fire_engine.core.rng import for_domain, set_world_seed
from fire_engine.procedural.flora.impostor import impostor_atlas, rasterize_impostor
from fire_engine.procedural.flora.leaves import Leaves, leaves_at_tips
from fire_engine.procedural.flora.skeleton import SkeletonBuilder

# ---------------------------------------------------------------------------
# Shared palettes (mirrors GnarledOakDef palettes — identical arrays to the
# production species so tests exercise the same rasterizer paths).
# ---------------------------------------------------------------------------
BARK_PALETTE = np.array([(40, 31, 22), (58, 46, 33), (79, 63, 45)], dtype=np.uint8)
LEAF_PALETTE = np.array(
    [(30, 44, 26), (44, 62, 34), (60, 80, 42), (80, 98, 52), (104, 116, 64)], dtype=np.uint8
)

# ---------------------------------------------------------------------------
# Helpers — match the style of test_tree_skeleton.py
# ---------------------------------------------------------------------------


def _make_rng(tag: str = "imp") -> np.random.Generator:
    set_world_seed(42)
    return for_domain("test", tag)


def _grow_oak(rng: np.random.Generator):
    """Minimal but realistic oak skeleton + leaves (echoes test_tree_skeleton)."""
    sb = SkeletonBuilder(rng)
    trunk = sb.trunk(height_m=5.5, base_radius_m=0.28, segments=4, wobble_m=0.35)
    limbs = sb.branches(
        trunk,
        count=(3, 5),
        t_range=(0.35, 0.95),
        pitch_set=(math.radians(80), math.radians(95)),
        length_ratio=(0.5, 0.8),
        length_scale_by_height=(1.0, 0.45),
        radius_ratio=0.5,
        upturn_rad=math.radians(18),
        segments=2,
    )
    twigs = sb.branches(
        limbs,
        count=(1, 3),
        pitch_set=(math.radians(85),),
        length_ratio=(0.4, 0.6),
        radius_ratio=0.5,
        upturn_rad=math.radians(25),
    )
    sk = sb.skeleton()
    ids = np.concatenate([limbs, twigs])
    leaves = leaves_at_tips(sk, ids, rng, cell_m=0.26, rounds=3, density=0.6)
    return sk, leaves


def _default_cell(rng_tag: str = "imp") -> np.ndarray:
    """Render one oak impostor cell with defaults; returns the (H,W,4) uint8 array."""
    rng = _make_rng(rng_tag)
    sk, leaves = _grow_oak(rng)
    return rasterize_impostor(sk, leaves, BARK_PALETTE, LEAF_PALETTE, rng)


# ---------------------------------------------------------------------------
# Class 1: output shape and dtype
# ---------------------------------------------------------------------------


class TestOutputShapeAndDtype:
    def test_default_cell_shape(self):
        """Default cell_wh=(64,96) → output shape (96, 64, 4)."""
        cell = _default_cell()
        assert cell.shape == (96, 64, 4), f"Expected (96, 64, 4), got {cell.shape}"

    def test_custom_cell_shape_square(self):
        """Explicit square cell_wh=(32, 32) → output shape (32, 32, 4)."""
        rng = _make_rng("sq")
        sk, leaves = _grow_oak(rng)
        cell = rasterize_impostor(sk, leaves, BARK_PALETTE, LEAF_PALETTE, rng, cell_wh=(32, 32))
        assert cell.shape == (32, 32, 4)

    def test_custom_cell_wh_respected(self):
        """Non-default cell_wh=(48, 80) → output shape (80, 48, 4)."""
        rng = _make_rng("wh")
        sk, leaves = _grow_oak(rng)
        cell = rasterize_impostor(sk, leaves, BARK_PALETTE, LEAF_PALETTE, rng, cell_wh=(48, 80))
        assert cell.shape == (80, 48, 4)

    def test_dtype_is_uint8(self):
        """Output array must be uint8."""
        cell = _default_cell()
        assert cell.dtype == np.uint8

    def test_four_channels(self):
        """Last dimension is always 4 (RGBA)."""
        cell = _default_cell()
        assert cell.shape[2] == 4


# ---------------------------------------------------------------------------
# Class 2: binary alpha
# ---------------------------------------------------------------------------


class TestBinaryAlpha:
    def test_alpha_only_0_or_255(self):
        """Alpha channel must be strictly binary — no partial transparency."""
        cell = _default_cell()
        alpha = cell[..., 3]
        unique = np.unique(alpha)
        assert set(unique.tolist()).issubset({0, 255}), f"Non-binary alpha values found: {unique}"

    def test_alpha_has_opaque_pixels(self):
        """A realistic oak must produce some opaque pixels (trunk + canopy)."""
        cell = _default_cell()
        assert (cell[..., 3] == 255).any()

    def test_alpha_has_transparent_pixels(self):
        """Background pixels must remain transparent — not a solid rectangle."""
        cell = _default_cell()
        assert (cell[..., 3] == 0).any()


# ---------------------------------------------------------------------------
# Class 3: trunk rasterization (empty/leafless skeleton)
# ---------------------------------------------------------------------------


class TestLeaflessSkeletonTrunkPaint:
    """Dead/bare tree with Leaves.empty() must still paint the trunk."""

    def _bare_cell(self, cell_wh=(64, 96)):
        rng = _make_rng("bare")
        sb = SkeletonBuilder(rng)
        sb.trunk(height_m=4.0, base_radius_m=0.25, segments=3, wobble_m=0.2)
        sk = sb.skeleton()
        empty_leaves = Leaves.empty()
        rng2 = _make_rng("bare2")
        return rasterize_impostor(
            sk, empty_leaves, BARK_PALETTE, LEAF_PALETTE, rng2, cell_wh=cell_wh
        )

    def test_bare_skeleton_has_opaque_pixels(self):
        """Trunk capsules must produce at least one opaque pixel."""
        cell = self._bare_cell()
        assert (cell[..., 3] == 255).any(), (
            "Leafless skeleton produced zero opaque pixels — trunk not rasterized"
        )

    def test_bare_skeleton_alpha_still_binary(self):
        """Binary alpha holds even when leaves=Leaves.empty()."""
        cell = self._bare_cell()
        alpha = cell[..., 3]
        assert set(np.unique(alpha).tolist()).issubset({0, 255})

    def test_base_row_opaque_for_bare_trunk(self):
        """Trunk base must touch the bottom image row (trunk base at ground = row H-1)."""
        # PIN: if the trunk base doesn't touch the bottom row the impostor
        # will float above the terrain at the crossfade.
        cell = self._bare_cell(cell_wh=(64, 96))
        H = cell.shape[0]
        # Bottom row is the last row; the docstring says trunk base is on it.
        assert cell[H - 1, :, 3].any(), (
            "SUSPECTED BUG or design gap: trunk base not touching bottom row"
        )


# ---------------------------------------------------------------------------
# Class 4: determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_sk_leaves_rng_seed_identical(self):
        """Identical inputs → byte-identical output (numpy array_equal)."""
        rng_a = _make_rng("det")
        sk_a, leaves_a = _grow_oak(rng_a)
        cell_a = rasterize_impostor(sk_a, leaves_a, BARK_PALETTE, LEAF_PALETTE, rng_a)

        rng_b = _make_rng("det")  # same seed → same state
        sk_b, leaves_b = _grow_oak(rng_b)
        cell_b = rasterize_impostor(sk_b, leaves_b, BARK_PALETTE, LEAF_PALETTE, rng_b)

        assert np.array_equal(cell_a, cell_b), (
            "rasterize_impostor is not deterministic under identical seed+skeleton"
        )

    def test_different_seed_differs(self):
        """Different seeds should produce visually distinct impostors."""
        rng_a = _make_rng("diff_a")
        sk_a, leaves_a = _grow_oak(rng_a)
        cell_a = rasterize_impostor(sk_a, leaves_a, BARK_PALETTE, LEAF_PALETTE, rng_a)

        set_world_seed(99)
        rng_b = for_domain("test", "diff_b")
        sk_b, leaves_b = _grow_oak(rng_b)
        cell_b = rasterize_impostor(sk_b, leaves_b, BARK_PALETTE, LEAF_PALETTE, rng_b)

        # Skeletons grown from different seeds should differ.
        assert not np.array_equal(cell_a, cell_b)


# ---------------------------------------------------------------------------
# Class 5: px_per_m scaling
# ---------------------------------------------------------------------------


class TestPxPerMScaling:
    """px_per_m: None = self-fit; explicit value scales the drawn footprint."""

    def _render(self, px_per_m, cell_wh=(64, 96)):
        rng = _make_rng("scale")
        sk, leaves = _grow_oak(rng)
        return rasterize_impostor(
            sk, leaves, BARK_PALETTE, LEAF_PALETTE, rng, cell_wh=cell_wh, px_per_m=px_per_m
        )

    def test_none_fits_tree_and_has_content(self):
        """px_per_m=None (self-fit) still produces opaque pixels."""
        cell = self._render(None)
        assert (cell[..., 3] == 255).any()

    def test_larger_px_per_m_more_opaque_pixels(self):
        """Larger px_per_m → larger drawn footprint → more opaque pixels.

        PIN: a 3x scale (px_per_m * 3) relative to a tight self-fit scale
        should have strictly more opaque pixels because the tree fills more
        of the canvas.  We derive a reasonable scale from the self-fit render
        to stay within cell bounds.
        """
        rng_ref = _make_rng("scale_ref")
        _sk, _leaves = _grow_oak(rng_ref)
        # A small but non-trivial explicit scale that keeps the trunk in view.
        small_scale = 4.0
        large_scale = 10.0

        rng_s = _make_rng("scale_s")
        sk_s, leaves_s = _grow_oak(rng_s)
        cell_small = rasterize_impostor(
            sk_s,
            leaves_s,
            BARK_PALETTE,
            LEAF_PALETTE,
            rng_s,
            cell_wh=(64, 96),
            px_per_m=small_scale,
        )

        rng_l = _make_rng("scale_s")  # same rng tag → same skeleton
        sk_l, leaves_l = _grow_oak(rng_l)
        cell_large = rasterize_impostor(
            sk_l,
            leaves_l,
            BARK_PALETTE,
            LEAF_PALETTE,
            rng_l,
            cell_wh=(64, 96),
            px_per_m=large_scale,
        )

        opaque_small = int((cell_small[..., 3] == 255).sum())
        opaque_large = int((cell_large[..., 3] == 255).sum())
        assert opaque_large > opaque_small, (
            f"Expected larger px_per_m to produce more opaque pixels; "
            f"small={opaque_small}, large={opaque_large}"
        )

    def test_explicit_px_per_m_dtype_invariant(self):
        """Explicit px_per_m must not break the binary-alpha invariant."""
        cell = self._render(px_per_m=8.0)
        alpha = cell[..., 3]
        assert set(np.unique(alpha).tolist()).issubset({0, 255})


# ---------------------------------------------------------------------------
# Class 6: hole_thresh
# ---------------------------------------------------------------------------


class TestHoleThresh:
    """Higher hole_thresh → noisier canopy holes → fewer opaque pixels in canopy."""

    def _render_thresh(self, thresh, rng_tag="hole"):
        rng = _make_rng(rng_tag)
        sk, leaves = _grow_oak(rng)
        return rasterize_impostor(sk, leaves, BARK_PALETTE, LEAF_PALETTE, rng, hole_thresh=thresh)

    def test_low_thresh_more_opaque_than_high(self):
        """hole_thresh=0.05 (dense) should have >= opaque px than thresh=0.8 (ragged).

        PIN direction: higher hole_thresh cuts more canopy → fewer opaque px.
        NOTE: the production code applies ``noise > hole_thresh * 0.8``, so the
        effective cut scales linearly with hole_thresh.
        """
        # Use same rng tag so the skeleton is identical; only the thresh differs.
        cell_low = self._render_thresh(0.05, rng_tag="hole")
        cell_high = self._render_thresh(0.80, rng_tag="hole")

        opaque_low = int((cell_low[..., 3] == 255).sum())
        opaque_high = int((cell_high[..., 3] == 255).sum())

        # NOTE: both renders consume the same initial rng state for skeleton
        # and leaves, but the hole noise is drawn AFTER leaves are computed so
        # both passes will consume the same leaf-generation draws; the pixel_noise
        # call is seeded by the same rng object, which means the draws are
        # sequentially identical only when the rng state matches at that point.
        # We therefore assert the direction (low thresh → denser canopy → >=),
        # not exact pixel counts.
        assert opaque_low >= opaque_high, (
            f"Expected low hole_thresh to produce >= opaque pixels; "
            f"low={opaque_low}, high={opaque_high}. "
            "SUSPECTED BUG if equal but thresh range is 0.05–0.80."
        )

    def test_extreme_high_thresh_still_has_trunk(self):
        """Even with hole_thresh=1.0 (everything cut) trunk bark must survive."""
        cell = self._render_thresh(1.0)
        # Trunk is drawn unconditionally before the leaf noise cut — it must
        # still be present.
        assert (cell[..., 3] == 255).any(), (
            "hole_thresh=1.0 eliminated ALL pixels including trunk — "
            "trunk paint should be unconditional"
        )

    def test_zero_thresh_has_dense_canopy(self):
        """hole_thresh=0.0 allows all canopy pixels through (noise > 0)."""
        cell = self._render_thresh(0.0)
        assert (cell[..., 3] == 255).any()


# ---------------------------------------------------------------------------
# Class 7: impostor_atlas
# ---------------------------------------------------------------------------


class TestImpostorAtlas:
    """impostor_atlas: concatenates cells left→right into a single strip."""

    def _make_cell(self, seed_tag: str, cell_wh=(64, 96)) -> np.ndarray:
        rng = _make_rng(seed_tag)
        sk, leaves = _grow_oak(rng)
        return rasterize_impostor(sk, leaves, BARK_PALETTE, LEAF_PALETTE, rng, cell_wh=cell_wh)

    # -- shape contracts -------------------------------------------------------

    def test_single_cell_atlas_shape(self):
        """Atlas of 1 cell has the same shape as the cell itself."""
        cell = self._make_cell("a1")
        atlas = impostor_atlas([cell])
        assert atlas.shape == cell.shape

    def test_two_cell_atlas_width(self):
        """Atlas of 2 equal cells: width == 2 × cell width."""
        c1 = self._make_cell("b1")
        c2 = self._make_cell("b2")
        atlas = impostor_atlas([c1, c2])
        assert atlas.shape[1] == c1.shape[1] + c2.shape[1]
        assert atlas.shape[0] == c1.shape[0]

    def test_n_cell_atlas_width(self):
        """Atlas of N cells: total width == sum of individual widths."""
        cells = [self._make_cell(f"n{k}") for k in range(4)]
        atlas = impostor_atlas(cells)
        expected_w = sum(c.shape[1] for c in cells)
        assert atlas.shape[1] == expected_w
        assert atlas.shape[0] == cells[0].shape[0]

    def test_atlas_height_equals_cell_height(self):
        """All cells same height → atlas height == that height."""
        cells = [self._make_cell(f"h{k}") for k in range(3)]
        atlas = impostor_atlas(cells)
        assert atlas.shape[0] == cells[0].shape[0]

    def test_atlas_dtype_uint8(self):
        """Atlas output is uint8."""
        cells = [self._make_cell(f"dt{k}") for k in range(2)]
        atlas = impostor_atlas(cells)
        assert atlas.dtype == np.uint8

    # -- pixel placement -------------------------------------------------------

    def test_cell_0_pixels_at_correct_offset(self):
        """Cell 0 pixels appear in atlas columns [0, cell_w)."""
        c0 = self._make_cell("p0")
        c1 = self._make_cell("p1")
        atlas = impostor_atlas([c0, c1])
        W = c0.shape[1]
        assert np.array_equal(atlas[:, :W, :], c0)

    def test_cell_1_pixels_at_correct_offset(self):
        """Cell 1 pixels appear in atlas columns [cell_w, 2*cell_w)."""
        c0 = self._make_cell("q0")
        c1 = self._make_cell("q1")
        atlas = impostor_atlas([c0, c1])
        W0 = c0.shape[1]
        W1 = c1.shape[1]
        assert np.array_equal(atlas[:, W0 : W0 + W1, :], c1)

    def test_last_cell_pixels_at_correct_offset(self):
        """Last cell (index 3) pixels appear in the final slice of the atlas."""
        cells = [self._make_cell(f"l{k}") for k in range(4)]
        atlas = impostor_atlas(cells)
        offset = sum(c.shape[1] for c in cells[:-1])
        last_w = cells[-1].shape[1]
        assert np.array_equal(atlas[:, offset : offset + last_w, :], cells[-1])

    # -- error handling --------------------------------------------------------

    def test_empty_cell_list_raises(self):
        """impostor_atlas([]) must raise ValueError (no cells)."""
        with pytest.raises(ValueError, match="no cells"):
            impostor_atlas([])

    # -- alpha contract --------------------------------------------------------

    def test_atlas_alpha_binary(self):
        """Binary-alpha invariant holds across the assembled atlas."""
        cells = [self._make_cell(f"ab{k}") for k in range(3)]
        atlas = impostor_atlas(cells)
        alpha = atlas[..., 3]
        assert set(np.unique(alpha).tolist()).issubset({0, 255})

    # -- determinism -----------------------------------------------------------

    def test_atlas_determinism(self):
        """Same cells in same order → byte-identical atlas."""

        def _build():
            cells = [self._make_cell(f"d{k}") for k in range(3)]
            return impostor_atlas(cells)

        a1 = _build()
        a2 = _build()
        assert np.array_equal(a1, a2)
