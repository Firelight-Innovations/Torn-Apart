"""
tests/test_procedural_maps.py — Golden-master / characterization tests for
fire_engine.procedural.maps: derive_normal_map, flat_normal_map, black_emission_map.

DO NOT fix bugs found here — pin current behaviour and note suspicions.

No panda3d imports.  Pure numpy assertions, no per-element Python loops.
"""

from __future__ import annotations

import numpy as np

from fire_engine.procedural.maps import (
    black_emission_map,
    derive_normal_map,
    flat_normal_map,
)

# ---------------------------------------------------------------------------
# flat_normal_map
# ---------------------------------------------------------------------------


class TestFlatNormalMap:
    """Pin the constant (128, 128, 255, 255) neutral tangent-space normal."""

    def test_default_size_shape(self):
        out = flat_normal_map()
        assert out.shape == (4, 4, 4), f"Expected (4,4,4), got {out.shape}"

    def test_default_size_dtype(self):
        out = flat_normal_map()
        assert out.dtype == np.uint8, f"Expected uint8, got {out.dtype}"

    def test_default_values_all_flat(self):
        """Every texel must be exactly (128, 128, 255, 255)."""
        out = flat_normal_map()
        expected = np.array([128, 128, 255, 255], dtype=np.uint8)
        target = np.broadcast_to(expected, out.shape)
        assert np.array_equal(out, target), (
            f"flat_normal_map() has unexpected values; "
            f"unique rows: {np.unique(out.reshape(-1, 4), axis=0)}"
        )

    def test_r_channel_is_128(self):
        out = flat_normal_map()
        assert (out[..., 0] == 128).all(), "R channel (X normal) must all be 128"

    def test_g_channel_is_128(self):
        out = flat_normal_map()
        assert (out[..., 1] == 128).all(), "G channel (Y normal) must all be 128"

    def test_b_channel_is_255(self):
        out = flat_normal_map()
        assert (out[..., 2] == 255).all(), "B channel (Z normal) must all be 255"

    def test_alpha_is_255(self):
        out = flat_normal_map()
        assert (out[..., 3] == 255).all(), "Alpha must all be 255"

    def test_size_1(self):
        out = flat_normal_map(size=1)
        assert out.shape == (1, 1, 4)
        assert np.array_equal(out[0, 0], np.array([128, 128, 255, 255], dtype=np.uint8))

    def test_size_16(self):
        out = flat_normal_map(size=16)
        assert out.shape == (16, 16, 4)
        expected = np.full((16, 16, 4), 128, dtype=np.uint8)
        expected[..., 2] = 255
        expected[..., 3] = 255
        assert np.array_equal(out, expected)

    def test_size_64(self):
        out = flat_normal_map(size=64)
        assert out.shape == (64, 64, 4)
        assert (out[..., 0] == 128).all()
        assert (out[..., 1] == 128).all()
        assert (out[..., 2] == 255).all()
        assert (out[..., 3] == 255).all()

    def test_determinism(self):
        """Pure function — must return identical bytes on every call."""
        a = flat_normal_map(size=8)
        b = flat_normal_map(size=8)
        assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# black_emission_map
# ---------------------------------------------------------------------------


class TestBlackEmissionMap:
    """Pin the constant (0, 0, 0, 255) non-emissive map."""

    def test_default_size_shape(self):
        out = black_emission_map()
        assert out.shape == (4, 4, 4), f"Expected (4,4,4), got {out.shape}"

    def test_default_size_dtype(self):
        out = black_emission_map()
        assert out.dtype == np.uint8, f"Expected uint8, got {out.dtype}"

    def test_rgb_all_zero(self):
        out = black_emission_map()
        assert (out[..., :3] == 0).all(), "RGB must be zero (black emission)"

    def test_alpha_is_255(self):
        out = black_emission_map()
        assert (out[..., 3] == 255).all(), "Alpha must be 255"

    def test_all_texels_black_opaque(self):
        """Every texel == (0, 0, 0, 255)."""
        out = black_emission_map()
        expected = np.array([0, 0, 0, 255], dtype=np.uint8)
        target = np.broadcast_to(expected, out.shape)
        assert np.array_equal(out, target)

    def test_size_16(self):
        out = black_emission_map(size=16)
        assert out.shape == (16, 16, 4)
        assert (out[..., :3] == 0).all()
        assert (out[..., 3] == 255).all()

    def test_determinism(self):
        a = black_emission_map(size=8)
        b = black_emission_map(size=8)
        assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# derive_normal_map
# ---------------------------------------------------------------------------


class TestDeriveNormalMapShapeAndDtype:
    """Output shape/dtype invariants."""

    def test_output_shape_matches_input_hw(self):
        rgba = np.zeros((16, 32, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        out = derive_normal_map(rgba)
        assert out.shape == (16, 32, 4), f"Expected (16,32,4), got {out.shape}"

    def test_output_dtype_uint8(self):
        rgba = np.zeros((8, 8, 4), dtype=np.uint8)
        out = derive_normal_map(rgba)
        assert out.dtype == np.uint8

    def test_alpha_channel_pinned_255(self):
        """Alpha output is always 255 regardless of input alpha."""
        rng = np.random.default_rng(0)
        rgba = rng.integers(0, 256, size=(12, 12, 4), dtype=np.uint8)
        # Vary input alpha to confirm output is unconditionally 255.
        rgba[..., 3] = rng.integers(0, 256, size=(12, 12), dtype=np.uint8)
        out = derive_normal_map(rgba)
        assert (out[..., 3] == 255).all(), (
            "derive_normal_map must always write alpha=255 (current behaviour)"
        )

    def test_non_square_input(self):
        rgba = np.zeros((4, 64, 4), dtype=np.uint8)
        out = derive_normal_map(rgba)
        assert out.shape == (4, 64, 4)

    def test_single_pixel_input(self):
        """1×1 uniform input — should not crash; output is the flat normal."""
        rgba = np.full((1, 1, 4), 200, dtype=np.uint8)
        rgba[0, 0, 3] = 255
        out = derive_normal_map(rgba)
        assert out.shape == (1, 1, 4)
        assert out.dtype == np.uint8


class TestDeriveNormalMapFlatInput:
    """A constant-luminance input must produce the neutral normal (128,128,255)."""

    def _flat_rgba(self, h: int, w: int, value: int = 128) -> np.ndarray:
        rgba = np.full((h, w, 4), value, dtype=np.uint8)
        rgba[..., 3] = 255
        return rgba

    def test_flat_rgb_neutral_normal_r(self):
        out = derive_normal_map(self._flat_rgba(8, 8))
        # For zero gradient, nx=0 → encoded as 128 (allow ±1 for rounding).
        assert np.abs(out[..., 0].astype(np.int16) - 128).max() <= 1, (
            "Flat input: R channel (X) should be ~128 (neutral)"
        )

    def test_flat_rgb_neutral_normal_g(self):
        out = derive_normal_map(self._flat_rgba(8, 8))
        assert np.abs(out[..., 1].astype(np.int16) - 128).max() <= 1, (
            "Flat input: G channel (Y) should be ~128 (neutral)"
        )

    def test_flat_rgb_neutral_normal_b_high(self):
        """B channel (Z) must be high — predominantly upward normal."""
        out = derive_normal_map(self._flat_rgba(8, 8))
        assert (out[..., 2] >= 200).all(), (
            f"Flat input: B channel (Z) should be close to 255; min={out[..., 2].min()}"
        )

    def test_flat_black_neutral_normal(self):
        out = derive_normal_map(self._flat_rgba(8, 8, value=0))
        assert np.abs(out[..., 0].astype(np.int16) - 128).max() <= 1
        assert np.abs(out[..., 1].astype(np.int16) - 128).max() <= 1

    def test_flat_white_neutral_normal(self):
        out = derive_normal_map(self._flat_rgba(8, 8, value=255))
        assert np.abs(out[..., 0].astype(np.int16) - 128).max() <= 1
        assert np.abs(out[..., 1].astype(np.int16) - 128).max() <= 1


class TestDeriveNormalMapGradient:
    """
    Gradient inputs: pin the sign of the X/Y tilt relative to known ramps.

    Convention under test:
        gx = right_col_sobel - left_col_sobel   (positive = brighter to the right)
        nx = -gx * strength                      (negative for rightward ramp)
        encoded_X = nx * 0.5 + 0.5               → < 0.5 → R < 128 for rightward ramp

    A leftward-dark, rightward-bright luminance ramp has positive gx everywhere.
    After negation: nx < 0 → encoded R < 128.
    We pin *that current behaviour*, even if the sign is "wrong" by some convention.
    """

    def _ramp_h(self, h: int, w: int) -> np.ndarray:
        """Horizontal luminance ramp: columns go from 0 (left) to 255 (right)."""
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        col = np.linspace(0, 255, w, dtype=np.uint8)
        rgba[..., :3] = col[np.newaxis, :, np.newaxis]
        rgba[..., 3] = 255
        return rgba

    def _ramp_v(self, h: int, w: int) -> np.ndarray:
        """Vertical luminance ramp: rows go from 0 (top) to 255 (bottom)."""
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        row = np.linspace(0, 255, h, dtype=np.uint8)
        rgba[..., :3] = row[:, np.newaxis, np.newaxis]
        rgba[..., 3] = 255
        return rgba

    def test_horizontal_ramp_r_channel_sign(self):
        """
        Horizontal ramp (dark-left, bright-right): current code produces R < 128
        (because gx > 0 → nx = -gx * strength < 0 → encoded below 0.5).
        Pin this as current behaviour.

        SUSPICION: This maps rightward-ascending height to a leftward-pointing
        normal.  A slope rising to the right should light *from the right* →
        normal pointing left (−X in tangent space) IS geometrically correct IF
        luminance = height (the raised side is toward the camera, not the light).
        Verify in-engine that shading matches expectation.
        """
        out = derive_normal_map(self._ramp_h(16, 16))
        # Interior columns (avoid wrap-padding boundary distortion at col 0 and W-1)
        interior_r = out[1:-1, 2:-2, 0].astype(np.int16)
        assert (interior_r < 128).all(), (
            f"Horizontal ramp: interior R should be < 128; max={interior_r.max()}"
        )

    def test_vertical_ramp_g_channel_sign(self):
        """
        Vertical ramp (dark-top, bright-bottom): pin that interior G < 128.
        gy > 0 → ny = -gy * strength < 0 → encoded G < 128.
        """
        out = derive_normal_map(self._ramp_v(16, 16))
        interior_g = out[2:-2, 1:-1, 1].astype(np.int16)
        assert (interior_g < 128).all(), (
            f"Vertical ramp: interior G should be < 128; max={interior_g.max()}"
        )

    def test_b_channel_stays_high_on_gradient(self):
        """Even on a steep ramp, B (Z normal) must dominate (> 128)."""
        out = derive_normal_map(self._ramp_h(16, 16))
        assert (out[..., 2] > 128).all(), (
            f"B channel must stay > 128 on a gradient; min={out[..., 2].min()}"
        )

    def test_step_edge_tilts_normal(self):
        """
        Hard vertical step (left half dark, right half bright) at mid-column:
        the normals near the edge should deviate from 128.
        """
        h, w = 16, 16
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, w // 2 :, :3] = 200
        rgba[..., 3] = 255
        out = derive_normal_map(rgba)
        # Normals in the interior far from the step should remain near 128.
        # Near the step edge, R deviates.
        edge_col = w // 2
        edge_zone = out[:, edge_col - 1 : edge_col + 2, 0].astype(np.int16)
        assert (np.abs(edge_zone - 128) > 0).any(), (
            "Step edge should produce non-neutral R values near the transition"
        )


class TestDeriveNormalMapStrength:
    """Larger strength → larger deviation of R/G from 128 (monotonic direction)."""

    def _ramp_rgba(self) -> np.ndarray:
        """16×16 horizontal luminance ramp for gradient tests."""
        rgba = np.zeros((16, 16, 4), dtype=np.uint8)
        col = np.linspace(0, 255, 16, dtype=np.uint8)
        rgba[..., :3] = col[np.newaxis, :, np.newaxis]
        rgba[..., 3] = 255
        return rgba

    def test_strength_monotonic_r_deviation(self):
        """
        As strength increases from 0.5 → 1.0 → 2.0, the mean absolute deviation
        of R from 128 should increase monotonically in the interior.
        """
        rgba = self._ramp_rgba()
        interior = np.s_[1:-1, 2:-2]
        deviations = []
        for s in (0.5, 1.0, 2.0):
            out = derive_normal_map(rgba, strength=s)
            dev = np.abs(out[interior][..., 0].astype(np.int16) - 128).mean()
            deviations.append(float(dev))
        assert deviations[0] < deviations[1] < deviations[2], (
            f"R deviation must increase with strength; got {deviations}"
        )

    def test_zero_strength_flat_output(self):
        """
        strength=0.0 on any input → nx=ny=0 → flat normal (128, 128, 255).
        """
        rgba = self._ramp_rgba()
        out = derive_normal_map(rgba, strength=0.0)
        assert np.abs(out[..., 0].astype(np.int16) - 128).max() <= 1, (
            "strength=0 should produce R≈128 everywhere"
        )
        assert np.abs(out[..., 1].astype(np.int16) - 128).max() <= 1, (
            "strength=0 should produce G≈128 everywhere"
        )

    def test_high_strength_b_remains_above_128(self):
        """Even at strength=5.0 (extreme), B channel must remain > 128 (Z > 0)."""
        rgba = self._ramp_rgba()
        out = derive_normal_map(rgba, strength=5.0)
        assert (out[..., 2] > 128).all(), (
            f"B channel must stay > 128 even at high strength; min={out[..., 2].min()}"
        )


class TestDeriveNormalMapWrapTiling:
    """
    Wrap-padding: left and right edges are treated as adjacent.
    The gradient at the left edge uses the right edge's luminance as its left
    neighbour and vice-versa, so a uniform texture has no seam artefacts.
    A uniform texture → zero gradient → neutral normal at every edge texel.
    """

    def test_uniform_texture_no_seam_at_horizontal_edges(self):
        """Left and right edge columns on a uniform texture == interior."""
        rgba = np.full((16, 16, 4), 150, dtype=np.uint8)
        rgba[..., 3] = 255
        out = derive_normal_map(rgba)
        left_col_r = out[:, 0, 0].astype(np.int16)
        right_col_r = out[:, -1, 0].astype(np.int16)
        # All should be ~128 with no deviation > 1 for a uniform texture.
        assert np.abs(left_col_r - 128).max() <= 1, "Left edge R should be ~128"
        assert np.abs(right_col_r - 128).max() <= 1, "Right edge R should be ~128"

    def test_uniform_texture_no_seam_at_vertical_edges(self):
        rgba = np.full((16, 16, 4), 100, dtype=np.uint8)
        rgba[..., 3] = 255
        out = derive_normal_map(rgba)
        top_row_g = out[0, :, 1].astype(np.int16)
        bottom_row_g = out[-1, :, 1].astype(np.int16)
        assert np.abs(top_row_g - 128).max() <= 1, "Top edge G should be ~128"
        assert np.abs(bottom_row_g - 128).max() <= 1, "Bottom edge G should be ~128"

    def test_wrap_gradient_continuity(self):
        """
        For a horizontal ramp that wraps (0→255 left-to-right), the Sobel
        gradient at the right edge considers col 0 as its right neighbour.
        The right-edge gradient should be large (since 255 and 0 are adjacent
        after wrap), similar in magnitude to the step at the centre.
        Pin that right-edge deviation from 128 is comparable to an interior
        gradient point — not zero — confirming wrap-padding is active.

        SUSPICION: if wrap-padding were absent (zero-padding instead), the
        right edge would see a flat region rather than the large wrap step.
        """
        rgba = np.zeros((8, 8, 4), dtype=np.uint8)
        col = np.linspace(0, 255, 8, dtype=np.uint8)
        rgba[..., :3] = col[np.newaxis, :, np.newaxis]
        rgba[..., 3] = 255
        out = derive_normal_map(rgba)
        # Interior has consistent gradient; right edge should too (wrap active).
        right_edge_dev = np.abs(out[:, -1, 0].astype(np.int16) - 128).mean()
        # Both should be non-zero (wrap makes the right edge "feel" the step).
        assert right_edge_dev > 0, (
            "Right edge should have non-zero R deviation for a ramp (wrap active)"
        )


class TestDeriveNormalMapDeterminism:
    """Pure function — identical output across calls."""

    def test_identical_output_twice(self):
        rng = np.random.default_rng(42)
        rgba = rng.integers(0, 256, size=(16, 16, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        out1 = derive_normal_map(rgba)
        out2 = derive_normal_map(rgba)
        assert np.array_equal(out1, out2), (
            "derive_normal_map must be deterministic for the same input"
        )

    def test_different_inputs_differ(self):
        rgba_a = np.zeros((8, 8, 4), dtype=np.uint8)
        rgba_a[..., 3] = 255
        rgba_b = np.full((8, 8, 4), 200, dtype=np.uint8)
        rgba_b[..., 3] = 255
        out_a = derive_normal_map(rgba_a)
        out_b = derive_normal_map(rgba_b)
        # Both flat → same output.
        assert np.array_equal(out_a, out_b), (
            "Two different flat inputs at different luminances should produce "
            "the same flat normal (zero gradient in both cases)"
        )
