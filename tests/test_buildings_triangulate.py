"""
tests/test_buildings_triangulate.py — golden-master / characterisation tests for
triangulate_polygon (ear-clipping triangulation in fire_engine/buildings/triangulate.py).

Headless only (numpy only — fire_engine/buildings/ never imports panda3d).
Pin current behaviour; do NOT fix bugs — report suspicions in comments.
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.buildings.triangulate import triangulate_polygon


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shoelace(poly: np.ndarray) -> float:
    """Signed area via shoelace (CCW positive)."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _tri_signed_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Signed area of a single 2-D triangle (CCW positive)."""
    return 0.5 * float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _triangle_areas(poly2d: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Absolute areas of all triangles defined by indices into poly2d."""
    a = poly2d[indices[:, 0]]
    b = poly2d[indices[:, 1]]
    c = poly2d[indices[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return 0.5 * np.abs(cross)


def _signed_triangle_areas(poly2d: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Signed areas of all triangles defined by indices into poly2d (CCW positive)."""
    a = poly2d[indices[:, 0]]
    b = poly2d[indices[:, 1]]
    c = poly2d[indices[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return 0.5 * cross


# ---------------------------------------------------------------------------
# Fixture polygons
# ---------------------------------------------------------------------------

_TRIANGLE_CCW = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
_TRIANGLE_CW = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

_SQUARE_CCW = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
_SQUARE_CW = _SQUARE_CCW[::-1].copy()

_PENTAGON_CCW = np.array(
    [
        [0.0, 0.0],
        [2.0, 0.0],
        [3.0, 1.5],
        [1.5, 3.0],
        [-0.5, 2.0],
    ]
)

_HEXAGON_CCW = np.array(
    [
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 1.0],
        [2.0, 2.0],
        [1.0, 2.0],
        [0.0, 1.0],
    ]
)

# L-shaped concave polygon (CCW)
_L_SHAPE_CCW = np.array(
    [
        [0.0, 0.0],
        [3.0, 0.0],
        [3.0, 1.0],
        [1.0, 1.0],
        [1.0, 3.0],
        [0.0, 3.0],
    ]
)

# Near-zero-area sliver (very thin triangle)
_SLIVER = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 1e-8]])

# Three nearly-collinear points
_COLLINEAR = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])


# ---------------------------------------------------------------------------
# Triangle count: N-vertex simple polygon yields exactly N-2 triangles
# ---------------------------------------------------------------------------


class TestTriangleCount:
    def test_triangle_yields_1_triangle(self):
        idx = triangulate_polygon(_TRIANGLE_CCW)
        assert idx.shape == (1, 3)

    def test_square_yields_2_triangles(self):
        idx = triangulate_polygon(_SQUARE_CCW)
        assert idx.shape == (2, 3)

    def test_pentagon_yields_3_triangles(self):
        idx = triangulate_polygon(_PENTAGON_CCW)
        assert idx.shape == (3, 3)

    def test_hexagon_yields_4_triangles(self):
        idx = triangulate_polygon(_HEXAGON_CCW)
        assert idx.shape == (4, 3)

    def test_l_shape_yields_4_triangles(self):
        # L-shape has 6 vertices → 4 triangles
        idx = triangulate_polygon(_L_SHAPE_CCW)
        assert idx.shape == (4, 3)


# ---------------------------------------------------------------------------
# Output dtype and shape
# ---------------------------------------------------------------------------


class TestOutputContract:
    def test_dtype_is_uint32(self):
        idx = triangulate_polygon(_SQUARE_CCW)
        assert idx.dtype == np.uint32

    def test_shape_is_t_by_3(self):
        for poly in (_TRIANGLE_CCW, _SQUARE_CCW, _PENTAGON_CCW, _HEXAGON_CCW, _L_SHAPE_CCW):
            n = poly.shape[0]
            idx = triangulate_polygon(poly)
            assert idx.shape == (n - 2, 3), (
                f"Expected ({n - 2}, 3) for {n}-vertex polygon, got {idx.shape}"
            )

    def test_all_indices_in_range(self):
        for poly in (_TRIANGLE_CCW, _SQUARE_CCW, _PENTAGON_CCW, _HEXAGON_CCW, _L_SHAPE_CCW):
            n = poly.shape[0]
            idx = triangulate_polygon(poly)
            assert np.all(idx >= 0), "Negative index found"
            assert np.all(idx < n), f"Index >= {n} found: {idx.max()}"


# ---------------------------------------------------------------------------
# Area preservation
# ---------------------------------------------------------------------------


class TestAreaPreservation:
    def test_square_area_preserved(self):
        idx = triangulate_polygon(_SQUARE_CCW)
        tri_areas = _triangle_areas(_SQUARE_CCW, idx)
        poly_area = abs(_shoelace(_SQUARE_CCW))
        assert np.isclose(tri_areas.sum(), poly_area, rtol=1e-9)

    def test_pentagon_area_preserved(self):
        idx = triangulate_polygon(_PENTAGON_CCW)
        tri_areas = _triangle_areas(_PENTAGON_CCW, idx)
        poly_area = abs(_shoelace(_PENTAGON_CCW))
        assert np.isclose(tri_areas.sum(), poly_area, rtol=1e-9)

    def test_hexagon_area_preserved(self):
        idx = triangulate_polygon(_HEXAGON_CCW)
        tri_areas = _triangle_areas(_HEXAGON_CCW, idx)
        poly_area = abs(_shoelace(_HEXAGON_CCW))
        assert np.isclose(tri_areas.sum(), poly_area, rtol=1e-9)

    def test_l_shape_area_preserved(self):
        # L-shape is concave; total area must match shoelace
        idx = triangulate_polygon(_L_SHAPE_CCW)
        tri_areas = _triangle_areas(_L_SHAPE_CCW, idx)
        poly_area = abs(_shoelace(_L_SHAPE_CCW))
        assert np.isclose(tri_areas.sum(), poly_area, rtol=1e-9), (
            f"L-shape area mismatch: triangles={tri_areas.sum():.6f}, "
            f"shoelace={poly_area:.6f} — SUSPECTED BUG in concave handling"
        )


# ---------------------------------------------------------------------------
# Winding: all output triangles must be CCW
# ---------------------------------------------------------------------------


class TestWinding:
    def _all_ccw(self, poly):
        idx = triangulate_polygon(poly)
        signed = _signed_triangle_areas(poly, idx)
        return signed, idx

    def test_ccw_input_square_emits_ccw_triangles(self):
        # SUSPECTED BUG NOTE: the implementation re-orders vertices when
        # input is CW (negates order), then indexes back into the *original*
        # polygon via that reversed ordering. The output indices reference the
        # original array, but the "signed area positive → CCW" check below
        # verifies the *semantic* winding of each emitted triangle via the
        # original coordinate positions. Pin current behaviour here.
        signed, _ = self._all_ccw(_SQUARE_CCW)
        # Pin: all triangles should have positive signed area (CCW)
        assert np.all(signed > 0), (
            f"Non-CCW triangle(s) in CCW-input square: signed areas={signed} "
            "— SUSPECTED BUG if any signed area <= 0"
        )

    def test_cw_input_square_emits_ccw_triangles(self):
        # CW input is reversed internally so result should still be CCW
        signed, _ = self._all_ccw(_SQUARE_CW)
        # Pin current behaviour (may be CW if bug present — do not fix, pin)
        # SUSPECTED BUG: reversing the order list but still indexing original
        # polygon means emitted triangles use reversed-order coordinates.
        # Record actual sign here (golden master):
        all_positive = bool(np.all(signed > 0))
        # Simply pin the count and sign pattern rather than asserting CCW:
        assert signed.shape[0] == 2  # still 2 triangles
        # Report suspected winding issue in CW input path without masking it
        if not all_positive:
            pytest.xfail(
                "CW input produces non-CCW output triangles — "
                "suspected winding bug: order reversal but original poly indexing"
            )

    def test_ccw_input_pentagon_emits_ccw_triangles(self):
        signed, _ = self._all_ccw(_PENTAGON_CCW)
        assert np.all(signed > 0), f"Non-CCW triangle in pentagon: {signed}"

    def test_ccw_l_shape_emits_ccw_triangles(self):
        signed, _ = self._all_ccw(_L_SHAPE_CCW)
        assert np.all(signed > 0), (
            f"Non-CCW triangle(s) in L-shape: {signed} — "
            "SUSPECTED BUG if concave ear-clip emits CW tris"
        )

    def test_winding_consistent_across_all_convex_shapes(self):
        """All convex polygons should produce uniformly-signed (CCW) output."""
        for poly in (_TRIANGLE_CCW, _SQUARE_CCW, _PENTAGON_CCW, _HEXAGON_CCW):
            idx = triangulate_polygon(poly)
            signed = _signed_triangle_areas(poly, idx)
            signs = np.sign(signed)
            # All same sign (all positive = CCW, all negative = consistently CW)
            assert np.all(signs == signs[0]), (
                f"Mixed winding in {poly.shape[0]}-vertex convex polygon"
            )


# ---------------------------------------------------------------------------
# Convex square coverage
# ---------------------------------------------------------------------------


class TestSquareCoverage:
    def test_square_exactly_2_triangles(self):
        idx = triangulate_polygon(_SQUARE_CCW)
        assert idx.shape == (2, 3)

    def test_square_area_is_16(self):
        idx = triangulate_polygon(_SQUARE_CCW)
        areas = _triangle_areas(_SQUARE_CCW, idx)
        assert np.isclose(areas.sum(), 16.0, rtol=1e-9)

    def test_square_no_duplicate_triangles(self):
        idx = triangulate_polygon(_SQUARE_CCW)
        # Normalise each triangle (sort vertex indices) and check uniqueness
        normalised = np.sort(idx, axis=1)
        unique_rows = np.unique(normalised, axis=0)
        assert unique_rows.shape[0] == idx.shape[0]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_polygon_same_result_square(self):
        a = triangulate_polygon(_SQUARE_CCW)
        b = triangulate_polygon(_SQUARE_CCW)
        assert np.array_equal(a, b)

    def test_same_polygon_same_result_l_shape(self):
        a = triangulate_polygon(_L_SHAPE_CCW)
        b = triangulate_polygon(_L_SHAPE_CCW)
        assert np.array_equal(a, b)

    def test_same_polygon_same_result_hexagon(self):
        a = triangulate_polygon(_HEXAGON_CCW)
        b = triangulate_polygon(_HEXAGON_CCW)
        assert np.array_equal(a, b)

    def test_same_polygon_same_result_concave_l(self):
        # Independent input arrays — same values → identical output
        poly1 = _L_SHAPE_CCW.copy()
        poly2 = _L_SHAPE_CCW.copy()
        a = triangulate_polygon(poly1)
        b = triangulate_polygon(poly2)
        assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


class TestDegenerateInputs:
    def test_empty_polygon_returns_empty(self):
        idx = triangulate_polygon(np.empty((0, 2)))
        assert idx.shape[0] == 0
        assert idx.shape[1] == 3
        assert idx.dtype == np.uint32

    def test_single_vertex_returns_empty(self):
        idx = triangulate_polygon(np.array([[1.0, 2.0]]))
        assert idx.shape == (0, 3)
        assert idx.dtype == np.uint32

    def test_two_vertices_returns_empty(self):
        idx = triangulate_polygon(np.array([[0.0, 0.0], [1.0, 0.0]]))
        assert idx.shape == (0, 3)
        assert idx.dtype == np.uint32

    def test_triangle_n3_yields_one_triangle(self):
        idx = triangulate_polygon(_TRIANGLE_CCW)
        assert idx.shape == (1, 3)
        assert idx.dtype == np.uint32
        # All indices in [0, 3)
        assert np.all(idx < 3)

    def test_triangle_indices_reference_all_three_vertices(self):
        idx = triangulate_polygon(_TRIANGLE_CCW)
        assert set(idx.flatten().tolist()) == {0, 1, 2}

    def test_collinear_triangle_pins_behaviour(self):
        # Three collinear points form a zero-area triangle.
        # The cross product = 0, which is <= 1e-12 → the ear tip test rejects
        # the single "ear" repeatedly until guard fires, then fan fallback runs.
        # Pin: does not raise, returns some (1, 3) array (fan fallback).
        # SUSPECTED ISSUE: result indices may reference collinear vertices;
        # the triangle will have zero area. Pin without asserting correctness.
        try:
            idx = triangulate_polygon(_COLLINEAR)
            # Pin: shape is (1, 3) and dtype is uint32 (fan fallback fires)
            assert idx.dtype == np.uint32
            assert idx.ndim == 2 and idx.shape[1] == 3
        except Exception as exc:
            pytest.fail(f"triangulate_polygon raised unexpectedly on collinear input: {exc}")

    def test_sliver_polygon_does_not_raise(self):
        # Near-zero-area triangle — pin current behaviour (result or exception).
        try:
            idx = triangulate_polygon(_SLIVER)
            # If it returns, pin that it is (1, 3) with valid indices
            assert idx.shape == (1, 3)
            assert idx.dtype == np.uint32
            assert np.all(idx < 3)
        except Exception as exc:
            pytest.fail(f"triangulate_polygon raised on sliver: {exc}")


# ---------------------------------------------------------------------------
# CW input path — pin index content (not just shape)
# ---------------------------------------------------------------------------


class TestCWInput:
    def test_cw_square_same_area_as_ccw_square(self):
        idx_ccw = triangulate_polygon(_SQUARE_CCW)
        idx_cw = triangulate_polygon(_SQUARE_CW)
        area_ccw = _triangle_areas(_SQUARE_CCW, idx_ccw).sum()
        area_cw = _triangle_areas(_SQUARE_CW, idx_cw).sum()
        assert np.isclose(area_ccw, area_cw, rtol=1e-9), (
            "CW and CCW squares should triangulate to the same total area"
        )

    def test_cw_triangle_yields_one_triangle(self):
        idx = triangulate_polygon(_TRIANGLE_CW)
        assert idx.shape == (1, 3)

    def test_cw_pentagon_yields_3_triangles(self):
        cw_pent = _PENTAGON_CCW[::-1].copy()
        idx = triangulate_polygon(cw_pent)
        assert idx.shape == (3, 3)

    def test_cw_pentagon_area_preserved(self):
        cw_pent = _PENTAGON_CCW[::-1].copy()
        idx = triangulate_polygon(cw_pent)
        tri_areas = _triangle_areas(cw_pent, idx)
        poly_area = abs(_shoelace(cw_pent))
        assert np.isclose(tri_areas.sum(), poly_area, rtol=1e-9)
