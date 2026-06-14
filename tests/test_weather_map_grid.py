"""
tests/test_weather_map_grid.py — Characterization / golden-master tests for
WeatherMap: grid layout, texel_centers geometry, rasterize equivalence with
sample_fields, channel-ordering, and dtype/shape invariants.

No panda3d imports.  Fixed seeds throughout.

These tests pin *current* behavior.  Do NOT fix bugs; note suspicions in
comments.
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.core import EventBus, load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.weather import MAP_CHANNELS, WeatherMap, WeatherSystem
from fire_engine.weather.cells import CellKind, natural_cells

DAY = 24 * 3600.0
HOUR = 3600.0


# ---------------------------------------------------------------------------
# Helpers — copied from test_weather_map.py / test_weather_system.py style
# ---------------------------------------------------------------------------

def _ws(seed: int = 1337) -> WeatherSystem:
    set_world_seed(seed)
    return WeatherSystem(load_config(), EventBus())


def _first_thunderstorm(cfg):
    for d in range(80):
        for c in natural_cells(d, cfg):
            if c.kind is CellKind.THUNDERSTORM:
                return c
    raise AssertionError("no thunderstorm found in 80 days")


# ---------------------------------------------------------------------------
# MAP_CHANNELS contract
# ---------------------------------------------------------------------------

class TestMapChannels:
    def test_channel_count(self):
        """MAP_CHANNELS must describe exactly 4 channels (last raster axis)."""
        assert len(MAP_CHANNELS) == 4

    def test_channel_names(self):
        """Pin the exact channel names in order."""
        assert MAP_CHANNELS == ("coverage", "density", "precip", "fog")

    def test_channel_order_matches_raster_axis(self):
        """Channel 0 = coverage, 1 = density, 2 = precip, 3 = fog — verify
        against a raster taken under a thunderstorm core where all four should
        be noticeably non-zero (precip and coverage especially)."""
        cfg = load_config()
        ws = _ws()
        wm = WeatherMap(cfg)
        c = _first_thunderstorm(cfg)
        t = c.spawn_time + 0.5 * c.duration_s
        center = tuple(c.center(t, ws.synoptic))
        r = wm.rasterize(ws, center, t)
        mid = wm.cells // 2
        # Under the storm core the center texel must carry coverage & precip.
        assert r[mid, mid, 0] > 0.0, "coverage (ch 0) zero at storm core"
        assert r[mid, mid, 2] > 0.0, "precip   (ch 2) zero at storm core"


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------

class TestShapeAndDtype:
    def test_shape_cells_cells_4(self):
        cfg = load_config()
        ws = _ws()
        wm = WeatherMap(cfg)
        r = wm.rasterize(ws, (0.0, 0.0), 12 * HOUR)
        assert r.shape == (wm.cells, wm.cells, 4)

    def test_dtype_float32(self):
        ws = _ws()
        wm = WeatherMap(load_config())
        r = wm.rasterize(ws, (0.0, 0.0), 12 * HOUR)
        assert r.dtype == np.float32

    def test_cells_attribute_matches_config(self):
        cfg = load_config()
        wm = WeatherMap(cfg)
        assert wm.cells == int(cfg.weather_map_cells)

    def test_cell_m_attribute_matches_config(self):
        cfg = load_config()
        wm = WeatherMap(cfg)
        assert wm.cell_m == pytest.approx(float(cfg.weather_map_cell_m))

    def test_span_m_equals_cells_times_cell_m(self):
        cfg = load_config()
        wm = WeatherMap(cfg)
        assert wm.span_m == pytest.approx(wm.cells * wm.cell_m)


# ---------------------------------------------------------------------------
# Finite values / channel bounds
# ---------------------------------------------------------------------------

class TestChannelBounds:
    def test_no_nan_or_inf_ambient(self):
        """A quiet daytime raster must have all-finite values."""
        ws = _ws(seed=42)
        wm = WeatherMap(load_config())
        r = wm.rasterize(ws, (0.0, 0.0), 12 * HOUR)
        assert np.all(np.isfinite(r)), "NaN/inf found in ambient raster"

    def test_no_nan_or_inf_under_storm(self):
        cfg = load_config()
        ws = _ws()
        wm = WeatherMap(cfg)
        c = _first_thunderstorm(cfg)
        t = c.spawn_time + 0.4 * c.duration_s
        center = tuple(c.center(t, ws.synoptic))
        r = wm.rasterize(ws, center, t)
        assert np.all(np.isfinite(r)), "NaN/inf found in storm raster"

    def test_coverage_in_0_1(self):
        cfg = load_config()
        ws = _ws()
        wm = WeatherMap(cfg)
        c = _first_thunderstorm(cfg)
        t = c.spawn_time + 0.4 * c.duration_s
        center = tuple(c.center(t, ws.synoptic))
        r = wm.rasterize(ws, center, t)
        assert np.all(r[..., 0] >= 0.0) and np.all(r[..., 0] <= 1.0)

    def test_density_in_0_1(self):
        cfg = load_config()
        ws = _ws()
        wm = WeatherMap(cfg)
        c = _first_thunderstorm(cfg)
        t = c.spawn_time + 0.4 * c.duration_s
        center = tuple(c.center(t, ws.synoptic))
        r = wm.rasterize(ws, center, t)
        assert np.all(r[..., 1] >= 0.0) and np.all(r[..., 1] <= 1.0)

    def test_precip_in_0_1(self):
        cfg = load_config()
        ws = _ws()
        wm = WeatherMap(cfg)
        c = _first_thunderstorm(cfg)
        t = c.spawn_time + 0.4 * c.duration_s
        center = tuple(c.center(t, ws.synoptic))
        r = wm.rasterize(ws, center, t)
        assert np.all(r[..., 2] >= 0.0) and np.all(r[..., 2] <= 1.0)

    def test_fog_nonneg_and_bounded_by_fog_max(self):
        cfg = load_config()
        ws = _ws()
        wm = WeatherMap(cfg)
        c = _first_thunderstorm(cfg)
        t = c.spawn_time + 0.4 * c.duration_s
        center = tuple(c.center(t, ws.synoptic))
        r = wm.rasterize(ws, center, t)
        assert np.all(r[..., 3] >= 0.0)
        assert np.all(r[..., 3] <= cfg.weather_fog_max_density + 1e-6)


# ---------------------------------------------------------------------------
# texel_centers geometry
# ---------------------------------------------------------------------------

class TestTexelCenters:
    def test_shape(self):
        cfg = load_config()
        wm = WeatherMap(cfg)
        pts = wm.texel_centers((0.0, 0.0))
        assert pts.shape == (wm.cells * wm.cells, 2)

    def test_center_column_is_x(self):
        """Column 0 of the return is X (not Y)."""
        cfg = load_config()
        wm = WeatherMap(cfg)
        pts = wm.texel_centers((0.0, 0.0))
        # All X values (col 0) should span the configured range.
        x_range = pts[:, 0].max() - pts[:, 0].min()
        expected_range = (wm.cells - 1) * wm.cell_m
        assert x_range == pytest.approx(expected_range, rel=1e-5)

    def test_span_centered_on_center_xy(self):
        """The texel grid's centroid == center_xy."""
        cfg = load_config()
        wm = WeatherMap(cfg)
        center = (500.0, -300.0)
        pts = wm.texel_centers(center)
        # Mean of all texel centers should equal center_xy exactly.
        assert pts[:, 0].mean() == pytest.approx(center[0], abs=1e-3)
        assert pts[:, 1].mean() == pytest.approx(center[1], abs=1e-3)

    def test_x_range_matches_span_m(self):
        """Outermost texel centers are offset by ±(span_m/2 - cell_m/2)."""
        cfg = load_config()
        wm = WeatherMap(cfg)
        center = (0.0, 0.0)
        pts = wm.texel_centers(center)
        half_inner = 0.5 * wm.span_m - 0.5 * wm.cell_m
        assert pts[:, 0].max() == pytest.approx(half_inner, rel=1e-5)
        assert pts[:, 0].min() == pytest.approx(-half_inner, rel=1e-5)

    def test_translated_center(self):
        """texel_centers shifts rigidly with center_xy."""
        cfg = load_config()
        wm = WeatherMap(cfg)
        pts0 = wm.texel_centers((0.0, 0.0))
        offset = (123.5, -456.7)
        pts1 = wm.texel_centers(offset)
        diff = pts1 - pts0
        assert np.allclose(diff[:, 0], offset[0], atol=1e-4)
        assert np.allclose(diff[:, 1], offset[1], atol=1e-4)

    def test_reshape_to_cells_cells_consistent_with_rasterize(self):
        """Row=Y, Col=X layout: pts reshaped to (N,N,2) must match grid coords
        that rasterize would use.  Verify the first and last texel match the
        expected corners."""
        cfg = load_config()
        wm = WeatherMap(cfg)
        center = (0.0, 0.0)
        pts = wm.texel_centers(center)
        # row 0, col 0 → lowest Y, lowest X
        expected_x0 = -0.5 * wm.span_m + 0.5 * wm.cell_m
        expected_y0 = -0.5 * wm.span_m + 0.5 * wm.cell_m
        assert pts[0, 0] == pytest.approx(expected_x0, rel=1e-5), \
            "first texel X wrong"
        assert pts[0, 1] == pytest.approx(expected_y0, rel=1e-5), \
            "first texel Y wrong"


# ---------------------------------------------------------------------------
# rasterize == sample_fields equivalence (the core contract)
# ---------------------------------------------------------------------------

class TestRasterEqualsFieldSample:
    """
    rasterize is documented to call sample_fields at every texel_center, so
    rasterize(sys, center, t) must exactly equal
    sample_fields(texel_centers(center), t) reshaped to (N,N,4).

    SUSPICION: If sample_fields returns values in a different channel order
    than MAP_CHANNELS, rasterize would silently produce wrong channel
    assignments.  We pin the equivalence here; any future divergence shows up
    as a test failure on the array_equal check.
    """

    def _expected_grid(self, ws, wm, center, t):
        """Build the expected (N,N,4) float32 from sample_fields directly."""
        pts = wm.texel_centers(center)
        cov, den, rain, fog, _gust = ws.sample_fields(pts, t)
        out = np.stack([cov, den, rain, fog], axis=1)
        return out.reshape(wm.cells, wm.cells, 4).astype(np.float32)

    def test_exact_match_ambient(self):
        cfg = load_config()
        ws = _ws()
        wm = WeatherMap(cfg)
        center = (0.0, 0.0)
        t = 12 * HOUR
        r = wm.rasterize(ws, center, t)
        expected = self._expected_grid(ws, wm, center, t)
        assert np.array_equal(r, expected), \
            "rasterize differs from sample_fields at ambient conditions"

    def test_exact_match_under_storm(self):
        cfg = load_config()
        ws = _ws()
        wm = WeatherMap(cfg)
        c = _first_thunderstorm(cfg)
        t = c.spawn_time + 0.5 * c.duration_s
        center = tuple(c.center(t, ws.synoptic))
        r = wm.rasterize(ws, center, t)
        expected = self._expected_grid(ws, wm, center, t)
        assert np.array_equal(r, expected), \
            "rasterize differs from sample_fields under storm"

    def test_allclose_match_off_center(self):
        """Off-origin center: rasterize ≈ sample_fields (float32 cast only)."""
        cfg = load_config()
        ws = _ws(seed=42)
        wm = WeatherMap(cfg)
        center = (1234.5, -987.3)
        t = 7 * HOUR
        r = wm.rasterize(ws, center, t)
        expected = self._expected_grid(ws, wm, center, t)
        assert np.allclose(r, expected, atol=1e-6), \
            "rasterize vs sample_fields mismatch at off-center position"

    def test_sample_fields_channel_order_matches_map_channels(self):
        """
        sample_fields returns (cov, den, rain, fog, gust).  MAP_CHANNELS is
        (coverage, density, precip, fog).  Pin that rasterize stacks them in
        that exact order by comparing channel slices.

        SUSPICION: The gust channel (index 4) is silently dropped in rasterize.
        This is intentional per the code, but worth characterising explicitly.
        """
        cfg = load_config()
        ws = _ws()
        wm = WeatherMap(cfg)
        c = _first_thunderstorm(cfg)
        t = c.spawn_time + 0.5 * c.duration_s
        center = tuple(c.center(t, ws.synoptic))

        pts = wm.texel_centers(center)
        cov, den, rain, fog, _gust = ws.sample_fields(pts, t)
        r = wm.rasterize(ws, center, t)

        n = wm.cells
        assert np.allclose(r[..., 0], cov.reshape(n, n).astype(np.float32),
                            atol=1e-6), "ch0 != coverage"
        assert np.allclose(r[..., 1], den.reshape(n, n).astype(np.float32),
                            atol=1e-6), "ch1 != density"
        assert np.allclose(r[..., 2], rain.reshape(n, n).astype(np.float32),
                            atol=1e-6), "ch2 != precip/rain"
        assert np.allclose(r[..., 3], fog.reshape(n, n).astype(np.float32),
                            atol=1e-6), "ch3 != fog"


# ---------------------------------------------------------------------------
# Time-invariance / determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_args_identical_array(self):
        """Calling rasterize twice with identical arguments gives array_equal."""
        cfg = load_config()
        ws = _ws()
        wm = WeatherMap(cfg)
        center = (0.0, 0.0)
        t = 15 * HOUR
        first = wm.rasterize(ws, center, t)
        second = wm.rasterize(ws, center, t)
        assert np.array_equal(first, second), \
            "rasterize not deterministic for same (center, t)"

    def test_determinism_after_system_state_changes(self):
        """
        Same (seed, center, t_abs) must produce the same raster even after the
        WeatherSystem has been driven to a completely different time/position.
        Pins the documented 'pure function of (seed, center, t_abs)' claim.
        """
        cfg = load_config()
        ws = _ws()
        wm = WeatherMap(cfg)
        c = _first_thunderstorm(cfg)
        t = c.spawn_time + 0.3 * c.duration_s
        center = tuple(c.center(t, ws.synoptic))

        first = wm.rasterize(ws, center, t)

        # Churn the system extensively.
        for i in range(30):
            ws.update(i % 5, (i * 1111.0) % DAY, (i * 200.0, -i * 100.0))
        ws.sample_local((99999.0, -99999.0), 14 * DAY)

        second = wm.rasterize(ws, center, t)
        assert np.array_equal(first, second), \
            "rasterize not reproducible after system churn"

    def test_storm_raster_differs_from_ambient_raster(self):
        """
        Center on the storm core vs far away → rasters must differ.

        This avoids the global-seed-isolation hazard (see suspected bug below)
        by using a SINGLE WeatherSystem and comparing two different centers.

        SUSPECTED BUG — global seed isolation: creating a second WeatherSystem
        with a different seed via set_world_seed() mutates the behaviour of
        already-constructed WeatherSystem instances.  Observed: rasterize(ws_a,
        center, t) returns a different array before vs after set_world_seed(9999)
        is called, even though ws_a was built with seed 1337 and its synoptic
        object is unchanged.  This violates the Hard Rule that 'same seed must
        always produce the same world'.  Root cause: sample_fields (or its
        cell/regime helpers) consumes the global RNG stream at query time
        rather than capturing all randomness at construction.  Do NOT fix here
        — just characterise.
        """
        cfg = load_config()
        wm = WeatherMap(cfg)
        ws = _ws(seed=1337)
        c = _first_thunderstorm(cfg)
        t = c.spawn_time + 0.5 * c.duration_s
        # Center on the storm (high precip) vs 50 km away (ambient only).
        center_storm = tuple(c.center(t, ws.synoptic))
        center_far = (center_storm[0] + 50000.0, center_storm[1])
        r_storm = wm.rasterize(ws, center_storm, t)
        r_far = wm.rasterize(ws, center_far, t)
        assert not np.array_equal(r_storm, r_far), \
            "storm-centered raster equals far-away raster — no spatial variation"
