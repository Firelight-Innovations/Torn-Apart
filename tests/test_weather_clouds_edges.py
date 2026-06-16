"""
tests/test_weather_clouds_edges.py — Golden-master / characterization tests for
weather/clouds.py (M9 WMO cloud genera).

PURPOSE
-------
Pin the CURRENT behaviour of classify_genus and cloud_layers so any future
refactor that changes observable output triggers a failure.  Do NOT fix bugs
here — suspect behaviour is noted in comments; the tests pin what the code
actually does today.

Reads:
  docs/systems/weather.md          (authoritative spec)
  fire_engine/weather/clouds.py    (implementation under test)
  tests/test_weather_clouds.py     (existing suite — do not duplicate)

No panda3d imports anywhere (weather/ is headless).
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.core import load_config
from fire_engine.world.weather.cells import Regime
from fire_engine.world.weather.clouds import (
    BAND_HIGH,
    BAND_LOW,
    BAND_MID,
    CloudGenus,
    CloudLayers,
    classify_genus,
    cloud_layers,
)

CFG = load_config()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _layers(**kw) -> CloudLayers:
    """Call cloud_layers with defaults, overriding named fields."""
    defaults = dict(coverage=0.5, density=0.5, precip=0.0, regime=Regime.MIXED, config=CFG)
    defaults.update(kw)
    return cloud_layers(**defaults)


# ===========================================================================
# 1. classify_genus — golden mapping across ALL three regimes
# ===========================================================================


class TestClassifyGenusGoldenMapping:
    """Pin the exact (high, mid, low) triple returned for representative inputs."""

    # --- HIGH_PRESSURE ---

    def test_hp_clear_sky(self):
        # Very low coverage/density, no precip → near-clear fair-weather sky.
        # High band is wispy CIRRUS; mid is thin ALTOCUMULUS; low is degenerate STRATUS.
        # (cov=0.05 is below both 0.18 CUMULUS and 0.55 STRATO thresholds)
        h, m, lo = classify_genus(0.05, 0.10, 0.0, Regime.HIGH_PRESSURE)
        assert h is CloudGenus.CIRRUS  # cov <= 0.55 → cirrus
        assert m is CloudGenus.ALTOCUMULUS  # always altocumulus at low cov
        assert lo is CloudGenus.STRATUS  # cov=0.05 ≤ 0.18 → degenerate stratus

    def test_hp_residual_cirrus(self):
        # Classic mares' tails: cov=0.08, den=0.30 → all three bands.
        h, m, lo = classify_genus(0.08, 0.30, 0.0, Regime.HIGH_PRESSURE)
        assert h is CloudGenus.CIRRUS
        assert m is CloudGenus.ALTOCUMULUS
        # cov=0.08 ≤ 0.18 → STRATUS (degenerate thin); CUMULUS threshold not reached
        assert lo is CloudGenus.STRATUS

    def test_hp_moderate_coverage(self):
        # Moderate cover on a high-pressure day: fair-weather cumulus.
        h, m, lo = classify_genus(0.35, 0.40, 0.0, Regime.HIGH_PRESSURE)
        assert h is CloudGenus.CIRRUS  # cov=0.35 ≤ 0.55 → CIRRUS (not CIRROSTRATUS)
        assert m is CloudGenus.ALTOCUMULUS  # cov=0.35 > 0.30 but still ALTOCUMULUS
        assert lo is CloudGenus.CUMULUS  # 0.18 < cov=0.35 < 0.55 → CUMULUS

    # --- MIXED ---

    def test_mixed_scattered_cumulus(self):
        # Light cover, no rain → fair-weather CUMULUS low.
        h, m, lo = classify_genus(0.30, 0.40, 0.0, Regime.MIXED)
        assert h is CloudGenus.CIRRUS
        assert m is CloudGenus.ALTOCUMULUS
        assert lo is CloudGenus.CUMULUS  # 0.18 < cov=0.30 < 0.55

    def test_mixed_overcast_no_rain(self):
        # High cover, no precip → STRATOCUMULUS low (lumpy overcast).
        # SUSPICION: the code uses `(cov > 0.55) | (frontal & (cov > 0.45))` for
        # STRATOCUMULUS. In MIXED regime, frontal=False, so only `cov > 0.55` fires.
        h, m, lo = classify_genus(0.80, 0.65, 0.0, Regime.MIXED)
        assert h is CloudGenus.CIRROSTRATUS  # cov=0.80 > 0.55
        assert m is CloudGenus.ALTOSTRATUS  # cov > 0.55 AND den > 0.55
        assert lo is CloudGenus.STRATOCUMULUS  # cov=0.80 > 0.55, no precip

    def test_mixed_light_rain(self):
        # pre=0.10 is just above the 0.05 rain-layer threshold → STRATUS (rain deck).
        h, m, lo = classify_genus(0.60, 0.65, 0.10, Regime.MIXED)
        assert lo is CloudGenus.STRATUS  # rain layer threshold (pre > 0.05)
        assert h is CloudGenus.CIRROSTRATUS  # cov=0.60 > 0.55
        assert m is CloudGenus.ALTOSTRATUS  # cov > 0.55 AND den > 0.55

    def test_mixed_heavy_rain_cumulonimbus(self):
        # pre=0.50 is just above 0.45 → CUMULONIMBUS tower.
        _, _, lo = classify_genus(0.90, 0.90, 0.50, Regime.MIXED)
        assert lo is CloudGenus.CUMULONIMBUS

    # --- FRONTAL ---

    def test_frontal_moderate_coverage(self):
        # FRONTAL with cov=0.50: the frontal branch in the code is
        # `frontal & (cov > 0.45)` which is True → STRATOCUMULUS (not CUMULUS).
        # NOTE: this differs from MIXED at the same cov=0.50 (which yields CUMULUS).
        # Pin current behaviour.
        h, m, lo = classify_genus(0.50, 0.55, 0.0, Regime.FRONTAL)
        assert lo is CloudGenus.STRATOCUMULUS  # frontal branch fires at cov > 0.45
        assert h is CloudGenus.CIRRUS  # cov=0.50 ≤ 0.55 → not cirrostratus
        # cov=0.50 > 0.30 so altocumulus is still altocumulus (cov+den threshold not met)
        # cov=0.50 NOT > 0.55 so altostratus branch doesn't fire
        assert m is CloudGenus.ALTOCUMULUS

    def test_frontal_full_overcast_stack(self):
        # Classic frontal overcast: cirrostratus / altostratus / stratocumulus.
        h, m, lo = classify_genus(0.85, 0.80, 0.0, Regime.FRONTAL)
        assert h is CloudGenus.CIRROSTRATUS  # cov > 0.55
        assert m is CloudGenus.ALTOSTRATUS  # cov > 0.55 AND den > 0.55
        assert lo is CloudGenus.STRATOCUMULUS  # cov > 0.55, no precip

    def test_frontal_moderate_precip_stratus(self):
        # FRONTAL with moderate precip (rain layer, not yet CUMULONIMBUS).
        h, m, lo = classify_genus(0.80, 0.75, 0.30, Regime.FRONTAL)
        assert lo is CloudGenus.STRATUS  # pre=0.30 > 0.05 → rain layer
        assert h is CloudGenus.CIRROSTRATUS
        assert m is CloudGenus.ALTOSTRATUS

    def test_frontal_storm_cumulonimbus(self):
        # FRONTAL THUNDERSTORM: high precip → CUMULONIMBUS tower.
        h, m, lo = classify_genus(0.98, 0.95, 0.95, Regime.FRONTAL)
        assert lo is CloudGenus.CUMULONIMBUS
        assert h is CloudGenus.CIRROSTRATUS
        assert m is CloudGenus.ALTOSTRATUS


# ===========================================================================
# 2. classify_genus — boundary / edge inputs
# ===========================================================================


class TestClassifyGenusBoundaries:
    """Pin the genus at exact boundary values."""

    def test_all_zeros(self):
        # coverage=0, density=0, precip=0 → all thresholds below minimum.
        h, m, lo = classify_genus(0.0, 0.0, 0.0, Regime.HIGH_PRESSURE)
        assert h is CloudGenus.CIRRUS  # cov=0 ≤ 0.55
        assert m is CloudGenus.ALTOCUMULUS  # default for mid
        assert lo is CloudGenus.STRATUS  # cov=0 ≤ 0.18 → degenerate stratus

    def test_all_ones(self):
        # coverage=1, density=1, precip=1 → top of every scale.
        h, m, lo = classify_genus(1.0, 1.0, 1.0, Regime.FRONTAL)
        assert lo is CloudGenus.CUMULONIMBUS  # pre=1.0 > 0.45 → storm tower
        assert h is CloudGenus.CIRROSTRATUS  # cov=1.0 > 0.55
        assert m is CloudGenus.ALTOSTRATUS  # cov AND den > 0.55

    def test_precip_exactly_at_cumulonimbus_threshold(self):
        # pre=0.45 is NOT > 0.45 (strict), so → STRATUS not CUMULONIMBUS.
        _, _, lo = classify_genus(0.9, 0.9, 0.45, Regime.FRONTAL)
        assert lo is CloudGenus.STRATUS  # strict threshold: 0.45 is NOT > 0.45

    def test_precip_just_above_cumulonimbus_threshold(self):
        # pre just above 0.45 tips into CUMULONIMBUS.
        _, _, lo = classify_genus(0.9, 0.9, 0.451, Regime.FRONTAL)
        assert lo is CloudGenus.CUMULONIMBUS

    def test_precip_exactly_at_rain_layer_threshold(self):
        # pre=0.05 is NOT > 0.05 (strict) → doesn't trigger rain layer.
        # With cov=0.7 (> 0.55), STRATOCUMULUS fires before we reach pre check.
        _, _, lo = classify_genus(0.7, 0.7, 0.05, Regime.MIXED)
        assert lo is CloudGenus.STRATOCUMULUS  # pre=0.05 NOT > 0.05 → no rain layer

    def test_precip_just_above_rain_layer_threshold(self):
        # pre=0.06 > 0.05 → STRATUS rain layer (overwrites STRATOCUMULUS).
        _, _, lo = classify_genus(0.7, 0.7, 0.06, Regime.MIXED)
        assert lo is CloudGenus.STRATUS  # rain layer wins over stratocumulus

    def test_cov_at_cumulus_threshold(self):
        # cov=0.18 is NOT > 0.18 (strict) → stays STRATUS not CUMULUS.
        _, _, lo = classify_genus(0.18, 0.3, 0.0, Regime.MIXED)
        assert lo is CloudGenus.STRATUS  # strict: 0.18 is NOT > 0.18

    def test_cov_just_above_cumulus_threshold(self):
        # cov=0.19 > 0.18 → CUMULUS.
        _, _, lo = classify_genus(0.19, 0.3, 0.0, Regime.MIXED)
        assert lo is CloudGenus.CUMULUS

    def test_frontal_strato_at_lower_cov_than_mixed(self):
        # In FRONTAL regime, STRATOCUMULUS fires at cov > 0.45 (not 0.55).
        # cov=0.50: FRONTAL → STRATOCUMULUS; MIXED → CUMULUS.
        _, _, l_frontal = classify_genus(0.50, 0.5, 0.0, Regime.FRONTAL)
        _, _, l_mixed = classify_genus(0.50, 0.5, 0.0, Regime.MIXED)
        assert l_frontal is CloudGenus.STRATOCUMULUS  # frontal branch: cov > 0.45
        assert l_mixed is CloudGenus.CUMULUS  # 0.18 < cov=0.50 ≤ 0.55


# ===========================================================================
# 3. classify_genus — vectorization
# ===========================================================================


class TestClassifyGenusVectorization:
    """
    classify_genus accepts ndarray inputs — pin scalar/array parity.
    The function is documented as vectorised; array in → object ndarray out.
    """

    def test_array_output_shape(self):
        cov = np.array([0.05, 0.30, 0.60, 0.90, 0.98])
        den = np.array([0.20, 0.40, 0.65, 0.70, 0.95])
        pre = np.array([0.00, 0.00, 0.00, 0.30, 0.90])
        h, m, lo = classify_genus(cov, den, pre, Regime.FRONTAL)
        assert h.shape == (5,)
        assert m.shape == (5,)
        assert lo.shape == (5,)
        assert h.dtype == object
        assert lo.dtype == object

    def test_array_scalar_parity_all_regimes(self):
        # For all three regimes, array call matches per-element scalar call.
        cov = np.array([0.05, 0.19, 0.50, 0.80, 0.95])
        den = np.array([0.10, 0.35, 0.55, 0.72, 0.95])
        pre = np.array([0.00, 0.00, 0.00, 0.10, 0.80])
        for regime in (Regime.HIGH_PRESSURE, Regime.MIXED, Regime.FRONTAL):
            ah, am, al = classify_genus(cov, den, pre, regime)
            for i in range(len(cov)):
                sh, sm, sl = classify_genus(float(cov[i]), float(den[i]), float(pre[i]), regime)
                assert ah[i] is sh, f"high mismatch at i={i}, regime={regime}"
                assert am[i] is sm, f"mid mismatch at i={i}, regime={regime}"
                assert al[i] is sl, f"low mismatch at i={i}, regime={regime}"

    def test_scalar_returns_tuple_not_array(self):
        # Scalar inputs → plain Python tuple of CloudGenus, not ndarrays.
        result = classify_genus(0.5, 0.5, 0.0, Regime.MIXED)
        assert isinstance(result, tuple)
        assert len(result) == 3
        for g in result:
            assert isinstance(g, CloudGenus)

    def test_single_element_array_returns_ndarrays(self):
        # A length-1 array → ndarrays even though logically scalar.
        cov = np.array([0.5])
        h, _, _ = classify_genus(cov, cov, np.array([0.0]), Regime.MIXED)
        assert isinstance(h, np.ndarray)
        assert h.shape == (1,)


# ===========================================================================
# 4. cloud_layers — structural invariants
# ===========================================================================


class TestCloudLayersStructure:
    """Pin the shape, finiteness, and sign of every field in CloudLayers."""

    def test_returns_cloudlayers_instance(self):
        L = _layers()
        assert isinstance(L, CloudLayers)

    def test_all_arrays_shape_3(self):
        L = _layers(coverage=0.6, density=0.6, precip=0.2)
        for name in ("base_altitude_m", "thickness_m", "coverage", "density", "detail_scale"):
            arr = getattr(L, name)
            assert arr.shape == (3,), f"{name} shape is {arr.shape}, expected (3,)"

    def test_all_arrays_finite(self):
        L = _layers(coverage=0.7, density=0.7, precip=0.5)
        for name in ("base_altitude_m", "thickness_m", "coverage", "density", "detail_scale"):
            arr = getattr(L, name)
            assert np.all(np.isfinite(arr)), f"{name} has non-finite values: {arr}"

    def test_altitudes_strictly_decreasing_by_index(self):
        # BAND_HIGH=0 is highest altitude (1400 m), BAND_LOW=2 is lowest (500 m).
        # NOTE: the CloudLayers docstring says "strictly increasing (low < mid < high)"
        # which is true by logical altitude, but the ARRAY is stored decreasing by index
        # (index 0 = HIGH = largest altitude). Pin the actual array ordering here.
        L = _layers(coverage=0.5, density=0.5, precip=0.0)
        assert L.base_altitude_m[BAND_HIGH] > L.base_altitude_m[BAND_MID], (
            "High band altitude must be above mid band altitude"
        )
        assert L.base_altitude_m[BAND_MID] > L.base_altitude_m[BAND_LOW], (
            "Mid band altitude must be above low band altitude"
        )

    def test_altitude_ordering_holds_under_storm(self):
        # Altitude ordering must hold even for CUMULONIMBUS (which deepens the
        # low slab's thickness, but does NOT change base_altitude_m).
        L = _layers(coverage=0.98, density=0.95, precip=1.0, regime=Regime.FRONTAL)
        assert L.base_altitude_m[BAND_HIGH] > L.base_altitude_m[BAND_MID]
        assert L.base_altitude_m[BAND_MID] > L.base_altitude_m[BAND_LOW]
        assert L.genus_low is CloudGenus.CUMULONIMBUS

    def test_altitudes_match_config_values(self):
        # base_altitude_m is read directly from config; pin the exact values from
        # config.toml (1400.0, 850.0, 500.0) to catch any config-wiring bugs.
        L = _layers()
        assert L.base_altitude_m[BAND_HIGH] == pytest.approx(1400.0)
        assert L.base_altitude_m[BAND_MID] == pytest.approx(850.0)
        assert L.base_altitude_m[BAND_LOW] == pytest.approx(500.0)

    def test_altitudes_invariant_to_regime(self):
        # base_altitude_m does NOT vary with regime (it's purely config-derived).
        # This is a golden pin: if the design changes to shift bands by regime,
        # this test should fail first.
        L_hp = _layers(coverage=0.1, density=0.1, precip=0.0, regime=Regime.HIGH_PRESSURE)
        L_mix = _layers(coverage=0.6, density=0.6, precip=0.0, regime=Regime.MIXED)
        L_fr = _layers(coverage=0.9, density=0.9, precip=0.5, regime=Regime.FRONTAL)
        np.testing.assert_array_equal(L_hp.base_altitude_m, L_mix.base_altitude_m)
        np.testing.assert_array_equal(L_mix.base_altitude_m, L_fr.base_altitude_m)

    def test_thicknesses_positive_finite(self):
        for pre in (0.0, 0.5, 1.0):
            L = _layers(coverage=0.8, density=0.8, precip=pre)
            assert np.all(L.thickness_m > 0.0), (
                f"All thicknesses must be positive (pre={pre}): {L.thickness_m}"
            )

    def test_detail_scales_invariant_to_inputs(self):
        # detail_scale is purely from config — pin that it never changes with inputs.
        L_a = _layers(coverage=0.0, density=0.0, precip=0.0, regime=Regime.HIGH_PRESSURE)
        L_b = _layers(coverage=1.0, density=1.0, precip=1.0, regime=Regime.FRONTAL)
        np.testing.assert_array_equal(
            L_a.detail_scale,
            L_b.detail_scale,
            err_msg="detail_scale must be constant (config-only)",
        )
        # Pin actual config values: [0.45, 0.85, 1.30]
        assert L_a.detail_scale[BAND_HIGH] == pytest.approx(0.45)
        assert L_a.detail_scale[BAND_MID] == pytest.approx(0.85)
        assert L_a.detail_scale[BAND_LOW] == pytest.approx(1.30)


# ===========================================================================
# 5. cloud_layers — coverage / density golden values
# ===========================================================================


class TestCloudLayersGoldenWeights:
    """
    Pin the computed coverage/density values at extremes, derived from the
    known formulae: high_cov = floor + w_high*cov = 0.06 + 0.35*cov (clamped),
    mid_cov = w_mid * cov * mid_present = 0.60 * cov * smoothstep(cov,0.30,0.65).
    """

    def test_high_coverage_at_cov_zero(self):
        # high_cov = 0.06 + 0.35*0.0 = 0.06 (floor only).
        L = _layers(coverage=0.0, density=0.0, precip=0.0)
        assert L.coverage[BAND_HIGH] == pytest.approx(0.06, abs=1e-9)

    def test_high_coverage_at_cov_one(self):
        # high_cov = 0.06 + 0.35*1.0 = 0.41 (floor + full weight).
        L = _layers(coverage=1.0, density=1.0, precip=0.0)
        assert L.coverage[BAND_HIGH] == pytest.approx(0.41, abs=1e-9)

    def test_mid_coverage_at_cov_zero(self):
        # mid_present = smoothstep(0.0, 0.30, 0.65) = 0.0 → mid_cov = 0.0.
        L = _layers(coverage=0.0, density=0.0, precip=0.0)
        assert L.coverage[BAND_MID] == pytest.approx(0.0, abs=1e-9)

    def test_mid_coverage_at_cov_one(self):
        # mid_present = smoothstep(1.0, 0.30, 0.65) = 1.0 → mid_cov = 0.60*1.0*1.0 = 0.60.
        # Golden: mid coverage never exceeds 0.60 (w_mid cap).
        L = _layers(coverage=1.0, density=1.0, precip=0.0)
        assert L.coverage[BAND_MID] == pytest.approx(0.60, abs=1e-9)

    def test_mid_coverage_never_exceeds_w_mid(self):
        # mid_cov = w_mid * cov * mid_present ≤ 0.60 always (since cov ≤ 1, mid_present ≤ 1).
        for cov in np.linspace(0.0, 1.0, 20):
            L = _layers(coverage=float(cov), density=float(cov), precip=0.0)
            assert L.coverage[BAND_MID] <= 0.60 + 1e-9

    def test_high_density_always_capped(self):
        # high_den = den_high * (0.6 + 0.4*cov); den_high = 0.30.
        # max: 0.30 * (0.6 + 0.4*1.0) = 0.30 * 1.0 = 0.30.
        # So high band density is always ≤ 0.30.
        for cov in (0.0, 0.3, 0.6, 0.9, 1.0):
            L = _layers(coverage=cov, density=1.0, precip=0.0)
            assert L.density[BAND_HIGH] <= 0.30 + 1e-9, (
                f"High band density {L.density[BAND_HIGH]:.4f} exceeds cap 0.30 at cov={cov}"
            )

    def test_low_coverage_boosted_by_precip(self):
        # low_cov = clip(cov + 0.35*pre, 0, 1).
        # At cov=0.5, pre=0.6: low_cov = clip(0.5 + 0.35*0.6, 0, 1) = clip(0.71, 0,1) = 0.71.
        L = _layers(coverage=0.5, density=0.5, precip=0.6)
        assert L.coverage[BAND_LOW] == pytest.approx(0.71, abs=1e-9)

    def test_low_coverage_clamped_at_one(self):
        # cov=1.0, pre=1.0: low_cov = clip(1.0 + 0.35, 0, 1) = 1.0.
        L = _layers(coverage=1.0, density=1.0, precip=1.0)
        assert L.coverage[BAND_LOW] == pytest.approx(1.0, abs=1e-9)


# ===========================================================================
# 6. cloud_layers — boundary inputs
# ===========================================================================


class TestCloudLayersBoundaries:
    """Pin genus + finite/positive-valued output at extremes."""

    @pytest.mark.parametrize("regime", [Regime.HIGH_PRESSURE, Regime.MIXED, Regime.FRONTAL])
    def test_all_zero_inputs_per_regime(self, regime):
        L = cloud_layers(0.0, 0.0, 0.0, regime, CFG)
        assert isinstance(L, CloudLayers)
        for arr in (L.base_altitude_m, L.thickness_m, L.coverage, L.density, L.detail_scale):
            assert np.all(np.isfinite(arr))
        # Pin genus at zero inputs — should map to the "degenerate" case.
        # In all regimes: cov=0 ≤ 0.18 → STRATUS for low; ALTOCUMULUS for mid;
        # cov=0 ≤ 0.55 → CIRRUS for high.
        assert L.genus_high is CloudGenus.CIRRUS
        assert L.genus_mid is CloudGenus.ALTOCUMULUS
        assert L.genus_low is CloudGenus.STRATUS

    @pytest.mark.parametrize("regime", [Regime.HIGH_PRESSURE, Regime.MIXED, Regime.FRONTAL])
    def test_all_one_inputs_per_regime(self, regime):
        L = cloud_layers(1.0, 1.0, 1.0, regime, CFG)
        assert isinstance(L, CloudLayers)
        for arr in (L.base_altitude_m, L.thickness_m, L.coverage, L.density, L.detail_scale):
            assert np.all(np.isfinite(arr))
            assert np.all(arr > 0.0), f"Expected all positive at max inputs: {arr}"
        # At max precip (1.0 > 0.45) → CUMULONIMBUS in all regimes.
        assert L.genus_low is CloudGenus.CUMULONIMBUS
        # At max coverage (1.0 > 0.55) → CIRROSTRATUS high.
        assert L.genus_high is CloudGenus.CIRROSTRATUS
        # At max cov+den (both > 0.55) → ALTOSTRATUS mid.
        assert L.genus_mid is CloudGenus.ALTOSTRATUS

    def test_storm_thickness_deeper_than_clear(self):
        # Cumulonimbus should deepen the low slab vs. a rainless frontal overcast.
        clear = cloud_layers(0.95, 0.90, 0.0, Regime.FRONTAL, CFG)
        storm = cloud_layers(0.95, 0.90, 1.0, Regime.FRONTAL, CFG)
        assert storm.thickness_m[BAND_LOW] > clear.thickness_m[BAND_LOW], (
            "Storm (CB) low-slab thickness must exceed clear overcast's"
        )

    def test_extreme_inputs_clipped_gracefully(self):
        # cloud_layers clips inputs to [0,1] before use; negative/overflow inputs
        # should not produce NaN, inf, or out-of-range coverage/density.
        # (Note: values beyond [0,1] are a caller bug; we pin that clipping works.)
        L = cloud_layers(-5.0, 2.0, -1.0, Regime.MIXED, CFG)
        for arr in (L.coverage, L.density):
            assert np.all(np.isfinite(arr))
            assert np.all(arr >= 0.0) and np.all(arr <= 1.0)


# ===========================================================================
# 7. Determinism — cloud_layers and classify_genus are pure functions
# ===========================================================================


class TestDeterminismEdges:
    """Determinism cases not covered by the existing test_weather_clouds.py suite."""

    def test_classify_genus_deterministic_all_regimes(self):
        inputs = [(0.08, 0.30, 0.0), (0.50, 0.50, 0.25), (0.98, 0.95, 0.95)]
        for cov, den, pre in inputs:
            for regime in (Regime.HIGH_PRESSURE, Regime.MIXED, Regime.FRONTAL):
                a = classify_genus(cov, den, pre, regime)
                b = classify_genus(cov, den, pre, regime)
                assert a == b

    def test_cloud_layers_deterministic_at_boundaries(self):
        for cov, pre in ((0.0, 0.0), (1.0, 1.0), (0.5, 0.5)):
            a = cloud_layers(cov, cov, pre, Regime.FRONTAL, CFG)
            b = cloud_layers(cov, cov, pre, Regime.FRONTAL, CFG)
            np.testing.assert_array_equal(a.base_altitude_m, b.base_altitude_m)
            np.testing.assert_array_equal(a.coverage, b.coverage)
            np.testing.assert_array_equal(a.density, b.density)
            np.testing.assert_array_equal(a.thickness_m, b.thickness_m)
            assert a.genus_low is b.genus_low
            assert a.genus_mid is b.genus_mid
            assert a.genus_high is b.genus_high
