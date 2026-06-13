"""
tests/test_weather_clouds.py — WMO cloud genera classification + layers (M9).

No panda3d imports anywhere in this file (weather/ is a headless package).

Coverage
--------
- classify_genus is deterministic and a pure function of the sampled fields.
- Each CellKind's characteristic sample maps to the expected genus family
  (THUNDERSTORM→CUMULONIMBUS, CLOUD_BANK→stratus family, SHOWER→cumulus/stratus,
  HIGH_PRESSURE residual→cirrus high).
- classify_genus vectorises (array in → array out, matching the scalar path).
- cloud_layers params are finite, ordered by altitude (high > mid > low), have
  the right shapes, and are continuous in the inputs (no NaN, no jumps).
- No panda3d import leaks into fire_engine/weather/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fire_engine.core import load_config
from fire_engine.weather.cells import Regime
from fire_engine.weather.clouds import (
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
# Determinism / purity
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_classify_repeatable(self):
        a = classify_genus(0.6, 0.7, 0.0, Regime.MIXED)
        b = classify_genus(0.6, 0.7, 0.0, Regime.MIXED)
        assert a == b

    def test_layers_repeatable(self):
        a = cloud_layers(0.6, 0.7, 0.2, Regime.MIXED, CFG)
        b = cloud_layers(0.6, 0.7, 0.2, Regime.MIXED, CFG)
        assert a.genus_low == b.genus_low
        np.testing.assert_array_equal(a.coverage, b.coverage)
        np.testing.assert_array_equal(a.density, b.density)
        np.testing.assert_array_equal(a.base_altitude_m, b.base_altitude_m)

    def test_pure_of_fields_only(self):
        # No hidden state: classifying B between two A calls doesn't change A.
        a1 = classify_genus(0.9, 0.9, 0.8, Regime.FRONTAL)
        _ = classify_genus(0.1, 0.1, 0.0, Regime.HIGH_PRESSURE)
        a2 = classify_genus(0.9, 0.9, 0.8, Regime.FRONTAL)
        assert a1 == a2


# ---------------------------------------------------------------------------
# CellKind → genus family
# ---------------------------------------------------------------------------

class TestGenusFamily:
    def test_thunderstorm_is_cumulonimbus(self):
        # A THUNDERSTORM core: heavy precip, full coverage/density.
        _, _, low = classify_genus(0.98, 0.95, 0.95, Regime.FRONTAL)
        assert low is CloudGenus.CUMULONIMBUS

    def test_cloud_bank_is_stratus_family(self):
        # A CLOUD_BANK: high coverage/density, NO precip → low flat deck.
        _, _, low = classify_genus(0.85, 0.70, 0.0, Regime.MIXED)
        assert low in (CloudGenus.STRATOCUMULUS, CloudGenus.STRATUS)

    def test_shower_is_rain_layer(self):
        # A SHOWER: moderate precip below the storm threshold → STRATUS rain
        # layer (the nimbostratus role), not a cumulonimbus tower.
        _, _, low = classify_genus(0.7, 0.7, 0.30, Regime.MIXED)
        assert low is CloudGenus.STRATUS

    def test_fair_weather_cumulus(self):
        # Partial cover, no rain → fair-weather CUMULUS heaps low.
        _, _, low = classify_genus(0.35, 0.45, 0.0, Regime.MIXED)
        assert low is CloudGenus.CUMULUS

    def test_high_pressure_residual_is_cirrus(self):
        # HIGH_PRESSURE near-clear: the high band is wispy CIRRUS.
        high, _, _ = classify_genus(0.08, 0.30, 0.0, Regime.HIGH_PRESSURE)
        assert high is CloudGenus.CIRRUS

    def test_frontal_overcast_stacks(self):
        # FRONTAL overcast stacks: cirrostratus veil / altostratus / strato deck.
        high, mid, low = classify_genus(0.85, 0.80, 0.0, Regime.FRONTAL)
        assert high is CloudGenus.CIRROSTRATUS
        assert mid is CloudGenus.ALTOSTRATUS
        assert low is CloudGenus.STRATOCUMULUS


# ---------------------------------------------------------------------------
# Vectorisation
# ---------------------------------------------------------------------------

class TestVectorised:
    def test_array_matches_scalar(self):
        cov = np.array([0.08, 0.35, 0.85, 0.98])
        den = np.array([0.30, 0.45, 0.80, 0.95])
        pre = np.array([0.00, 0.00, 0.00, 0.95])
        high, mid, low = classify_genus(cov, den, pre, Regime.FRONTAL)
        assert high.shape == low.shape == (4,)
        for i in range(4):
            sh, sm, sl = classify_genus(
                float(cov[i]), float(den[i]), float(pre[i]), Regime.FRONTAL)
            assert high[i] is sh
            assert mid[i] is sm
            assert low[i] is sl

    def test_array_storm_tower(self):
        cov = np.full(5, 0.95)
        den = np.full(5, 0.95)
        pre = np.linspace(0.0, 1.0, 5)
        _, _, low = classify_genus(cov, den, pre, Regime.FRONTAL)
        assert low[-1] is CloudGenus.CUMULONIMBUS   # full precip → tower
        assert low[0] is not CloudGenus.CUMULONIMBUS  # no precip → not a tower


# ---------------------------------------------------------------------------
# Layer params: shape, finiteness, ordering
# ---------------------------------------------------------------------------

class TestLayers:
    def test_shapes(self):
        L = cloud_layers(0.6, 0.6, 0.3, Regime.MIXED, CFG)
        assert isinstance(L, CloudLayers)
        for arr in (L.base_altitude_m, L.thickness_m, L.coverage,
                    L.density, L.detail_scale):
            assert arr.shape == (3,)
            assert np.all(np.isfinite(arr))

    def test_altitudes_ordered_high_above_low(self):
        L = cloud_layers(0.7, 0.7, 0.0, Regime.MIXED, CFG)
        # Band 0=high, 1=mid, 2=low → strictly decreasing altitude.
        assert L.base_altitude_m[BAND_HIGH] > L.base_altitude_m[BAND_MID]
        assert L.base_altitude_m[BAND_MID] > L.base_altitude_m[BAND_LOW]

    @pytest.mark.parametrize("cov,den,pre,regime", [
        (0.0, 0.0, 0.0, Regime.HIGH_PRESSURE),
        (0.08, 0.30, 0.0, Regime.HIGH_PRESSURE),
        (0.40, 0.52, 0.0, Regime.MIXED),
        (0.75, 0.72, 0.0, Regime.FRONTAL),
        (0.95, 0.90, 0.5, Regime.FRONTAL),
        (1.0, 1.0, 1.0, Regime.FRONTAL),
    ])
    def test_weights_in_range(self, cov, den, pre, regime):
        L = cloud_layers(cov, den, pre, regime, CFG)
        assert np.all(L.coverage >= 0.0) and np.all(L.coverage <= 1.0)
        assert np.all(L.density >= 0.0) and np.all(L.density <= 1.0)
        assert np.all(L.detail_scale > 0.0)

    def test_cirrus_always_thin(self):
        # High band density is always capped low (ice cloud is thin), for any
        # input — even a full overcast.
        for cov in (0.0, 0.3, 0.6, 1.0):
            L = cloud_layers(cov, 1.0, 0.0, Regime.FRONTAL, CFG)
            assert L.density[BAND_HIGH] <= 0.5

    def test_cirrus_present_fair_weather(self):
        # The high band has a non-zero residual even when the sky is clear.
        L = cloud_layers(0.0, 0.0, 0.0, Regime.HIGH_PRESSURE, CFG)
        assert L.coverage[BAND_HIGH] > 0.0

    def test_clear_low_deck_thin(self):
        L = cloud_layers(0.08, 0.30, 0.0, Regime.HIGH_PRESSURE, CFG)
        assert L.coverage[BAND_LOW] < 0.3

    def test_storm_deepens_low_slab(self):
        clear = cloud_layers(0.95, 0.90, 0.0, Regime.FRONTAL, CFG)
        storm = cloud_layers(0.95, 0.90, 1.0, Regime.FRONTAL, CFG)
        # A cumulonimbus tower has a deeper low slab than a rainless overcast.
        assert storm.thickness_m[BAND_LOW] > clear.thickness_m[BAND_LOW]


# ---------------------------------------------------------------------------
# Continuity (no jumps in the continuous layer weights)
# ---------------------------------------------------------------------------

class TestContinuity:
    def test_coverage_sweep_continuous(self):
        # Sweep sampled coverage; per-band coverage must move smoothly (no
        # discontinuity larger than a small step) so drifting cells don't pop.
        cov = np.linspace(0.0, 1.0, 400)
        low = np.array([
            cloud_layers(float(c), 0.6, 0.0, Regime.MIXED, CFG).coverage[BAND_LOW]
            for c in cov
        ])
        assert np.all(np.isfinite(low))
        assert np.max(np.abs(np.diff(low))) < 0.05

    def test_precip_sweep_continuous(self):
        pre = np.linspace(0.0, 1.0, 400)
        thick = np.array([
            cloud_layers(0.9, 0.9, float(p), Regime.FRONTAL, CFG).thickness_m[BAND_LOW]
            for p in pre
        ])
        assert np.all(np.isfinite(thick))
        # Thickness grows smoothly with precip (bounded per-step change).
        span = thick.max() - thick.min()
        assert span > 0.0
        assert np.max(np.abs(np.diff(thick))) < 0.05 * span


# ---------------------------------------------------------------------------
# Headless guarantee
# ---------------------------------------------------------------------------

def test_no_panda3d_import_in_weather_clouds():
    src = (Path(__file__).resolve().parents[1]
           / "fire_engine" / "weather" / "clouds.py").read_text(encoding="utf-8")
    assert "panda3d" not in src
    assert "import panda3d" not in src
