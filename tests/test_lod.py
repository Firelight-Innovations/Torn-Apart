"""
tests/test_lod.py — Characterization / golden-master tests for core/lod.py.

Pins the CURRENT behaviour of LODPolicy.band_for without fixing anything.
Suspicious behaviour is noted in comments and reported to the caller.

Conventions mirror tests/test_math3d.py:
  - Class-per-concern, method-per-case.
  - No magic numbers duplicated from source: default bands are read directly
    from LODPolicy.bands and assertions are written relative to those values.
  - Pure-function, headless — no panda3d / fire_engine.world / lighting.gpu.
"""

from __future__ import annotations

import math

import pytest

from fire_engine.core.lod import LODPolicy

# ---------------------------------------------------------------------------
# Helpers — read default bands from source once; tests refer to these names.
# ---------------------------------------------------------------------------

_DEFAULT_POLICY = LODPolicy()
_BANDS = _DEFAULT_POLICY.bands  # (32.0, 96.0, 192.0, 512.0)
_LAST_BAND_IDX = len(_BANDS)  # 4  — "beyond last threshold"


# ===========================================================================
# Default-bands: below / above / at each threshold
# ===========================================================================


class TestDefaultBandsBelow:
    """Distances strictly below the first threshold → band 0."""

    def test_zero_is_band_0(self):
        assert _DEFAULT_POLICY.band_for(0.0) == 0

    def test_one_meter_is_band_0(self):
        assert _DEFAULT_POLICY.band_for(1.0) == 0

    def test_just_below_first_threshold_is_band_0(self):
        # One float ULP below bands[0] is still band 0
        import struct

        bits = struct.unpack("Q", struct.pack("d", _BANDS[0]))[0]
        just_below = struct.unpack("d", struct.pack("Q", bits - 1))[0]
        assert _DEFAULT_POLICY.band_for(just_below) == 0

    def test_midpoint_between_band0_and_band1(self):
        mid = (_BANDS[0] + _BANDS[1]) / 2.0
        assert _DEFAULT_POLICY.band_for(mid) == 1


class TestDefaultBandsBoundaries:
    """
    Pin boundary inclusion: band_for uses strict '<', so each threshold value
    itself falls into the *upper* band (i.e. distance == threshold → band i+1).
    """

    def test_exactly_on_first_threshold_is_band_1(self):
        # distance == bands[0] → NOT band 0 → band 1
        assert _DEFAULT_POLICY.band_for(_BANDS[0]) == 1

    def test_exactly_on_second_threshold_is_band_2(self):
        assert _DEFAULT_POLICY.band_for(_BANDS[1]) == 2

    def test_exactly_on_third_threshold_is_band_3(self):
        assert _DEFAULT_POLICY.band_for(_BANDS[2]) == 3

    def test_exactly_on_last_threshold_is_last_band_idx(self):
        # distance == bands[-1] falls into the "beyond last" band
        assert _DEFAULT_POLICY.band_for(_BANDS[-1]) == _LAST_BAND_IDX


class TestDefaultBandsBetweenThresholds:
    """Distances between adjacent thresholds."""

    def test_between_band0_and_band1(self):
        d = (_BANDS[0] + _BANDS[1]) / 2.0
        assert _DEFAULT_POLICY.band_for(d) == 1

    def test_between_band1_and_band2(self):
        d = (_BANDS[1] + _BANDS[2]) / 2.0
        assert _DEFAULT_POLICY.band_for(d) == 2

    def test_between_band2_and_band3(self):
        d = (_BANDS[2] + _BANDS[3]) / 2.0
        assert _DEFAULT_POLICY.band_for(d) == 3


class TestDefaultBandsBeyondLast:
    """Distances beyond the last threshold → len(bands)."""

    def test_just_above_last_threshold_is_last_band_idx(self):
        assert _DEFAULT_POLICY.band_for(_BANDS[-1] + 0.001) == _LAST_BAND_IDX

    def test_very_large_distance_is_last_band_idx(self):
        assert _DEFAULT_POLICY.band_for(1_000_000.0) == _LAST_BAND_IDX

    def test_inf_is_last_band_idx(self):
        # math.inf < any finite threshold is False, so falls through to len(bands)
        assert _DEFAULT_POLICY.band_for(math.inf) == _LAST_BAND_IDX


# ===========================================================================
# Edge inputs: negative distance, -inf
# ===========================================================================


class TestEdgeDistances:
    """
    Pin current behaviour for out-of-range inputs (negative, -inf).
    These are not documented as valid; we capture what actually happens.

    SUSPECTED BUG: negative distances return band 0, which is the "full detail"
    band. This is probably correct by accident (negative < threshold), but the
    docstring says "distance from camera", implying non-negative values only.
    No explicit guard exists — callers could silently get band 0 on bad input.
    """

    def test_negative_small_is_band_0(self):
        # -1.0 < bands[0] → band 0 (no guard)
        assert _DEFAULT_POLICY.band_for(-1.0) == 0

    def test_negative_large_is_band_0(self):
        assert _DEFAULT_POLICY.band_for(-9999.0) == 0

    def test_neg_inf_is_band_0(self):
        # -inf < any finite threshold → band 0
        assert _DEFAULT_POLICY.band_for(-math.inf) == 0


# ===========================================================================
# Determinism
# ===========================================================================


class TestDeterminism:
    """Same distance → same band, every call, no state mutation."""

    def test_same_distance_same_result_repeated(self):
        policy = LODPolicy()
        d = (_BANDS[0] + _BANDS[1]) / 2.0
        results = [policy.band_for(d) for _ in range(100)]
        assert len(set(results)) == 1

    def test_independent_instances_agree(self):
        p1 = LODPolicy()
        p2 = LODPolicy()
        for d in [0.0, 10.0, 50.0, 100.0, 300.0, 600.0]:
            assert p1.band_for(d) == p2.band_for(d)


# ===========================================================================
# Custom bands
# ===========================================================================


class TestSingleEntryBands:
    """bands with exactly one threshold → bands 0 or 1."""

    def test_below_single_threshold_is_band_0(self):
        p = LODPolicy(bands=(100.0,))
        assert p.band_for(50.0) == 0

    def test_at_single_threshold_is_band_1(self):
        p = LODPolicy(bands=(100.0,))
        assert p.band_for(100.0) == 1

    def test_above_single_threshold_is_band_1(self):
        p = LODPolicy(bands=(100.0,))
        assert p.band_for(200.0) == 1


class TestTwoEntryBands:
    """bands with two thresholds — three possible bands: 0, 1, 2."""

    def test_below_first_is_0(self):
        p = LODPolicy(bands=(10.0, 50.0))
        assert p.band_for(5.0) == 0

    def test_between_is_1(self):
        p = LODPolicy(bands=(10.0, 50.0))
        assert p.band_for(25.0) == 1

    def test_above_second_is_2(self):
        p = LODPolicy(bands=(10.0, 50.0))
        assert p.band_for(75.0) == 2

    def test_exactly_second_threshold_is_2(self):
        # boundary falls in upper band (strict <)
        p = LODPolicy(bands=(10.0, 50.0))
        assert p.band_for(50.0) == 2


class TestMonotonicBandFor:
    """band_for must be monotonically non-decreasing as distance grows."""

    def test_monotone_over_sweep(self):
        p = LODPolicy(bands=(20.0, 80.0, 200.0))
        prev = p.band_for(0.0)
        for d in range(1, 500):
            cur = p.band_for(float(d))
            assert cur >= prev, (
                f"band_for not monotone: band_for({d - 1}) = {prev} > band_for({d}) = {cur}"
            )
            prev = cur


# ===========================================================================
# Edge construction: empty bands, unsorted bands, duplicate thresholds
# ===========================================================================


class TestEdgeConstruction:
    """
    Pin whatever LODPolicy does with unusual band tuples.
    LODPolicy is a frozen dataclass with no __post_init__ validation,
    so these all construct without error.
    """

    def test_empty_bands_always_returns_0(self):
        """
        With no thresholds the for-loop never executes and len(bands)=0 is
        returned for every distance.  This means band_for always returns 0,
        which is the "full detail" band — semantically reasonable but untested
        by callers who expect a non-trivial LOD policy.

        REPORTED: no validation guard; constructing LODPolicy(bands=()) silently
        produces a degenerate policy that assigns full detail to every object.
        """
        p = LODPolicy(bands=())
        assert p.band_for(0.0) == 0
        assert p.band_for(999.0) == 0

    def test_unsorted_bands_produces_wrong_results(self):
        """
        band_for iterates linearly and returns on the first threshold that
        exceeds the distance.  With unsorted bands the returned band index is
        the position of the first matching threshold in the tuple, NOT the
        logically correct band.

        Example: bands=(100.0, 50.0), distance=60.0
          → 60 < 100? yes → returns 0  (wrong; 60 is between 50 and 100)

        SUSPECTED BUG: no assertion in __post_init__ enforces sorted order.
        Callers who pass unsorted bands will get silently wrong band indices.
        """
        p = LODPolicy(bands=(100.0, 50.0))
        # Pin the current (wrong) result without asserting it is correct
        result = p.band_for(60.0)
        assert result == 0  # 60 < 100 → first threshold matches → band 0

    def test_duplicate_thresholds_skips_a_band(self):
        """
        bands=(50.0, 50.0, 200.0): distance exactly 50 returns band 1 (hits
        the second threshold in the loop at index 1, not index 0, because the
        first 50.0 is not strictly less-than 50.0).

        Wait — actually distance=50.0 < 50.0 is False for the first entry, so
        the loop moves to i=1 where 50.0 < 50.0 is also False, then i=2 where
        50.0 < 200.0 is True → returns 2.  Band 1 is effectively unreachable
        for any distance equal to 50.0.

        Pin this as-is; there is no validation to prevent it.
        """
        p = LODPolicy(bands=(50.0, 50.0, 200.0))
        # distance just below first dup → band 0
        assert p.band_for(49.9) == 0
        # distance exactly on dup threshold → band 2 (band 1 is skipped)
        assert p.band_for(50.0) == 2
        # distance between dup and last → band 2
        assert p.band_for(100.0) == 2
        # distance above last → band 3
        assert p.band_for(300.0) == 3

    def test_frozen_dataclass_rejects_mutation(self):
        """LODPolicy is a frozen dataclass; mutation raises FrozenInstanceError."""
        import dataclasses

        p = LODPolicy()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            p.bands = (1.0, 2.0)  # type: ignore[misc]
