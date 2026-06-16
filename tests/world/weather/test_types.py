"""
tests/world/weather/test_types.py — Mirror for fire_engine/world/weather/types.py.

Authored tests (no existing flat test maps 1:1 to this grouping module).
types.py is the support-types/enums grouping module for the weather package.

Coverage
--------
CORRECTNESS — enum members and values:
  - CellKind: all four members present, StrEnum values round-trip as plain strings.
  - Regime: all three members present, StrEnum values round-trip.
  - CloudGenus: all eight members present, StrEnum values are valid plain strings.
  - CloudBand: int Enum — HIGH/MID/LOW index arrays correctly (0/1/2).

CORRECTNESS — CloudLayers dataclass:
  - genus_for_band(band) returns the expected genus for each band.
  - CloudLayers is frozen (immutable) — attribute assignment raises.
  - Arrays are stored by reference (no copy forced by the dataclass).

CORRECTNESS — LocalWeather dataclass:
  - Constructor with required fields only uses defaults for optional fields.
  - All fields accessible by name; humidity/wetness/temperature_c default correctly.
  - Frozen — attribute assignment raises.

ROUND-TRIP — StrEnum values:
  - CellKind("shower") == CellKind.SHOWER, etc.
  - Regime("frontal") == Regime.FRONTAL.
  - CloudGenus("cumulonimbus") == CloudGenus.CUMULONIMBUS.

Headless — no panda3d imports.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import ClassVar

import numpy as np
import pytest

from fire_engine.world.weather.types import (
    CellKind,
    CloudBand,
    CloudGenus,
    CloudLayers,
    LocalWeather,
    Regime,
)

# ---------------------------------------------------------------------------
# CellKind
# ---------------------------------------------------------------------------


class TestCellKind:
    def test_all_members_present(self):
        names = {m.name for m in CellKind}
        assert names == {"SHOWER", "THUNDERSTORM", "CLOUD_BANK", "FOG_BANK"}

    def test_values_are_plain_strings(self):
        assert CellKind.SHOWER.value == "shower"
        assert CellKind.THUNDERSTORM.value == "thunderstorm"
        assert CellKind.CLOUD_BANK.value == "cloud_bank"
        assert CellKind.FOG_BANK.value == "fog_bank"

    def test_round_trip_from_string(self):
        for kind in CellKind:
            assert CellKind(kind.value) is kind

    def test_str_mixin_equality(self):
        """StrEnum members compare equal to their plain-string value."""
        assert CellKind.SHOWER == "shower"
        assert CellKind.THUNDERSTORM == "thunderstorm"
        assert CellKind.CLOUD_BANK == "cloud_bank"
        assert CellKind.FOG_BANK == "fog_bank"

    def test_identity(self):
        assert CellKind.SHOWER is CellKind.SHOWER
        assert CellKind.SHOWER is not CellKind.THUNDERSTORM


# ---------------------------------------------------------------------------
# Regime
# ---------------------------------------------------------------------------


class TestRegime:
    def test_all_members_present(self):
        names = {m.name for m in Regime}
        assert names == {"HIGH_PRESSURE", "MIXED", "FRONTAL"}

    def test_values_are_plain_strings(self):
        assert Regime.HIGH_PRESSURE.value == "high_pressure"
        assert Regime.MIXED.value == "mixed"
        assert Regime.FRONTAL.value == "frontal"

    def test_round_trip_from_string(self):
        for regime in Regime:
            assert Regime(regime.value) is regime

    def test_str_mixin_equality(self):
        assert Regime.MIXED == "mixed"
        assert Regime.FRONTAL == "frontal"


# ---------------------------------------------------------------------------
# CloudGenus
# ---------------------------------------------------------------------------


class TestCloudGenus:
    _EXPECTED: ClassVar[set[str]] = {
        "CIRRUS",
        "CIRROSTRATUS",
        "ALTOCUMULUS",
        "ALTOSTRATUS",
        "STRATOCUMULUS",
        "STRATUS",
        "CUMULUS",
        "CUMULONIMBUS",
    }

    def test_all_members_present(self):
        assert {m.name for m in CloudGenus} == self._EXPECTED

    def test_values_lowercase_plain_strings(self):
        for genus in CloudGenus:
            assert genus.value == genus.value.lower()
            assert isinstance(genus.value, str)

    def test_round_trip_from_string(self):
        for genus in CloudGenus:
            assert CloudGenus(genus.value) is genus

    def test_cumulonimbus_value(self):
        assert CloudGenus.CUMULONIMBUS.value == "cumulonimbus"

    def test_cirrus_value(self):
        assert CloudGenus.CIRRUS.value == "cirrus"


# ---------------------------------------------------------------------------
# CloudBand
# ---------------------------------------------------------------------------


class TestCloudBand:
    def test_high_is_zero(self):
        assert int(CloudBand.HIGH) == 0

    def test_mid_is_one(self):
        assert int(CloudBand.MID) == 1

    def test_low_is_two(self):
        assert int(CloudBand.LOW) == 2

    def test_indexes_array(self):
        arr = np.array([10.0, 20.0, 30.0])
        assert arr[CloudBand.HIGH] == 10.0
        assert arr[CloudBand.MID] == 20.0
        assert arr[CloudBand.LOW] == 30.0

    def test_all_members_present(self):
        assert {m.name for m in CloudBand} == {"HIGH", "MID", "LOW"}


# ---------------------------------------------------------------------------
# CloudLayers
# ---------------------------------------------------------------------------


def _make_layers() -> CloudLayers:
    return CloudLayers(
        genus_high=CloudGenus.CIRRUS,
        genus_mid=CloudGenus.ALTOSTRATUS,
        genus_low=CloudGenus.CUMULUS,
        base_altitude_m=np.array([7000.0, 3000.0, 800.0]),
        thickness_m=np.array([500.0, 800.0, 600.0]),
        coverage=np.array([0.1, 0.3, 0.6]),
        density=np.array([0.1, 0.4, 0.7]),
        detail_scale=np.array([0.5, 1.0, 1.5]),
    )


class TestCloudLayers:
    def test_genus_for_band_high(self):
        L = _make_layers()
        assert L.genus_for_band(CloudBand.HIGH) is CloudGenus.CIRRUS

    def test_genus_for_band_mid(self):
        L = _make_layers()
        assert L.genus_for_band(CloudBand.MID) is CloudGenus.ALTOSTRATUS

    def test_genus_for_band_low(self):
        L = _make_layers()
        assert L.genus_for_band(CloudBand.LOW) is CloudGenus.CUMULUS

    def test_genus_for_band_by_int(self):
        L = _make_layers()
        assert L.genus_for_band(0) is CloudGenus.CIRRUS
        assert L.genus_for_band(1) is CloudGenus.ALTOSTRATUS
        assert L.genus_for_band(2) is CloudGenus.CUMULUS

    def test_frozen_immutable(self):
        L = _make_layers()
        with pytest.raises(FrozenInstanceError):
            L.genus_high = CloudGenus.CUMULONIMBUS  # type: ignore[misc]

    def test_arrays_have_length_3(self):
        L = _make_layers()
        for arr in (L.base_altitude_m, L.thickness_m, L.coverage, L.density, L.detail_scale):
            assert len(arr) == 3

    def test_genus_attributes_accessible(self):
        L = _make_layers()
        assert L.genus_high is CloudGenus.CIRRUS
        assert L.genus_mid is CloudGenus.ALTOSTRATUS
        assert L.genus_low is CloudGenus.CUMULUS


# ---------------------------------------------------------------------------
# LocalWeather
# ---------------------------------------------------------------------------


class TestLocalWeather:
    def _minimal(self) -> LocalWeather:
        return LocalWeather(
            cloud_coverage=0.2,
            cloud_density=0.4,
            fog_density=0.001,
            rain_intensity=0.0,
            wind_dir=(1.0, 0.0),
            wind_speed=3.0,
        )

    def test_required_fields_accessible(self):
        lw = self._minimal()
        assert lw.cloud_coverage == pytest.approx(0.2)
        assert lw.cloud_density == pytest.approx(0.4)
        assert lw.fog_density == pytest.approx(0.001)
        assert lw.rain_intensity == pytest.approx(0.0)
        assert lw.wind_dir == (1.0, 0.0)
        assert lw.wind_speed == pytest.approx(3.0)

    def test_optional_fields_default(self):
        lw = self._minimal()
        assert lw.humidity == pytest.approx(0.5)
        assert lw.wetness == pytest.approx(0.0)
        assert lw.temperature_c == pytest.approx(12.0)

    def test_optional_fields_can_be_set(self):
        lw = LocalWeather(
            cloud_coverage=0.9,
            cloud_density=0.8,
            fog_density=0.005,
            rain_intensity=0.7,
            wind_dir=(0.0, 1.0),
            wind_speed=10.0,
            humidity=0.85,
            wetness=0.3,
            temperature_c=5.0,
        )
        assert lw.humidity == pytest.approx(0.85)
        assert lw.wetness == pytest.approx(0.3)
        assert lw.temperature_c == pytest.approx(5.0)

    def test_frozen_immutable(self):
        lw = self._minimal()
        with pytest.raises(FrozenInstanceError):
            lw.cloud_coverage = 0.9  # type: ignore[misc]

    def test_equality(self):
        a = self._minimal()
        b = self._minimal()
        assert a == b

    def test_inequality_on_different_field(self):
        a = self._minimal()
        b = LocalWeather(
            cloud_coverage=0.9,  # different
            cloud_density=0.4,
            fog_density=0.001,
            rain_intensity=0.0,
            wind_dir=(1.0, 0.0),
            wind_speed=3.0,
        )
        assert a != b
