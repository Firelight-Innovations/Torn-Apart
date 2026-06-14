"""
tests/test_weather_classify.py — Golden-master / characterisation tests for
classify() and WeatherType.

DO NOT fix bugs found here; only pin current behaviour.

Coverage
--------
- WeatherType enum members, str mixin, value equality.
- classify() priority order: fog > storm > rain > overcast > cloudy > clear.
- All threshold boundaries: both sides of every literal in classify().
- All-zero (bare) LocalWeather → CLEAR.
- Representative combined / realistic states.

No panda3d imports anywhere in this file.
"""

from __future__ import annotations

import pytest

from fire_engine.world.weather.classify import WeatherType, classify
from fire_engine.world.weather.system import LocalWeather

# ---------------------------------------------------------------------------
# Threshold constants (read directly from the source; if the module ever
# exposes them as public names we can import instead).
# ---------------------------------------------------------------------------
_FOG_THRESHOLD = 0.008  # fog_density > this → FOG
_STORM_RAIN = 0.55  # rain_intensity > this (AND wind > _STORM_WIND) → STORM
_STORM_WIND = 9.0  # wind_speed > this (AND rain > _STORM_RAIN) → STORM
_RAIN_THRESHOLD = 0.05  # rain_intensity > this → RAIN
_OVERCAST_COV = 0.7  # cloud_coverage > this → OVERCAST
_CLOUDY_COV = 0.3  # cloud_coverage > this → CLOUDY

# A neutral wind direction used wherever direction is irrelevant.
_WIND_DIR = (1.0, 0.0)


def _lw(
    cloud_coverage: float = 0.0,
    cloud_density: float = 0.0,
    fog_density: float = 0.0,
    rain_intensity: float = 0.0,
    wind_dir: tuple[float, float] = _WIND_DIR,
    wind_speed: float = 0.0,
) -> LocalWeather:
    """Minimal LocalWeather factory (optional fields use dataclass defaults)."""
    return LocalWeather(
        cloud_coverage=cloud_coverage,
        cloud_density=cloud_density,
        fog_density=fog_density,
        rain_intensity=rain_intensity,
        wind_dir=wind_dir,
        wind_speed=wind_speed,
    )


# ===========================================================================
# WeatherType enum
# ===========================================================================


class TestWeatherTypeEnum:
    def test_all_members_present(self):
        members = {m.name for m in WeatherType}
        assert members == {"CLEAR", "CLOUDY", "OVERCAST", "FOG", "RAIN", "STORM"}

    def test_values_are_legacy_strings(self):
        """Values must not change — saves and WeatherChangedEvent use them."""
        assert WeatherType.CLEAR.value == "clear"
        assert WeatherType.CLOUDY.value == "cloudy"
        assert WeatherType.OVERCAST.value == "overcast"
        assert WeatherType.FOG.value == "fog"
        assert WeatherType.RAIN.value == "rain"
        assert WeatherType.STORM.value == "storm"

    def test_str_mixin_str_returns_name_not_value(self):
        """CURRENT BEHAVIOUR (potential bug): str(member) returns
        'WeatherType.CLEAR', NOT the plain .value 'clear'.
        The class is declared as ``class WeatherType(str, Enum)`` so the str
        mixin IS present (``isinstance(WeatherType.CLEAR, str)`` is True and
        ``WeatherType.CLEAR == "clear"`` is True via __eq__), but Python's
        default Enum.__str__ overrides str.__str__ and produces the
        'ClassName.MEMBER' form.  Pin both sides of that gap here.
        """
        # .value IS the plain string
        for member in WeatherType:
            assert member.value == member.value.lower()  # sanity

        # str() is NOT the plain value under current Python/Enum behaviour
        assert str(WeatherType.CLEAR) == "WeatherType.CLEAR"
        assert str(WeatherType.RAIN) == "WeatherType.RAIN"

        # But equality with the plain string still works (str mixin __eq__)
        assert WeatherType.CLEAR == "clear"
        assert WeatherType.RAIN == "rain"
        assert WeatherType.STORM == "storm"

    def test_equality_by_identity(self):
        assert WeatherType.RAIN is WeatherType.RAIN
        assert WeatherType.RAIN != WeatherType.STORM

    def test_membership_by_value(self):
        assert WeatherType("rain") is WeatherType.RAIN
        assert WeatherType("storm") is WeatherType.STORM


# ===========================================================================
# classify() — all-zero baseline
# ===========================================================================


class TestClassifyBaseline:
    def test_all_zero_is_clear(self):
        assert classify(_lw()) is WeatherType.CLEAR

    def test_zero_coverage_non_zero_density_is_clear(self):
        # cloud_density alone has no threshold — coverage drives overcast/cloudy.
        assert classify(_lw(cloud_density=1.0)) is WeatherType.CLEAR


# ===========================================================================
# FOG threshold boundary  (fog_density: > 0.008)
# ===========================================================================


class TestFogThreshold:
    def test_fog_density_exactly_at_threshold_is_not_fog(self):
        # Strict >: at == 0.008 should NOT trigger FOG.
        assert classify(_lw(fog_density=_FOG_THRESHOLD)) is WeatherType.CLEAR

    def test_fog_density_just_below_threshold_is_not_fog(self):
        assert classify(_lw(fog_density=_FOG_THRESHOLD - 1e-9)) is WeatherType.CLEAR

    def test_fog_density_just_above_threshold_is_fog(self):
        assert classify(_lw(fog_density=_FOG_THRESHOLD + 1e-9)) is WeatherType.FOG

    def test_fog_density_well_above_threshold_is_fog(self):
        assert classify(_lw(fog_density=0.05)) is WeatherType.FOG


# ===========================================================================
# FOG priority — wins over storm, rain, overcast, cloudy
# ===========================================================================


class TestFogPriority:
    def test_fog_beats_storm(self):
        """fog_density over threshold + storm-strength rain + wind → FOG."""
        lw = _lw(
            fog_density=_FOG_THRESHOLD + 0.01,
            rain_intensity=_STORM_RAIN + 0.1,
            wind_speed=_STORM_WIND + 1.0,
            cloud_coverage=0.99,
        )
        assert classify(lw) is WeatherType.FOG

    def test_fog_beats_rain(self):
        lw = _lw(
            fog_density=_FOG_THRESHOLD + 0.01,
            rain_intensity=_RAIN_THRESHOLD + 0.1,
        )
        assert classify(lw) is WeatherType.FOG

    def test_fog_beats_overcast(self):
        lw = _lw(
            fog_density=_FOG_THRESHOLD + 0.01,
            cloud_coverage=_OVERCAST_COV + 0.1,
        )
        assert classify(lw) is WeatherType.FOG

    def test_fog_beats_cloudy(self):
        lw = _lw(
            fog_density=_FOG_THRESHOLD + 0.01,
            cloud_coverage=_CLOUDY_COV + 0.1,
        )
        assert classify(lw) is WeatherType.FOG


# ===========================================================================
# STORM threshold boundary  (rain_intensity > 0.55 AND wind_speed > 9.0)
# ===========================================================================


class TestStormThreshold:
    def test_storm_requires_both_rain_and_wind(self):
        """High rain alone without strong wind is RAIN, not STORM."""
        lw = _lw(rain_intensity=_STORM_RAIN + 0.1, wind_speed=_STORM_WIND)
        # wind_speed == 9.0 is NOT > 9.0 → should be RAIN
        assert classify(lw) is WeatherType.RAIN

    def test_storm_requires_both_rain_and_wind2(self):
        """Strong wind alone without high rain is not STORM."""
        lw = _lw(rain_intensity=_STORM_RAIN, wind_speed=_STORM_WIND + 0.1)
        # rain_intensity == 0.55 is NOT > 0.55 → should not be STORM
        # also not > _RAIN_THRESHOLD (0.55 > 0.05) → RAIN
        assert classify(lw) is WeatherType.RAIN

    def test_storm_rain_just_below_threshold(self):
        """rain_intensity = 0.55 (not strictly >) should fall through to RAIN."""
        lw = _lw(rain_intensity=_STORM_RAIN, wind_speed=_STORM_WIND + 1.0)
        assert classify(lw) is WeatherType.RAIN

    def test_storm_wind_just_below_threshold(self):
        """wind_speed = 9.0 (not strictly >) should fall through to RAIN."""
        lw = _lw(rain_intensity=_STORM_RAIN + 0.1, wind_speed=_STORM_WIND)
        assert classify(lw) is WeatherType.RAIN

    def test_storm_both_just_above_threshold(self):
        """Both strictly above → STORM."""
        lw = _lw(
            rain_intensity=_STORM_RAIN + 1e-9,
            wind_speed=_STORM_WIND + 1e-9,
        )
        assert classify(lw) is WeatherType.STORM

    def test_storm_priority_over_rain(self):
        """STORM check comes before RAIN check in the chain."""
        lw = _lw(
            rain_intensity=_STORM_RAIN + 0.1,
            wind_speed=_STORM_WIND + 1.0,
        )
        result = classify(lw)
        assert result is WeatherType.STORM
        assert result is not WeatherType.RAIN


# ===========================================================================
# RAIN threshold boundary  (rain_intensity > 0.05)
# ===========================================================================


class TestRainThreshold:
    def test_rain_intensity_exactly_at_threshold_is_not_rain(self):
        # Strict >: 0.05 exactly should NOT trigger RAIN.
        assert classify(_lw(rain_intensity=_RAIN_THRESHOLD)) is WeatherType.CLEAR

    def test_rain_intensity_just_below_threshold_is_not_rain(self):
        assert classify(_lw(rain_intensity=_RAIN_THRESHOLD - 1e-9)) is WeatherType.CLEAR

    def test_rain_intensity_just_above_threshold_is_rain(self):
        assert classify(_lw(rain_intensity=_RAIN_THRESHOLD + 1e-9)) is WeatherType.RAIN

    def test_rain_priority_over_overcast(self):
        """Rain check comes before overcast in the chain."""
        lw = _lw(rain_intensity=_RAIN_THRESHOLD + 0.1, cloud_coverage=_OVERCAST_COV + 0.1)
        assert classify(lw) is WeatherType.RAIN


# ===========================================================================
# OVERCAST threshold boundary  (cloud_coverage > 0.7)
# ===========================================================================


class TestOvercastThreshold:
    def test_coverage_exactly_at_overcast_threshold_is_not_overcast(self):
        # Strict >: 0.7 exactly should NOT trigger OVERCAST.
        result = classify(_lw(cloud_coverage=_OVERCAST_COV))
        # 0.7 > 0.3 → CLOUDY
        assert result is WeatherType.CLOUDY

    def test_coverage_just_below_overcast_is_cloudy(self):
        result = classify(_lw(cloud_coverage=_OVERCAST_COV - 1e-9))
        assert result is WeatherType.CLOUDY

    def test_coverage_just_above_overcast_is_overcast(self):
        result = classify(_lw(cloud_coverage=_OVERCAST_COV + 1e-9))
        assert result is WeatherType.OVERCAST

    def test_overcast_priority_over_cloudy(self):
        """Overcast check comes before cloudy in the chain."""
        lw = _lw(cloud_coverage=0.85)
        assert classify(lw) is WeatherType.OVERCAST


# ===========================================================================
# CLOUDY threshold boundary  (cloud_coverage > 0.3)
# ===========================================================================


class TestCloudyThreshold:
    def test_coverage_exactly_at_cloudy_threshold_is_not_cloudy(self):
        # Strict >: 0.3 exactly should NOT trigger CLOUDY → falls through to CLEAR.
        assert classify(_lw(cloud_coverage=_CLOUDY_COV)) is WeatherType.CLEAR

    def test_coverage_just_below_cloudy_is_clear(self):
        assert classify(_lw(cloud_coverage=_CLOUDY_COV - 1e-9)) is WeatherType.CLEAR

    def test_coverage_just_above_cloudy_is_cloudy(self):
        assert classify(_lw(cloud_coverage=_CLOUDY_COV + 1e-9)) is WeatherType.CLOUDY


# ===========================================================================
# Realistic combined states
# ===========================================================================


class TestRealisticStates:
    def test_light_drizzle(self):
        """Low rain, no storm wind, heavy cloud → RAIN."""
        lw = _lw(
            cloud_coverage=0.85,
            cloud_density=0.70,
            fog_density=0.003,
            rain_intensity=0.12,
            wind_speed=4.0,
        )
        assert classify(lw) is WeatherType.RAIN

    def test_heavy_thunderstorm(self):
        """Full rain + gale-force wind → STORM."""
        lw = _lw(
            cloud_coverage=0.99,
            cloud_density=0.95,
            fog_density=0.006,
            rain_intensity=0.90,
            wind_speed=15.0,
        )
        assert classify(lw) is WeatherType.STORM

    def test_dense_fog_at_night(self):
        """High fog coefficient (calm, no rain) → FOG regardless of coverage."""
        lw = _lw(
            cloud_coverage=0.50,
            fog_density=0.025,
            rain_intensity=0.0,
            wind_speed=0.5,
        )
        assert classify(lw) is WeatherType.FOG

    def test_clear_sunny_day(self):
        """Low coverage, no precip, no fog → CLEAR."""
        lw = _lw(
            cloud_coverage=0.10,
            cloud_density=0.30,
            fog_density=0.0008,
            rain_intensity=0.0,
            wind_speed=3.0,
        )
        assert classify(lw) is WeatherType.CLEAR

    def test_overcast_no_rain(self):
        """Heavy cloud cover but dry and no fog → OVERCAST."""
        lw = _lw(
            cloud_coverage=0.90,
            cloud_density=0.80,
            fog_density=0.003,
            rain_intensity=0.0,
            wind_speed=5.0,
        )
        assert classify(lw) is WeatherType.OVERCAST

    def test_foggy_storm_fog_wins(self):
        """Even at storm-strength values, fog takes priority."""
        lw = _lw(
            fog_density=0.030,  # >> 0.008
            rain_intensity=0.80,  # >> 0.55
            wind_speed=12.0,  # >> 9.0
            cloud_coverage=0.99,
        )
        assert classify(lw) is WeatherType.FOG
