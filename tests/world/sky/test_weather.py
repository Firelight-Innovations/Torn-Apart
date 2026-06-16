"""
tests/world/sky/test_weather.py — Mirror tests for fire_engine/world/sky/weather.py.

This module is a compatibility shim that re-exports the weather public names
from fire_engine.world.weather into the legacy fire_engine.world.sky.weather
namespace.  Tests verify:
 - All expected names are present and are the canonical objects (identity / type)
 - BLEND_SECONDS and HYSTERESIS_SECONDS are positive floats
 - WeatherType is an enum with CLEAR, RAIN, STORM members
 - WeatherSystem and LocalWeather are the actual classes from world.weather
 - classify is callable and returns a WeatherType

No panda3d imports. All tests headless.
"""

from __future__ import annotations

import inspect

import fire_engine.world.sky.weather as shim

# Import the canonical sources for identity checks
from fire_engine.world.weather.classify import WeatherType as _CanonicalWeatherType
from fire_engine.world.weather.classify import classify as _canonical_classify
from fire_engine.world.weather.system import (
    BLEND_SECONDS as _CANONICAL_BLEND,
)
from fire_engine.world.weather.system import (
    HYSTERESIS_SECONDS as _CANONICAL_HYSTERESIS,
)
from fire_engine.world.weather.system import (
    LocalWeather as _CanonicalLocalWeather,
)
from fire_engine.world.weather.system import (
    WeatherSystem as _CanonicalWeatherSystem,
)

# ---------------------------------------------------------------------------
# __all__ presence
# ---------------------------------------------------------------------------


class TestAll:
    def test_all_defined(self):
        assert hasattr(shim, "__all__")

    def test_all_contains_expected_names(self):
        expected = {
            "BLEND_SECONDS",
            "HYSTERESIS_SECONDS",
            "LocalWeather",
            "WeatherSystem",
            "WeatherType",
            "classify",
        }
        assert expected.issubset(set(shim.__all__))


# ---------------------------------------------------------------------------
# Re-exported names are the canonical objects
# ---------------------------------------------------------------------------


class TestReExports:
    def test_weather_type_is_canonical(self):
        assert shim.WeatherType is _CanonicalWeatherType

    def test_weather_system_is_canonical(self):
        assert shim.WeatherSystem is _CanonicalWeatherSystem

    def test_local_weather_is_canonical(self):
        assert shim.LocalWeather is _CanonicalLocalWeather

    def test_classify_is_canonical(self):
        assert shim.classify is _canonical_classify

    def test_blend_seconds_is_canonical(self):
        assert shim.BLEND_SECONDS is _CANONICAL_BLEND

    def test_hysteresis_seconds_is_canonical(self):
        assert shim.HYSTERESIS_SECONDS is _CANONICAL_HYSTERESIS


# ---------------------------------------------------------------------------
# BLEND_SECONDS and HYSTERESIS_SECONDS
# ---------------------------------------------------------------------------


class TestBlendSeconds:
    def test_blend_seconds_positive(self):
        assert shim.BLEND_SECONDS > 0.0

    def test_hysteresis_seconds_positive(self):
        assert shim.HYSTERESIS_SECONDS > 0.0

    def test_blend_seconds_is_float_or_int(self):
        assert isinstance(shim.BLEND_SECONDS, (int, float))

    def test_hysteresis_seconds_is_float_or_int(self):
        assert isinstance(shim.HYSTERESIS_SECONDS, (int, float))


# ---------------------------------------------------------------------------
# WeatherType enum behavior
# ---------------------------------------------------------------------------


class TestWeatherType:
    def test_is_enum(self):
        import enum

        assert issubclass(shim.WeatherType, enum.Enum)

    def test_clear_member_exists(self):
        assert hasattr(shim.WeatherType, "CLEAR")

    def test_rain_member_exists(self):
        assert hasattr(shim.WeatherType, "RAIN")

    def test_storm_member_exists(self):
        assert hasattr(shim.WeatherType, "STORM")

    def test_clear_value_is_string(self):
        assert isinstance(shim.WeatherType.CLEAR.value, str)


# ---------------------------------------------------------------------------
# WeatherSystem and LocalWeather are proper classes
# ---------------------------------------------------------------------------


class TestWeatherSystemClass:
    def test_weather_system_is_class(self):
        assert inspect.isclass(shim.WeatherSystem)

    def test_local_weather_is_class(self):
        assert inspect.isclass(shim.LocalWeather)


# ---------------------------------------------------------------------------
# classify callable behavior
# ---------------------------------------------------------------------------


def _lw(
    coverage: float = 0.0,
    density: float = 0.0,
    rain: float = 0.0,
    wind_speed: float = 0.0,
    fog: float = 0.0,
) -> shim.LocalWeather:
    """Helper: construct a LocalWeather for classify tests."""
    return shim.LocalWeather(
        cloud_coverage=coverage,
        cloud_density=density,
        fog_density=fog,
        rain_intensity=rain,
        wind_dir=(1.0, 0.0),
        wind_speed=wind_speed,
    )


class TestClassify:
    def test_classify_is_callable(self):
        assert callable(shim.classify)

    def test_classify_clear_at_zero_coverage(self):
        result = shim.classify(_lw(coverage=0.0, density=0.0))
        assert result == shim.WeatherType.CLEAR

    def test_classify_cloudy_at_mid_coverage(self):
        result = shim.classify(_lw(coverage=0.5, density=0.5))
        assert result == shim.WeatherType.CLOUDY

    def test_classify_overcast_at_high_coverage(self):
        result = shim.classify(_lw(coverage=0.9, density=0.9))
        assert result == shim.WeatherType.OVERCAST

    def test_classify_rain(self):
        result = shim.classify(_lw(coverage=0.9, density=0.9, rain=0.3))
        assert result == shim.WeatherType.RAIN

    def test_classify_storm(self):
        result = shim.classify(_lw(coverage=0.9, density=0.9, rain=0.8, wind_speed=12.0))
        assert result == shim.WeatherType.STORM

    def test_classify_fog(self):
        result = shim.classify(_lw(fog=0.02))
        assert result == shim.WeatherType.FOG

    def test_classify_returns_weather_type(self):
        for cov in (0.0, 0.3, 0.6, 0.9):
            result = shim.classify(_lw(coverage=cov, density=cov))
            assert isinstance(result, shim.WeatherType)
