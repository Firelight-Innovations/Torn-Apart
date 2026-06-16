"""
tests/world/weather/_impl/test__save.py — Mirror for
fire_engine/world/weather/_impl/_save.py.

Authored tests covering the LocalWeather and StormCell serialisation helpers,
plus get_delta / apply_delta exercised indirectly through WeatherSystem.

Headless — no panda3d imports.

Coverage
--------
CORRECTNESS — local_to_dict / local_from_dict round-trip:
  - All nine fields survive encode→decode with bit-exact float equality.
  - Legacy dict (missing humidity/wetness/temperature_c) loads without error,
    applying the documented defaults.
  - Output dict contains only plain primitives (no numpy, no live objects).

CORRECTNESS — cell_to_dict / cell_from_dict round-trip:
  - All eight StormCell fields survive encode→decode exactly.
  - Malformed dicts raise KeyError/ValueError (not crash silently).

CORRECTNESS — get_delta / apply_delta via WeatherSystem:
  - Natural (no override, no summons) → get_delta() == {}.
  - After a force_weather() call, delta contains "override" key.
  - apply_delta on a fresh WeatherSystem reproduces override state.
"""

from __future__ import annotations

import pytest

from fire_engine.core import load_config
from fire_engine.core.rng import set_world_seed
from fire_engine.world.weather._impl._save import (
    cell_from_dict,
    cell_to_dict,
    local_from_dict,
    local_to_dict,
)
from fire_engine.world.weather.cells import CellKind, StormCell
from fire_engine.world.weather.types import LocalWeather

# ---------------------------------------------------------------------------
# LocalWeather round-trip
# ---------------------------------------------------------------------------


def _lw_full() -> LocalWeather:
    return LocalWeather(
        cloud_coverage=0.72,
        cloud_density=0.55,
        fog_density=0.003,
        rain_intensity=0.45,
        wind_dir=(0.6, 0.8),
        wind_speed=7.5,
        humidity=0.68,
        wetness=0.22,
        temperature_c=9.0,
    )


class TestLocalWeatherRoundTrip:
    def test_all_fields_survive_encode_decode(self):
        lw = _lw_full()
        d = local_to_dict(lw)
        lw2 = local_from_dict(d)
        assert lw2.cloud_coverage == pytest.approx(lw.cloud_coverage)
        assert lw2.cloud_density == pytest.approx(lw.cloud_density)
        assert lw2.fog_density == pytest.approx(lw.fog_density)
        assert lw2.rain_intensity == pytest.approx(lw.rain_intensity)
        assert lw2.wind_dir[0] == pytest.approx(lw.wind_dir[0])
        assert lw2.wind_dir[1] == pytest.approx(lw.wind_dir[1])
        assert lw2.wind_speed == pytest.approx(lw.wind_speed)
        assert lw2.humidity == pytest.approx(lw.humidity)
        assert lw2.wetness == pytest.approx(lw.wetness)
        assert lw2.temperature_c == pytest.approx(lw.temperature_c)

    def test_output_contains_only_primitives(self):
        d = local_to_dict(_lw_full())
        for key, val in d.items():
            if isinstance(val, list):
                for v in val:
                    assert isinstance(v, float), f"d[{key!r}] list item is {type(v)}"
            else:
                assert isinstance(val, float), f"d[{key!r}] is {type(val)}"

    def test_wind_dir_serialised_as_two_element_list(self):
        d = local_to_dict(_lw_full())
        assert isinstance(d["wind_dir"], list) and len(d["wind_dir"]) == 2

    def test_legacy_dict_missing_optional_fields(self):
        """A dict saved before humidity/wetness/temperature_c were added still loads."""
        legacy = {
            "cloud_coverage": 0.3,
            "cloud_density": 0.5,
            "fog_density": 0.001,
            "rain_intensity": 0.0,
            "wind_dir": [1.0, 0.0],
            "wind_speed": 5.0,
            # no humidity, wetness, temperature_c
        }
        lw = local_from_dict(legacy)
        assert lw.humidity == pytest.approx(0.5)  # documented default
        assert lw.wetness == pytest.approx(0.0)  # documented default
        assert lw.temperature_c == pytest.approx(12.0)  # documented default

    def test_round_trip_produces_equal_objects(self):
        lw = _lw_full()
        assert local_from_dict(local_to_dict(lw)) == lw


# ---------------------------------------------------------------------------
# StormCell round-trip
# ---------------------------------------------------------------------------


def _cell() -> StormCell:
    return StormCell(
        id="s:7",
        kind=CellKind.THUNDERSTORM,
        spawn_time=12345.0,
        spawn_pos=(100.0, -50.0),
        duration_s=7200.0,
        radius_m=900.0,
        peak_intensity=0.95,
        drift_bias=(0.3, -0.1),
    )


class TestStormCellRoundTrip:
    def test_all_fields_survive_encode_decode(self):
        c = _cell()
        d = cell_to_dict(c)
        c2 = cell_from_dict(d)
        assert c2.id == c.id
        assert c2.kind is c.kind
        assert c2.spawn_time == pytest.approx(c.spawn_time)
        assert c2.spawn_pos[0] == pytest.approx(c.spawn_pos[0])
        assert c2.spawn_pos[1] == pytest.approx(c.spawn_pos[1])
        assert c2.duration_s == pytest.approx(c.duration_s)
        assert c2.radius_m == pytest.approx(c.radius_m)
        assert c2.peak_intensity == pytest.approx(c.peak_intensity)
        assert c2.drift_bias[0] == pytest.approx(c.drift_bias[0])
        assert c2.drift_bias[1] == pytest.approx(c.drift_bias[1])

    def test_kind_is_cell_kind_member_after_decode(self):
        c2 = cell_from_dict(cell_to_dict(_cell()))
        assert isinstance(c2.kind, CellKind)
        assert c2.kind is CellKind.THUNDERSTORM

    def test_output_dict_has_expected_keys(self):
        d = cell_to_dict(_cell())
        assert set(d.keys()) == {
            "id",
            "kind",
            "spawn_time",
            "spawn_pos",
            "duration_s",
            "radius_m",
            "peak_intensity",
            "drift_bias",
        }

    def test_malformed_dict_raises(self):
        with pytest.raises((KeyError, ValueError, TypeError, IndexError)):
            cell_from_dict({"id": "s:0"})  # missing required fields

    def test_all_cell_kinds_round_trip(self):
        for kind in CellKind:
            c = StormCell(
                id="s:0",
                kind=kind,
                spawn_time=0.0,
                spawn_pos=(0.0, 0.0),
                duration_s=3600.0,
                radius_m=500.0,
                peak_intensity=0.8,
                drift_bias=(0.0, 0.0),
            )
            c2 = cell_from_dict(cell_to_dict(c))
            assert c2.kind is kind


# ---------------------------------------------------------------------------
# get_delta / apply_delta via WeatherSystem
# ---------------------------------------------------------------------------


class TestGetApplyDeltaViaSystem:
    def _fresh(self, seed: int = 1337):
        from fire_engine.world.weather import WeatherSystem

        set_world_seed(seed)
        return WeatherSystem(load_config())

    def test_natural_delta_is_empty(self):
        ws = self._fresh()
        ws.update(1, 6 * 3600.0, (0.0, 0.0))
        assert ws.get_delta() == {}

    def test_override_delta_has_override_key(self):
        from fire_engine.world.weather import WeatherType

        ws = self._fresh()
        ws.update(0, 3600.0, (0.0, 0.0))
        ws.force_weather(WeatherType.RAIN)
        ws.update(0, 3660.0, (0.0, 0.0))
        delta = ws.get_delta()
        assert "override" in delta
        assert delta["override"] == "rain"

    def test_apply_delta_restores_override(self):
        from fire_engine.world.weather import WeatherType

        ws1 = self._fresh()
        ws1.update(0, 3600.0, (0.0, 0.0))
        ws1.force_weather(WeatherType.STORM)
        ws1.update(0, 3660.0, (0.0, 0.0))
        delta = ws1.get_delta()

        ws2 = self._fresh()
        ws2.apply_delta(delta)
        assert ws2.current is WeatherType.STORM
