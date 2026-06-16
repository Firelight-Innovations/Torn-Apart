"""
tests/render/overlay/test__overlay_environment.py — Headless tests for
render/overlay/_overlay_environment.py.

Tests the pure helper functions (_fmt, cycle_weather, build_environment)
using SimpleNamespace fakes.  No panda3d required.
"""

from __future__ import annotations

import types

from fire_engine.render.overlay._overlay_environment import (
    _fmt,
    build_environment,
    cycle_weather,
)


class TestFmt:
    """_fmt produces compact display strings for scalar values."""

    def test_bool_true(self) -> None:
        assert _fmt(True) == "true"

    def test_bool_false(self) -> None:
        assert _fmt(False) == "false"

    def test_float_three_decimal_places(self) -> None:
        result = _fmt(3.14159)
        assert result == "3.142"

    def test_float_zero(self) -> None:
        assert _fmt(0.0) == "0.000"

    def test_float_negative(self) -> None:
        assert _fmt(-1.5) == "-1.500"

    def test_int_becomes_str(self) -> None:
        result = _fmt(42)
        assert result == "42"

    def test_string_passthrough(self) -> None:
        assert _fmt("hello") == "hello"

    def test_none_becomes_str_none(self) -> None:
        assert _fmt(None) == "None"


class TestCycleWeather:
    """cycle_weather advances _wx cyclically and calls weather.force_weather."""

    def _make_fake_overlay(self, weather_types: list, wx: int = 0) -> types.SimpleNamespace:
        fake = types.SimpleNamespace()
        fake._weather_types = weather_types
        fake._wx = wx
        return fake

    def _make_fake_weather(self) -> types.SimpleNamespace:
        fake = types.SimpleNamespace()
        fake.force_weather_calls: list = []

        def force_weather(wtype: object) -> None:
            fake.force_weather_calls.append(wtype)

        fake.force_weather = force_weather
        return fake

    def test_advances_wx_by_one(self) -> None:
        overlay = self._make_fake_overlay(["CLEAR", "RAIN", "STORM"], wx=0)
        weather = self._make_fake_weather()
        cycle_weather(overlay, weather)
        assert overlay._wx == 1

    def test_wraps_around_at_end(self) -> None:
        overlay = self._make_fake_overlay(["CLEAR", "RAIN", "STORM"], wx=2)
        weather = self._make_fake_weather()
        cycle_weather(overlay, weather)
        assert overlay._wx == 0

    def test_calls_force_weather_with_correct_type(self) -> None:
        types_list = ["CLEAR", "RAIN", "STORM"]
        overlay = self._make_fake_overlay(types_list, wx=0)
        weather = self._make_fake_weather()
        cycle_weather(overlay, weather)
        assert weather.force_weather_calls == ["RAIN"]

    def test_does_nothing_when_weather_types_empty(self) -> None:
        overlay = self._make_fake_overlay([], wx=0)
        weather = self._make_fake_weather()
        cycle_weather(overlay, weather)
        assert overlay._wx == 0
        assert weather.force_weather_calls == []

    def test_full_cycle_returns_to_start(self) -> None:
        types_list = ["A", "B", "C"]
        overlay = self._make_fake_overlay(types_list, wx=0)
        weather = self._make_fake_weather()
        for _ in range(len(types_list)):
            cycle_weather(overlay, weather)
        assert overlay._wx == 0

    def test_force_weather_called_correct_number_of_times(self) -> None:
        types_list = ["A", "B"]
        overlay = self._make_fake_overlay(types_list, wx=0)
        weather = self._make_fake_weather()
        cycle_weather(overlay, weather)
        cycle_weather(overlay, weather)
        assert len(weather.force_weather_calls) == 2


class TestBuildEnvironment:
    """build_environment returns (sections, buttons) with the right structure."""

    def _make_fake_sky(self) -> types.SimpleNamespace:
        sky = types.SimpleNamespace()
        sky.state = types.SimpleNamespace(
            cloud_coverage=0.5,
            fog_density=0.001,
            rain_intensity=0.0,
        )
        weather = types.SimpleNamespace()
        weather.current = types.SimpleNamespace(value="CLEAR")
        weather.force_weather_calls: list = []

        def force_weather(wtype: object) -> None:
            weather.force_weather_calls.append(wtype)

        weather.force_weather = force_weather
        sky.weather = weather
        return sky

    def _make_fake_clock(self) -> types.SimpleNamespace:
        clock = types.SimpleNamespace()
        clock.game_time_of_day = 7200.0  # 2 hours
        clock.game_time_scale = 60.0
        clock.game_day = 3
        return clock

    def _make_fake_overlay(self) -> types.SimpleNamespace:
        fake = types.SimpleNamespace()
        fake._weather_types = ["CLEAR", "RAIN"]
        fake._wx = 0
        return fake

    def test_returns_tuple_of_sections_and_buttons(self) -> None:
        overlay = self._make_fake_overlay()
        sky = self._make_fake_sky()
        clock = self._make_fake_clock()
        result = build_environment(overlay, sky, clock)
        assert isinstance(result, tuple)
        assert len(result) == 2
        sections, buttons = result
        assert isinstance(sections, list)
        assert isinstance(buttons, list)

    def test_cycle_weather_button_present_when_weather_has_force(self) -> None:
        from fire_engine.devtools import Button

        overlay = self._make_fake_overlay()
        sky = self._make_fake_sky()
        clock = self._make_fake_clock()
        _, buttons = build_environment(overlay, sky, clock)
        assert any(isinstance(b, Button) for b in buttons)
        cycle_buttons = [b for b in buttons if isinstance(b, Button) and "Cycle" in b.label]
        assert len(cycle_buttons) == 1

    def test_sections_not_empty(self) -> None:
        overlay = self._make_fake_overlay()
        sky = self._make_fake_sky()
        clock = self._make_fake_clock()
        sections, _ = build_environment(overlay, sky, clock)
        assert len(sections) >= 1

    def test_no_buttons_when_weather_has_no_force_weather(self) -> None:
        overlay = self._make_fake_overlay()
        sky = self._make_fake_sky()
        del sky.weather.force_weather  # remove the method
        clock = self._make_fake_clock()
        _, buttons = build_environment(overlay, sky, clock)
        assert buttons == []

    def test_cycle_button_invokes_cycle_weather(self) -> None:
        overlay = self._make_fake_overlay()
        sky = self._make_fake_sky()
        clock = self._make_fake_clock()
        _, buttons = build_environment(overlay, sky, clock)
        # Find and invoke the cycle button
        for btn in buttons:
            if hasattr(btn, "label") and "Cycle" in btn.label:
                btn.on_click()
                break
        # overlay._wx should have advanced
        assert overlay._wx == 1
