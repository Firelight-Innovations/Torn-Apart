"""
sky/weather.py — Compatibility shim for the relocated weather system.

The weather model moved out of ``sky/`` into its own headless package
``fire_engine/world/weather/`` when it became spatial (storm cells that drift on the
synoptic flow, sampled at the player's position) instead of a single global
Markov state.  This module re-exports the public names from their new home so
existing imports (``from fire_engine.world.sky.weather import WeatherSystem,
WeatherType``) keep working.

New code should import from :mod:`fire_engine.world.weather` directly.

Docs: docs/systems/world.sky.md
"""

from __future__ import annotations

from fire_engine.world.weather.classify import WeatherType, classify
from fire_engine.world.weather.system import (
    BLEND_SECONDS,
    HYSTERESIS_SECONDS,
    LocalWeather,
    WeatherSystem,
)

__all__ = [
    "BLEND_SECONDS",
    "HYSTERESIS_SECONDS",
    "LocalWeather",
    "WeatherSystem",
    "WeatherType",
    "classify",
]
