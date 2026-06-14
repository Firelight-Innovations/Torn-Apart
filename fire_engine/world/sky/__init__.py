"""
fire_engine.world.sky — Procedural sky + weather (Layer 1 — Services).

Headless peer of ``lighting/``: computes the per-frame :class:`SkyState`
(sun/moon directions, sky gradient colors, blended weather parameters, and
the terrain light scale) that the render layer (``world/``) consumes.  This
package never imports panda3d — everything is testable without a window.

Public API summary
------------------
SkyState
    Frozen per-frame snapshot: celestial directions, colors, weather, fog,
    wind, ``terrain_light_scale``.
SkySystem
    Composer.  ``update()`` once per frame; ``state`` property for the last
    snapshot; ``weather`` attribute is the owned WeatherSystem.
WeatherType, LocalWeather, WeatherSystem
    Spatial storm-cell weather (Saveable, ``save_key="weather"``) — re-exported
    from :mod:`fire_engine.world.weather`; ``update()`` samples at the player and
    blends a ``force_weather`` dev override over 20 game minutes.
sun_direction, moon_direction
    Pure time-of-day → unit Vec3 celestial geometry (Z-up).

Quick-start example
-------------------
::

    from fire_engine.core import Clock, EventBus, load_config, set_world_seed
    from fire_engine.world.sky import SkySystem, WeatherType

    cfg = load_config()
    set_world_seed(cfg.world_seed)
    bus = EventBus()
    clock = Clock(fixed_dt=cfg.fixed_dt, bus=bus)

    sky = SkySystem(cfg, clock, bus)
    state = sky.update()                       # once per frame
    sky.weather.force_weather(WeatherType.STORM)   # dev override
    sky.weather.force_weather(None)                # back to the schedule
"""

from fire_engine.world.sky.celestial import moon_direction, sun_direction
from fire_engine.world.sky.sky_state import SkyState, SkySystem
from fire_engine.world.sky.weather_map_pack import pack_weather_map
from fire_engine.world.weather import LocalWeather, WeatherSystem, WeatherType

__all__ = [
    "LocalWeather",
    "SkyState",
    "SkySystem",
    "WeatherSystem",
    "WeatherType",
    "moon_direction",
    "pack_weather_map",
    "sun_direction",
]
