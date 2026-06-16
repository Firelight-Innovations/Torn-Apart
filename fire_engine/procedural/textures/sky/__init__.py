"""
procedural/textures/sky — sky and atmospheric texture definitions.

Registers:
    ``"moon_surface"``    — 256×256 RGBA lunar disc (maria + craters).
    ``"night_sky"``       — 1024×512 RGBA equirect star field + galaxy band.
    ``"night_sky_cube"``  — (6, 512, 512, 4) RGBA cube-map star field.
    ``"rain_streak"``     — 128×512 RGBA tiling rain streaks (U+V tileable).

Private sub-package of ``fire_engine.procedural.textures``; import the
parent package or ``fire_engine.procedural`` instead of this directly.

Docs: docs/systems/procedural.md
"""

from fire_engine.procedural.textures.sky import (
    moon_surface,
    night_sky,  # registers "night_sky" + "night_sky_cube" via re-export
    rain_streak,
)

__all__ = ["moon_surface", "night_sky", "rain_streak"]
