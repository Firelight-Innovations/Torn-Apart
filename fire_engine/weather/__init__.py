"""
fire_engine/weather — Spatial, volumetric weather simulation (headless).

Layer 1 service, peer of ``sky/`` and ``wind/``.  Owns the synoptic flow
(M1), storm cells / regimes (M2), the weather-map raster (M3), emergent
humidity/fog (M5) and lightning scheduling (M7).  Never imports panda3d;
render bridges live in ``fire_engine/world/``.

See ``docs/systems/weather.md`` for the system contract.
"""

from fire_engine.weather.synoptic import Synoptic

__all__ = ["Synoptic"]
