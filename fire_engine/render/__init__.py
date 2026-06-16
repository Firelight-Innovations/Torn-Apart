"""
fire_engine.render — Render API: Unity-clone object model and Panda3D application shell.

Exports the full public API for the render package (formerly `world`).  Import from here rather
than from submodules directly whenever practical.

The object model deliberately copies the Unity API (same names, snake_case)
with Z-up coordinates (forward=+Y, right=+X, up=+Z) and batched execution
semantics (see registry.py — ticks are batched by component type, not per-object).

All non-render files (transform, component, gameobject, registry) are pure
Python/numpy — zero panda3d imports — so the object model is fully
headless-testable.  Panda3D types are only touched in app.py and camera.py.

Quick-start example
-------------------
    from fire_engine.render import (
        Transform, Space,
        Component, GameObject,
        ComponentRegistry, instantiate, destroy,
        find_with_tag, find_objects_with_tag,
    )
    from fire_engine.core.math3d import Vec3, Quat

    class Spinner(Component):
        def update(self, dt):
            self.transform.rotate(
                Quat.from_axis_angle(Vec3.UP, dt),
            )

    go = instantiate(position=Vec3(0, 0, 5))
    go.add_component(Spinner)

    from fire_engine.core.clock import Clock
    from fire_engine.core.event_bus import EventBus
    clock = Clock(fixed_dt=0.02, bus=EventBus())
    clock.update(0.016)
    ComponentRegistry.run_frame(clock)  # awake + start + update
"""

from fire_engine.render.component import Component
from fire_engine.render.gameobject import GameObject
from fire_engine.render.registry import (
    ComponentRegistry,
    destroy,
    find_objects_with_tag,
    find_with_tag,
    instantiate,
)
from fire_engine.render.transform import Space, Transform

# App is exported but has panda3d as a hard dependency — only import if
# panda3d is installed (headless tests skip app.py via the import rule).
try:
    from fire_engine.render.app import App
except ImportError:
    App = None  # type: ignore[assignment,misc]

# --- bridges (panda3d-backed; guarded so the package imports headless) ---
try:
    from fire_engine.render.bridges.texture_bridge import to_panda_texture
except ImportError:
    to_panda_texture = None  # type: ignore[assignment]

try:
    from fire_engine.render.bridges.resource_adapter import register_panda_loaders
except ImportError:
    register_panda_loaders = None  # type: ignore[assignment]

try:
    from fire_engine.render.bridges.geometry_bridge import to_geom, to_geom_node
except ImportError:
    to_geom = None  # type: ignore[assignment]
    to_geom_node = None  # type: ignore[assignment]

try:
    from fire_engine.render.overlay.devtools_overlay import DevOverlay
except ImportError:
    DevOverlay = None  # type: ignore[assignment,misc]

try:
    from fire_engine.render.sky.sky_renderer import SkyRendererComponent
except ImportError:
    SkyRendererComponent = None  # type: ignore[assignment,misc]

try:
    from fire_engine.render.sky.weather_renderer import WeatherMapComponent
except ImportError:
    WeatherMapComponent = None  # type: ignore[assignment,misc]

try:
    from fire_engine.render.sky.rain_renderer import RainRendererComponent
except ImportError:
    RainRendererComponent = None  # type: ignore[assignment,misc]

__all__ = [
    # Application shell (may be None when panda3d not installed)
    "App",
    # Component lifecycle base
    "Component",
    # Registry + Unity statics
    "ComponentRegistry",
    # Dev tools: in-game DirectGUI debug overlay renderer
    "DevOverlay",
    # Entity container
    "GameObject",
    # M6: volumetric rain (cover-culled + storm-gated)
    "RainRendererComponent",
    # Sky/weather: dome + volumetric clouds + fog renderer
    "SkyRendererComponent",
    # Transform hierarchy
    "Space",
    "Transform",
    # M4: spatial weather-map texture upload + uniforms
    "WeatherMapComponent",
    "destroy",
    "find_objects_with_tag",
    "find_with_tag",
    "instantiate",
    # --- bridges (panda3d-backed; may be None when panda3d not installed) ---
    "register_panda_loaders",  # Phase 5: inject panda3d asset loaders into ResourceManager
    "to_geom",  # Phase 3: MeshArrays → Panda3D Geom (bulk writes)
    "to_geom_node",  # Phase 3: MeshArrays → Panda3D GeomNode
    "to_panda_texture",  # Phase 2: procedural RGBA → Panda3D Texture
]
