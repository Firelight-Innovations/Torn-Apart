"""
torn_apart.world — World API: Unity-clone object model and Panda3D application shell.

Exports the full public API for the world package.  Import from here rather
than from submodules directly whenever practical.

The object model deliberately copies the Unity API (same names, snake_case)
with Z-up coordinates (forward=+Y, right=+X, up=+Z) and batched execution
semantics (see registry.py — ticks are batched by component type, not per-object).

All non-render files (transform, component, gameobject, registry) are pure
Python/numpy — zero panda3d imports — so the object model is fully
headless-testable.  Panda3D types are only touched in app.py and camera.py.

Quick-start example
-------------------
    from torn_apart.world import (
        Transform, Space,
        Component, GameObject,
        ComponentRegistry, instantiate, destroy,
        find_with_tag, find_objects_with_tag,
    )
    from torn_apart.core.math3d import Vec3, Quat

    class Spinner(Component):
        def update(self, dt):
            self.transform.rotate(
                Quat.from_axis_angle(Vec3.UP, dt),
            )

    go = instantiate(position=Vec3(0, 0, 5))
    go.add_component(Spinner)

    from torn_apart.core.clock import Clock
    from torn_apart.core.event_bus import EventBus
    clock = Clock(fixed_dt=0.02, bus=EventBus())
    clock.update(0.016)
    ComponentRegistry.run_frame(clock)  # awake + start + update
"""

from torn_apart.world.transform  import Transform, Space
from torn_apart.world.component  import Component
from torn_apart.world.gameobject import GameObject
from torn_apart.world.registry   import (
    ComponentRegistry,
    instantiate,
    destroy,
    find_with_tag,
    find_objects_with_tag,
)
# App is exported but has panda3d as a hard dependency — only import if
# panda3d is installed (headless tests skip app.py via the import rule).
try:
    from torn_apart.world.app import App
except ImportError:
    App = None  # type: ignore[assignment,misc]

# --- bridges (panda3d-backed; guarded so the package imports headless) ---
try:
    from torn_apart.world.texture_bridge import to_panda_texture
except ImportError:
    to_panda_texture = None  # type: ignore[assignment,misc]

try:
    from torn_apart.world.resource_adapter import register_panda_loaders
except ImportError:
    register_panda_loaders = None  # type: ignore[assignment,misc]

try:
    from torn_apart.world.geometry_bridge import to_geom, to_geom_node
except ImportError:
    to_geom = None  # type: ignore[assignment,misc]
    to_geom_node = None  # type: ignore[assignment,misc]

__all__ = [
    # Transform hierarchy
    "Transform",
    "Space",
    # Component lifecycle base
    "Component",
    # Entity container
    "GameObject",
    # Registry + Unity statics
    "ComponentRegistry",
    "instantiate",
    "destroy",
    "find_with_tag",
    "find_objects_with_tag",
    # Application shell (may be None when panda3d not installed)
    "App",
    # --- bridges (panda3d-backed; may be None when panda3d not installed) ---
    "to_panda_texture",        # Phase 2: procedural RGBA → Panda3D Texture
    "register_panda_loaders",  # Phase 5: inject panda3d asset loaders into ResourceManager
    "to_geom",                 # Phase 3: MeshArrays → Panda3D Geom (bulk writes)
    "to_geom_node",            # Phase 3: MeshArrays → Panda3D GeomNode
]
