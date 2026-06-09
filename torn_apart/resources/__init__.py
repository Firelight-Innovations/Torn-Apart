"""
torn_apart.resources — Hand-crafted asset loading with reference-counted cache.

This package is the ONLY place raw asset file I/O occurs for hand-crafted
assets (landmark models, player hands, audio, static PNG textures).

Procedural environment textures do NOT route through here — they are generated
by the Procedural API (``torn_apart.procedural``) and bridged to panda3d via
``world/texture_bridge.py``.  See ARCHITECTURE.md §5.3.

Panda3D-free
------------
``resources/`` never imports panda3d.  The actual Panda3D-backed loaders are
injected at boot by ``world/resource_adapter.register_panda_loaders(manager)``
(inversion of control).  This keeps ``resources/`` fully headless-testable.

Quick start
-----------
    # Boot (world/app.py):
    from torn_apart.resources import default_manager
    from torn_apart.world.resource_adapter import register_panda_loaders
    register_panda_loaders(default_manager)

    # Usage anywhere in the engine:
    from torn_apart.resources import load, acquire, release
    handle = acquire(load("assets/models/landmark_church.egg"))
    nodepath = handle.resource   # live Panda3D NodePath
    ...
    release(handle)
"""

from torn_apart.resources.loaders import (
    LoaderCallable,
    UnknownResourceFormatError,
    register_loader,
    dispatch,
    registered_suffixes,
)
from torn_apart.resources.manager import (
    Handle,
    ResourceManager,
    default_manager,
    load,
    acquire,
    release,
    unload_unreferenced,
)

__all__ = [
    # Error
    "UnknownResourceFormatError",
    # Loader registry
    "LoaderCallable",
    "register_loader",
    "dispatch",
    "registered_suffixes",
    # Handle + manager class
    "Handle",
    "ResourceManager",
    # Module-level default instance
    "default_manager",
    # Convenience functions (operate on default_manager)
    "load",
    "acquire",
    "release",
    "unload_unreferenced",
]
