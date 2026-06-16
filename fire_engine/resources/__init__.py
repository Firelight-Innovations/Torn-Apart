"""
fire_engine.resources — Hand-crafted asset loading with reference-counted cache.

This package is the ONLY place raw asset file I/O occurs for hand-crafted
assets (landmark models, player hands, audio, static PNG textures).

Procedural environment textures do NOT route through here — they are generated
by the Procedural API (``fire_engine.procedural``) and bridged to panda3d via
``world/texture_bridge.py``.  See ARCHITECTURE.md §5.3.

Panda3D-free
------------
``resources/`` never imports panda3d.  The actual Panda3D-backed loaders are
injected at boot by ``world/resource_adapter.register_panda_loaders(manager)``
(inversion of control).  This keeps ``resources/`` fully headless-testable.

Quick start
-----------
    # Boot (world/app.py):
    from fire_engine.resources import default_manager
    from fire_engine.render.bridges.resource_adapter import register_panda_loaders
    register_panda_loaders(default_manager)

    # Usage anywhere in the engine:
    from fire_engine.resources import load, acquire, release
    handle = acquire(load("assets/models/landmark_church.egg"))
    nodepath = handle.resource   # live Panda3D NodePath
    ...
    release(handle)
"""

from fire_engine.resources.loaders import (
    LoaderCallable,
    UnknownResourceFormatError,
    dispatch,
    register_loader,
    registered_suffixes,
)
from fire_engine.resources.manager import (
    Handle,
    ResourceManager,
    acquire,
    default_manager,
    load,
    release,
    unload_unreferenced,
)

__all__ = [
    "Handle",
    "LoaderCallable",
    "ResourceManager",
    "UnknownResourceFormatError",
    "acquire",
    "default_manager",
    "dispatch",
    "load",
    "register_loader",
    "registered_suffixes",
    "release",
    "unload_unreferenced",
]
