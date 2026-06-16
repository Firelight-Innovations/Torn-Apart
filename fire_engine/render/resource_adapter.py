"""
world/resource_adapter.py — Panda3D-backed asset loaders for the Resource Manager.

This module bridges ``fire_engine.resources`` (panda3d-free) with the real
Panda3D/Pillow asset-loading APIs.  It lives in ``world/`` because panda3d
imports are **only** allowed in ``world/`` and ``lighting/`` per
ARCHITECTURE.md §3 and CLAUDE.md hard rule 1.

Boot Registration (Inversion of Control)
-----------------------------------------
``register_panda_loaders(manager)`` is called once during the application boot
sequence (inside ``world/app.py`` or ``main.py``) and injects the real loader
callables into ``fire_engine.resources.loaders`` via
``resources.loaders.register_loader()``.

This keeps ``resources/`` completely panda3d-free while still supporting real
asset loading at runtime.

Supported Formats Registered
-----------------------------
Models  (Panda3D Loader):   ``.egg``, ``.bam``, ``.gltf``, ``.glb``
Audio   (Panda3D Audio):    ``.ogg``, ``.wav``
Textures (Pillow / p3d):    ``.png``, ``.jpg``

Usage
-----
    from fire_engine.resources import default_manager
    from fire_engine.render.resource_adapter import register_panda_loaders

    register_panda_loaders(default_manager)

    # Now model/audio/texture loading works:
    from fire_engine.resources import load, acquire
    handle = acquire(load("assets/models/church.egg"))
    nodepath = handle.resource   # Panda3D NodePath
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Panda3D imports are ALLOWED here — this module lives in world/
from panda3d.core import Loader as P3DLoader

if TYPE_CHECKING:
    from fire_engine.resources.manager import ResourceManager


# ---------------------------------------------------------------------------
# Internal helpers: one loader function per format family
# ---------------------------------------------------------------------------


def _get_global_loader() -> P3DLoader:
    """
    Retrieve the global Panda3D ``Loader`` instance.

    Panda3D maintains a module-level ``loader`` singleton once a ``ShowBase``
    (or ``LoaderOptions``-backed loader) has been created.  This helper falls
    back to constructing a bare ``Loader`` if the global isn't available, which
    allows the model loader to work even in an offscreen/headless context (as
    long as a ``ShowBase`` was initialised first).

    Returns
    -------
    Loader
        The active Panda3D Loader instance.

    Raises
    ------
    RuntimeError
        If Panda3D's global loader is not yet initialised (i.e. no ShowBase
        was created before calling this function).
    """
    # Panda3D places a ``loader`` in the builtins namespace when ShowBase starts.
    import builtins

    global_loader = getattr(builtins, "loader", None)
    if global_loader is None:
        raise RuntimeError(
            "Panda3D global loader is not available. "
            "Ensure a ShowBase (or App) instance has been created before "
            "calling resource_adapter loaders."
        )
    return global_loader


def _load_model(path: str) -> Any:
    """
    Load a 3D model file via Panda3D's Loader.

    Parameters
    ----------
    path : str
        Path to a ``.egg``, ``.bam``, ``.gltf``, or ``.glb`` file.
        May be a Windows-style or POSIX path; converted to a Panda3D
        ``Filename`` so the loader finds it regardless of the model-path
        search list.

    Returns
    -------
    NodePath
        Panda3D ``NodePath`` wrapping the loaded model.  The model is NOT
        parented to the render tree — the caller (World API) must parent it.

    Raises
    ------
    IOError / RuntimeError
        If the file does not exist or Panda3D cannot parse it.
    """
    from panda3d.core import Filename

    p3d_loader = _get_global_loader()
    # Use fromOsSpecific so absolute Windows paths (C:\...) are interpreted
    # correctly rather than being routed through Panda3D's model-path search.
    panda_filename = Filename.fromOsSpecific(path)
    nodepath = p3d_loader.loadModel(panda_filename)
    if nodepath is None:
        raise OSError(f"Panda3D Loader returned None for model: {path!r}")
    return nodepath


def _load_audio(path: str) -> Any:
    """
    Load an audio file via Panda3D's audio manager.

    Parameters
    ----------
    path : str
        Path to a ``.ogg`` or ``.wav`` file.

    Returns
    -------
    AudioSound
        Panda3D ``AudioSound`` instance (not yet playing).

    Raises
    ------
    IOError / RuntimeError
        If the file does not exist or cannot be decoded.
    """
    import builtins

    base = getattr(builtins, "base", None)
    if base is None:
        raise RuntimeError(
            "Panda3D ShowBase ('base') is not available. "
            "Ensure App has been initialised before loading audio."
        )
    sound = base.loader.loadSfx(path)
    if sound is None:
        raise OSError(f"Panda3D audio loader returned None for: {path!r}")
    return sound


def _load_texture_image(path: str) -> Any:
    """
    Load a static PNG/JPG texture for a hand-crafted asset.

    Uses Pillow to decode the file to a raw RGBA ``bytes`` blob wrapped in
    a small descriptor dict so the result is panda3d-free and can be used
    headlessly.  ``world/texture_bridge.py`` can promote this to a Panda3D
    ``Texture`` if needed.

    The descriptor format::

        {
            "width":  int,
            "height": int,
            "mode":   str,   # e.g. "RGBA"
            "data":   bytes, # raw pixel data
        }

    Parameters
    ----------
    path : str
        Path to a ``.png`` or ``.jpg`` file.

    Returns
    -------
    dict
        Image descriptor as above.

    Raises
    ------
    ImportError
        If Pillow (``PIL``) is not installed.
    IOError
        If the file cannot be opened.
    """
    from PIL import Image

    img = Image.open(path).convert("RGBA")
    return {
        "width": img.width,
        "height": img.height,
        "mode": img.mode,
        "data": img.tobytes(),
    }


# ---------------------------------------------------------------------------
# Public registration function
# ---------------------------------------------------------------------------


def register_panda_loaders(resource_manager: ResourceManager) -> None:
    """
    Inject panda3d-backed loader callables into the Resource Manager's loader
    registry.

    Call this once during application boot (``world/app.py``) **after**
    ``ShowBase`` has been initialised (so the global Panda3D ``Loader``
    singleton exists).

    Parameters
    ----------
    resource_manager : ResourceManager
        The ``ResourceManager`` instance whose underlying loaders module will
        receive the registered callables.  Pass
        ``fire_engine.resources.default_manager`` for the production singleton.

    Example
    -------
        # world/app.py boot sequence:
        from fire_engine.resources import default_manager
        from fire_engine.render.resource_adapter import register_panda_loaders

        register_panda_loaders(default_manager)

        # Now these all work:
        from fire_engine.resources import load, acquire
        h = acquire(load("assets/models/landmark_church.egg"))
        nodepath = h.resource   # Panda3D NodePath
    """
    _loaders = resource_manager._loaders  # honours the IoC contract; may be a test fake

    # --- 3D models ---
    _loaders.register_loader(".egg", _load_model)
    _loaders.register_loader(".bam", _load_model)
    _loaders.register_loader(".gltf", _load_model)
    _loaders.register_loader(".glb", _load_model)

    # --- Audio ---
    _loaders.register_loader(".ogg", _load_audio)
    _loaders.register_loader(".wav", _load_audio)

    # --- Static hand-crafted textures ---
    _loaders.register_loader(".png", _load_texture_image)
    _loaders.register_loader(".jpg", _load_texture_image)
