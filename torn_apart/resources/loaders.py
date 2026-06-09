"""
resources/loaders.py — Loader registry for hand-crafted asset file formats.

This module owns a **dispatch table** keyed by file-suffix (e.g. ``".egg"``,
``".png"``) and a ``LoaderProtocol`` type alias describing the callable
signature each loader must satisfy.

Design: Inversion of Control
-----------------------------
``resources/`` must NOT import panda3d (ARCHITECTURE.md §3, CLAUDE.md hard
rule 1).  The actual Panda3D-backed loader callables are injected at boot by
``world/resource_adapter.py`` via ``register_loader()``.  Until they are
registered, the dispatch table contains ``None`` for model/audio/texture
suffixes — calling ``dispatch(path)`` on an unregistered suffix raises
``UnknownResourceFormatError`` (see error handling below).

Supported suffix groups (dispatch stubs, filled at boot):
  Models  : ``.egg``, ``.bam``, ``.gltf``, ``.glb``   — Panda3D loader
  Audio   : ``.ogg``, ``.wav``                          — Panda3D audio loader
  Textures: ``.png``, ``.jpg``                          — static hand-crafted
            (procedural environment textures come from the Procedural API,
             never from here — see ARCHITECTURE.md §5.3)

Usage
-----
    # At boot, world/resource_adapter.py injects real loaders:
    from torn_apart.resources.loaders import register_loader
    register_loader(".egg", my_panda3d_load_function)

    # At runtime, ResourceManager uses dispatch:
    from torn_apart.resources.loaders import dispatch
    resource_obj = dispatch("assets/models/landmark.egg")
"""

from __future__ import annotations

from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Public type alias
# ---------------------------------------------------------------------------

#: Callable signature every loader must satisfy.
#: Receives the normalised absolute-or-relative asset path; returns the loaded
#: resource object (type depends on the format — NodePath, AudioSound, etc.).
LoaderCallable = Callable[[str], Any]


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class UnknownResourceFormatError(Exception):
    """
    Raised when ``dispatch()`` is called with a file suffix that has no
    registered loader.

    This covers two cases:
    1. The suffix is completely unknown (e.g. ``.xyz``).
    2. The suffix is known (e.g. ``.egg``) but no loader has been registered
       yet — typically because ``world/resource_adapter.register_panda_loaders``
       has not been called yet (boot ordering error).

    Attributes
    ----------
    path   : str — the file path that triggered the error.
    suffix : str — the extracted file suffix.

    Example
    -------
        try:
            obj = dispatch("models/npc.xyz")
        except UnknownResourceFormatError as e:
            print(e.suffix)   # ".xyz"
    """

    def __init__(self, path: str, suffix: str) -> None:
        self.path   = path
        self.suffix = suffix
        super().__init__(
            f"No loader registered for suffix {suffix!r} "
            f"(path: {path!r}). "
            f"Known suffixes require boot-time registration via "
            f"world.resource_adapter.register_panda_loaders()."
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Module-level dispatch table: suffix (lowercase, with dot) → loader callable.
# Entries are initialised to None for known suffixes to distinguish
# "recognised but not yet registered" from "completely unknown".
_LOADERS: dict[str, Optional[LoaderCallable]] = {
    # --- 3D models (Panda3D loader registered by world/resource_adapter.py) ---
    ".egg":  None,
    ".bam":  None,
    ".gltf": None,
    ".glb":  None,
    # --- Audio (Panda3D loader registered by world/resource_adapter.py) ---
    ".ogg":  None,
    ".wav":  None,
    # --- Static hand-crafted textures (Pillow/Panda3D registered at boot) ---
    ".png":  None,
    ".jpg":  None,
}


def register_loader(suffix: str, loader: LoaderCallable) -> None:
    """
    Register (or replace) the loader callable for a given file suffix.

    This is the **only** way to wire in panda3d-backed loading; ``resources/``
    never imports panda3d directly.  Called by
    ``world.resource_adapter.register_panda_loaders(manager)`` during the
    application boot sequence (``world/app.py``), and in tests via a fake
    loader.

    Parameters
    ----------
    suffix : str
        Lowercase file suffix including the leading dot, e.g. ``".egg"``.
        Will be normalised to lowercase automatically.
    loader : LoaderCallable
        A callable ``(path: str) -> object``.  ``path`` is the normalised
        string passed to ``ResourceManager.load()``.  The return value is
        wrapped in a ``Handle`` and cached.

    Example
    -------
        from torn_apart.resources.loaders import register_loader

        def my_loader(path: str) -> object:
            import json
            with open(path) as f:
                return json.load(f)

        register_loader(".json", my_loader)
    """
    _LOADERS[suffix.lower()] = loader


def dispatch(path: str) -> Any:
    """
    Look up the loader for *path*'s suffix and invoke it.

    Parameters
    ----------
    path : str
        File path (relative or absolute).  The suffix is extracted from the
        last ``.``-delimited segment, lowercased.

    Returns
    -------
    object
        The resource object returned by the registered loader callable.

    Raises
    ------
    UnknownResourceFormatError
        If no loader is registered for the suffix **or** if the suffix is
        known but the loader slot is still ``None`` (boot registration not
        done yet).

    Example
    -------
        from torn_apart.resources.loaders import register_loader, dispatch

        register_loader(".fake", lambda p: {"loaded": p})
        obj = dispatch("assets/test.fake")
        # obj == {"loaded": "assets/test.fake"}
    """
    dot_idx = path.rfind(".")
    if dot_idx == -1:
        suffix = ""
    else:
        suffix = path[dot_idx:].lower()

    if suffix not in _LOADERS or _LOADERS[suffix] is None:
        raise UnknownResourceFormatError(path, suffix)

    loader = _LOADERS[suffix]
    assert loader is not None  # type narrowing
    return loader(path)


def registered_suffixes() -> list[str]:
    """
    Return the list of suffixes that have a non-None loader registered.

    Useful for diagnostics and tests.

    Returns
    -------
    list[str]
        Sorted list of lowercase suffix strings (e.g. ``['.egg', '.png']``).
    """
    return sorted(s for s, fn in _LOADERS.items() if fn is not None)
