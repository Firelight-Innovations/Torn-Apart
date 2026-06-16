"""
resources/types.py — trivial support types for the resource manager.

Holds :class:`Handle`, the reference-counted wrapper a
:class:`~fire_engine.resources.manager.ResourceManager` hands back from
``load()``. Grouping module (one-public-class rule exempt).

Docs: docs/systems/resources.md
"""

from __future__ import annotations

from typing import Any


class Handle:
    """
    A reference-counted wrapper around a loaded resource.

    Attributes
    ----------
    resource : object
        The raw loaded object (e.g. a Panda3D ``NodePath``, ``AudioSound``,
        Pillow ``Image``, or whatever the registered loader returned).
    path : str
        Normalised path string used as the cache key.
    refcount : int
        Current reference count.  Starts at 0 on construction.  Call
        ``ResourceManager.acquire(handle)`` to increment,
        ``ResourceManager.release(handle)`` to decrement.

    Example
    -------
        handle = manager.load("assets/models/player_hands.egg")
        manager.acquire(handle)
        obj = handle.resource   # live resource
        manager.release(handle)

    Docs: docs/systems/resources.md
    """

    __slots__ = ("path", "refcount", "resource")

    def __init__(self, resource: Any, path: str) -> None:
        self.resource: Any = resource
        self.path: str = path
        self.refcount: int = 0

    def __repr__(self) -> str:
        return (
            f"Handle(path={self.path!r}, refcount={self.refcount}, "
            f"resource={type(self.resource).__name__})"
        )
