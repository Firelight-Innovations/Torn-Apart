"""
devtools/selection.py — the currently-selected GameObject, with a change counter.

Selection is shared engine-debug state: the picker writes it (click in the
viewport), the Inspector tool reads it (shows the selected object's fields), and
the overlay renderer reads it (draws the selection outline).  Keeping it in one
small headless object means none of those three need to know about each other.

No panda3d imports — headless-testable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    # Duck-typed at runtime; imported only for type-checkers so this module
    # never pulls world/ (and therefore panda3d) into the import graph.
    from torn_apart.world.gameobject import GameObject


class Selection:
    """
    Holds the currently-selected GameObject and a monotonically increasing
    ``revision`` that ticks every time the selection *changes*.

    Tools (e.g. the Inspector) compare ``revision`` to know when to rebuild
    their panel structure, rather than diffing the object identity themselves.

    Attributes
    ----------
    current : GameObject | None
        The selected object, or ``None`` when nothing is selected.

    Example
    -------
        sel = Selection()
        sel.set(some_gameobject)
        assert sel.current is some_gameobject
        sel.on_change(lambda go: print("selected", go.name if go else None))
    """

    __slots__ = ("_current", "_revision", "_listeners")

    def __init__(self) -> None:
        self._current: "Optional[GameObject]" = None
        self._revision: int = 0
        self._listeners: list[Callable[["Optional[GameObject]"], None]] = []

    @property
    def current(self) -> "Optional[GameObject]":
        """The selected GameObject, or None."""
        return self._current

    @property
    def revision(self) -> int:
        """Counter that increments on every selection change (never decreases)."""
        return self._revision

    def set(self, go: "Optional[GameObject]") -> None:
        """
        Select ``go`` (or clear with ``None``).

        No-op when ``go`` is already the current selection (so ``revision`` only
        moves on real changes).  Notifies any registered change listeners.

        Parameters
        ----------
        go : GameObject | None
        """
        if go is self._current:
            return
        self._current = go
        self._revision += 1
        for cb in self._listeners:
            cb(go)

    def clear(self) -> None:
        """Deselect (equivalent to ``set(None)``)."""
        self.set(None)

    def on_change(
        self, callback: "Callable[[Optional[GameObject]], None]"
    ) -> None:
        """
        Register a callback fired with the new selection whenever it changes.

        Parameters
        ----------
        callback : Callable[[GameObject | None], None]
        """
        self._listeners.append(callback)
