"""
wind/protocols.py — Structural protocols for the wind package.

Groups the :class:`WindModifier` ``Protocol`` used as the extension seam for
pluggable in-place wind-field modifiers.  Behavioural implementations (e.g.
:class:`~fire_engine.world.wind.modifiers.GustFront`) live in ``modifiers.py``.

Docs: docs/systems/world.wind.md
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["WindModifier"]


@runtime_checkable
class WindModifier(Protocol):
    """
    In-place modifier of the composed wind field, applied before publish.

    Implementations mutate ``vx``, ``vy`` and ``turb`` (all same-shaped
    ``float32`` ``(cells, cells)`` arrays, indexed ``[x, y]``) in place; the
    return value is ignored.  ``X`` / ``Y`` are the matching cell-centre world
    coordinate meshes (meters); ``t`` is the field's evaluation time (seconds).

    Keep implementations a **pure function of their own seed/config and ``t``**
    (no accumulated state) to preserve the field's determinism and
    zero-save-bytes guarantee.

    Example
    -------
    >>> class Calm:                          # zero out all wind in a region
    ...     def apply(self, X, Y, t, vx, vy, turb):
    ...         mask = (X**2 + Y**2) < 100.0
    ...         vx[mask] = 0.0; vy[mask] = 0.0
    """

    def apply(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        t: float,
        vx: np.ndarray,
        vy: np.ndarray,
        turb: np.ndarray,
    ) -> None:
        """Mutate ``vx`` / ``vy`` / ``turb`` in place for time ``t``."""
        ...
