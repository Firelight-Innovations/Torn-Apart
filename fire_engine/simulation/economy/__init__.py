"""Economy API — per-settlement supply/demand pricing.

Session 1 stub. See docs/ARCHITECTURE.md §5.9. NPCs and the player use the
identical pricing path. Trade routes, hired managers, arbitrage.

Imports allowed (ARCHITECTURE.md §4a.2): procedural, core.
"""

from __future__ import annotations

import numpy as np

__all__ = ["GoodDef"]


class GoodDef:
    """Stub for a tradeable good definition.

    Implement in a later session per ARCHITECTURE.md §5.9.
    """

    def generate(self, rng: np.random.Generator, **params: object) -> None:
        raise NotImplementedError(
            "GoodDef.generate is future scope — see ARCHITECTURE.md §5.9 "
            "(Economy API). Not part of Session 1."
        )
