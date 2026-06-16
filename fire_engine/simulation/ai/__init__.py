"""AI API — tiered simulation for 10k+ agents.

Session 1 stub. See docs/ARCHITECTURE.md §5.8. Three tiers (Active / Regional /
World Map) with promotion-demotion as the player moves. The World-Map tier must
be a single array-based numpy pass over all agents, never 10k Python objects.

Imports allowed (ARCHITECTURE.md §4a.2): world, procedural, core.
"""

from __future__ import annotations

import numpy as np

__all__ = ["NPCArchetype"]


class NPCArchetype:
    """Stub for an NPC archetype (procedural character type).

    Implement in a later session per ARCHITECTURE.md §5.8.
    """

    def generate(self, rng: np.random.Generator, **params: object) -> None:
        raise NotImplementedError(
            "NPCArchetype.generate is future scope — see ARCHITECTURE.md §5.8 "
            "(AI API). Not part of Session 1."
        )
