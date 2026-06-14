"""Politics API — factions, settlement ownership, world-map events.

Session 1 stub. See docs/ARCHITECTURE.md §5.10. Publishes events consumed by
Economy (war disrupts routes) and AI (allegiance, morale).

Imports allowed (ARCHITECTURE.md §4a.2): procedural, core.
"""

__all__ = ["FactionDef"]


class FactionDef:
    """Stub for a faction definition.

    Implement in a later session per ARCHITECTURE.md §5.10.
    """

    def generate(self, rng, **params):  # noqa: ANN001, ANN003
        raise NotImplementedError(
            "FactionDef.generate is future scope — see ARCHITECTURE.md §5.10 "
            "(Politics API). Not part of Session 1."
        )
