"""Building Manager API — procedural buildings from BuildingDef scripts.

Session 1 stub. See docs/ARCHITECTURE.md §5.7. Buildings are generated at
runtime from `BuildingDef` (a `ProceduralDef` subclass): footprint, floors,
room layout, facade, furniture rules — all from blocks + primitives with
procedural textures. Only hand-crafted landmarks come from the Resource Manager.

Imports allowed (ARCHITECTURE.md §4a.2): procedural, terrain, core.
"""

__all__ = ["BuildingDef"]


class BuildingDef:
    """Stub for a procedural building archetype.

    Implement in a later session per ARCHITECTURE.md §5.7. A BuildingDef will
    subclass procedural.ProceduralDef and emit block/primitive geometry.
    """

    def generate(self, rng, **params):  # noqa: D401, ANN001, ANN003
        raise NotImplementedError(
            "BuildingDef.generate is future scope — see ARCHITECTURE.md §5.7 "
            "(Building Manager API). Not part of Session 1."
        )
