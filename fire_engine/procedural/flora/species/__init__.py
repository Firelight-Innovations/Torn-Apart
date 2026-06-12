"""
fire_engine.procedural.flora.species — built-in tree/bush species scripts.

One file per species; importing this package registers them all
(``@register_def`` fires at import time).  To author a new species, copy
``gnarled_oak.py`` (the annotated reference) and add an import line here —
full guide in ``docs/content/tree_species_authoring.md``.
"""

from fire_engine.procedural.flora.species.gnarled_oak import GnarledOakDef
from fire_engine.procedural.flora.species.dead_tree import DeadTreeDef
from fire_engine.procedural.flora.species.scrub_bush import ScrubBushDef
from fire_engine.procedural.flora.species.berry_bush import BerryBushDef

__all__ = ["GnarledOakDef", "DeadTreeDef", "ScrubBushDef", "BerryBushDef"]
