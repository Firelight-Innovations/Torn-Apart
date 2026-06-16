"""
procedural/flora/species/gnarled_oak.py — "tree_gnarled_oak" species script.

The REFERENCE tree species: a thick crooked wasteland oak with tiered
near-right-angle limbs (the blocky *Dynamic Trees* silhouette), upturned
twigs and a ragged olive canopy.  Copy this file to author a new tree —
every knob below is a deliberate example of the SkeletonBuilder API
(authoring guide: ``docs/content/tree_species_authoring.md``).

Registered as ``"tree_gnarled_oak"`` at import time.

Usage
-----
::

    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural import get

    set_world_seed(1337)
    oaks = get("tree_gnarled_oak")        # TreeVariantSet, 8 unique meshes
    # Preview: python tools/preview_tree.py tree_gnarled_oak --obj --png

Docs: docs/systems/procedural.flora.species.md
"""

from __future__ import annotations

import math

import numpy as np

from fire_engine.procedural.defs import register_def
from fire_engine.procedural.flora.leaves import Leaves, leaves_at_tips
from fire_engine.procedural.flora.skeleton import SkeletonBuilder, TreeSkeleton
from fire_engine.procedural.flora.species_def import TreeSpeciesDef

__all__ = ["GnarledOakDef"]

_D = math.radians


@register_def
class GnarledOakDef(TreeSpeciesDef):
    """
    Gnarled wasteland oak — 5–7 m, crooked trunk, blocky tiered limbs.

    Registered name: ``"tree_gnarled_oak"``.  8 variants per world.

    Docs: docs/systems/procedural.flora.species.md
    """

    name = "tree_gnarled_oak"
    variants = 8

    # Oak bark: deep umber shadow → grey-brown lit face.
    BARK_PALETTE = np.array([(40, 31, 22), (58, 46, 33), (79, 63, 45)], dtype=np.uint8)
    # Recovering-wasteland olive canopy, dark under-storey → pale crown.
    LEAF_PALETTE = np.array(
        [(30, 44, 26), (44, 62, 34), (60, 80, 42), (80, 98, 52), (104, 116, 64)], dtype=np.uint8
    )
    LEAF_HOLE_THRESH = 0.16

    def grow(self, rng: np.random.Generator, variant: int) -> tuple[TreeSkeleton, Leaves]:
        """Crooked trunk → near-90° limbs (shorter near the crown) → twigs.

        Docs: docs/systems/procedural.flora.species.md
        """
        sb = SkeletonBuilder(rng)
        trunk = sb.trunk(
            height_m=5.5 + float(rng.uniform(-1.0, 1.5)),
            base_radius_m=0.28,
            tip_radius_m=0.07,
            segments=4,
            wobble_m=0.35,
            lean_rad=_D(float(rng.uniform(0.0, 6.0))),
        )
        limbs = sb.branches(
            trunk,
            count=(1, 2),
            t_range=(0.35, 0.95),
            pitch_set=(_D(80), _D(95)),  # blocky right angles
            pitch_jitter_rad=_D(8),
            yaw_mode="spiral",
            length_ratio=(0.35, 0.55),
            length_scale_by_height=(1.0, 0.45),
            radius_ratio=0.5,
            upturn_rad=_D(18),
            segments=2,
        )
        twigs = sb.branches(
            limbs,
            count=(1, 2),
            pitch_set=(_D(85),),
            length_ratio=(0.35, 0.5),
            radius_ratio=0.5,
            upturn_rad=_D(25),
        )
        sk = sb.skeleton()
        # CA canopy: each twig tip hydrates a ~0.8 m dome of individual
        # leaves; the ragged-oak silhouette is the twig layout itself.
        leaves = leaves_at_tips(
            sk,
            np.concatenate([limbs, twigs]),
            rng,
            cell_m=0.26,
            rounds=3,
            density=0.85,
            leaf_size_m=(0.12, 0.18),
            max_leaves=420,
        )
        return sk, leaves
