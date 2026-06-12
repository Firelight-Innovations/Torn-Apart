"""
procedural/flora/species/dead_tree.py — "tree_dead" species script.

A tall bare snag: twisted drooping limbs, almost no foliage (0–2 dry tufts
per tree), washed grey-brown bark.  Demonstrates the **leafless** species
path — most variants return very few or zero leaf clusters, and the
posterise ramps sell the dead wood.  Post-apocalyptic treelines mix these
with living oaks via volume ``species_mix`` params.

Registered as ``"tree_dead"`` at import time.

Usage
-----
::

    set_world_seed(1337)
    snags = get("tree_dead")              # TreeVariantSet, 6 unique meshes
    # Preview: python tools/preview_tree.py tree_dead --obj --png
"""

from __future__ import annotations

import math

import numpy as np

from fire_engine.procedural.defs import register_def
from fire_engine.procedural.flora.leaves import (
    LeafClusters,
    leaf_clusters_at_tips,
)
from fire_engine.procedural.flora.skeleton import SkeletonBuilder, TreeSkeleton
from fire_engine.procedural.flora.species_def import TreeSpeciesDef

__all__ = ["DeadTreeDef"]

_D = math.radians


@register_def
class DeadTreeDef(TreeSpeciesDef):
    """
    Dead snag — 6–9 m bare twisted tree, 0–2 dry leaf tufts.

    Registered name: ``"tree_dead"``.  6 variants per world.
    """

    name = "tree_dead"
    variants = 6

    # Weathered grey driftwood ramp.
    BARK_PALETTE = np.array([(42, 38, 32), (62, 57, 48), (84, 78, 66)],
                            dtype=np.uint8)
    # The few surviving tufts read dry and dusty.
    LEAF_PALETTE = np.array([(52, 48, 30), (72, 66, 40), (94, 86, 52),
                             (112, 102, 64)], dtype=np.uint8)
    LEAF_HOLE_THRESH = 0.30          # scraggly, mostly gaps

    def grow(self, rng: np.random.Generator,
             variant: int) -> tuple[TreeSkeleton, LeafClusters]:
        """Tall lean trunk → sparse drooping limbs → crooked twig ends."""
        sb = SkeletonBuilder(rng)
        trunk = sb.trunk(height_m=6.5 + float(rng.uniform(-0.5, 2.5)),
                         base_radius_m=0.24, tip_radius_m=0.04,
                         segments=5, wobble_m=0.5,
                         lean_rad=_D(float(rng.uniform(2.0, 10.0))))
        limbs = sb.branches(trunk, count=(0, 2), t_range=(0.4, 0.98),
                            pitch_set=(_D(70), _D(95), _D(110)),
                            pitch_jitter_rad=_D(15),    # twisted, irregular
                            yaw_mode="random",
                            length_ratio=(0.25, 0.45),
                            length_scale_by_height=(1.0, 0.55),
                            radius_ratio=0.45,
                            droop_rad=_D(12),           # dead limbs sag
                            bend_rad=0.3, segments=2)
        twigs = sb.branches(limbs, count=(0, 2),
                            pitch_set=(_D(60), _D(100)),
                            length_ratio=(0.3, 0.5),
                            radius_ratio=0.5, bend_rad=0.35)
        sk = sb.skeleton()
        # Nearly leafless: a couple of dry tufts cling to some variants.
        ids = np.concatenate([limbs, twigs]) if twigs.size or limbs.size \
            else trunk
        leaves = leaf_clusters_at_tips(sk, ids, rng,
                                       radius_m=(0.25, 0.45),
                                       per_tip=(0, 1), sway_min=0.9)
        if leaves.n_clusters > 2:               # cap at 2 tufts per snag
            keep = rng.choice(leaves.n_clusters, size=2, replace=False)
            leaves = LeafClusters(center=leaves.center[keep],
                                  radius=leaves.radius[keep],
                                  sway=leaves.sway[keep])
        return sk, leaves
