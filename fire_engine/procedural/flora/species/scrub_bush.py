"""
procedural/flora/species/scrub_bush.py — "bush_scrub" species script.

Dry wasteland scrub: a stub trunk with 4–7 splayed crooked stems and dusty
olive foliage.  Demonstrates the **bush path** of the one shared generator —
a bush is just a tree with a near-zero trunk and absolute stem lengths
(``length_m`` instead of ``length_ratio``, because ratios of a 0.15 m stub
are meaningless).

Registered as ``"bush_scrub"`` at import time.

Usage
-----
::

    set_world_seed(1337)
    scrub = get("bush_scrub")             # TreeVariantSet, 6 unique meshes
    # Preview: python tools/preview_tree.py bush_scrub --obj --png

Docs: docs/systems/procedural.flora.species.md
"""

from __future__ import annotations

import math

import numpy as np

from fire_engine.procedural.defs import register_def
from fire_engine.procedural.flora.leaves import Leaves, leaves_at_tips
from fire_engine.procedural.flora.skeleton import SkeletonBuilder, TreeSkeleton
from fire_engine.procedural.flora.species_def import TreeSpeciesDef

__all__ = ["ScrubBushDef"]

_D = math.radians


@register_def
class ScrubBushDef(TreeSpeciesDef):
    """
    Dry scrub bush — ≈1 m of splayed woody stems, dusty olive foliage.

    Registered name: ``"bush_scrub"``.  6 variants per world.

    Docs: docs/systems/procedural.flora.species.md
    """

    name = "bush_scrub"
    variants = 6
    impostor_cell = (48, 48)  # bushes are square-ish, smaller sprites

    BARK_PALETTE = np.array([(46, 38, 26), (64, 53, 36), (82, 68, 47)], dtype=np.uint8)
    # Dry olive, halfway to the dead ramp — wasteland scrub, not garden box.
    LEAF_PALETTE = np.array(
        [(40, 46, 28), (56, 62, 36), (74, 78, 46), (94, 96, 56)], dtype=np.uint8
    )
    LEAF_HOLE_THRESH = 0.22

    def grow(self, rng: np.random.Generator, variant: int) -> tuple[TreeSkeleton, Leaves]:
        """Stub trunk → splayed stems (absolute lengths) → foliage.

        Docs: docs/systems/procedural.flora.species.md
        """
        sb = SkeletonBuilder(rng)
        stub = sb.trunk(height_m=0.15, base_radius_m=0.06, segments=1, wobble_m=0.0)
        stems = sb.branches(
            stub,
            count=(4, 7),
            t_range=(0.6, 1.0),
            pitch_set=(_D(50), _D(70)),  # splayed, not flat
            pitch_jitter_rad=_D(10),
            yaw_mode="random",
            length_m=(0.5, 0.9),  # absolute — stub trunk
            radius_ratio=0.7,
            min_radius_m=0.015,
            upturn_rad=_D(20),
            bend_rad=0.25,
            segments=2,
        )
        sk = sb.skeleton()
        # Small dry leaves in loose tufts — scrub stays see-through.
        leaves = leaves_at_tips(
            sk,
            stems,
            rng,
            cell_m=0.16,
            rounds=2,
            density=0.8,
            leaf_size_m=(0.07, 0.11),
            sway_min=0.8,
            max_leaves=240,
        )
        return sk, leaves
