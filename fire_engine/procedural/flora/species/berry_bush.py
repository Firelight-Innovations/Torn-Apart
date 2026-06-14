"""
procedural/flora/species/berry_bush.py — "bush_berry" species script.

A rounder, denser, living-green bush speckled with washed-red berries.
Demonstrates **texture customization**: the berry pass is two class
attributes (``BERRY_COLOR`` / ``BERRY_DENSITY``) consumed by the shared
atlas pipeline, plus a per-world hue-drift override of :meth:`palettes`.
Future forage gameplay can key off the species name.

Registered as ``"bush_berry"`` at import time.

Usage
-----
::

    set_world_seed(1337)
    berries = get("bush_berry")           # TreeVariantSet, 6 unique meshes
    # Preview: python tools/preview_tree.py bush_berry --obj --png
"""

from __future__ import annotations

import math

import numpy as np

from fire_engine.procedural.defs import register_def
from fire_engine.procedural.flora.leaves import Leaves, leaves_at_tips
from fire_engine.procedural.flora.skeleton import SkeletonBuilder, TreeSkeleton
from fire_engine.procedural.flora.species_def import TreeSpeciesDef

__all__ = ["BerryBushDef"]

_D = math.radians


@register_def
class BerryBushDef(TreeSpeciesDef):
    """
    Berry bush — compact living-green dome flecked with red berries.

    Registered name: ``"bush_berry"``.  6 variants per world.
    """

    name = "bush_berry"
    variants = 6
    impostor_cell = (48, 48)

    BARK_PALETTE = np.array([(42, 34, 24), (58, 48, 33), (74, 62, 43)], dtype=np.uint8)
    # The one genuinely alive ramp in the wasteland palette.
    LEAF_PALETTE = np.array(
        [(28, 48, 26), (40, 66, 34), (54, 84, 42), (72, 102, 52), (92, 118, 62)], dtype=np.uint8
    )
    LEAF_HOLE_THRESH = 0.12  # dense dome, few gaps
    BERRY_COLOR = (168, 86, 72)  # washed poppy red (flower-atlas kin)
    BERRY_DENSITY = 0.035

    def palettes(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Drift the green channel a touch per world (±8) — no two worlds'
        berry thickets read quite the same hue."""
        drift = int(rng.integers(-8, 9))
        leaf = self.LEAF_PALETTE.astype(np.int16)
        leaf[:, 1] = np.clip(leaf[:, 1] + drift, 0, 255)
        return {"bark": self.BARK_PALETTE, "leaf": leaf.astype(np.uint8)}

    def grow(self, rng: np.random.Generator, variant: int) -> tuple[TreeSkeleton, Leaves]:
        """Stub trunk → upcurled stems → dense overlapping foliage dome."""
        sb = SkeletonBuilder(rng)
        stub = sb.trunk(height_m=0.12, base_radius_m=0.05, segments=1, wobble_m=0.0)
        stems = sb.branches(
            stub,
            count=(5, 8),
            t_range=(0.5, 1.0),
            pitch_set=(_D(35), _D(55)),  # rounder dome
            pitch_jitter_rad=_D(8),
            yaw_mode="spiral",  # even all-around fill
            length_m=(0.35, 0.6),
            radius_ratio=0.7,
            min_radius_m=0.012,
            upturn_rad=_D(30),
            bend_rad=0.2,
            segments=2,
        )
        sk = sb.skeleton()
        # Dense dome: high density + 2 leaves per cell close ranks.
        leaves = leaves_at_tips(
            sk,
            stems,
            rng,
            cell_m=0.15,
            rounds=2,
            density=0.9,
            per_cell=(1, 2),
            leaf_size_m=(0.06, 0.10),
            sway_min=0.75,
            max_leaves=280,
        )
        return sk, leaves
