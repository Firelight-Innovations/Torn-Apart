"""
Static-occluder helpers for TreeRendererComponent, extracted from
tree_renderer.py to satisfy the ≤500-line module limit.

Each function takes the component instance as ``self_obj`` and operates on it
directly, preserving identical runtime behaviour.

Docs: docs/systems/render.vegetation.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from fire_engine.lighting import TreeOccluderSet

if TYPE_CHECKING:
    from fire_engine.render.vegetation.tree_renderer import TreeRendererComponent

__all__ = ["push_occluders", "species_canopy_sigma", "species_splat_rgb"]


def push_occluders(self_obj: TreeRendererComponent) -> None:
    """Merge every volume's occluder set and hand it to the pipeline.

    Docs: docs/systems/render.vegetation._impl.md
    """
    if self_obj.lighting_pipeline is None:
        return
    sets = [s for s in self_obj._volume_occluders.values() if s.count]
    self_obj.lighting_pipeline.set_static_occluders(TreeOccluderSet.merge(sets) if sets else None)


def species_canopy_sigma(self_obj: TreeRendererComponent, name: str, vs: Any) -> float:
    """
    Per-meter canopy extinction for a species at scale 1.0.

    How thick the leaves are, measured from the actual meshes: mean
    one-sided leaf area over the variant pool
    (``procedural.flora.mesh_leaf_area_m2``) ÷ the canopy ellipsoid
    volume from the pool extents, × 0.5 (randomly-oriented flat cards
    present half their area to any direction).  Transmittance through
    ``X`` meters of crown centre is then ``exp(-sigma·X)`` — a leafy
    oak shades hard, a two-tuft snag barely dims the ground.  Cached
    per species; deterministic (meshes are seeded procedural content).

    Docs: docs/systems/render.vegetation._impl.md
    """
    sigma = self_obj._species_sigma.get(name)
    if sigma is None:
        from fire_engine.lighting.occluders import CANOPY_HALF_HEIGHT_FRAC
        from fire_engine.procedural.flora import mesh_leaf_area_m2

        leaf_area = float(np.mean([mesh_leaf_area_m2(m) for m in vs.meshes]))
        cv = CANOPY_HALF_HEIGHT_FRAC * float(vs.max_height_m)
        r = float(vs.max_radius_m)
        volume = (4.0 / 3.0) * np.pi * r * r * max(cv, 1e-3)
        sigma = 0.5 * leaf_area / max(volume, 1e-3)
        self_obj._species_sigma[name] = sigma
    return sigma


def species_splat_rgb(self_obj: TreeRendererComponent, name: str, vs: Any) -> tuple[Any, Any]:
    """
    Mean linear bark/leaf splat colours for a species, from its atlas.

    The atlas is bark on the left half (opaque) and the leaf card on the
    right half (binary alpha) — averaging each half gives the GI bounce
    colour for trunk/canopy cells without any new species API.  Cached
    per species; deterministic (the atlas is seeded procedural content).

    Docs: docs/systems/render.vegetation._impl.md
    """
    cached = self_obj._species_occ_rgb.get(name)
    if cached is None:
        atlas = vs.atlas.astype(np.float32) / 255.0
        half = atlas.shape[1] // 2
        bark = atlas[:, :half, :3].reshape(-1, 3).mean(axis=0)
        leaf_px = atlas[:, half:, :].reshape(-1, 4)
        sel = leaf_px[:, 3] > 0.5
        leaf = leaf_px[sel, :3].mean(axis=0) if bool(sel.any()) else bark
        # sRGB → linear (the cascade albedo channel is linear).
        cached = ((bark**2.2).astype(np.float32), (leaf**2.2).astype(np.float32))
        self_obj._species_occ_rgb[name] = cached
    return cached
