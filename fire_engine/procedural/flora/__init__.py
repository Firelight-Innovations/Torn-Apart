"""
fire_engine.procedural.flora — procedural 3-D tree/bush generation.

The species-script pipeline (headless, deterministic, numpy-only):

1. A **species script** (``flora/species/*.py``) subclasses
   :class:`TreeSpeciesDef` and grows a branch skeleton with
   :class:`SkeletonBuilder` — the "node-graph editor in Python".
2. :func:`leaf_clusters_at_tips` foliates the branch tips.
3. The shared mesher (:func:`mesh_branches` / :func:`mesh_leaf_clusters` /
   :func:`merge_parts`) emits :class:`TreeMesh` arrays in the engine's
   V3N3T2C4 layout, with per-vertex wind-sway weights in ``color.a``.
4. ``atlas.py`` composes the species' bark + leaf pixel-art texture;
   ``impostor.py`` software-rasterizes far-LOD sprites.
5. ``registry.get("tree_<species>")`` returns the cached
   :class:`TreeVariantSet` — a per-world-seed pool of unique variant meshes.

Authoring guide: ``docs/content/tree_species_authoring.md``.
Renderer: ``world/tree_renderer.py`` (instanced; placement from
``zones/tree_placement.py``).
"""

from fire_engine.procedural.flora.skeleton import (
    SkeletonBuilder,
    TreeSkeleton,
    validate_skeleton,
)
from fire_engine.procedural.flora.leaves import (
    LeafClusters,
    leaf_clusters_at_tips,
)
from fire_engine.procedural.flora.mesher import (
    TreeMesh,
    merge_parts,
    mesh_branches,
    mesh_leaf_clusters,
)
from fire_engine.procedural.flora.atlas import (
    AtlasLayout,
    bark_texture,
    compose_atlas,
    leaf_texture,
)
from fire_engine.procedural.flora.impostor import (
    impostor_atlas,
    rasterize_impostor,
)
from fire_engine.procedural.flora.species_def import (
    TreeSpeciesDef,
    TreeVariantSet,
)
# Importing the species sub-package registers all built-in species defs.
import fire_engine.procedural.flora.species  # noqa: F401

__all__ = [
    "SkeletonBuilder",
    "TreeSkeleton",
    "validate_skeleton",
    "LeafClusters",
    "leaf_clusters_at_tips",
    "TreeMesh",
    "mesh_branches",
    "mesh_leaf_clusters",
    "merge_parts",
    "AtlasLayout",
    "bark_texture",
    "leaf_texture",
    "compose_atlas",
    "rasterize_impostor",
    "impostor_atlas",
    "TreeSpeciesDef",
    "TreeVariantSet",
]
