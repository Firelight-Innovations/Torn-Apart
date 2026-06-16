"""
tests/procedural/flora/test_species_def.py
— Tests for fire_engine/procedural/flora/species_def.py.

Covers TreeSpeciesDef base class and its generate() pipeline, producing
TreeVariantSet.  Uses a minimal concrete species to exercise the pipeline
without depending on specific species scripts.
Headless — no panda3d imports.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fire_engine.core.rng import set_world_seed
from fire_engine.procedural.flora.leaves import Leaves, leaves_at_tips
from fire_engine.procedural.flora.skeleton import SkeletonBuilder, TreeSkeleton
from fire_engine.procedural.flora.species_def import TreeSpeciesDef, TreeVariantSet

# ---------------------------------------------------------------------------
# Minimal concrete species for testing
# ---------------------------------------------------------------------------


class _TinyTreeDef(TreeSpeciesDef):
    """Minimal species: 2 variants, short trunk, a few limbs."""

    name = "_test_tiny_tree"
    variants = 2
    impostor_cell = (32, 48)

    BARK_PALETTE = np.array([(40, 30, 20), (70, 55, 38)], dtype=np.uint8)
    LEAF_PALETTE = np.array([(30, 50, 20), (55, 85, 35)], dtype=np.uint8)

    def grow(self, rng: np.random.Generator, variant: int) -> tuple[TreeSkeleton, Leaves]:
        sb = SkeletonBuilder(rng)
        trunk = sb.trunk(height_m=3.0, base_radius_m=0.18, segments=3, wobble_m=0.1)
        limbs = sb.branches(
            trunk,
            count=(2, 3),
            t_range=(0.4, 0.85),
            pitch_set=(math.radians(80),),
            length_ratio=(0.35, 0.55),
            radius_ratio=0.5,
        )
        sk = sb.skeleton()
        lv = leaves_at_tips(sk, limbs, rng)
        return sk, lv


def _generate_tiny(seed: int = 1337) -> TreeVariantSet:
    set_world_seed(seed)
    from fire_engine.procedural.registry import clear_cache, get, register

    register(_TinyTreeDef())
    clear_cache()
    return get("_test_tiny_tree")


# ---------------------------------------------------------------------------
# TreeVariantSet output structure
# ---------------------------------------------------------------------------


class TestTreeSpeciesDefOutput:
    def test_returns_tree_variant_set(self):
        vs = _generate_tiny()
        assert isinstance(vs, TreeVariantSet)

    def test_name_matches_def(self):
        vs = _generate_tiny()
        assert vs.name == "_test_tiny_tree"

    def test_n_variants_matches_class_attr(self):
        vs = _generate_tiny()
        assert vs.n_variants == _TinyTreeDef.variants

    def test_atlas_shape_and_dtype(self):
        vs = _generate_tiny()
        assert vs.atlas.ndim == 3
        assert vs.atlas.shape[2] == 4
        assert vs.atlas.dtype == np.uint8

    def test_impostors_shape_and_dtype(self):
        vs = _generate_tiny()
        assert vs.impostors.ndim == 3
        assert vs.impostors.shape[2] == 4
        assert vs.impostors.dtype == np.uint8

    def test_max_height_positive(self):
        vs = _generate_tiny()
        assert vs.max_height_m > 0.0

    def test_max_radius_positive(self):
        vs = _generate_tiny()
        assert vs.max_radius_m > 0.0

    def test_impostor_dimensions_positive(self):
        vs = _generate_tiny()
        assert vs.impostor_width_m > 0.0
        assert vs.impostor_height_m > 0.0

    def test_all_meshes_have_positions(self):
        vs = _generate_tiny()
        for mesh in vs.meshes:
            assert mesh.positions.ndim == 2
            assert mesh.positions.shape[1] == 3
            assert len(mesh.positions) > 0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestTreeSpeciesDefDeterminism:
    def test_same_seed_identical_atlas(self):
        vs1 = _generate_tiny(seed=42)
        vs2 = _generate_tiny(seed=42)
        assert np.array_equal(vs1.atlas, vs2.atlas)

    def test_same_seed_same_variant_count(self):
        vs1 = _generate_tiny(seed=42)
        vs2 = _generate_tiny(seed=42)
        assert vs1.n_variants == vs2.n_variants

    def test_different_seeds_different_atlas(self):
        vs1 = _generate_tiny(seed=1)
        vs2 = _generate_tiny(seed=2)
        assert not np.array_equal(vs1.atlas, vs2.atlas)


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


class TestTreeSpeciesDefHooks:
    def test_grow_not_implemented_raises(self):
        """Base TreeSpeciesDef.grow() raises NotImplementedError."""
        base = TreeSpeciesDef.__new__(TreeSpeciesDef)
        rng = np.random.default_rng(0)
        with pytest.raises(NotImplementedError):
            base.grow(rng, 0)

    def test_palettes_returns_bark_and_leaf(self):
        """Default palettes() returns the class-level arrays."""
        obj = _TinyTreeDef()
        rng = np.random.default_rng(0)
        pal = obj.palettes(rng)
        assert "bark" in pal
        assert "leaf" in pal
        assert np.array_equal(pal["bark"], _TinyTreeDef.BARK_PALETTE)
        assert np.array_equal(pal["leaf"], _TinyTreeDef.LEAF_PALETTE)

    def test_variants_param_override(self):
        """variants= keyword argument overrides pool size."""
        set_world_seed(5)
        from fire_engine.procedural.registry import clear_cache, get, register

        register(_TinyTreeDef())
        clear_cache()
        vs = get("_test_tiny_tree", variants=1)
        assert vs.n_variants == 1
