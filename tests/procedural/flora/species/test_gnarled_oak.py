"""
tests/procedural/flora/species/test_gnarled_oak.py
— Tests for fire_engine/procedural/flora/species/gnarled_oak.py.

Covers GnarledOakDef: registration, TreeVariantSet output, and determinism.
Headless — no panda3d imports.
"""

from __future__ import annotations

import numpy as np

from fire_engine.core.rng import set_world_seed
from fire_engine.procedural.flora.species_def import TreeVariantSet


def _gen(seed: int = 1337) -> TreeVariantSet:
    set_world_seed(seed)
    from fire_engine.procedural import clear_cache, get

    clear_cache()
    return get("tree_gnarled_oak")


class TestGnarledOakRegistration:
    def test_name_is_tree_gnarled_oak(self):
        from fire_engine.procedural.flora.species.gnarled_oak import GnarledOakDef

        assert GnarledOakDef.name == "tree_gnarled_oak"

    def test_registered_at_import(self):
        # Importing the species registers it; get() must succeed
        import fire_engine.procedural.flora.species.gnarled_oak  # noqa: F401
        from fire_engine.procedural.registry import get

        set_world_seed(0)
        vs = get("tree_gnarled_oak")
        assert isinstance(vs, TreeVariantSet)


class TestGnarledOakOutput:
    def test_returns_variant_set(self):
        assert isinstance(_gen(), TreeVariantSet)

    def test_name_on_variant_set(self):
        assert _gen().name == "tree_gnarled_oak"

    def test_variants_pool_size(self):
        from fire_engine.procedural.flora.species.gnarled_oak import GnarledOakDef

        vs = _gen()
        assert vs.n_variants == GnarledOakDef.variants

    def test_atlas_shape(self):
        vs = _gen()
        assert vs.atlas.ndim == 3
        assert vs.atlas.shape[2] == 4
        assert vs.atlas.dtype == np.uint8

    def test_impostors_shape(self):
        vs = _gen()
        assert vs.impostors.ndim == 3
        assert vs.impostors.shape[2] == 4

    def test_max_height_plausible(self):
        """Gnarled oak should be between 3 m and 15 m."""
        vs = _gen()
        assert 3.0 <= vs.max_height_m <= 15.0, f"Unexpected height: {vs.max_height_m}"

    def test_all_meshes_non_empty(self):
        vs = _gen()
        for mesh in vs.meshes:
            assert len(mesh.positions) > 0


class TestGnarledOakDeterminism:
    def test_same_seed_identical_atlas(self):
        vs1 = _gen(seed=42)
        vs2 = _gen(seed=42)
        assert np.array_equal(vs1.atlas, vs2.atlas)

    def test_different_seeds_different_atlas(self):
        vs1 = _gen(seed=1)
        vs2 = _gen(seed=2)
        assert not np.array_equal(vs1.atlas, vs2.atlas)
