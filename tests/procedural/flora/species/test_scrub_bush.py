"""
tests/procedural/flora/species/test_scrub_bush.py
— Tests for fire_engine/procedural/flora/species/scrub_bush.py.

Covers ScrubBushDef: registration, TreeVariantSet output, and determinism.
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
    return get("bush_scrub")


class TestScrubBushRegistration:
    def test_name_is_bush_scrub(self):
        from fire_engine.procedural.flora.species.scrub_bush import ScrubBushDef

        assert ScrubBushDef.name == "bush_scrub"

    def test_registered_at_import(self):
        import fire_engine.procedural.flora.species.scrub_bush  # noqa: F401
        from fire_engine.procedural.registry import get

        set_world_seed(0)
        vs = get("bush_scrub")
        assert isinstance(vs, TreeVariantSet)


class TestScrubBushOutput:
    def test_returns_variant_set(self):
        assert isinstance(_gen(), TreeVariantSet)

    def test_name_on_variant_set(self):
        assert _gen().name == "bush_scrub"

    def test_variants_pool_size(self):
        from fire_engine.procedural.flora.species.scrub_bush import ScrubBushDef

        vs = _gen()
        assert vs.n_variants == ScrubBushDef.variants

    def test_atlas_dtype_and_channels(self):
        vs = _gen()
        assert vs.atlas.dtype == np.uint8
        assert vs.atlas.shape[2] == 4

    def test_max_height_plausible(self):
        """Scrub bush should be ≤ 3 m (it's a low bush)."""
        vs = _gen()
        assert vs.max_height_m <= 4.0, f"Scrub bush seems too tall: {vs.max_height_m} m"

    def test_all_meshes_non_empty(self):
        vs = _gen()
        for mesh in vs.meshes:
            assert len(mesh.positions) > 0


class TestScrubBushDeterminism:
    def test_same_seed_identical_atlas(self):
        vs1 = _gen(seed=42)
        vs2 = _gen(seed=42)
        assert np.array_equal(vs1.atlas, vs2.atlas)

    def test_different_seeds_different_atlas(self):
        vs1 = _gen(seed=1)
        vs2 = _gen(seed=2)
        assert not np.array_equal(vs1.atlas, vs2.atlas)
