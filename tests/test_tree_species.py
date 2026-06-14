"""
tests/test_tree_species.py — TreeSpeciesDef pipeline + built-in species.

Pins the registry contract (cache identity, determinism per world seed),
the TreeVariantSet invariants (pool size, distinct variants, V3N3T2C4 mesh
contract) and the texture contracts (bark opaque, leaf/impostor binary
alpha, impostor trunk base on the bottom row) for all four built-in
species scripts.
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.core.rng import set_world_seed
from fire_engine.procedural import clear_cache, get
from fire_engine.procedural.flora import TreeVariantSet

SPECIES = ("tree_gnarled_oak", "tree_dead", "bush_scrub", "bush_berry")
POOL_SIZES = {"tree_gnarled_oak": 8, "tree_dead": 6, "bush_scrub": 6, "bush_berry": 6}


@pytest.fixture(autouse=True)
def _seeded():
    _ensure_species_registered()
    set_world_seed(1337)
    clear_cache()
    yield
    clear_cache()


def _ensure_species_registered() -> None:
    """Re-register the species defs if another test reset the registry
    (test_procedural's _fresh_registry re-registers only its own subset —
    the same workaround test_sky_atmosphere uses for moon_surface)."""
    from fire_engine.procedural import registry

    if "tree_gnarled_oak" in registry._registry:
        return
    from fire_engine.procedural.flora.species import (
        BerryBushDef,
        DeadTreeDef,
        GnarledOakDef,
        ScrubBushDef,
    )

    for cls in (GnarledOakDef, DeadTreeDef, ScrubBushDef, BerryBushDef):
        registry.register(cls())


class TestRegistry:
    @pytest.mark.parametrize("name", SPECIES)
    def test_round_trip_and_cache_identity(self, name):
        vs1 = get(name)
        vs2 = get(name)
        assert isinstance(vs1, TreeVariantSet)
        assert vs1 is vs2  # registry cache identity
        assert vs1.name == name

    @pytest.mark.parametrize("name", SPECIES)
    def test_pool_size(self, name):
        assert get(name).n_variants == POOL_SIZES[name]

    def test_variants_param_override(self):
        vs = get("tree_gnarled_oak", variants=3)
        assert vs.n_variants == 3


class TestDeterminism:
    def test_same_seed_byte_identical(self):
        vs1 = get("tree_gnarled_oak")
        set_world_seed(1337)
        clear_cache()
        vs2 = get("tree_gnarled_oak")
        for m1, m2 in zip(vs1.meshes, vs2.meshes):
            assert np.array_equal(m1.positions, m2.positions)
            assert np.array_equal(m1.colors, m2.colors)
            assert np.array_equal(m1.indices, m2.indices)
        assert np.array_equal(vs1.atlas, vs2.atlas)
        assert np.array_equal(vs1.impostors, vs2.impostors)

    def test_different_seed_differs(self):
        vs1 = get("tree_gnarled_oak")
        set_world_seed(31337)
        clear_cache()
        vs2 = get("tree_gnarled_oak")
        assert not np.array_equal(vs1.meshes[0].positions, vs2.meshes[0].positions)

    def test_variants_mutually_distinct(self):
        vs = get("tree_gnarled_oak")
        m0 = vs.meshes[0]
        assert any(
            m.n_vertices != m0.n_vertices or not np.array_equal(m.positions, m0.positions)
            for m in vs.meshes[1:]
        )


class TestVariantSetInvariants:
    @pytest.mark.parametrize("name", SPECIES)
    def test_mesh_contract(self, name):
        vs = get(name)
        for m in vs.meshes:
            assert m.n_vertices > 0
            assert m.positions.dtype == np.float32
            assert m.indices.dtype == np.uint32
            assert int(m.indices.max()) < m.n_vertices
            assert np.allclose(np.linalg.norm(m.normals, axis=1), 1.0, atol=1e-4)
            assert (m.colors[:, 3] >= 0.0).all() and (m.colors[:, 3] <= 1.0).all()
            assert m.height_m <= vs.max_height_m + 1e-6
            assert m.radius_m <= vs.max_radius_m + 1e-6

    @pytest.mark.parametrize("name", SPECIES)
    def test_atlas_contract(self, name):
        atlas = get(name).atlas
        assert atlas.dtype == np.uint8 and atlas.shape == (64, 64, 4)
        hw = atlas.shape[1] // 2
        assert (atlas[:, :hw, 3] == 255).all()  # bark opaque
        leaf_a = atlas[:, hw:, 3]
        assert ((leaf_a == 0) | (leaf_a == 255)).all()  # leaf binary
        assert leaf_a.any()  # some foliage texels

    @pytest.mark.parametrize("name", SPECIES)
    def test_impostor_contract(self, name):
        vs = get(name)
        imp = vs.impostors
        assert imp.dtype == np.uint8
        cell_w = imp.shape[1] // vs.n_variants
        assert imp.shape[1] == cell_w * vs.n_variants
        a = imp[..., 3]
        assert ((a == 0) | (a == 255)).all()  # binary alpha
        assert imp[-1, :, 3].any()  # base on bottom row
        # Every cell has content (no blank variant sprite).
        for k in range(vs.n_variants):
            assert imp[:, k * cell_w : (k + 1) * cell_w, 3].any()

    def test_dead_tree_nearly_leafless(self):
        """The leafless species path: snags carry almost no foliage."""
        dead = get("tree_dead")
        oak = get("tree_gnarled_oak")

        def leaf_count(m):
            # Leaf cards span the leaf half, so only they own u=1.0
            # corners (bark stops at u=0.5): 2 far corners per leaf.
            return int((m.uvs[:, 0] > 0.55).sum()) // 2

        # Hard cap from the species script (max_leaves=36 per tuft bake)...
        assert all(leaf_count(m) <= 36 for m in dead.meshes)
        # ...and an order of magnitude under the living oak's canopy.
        assert max(leaf_count(m) for m in dead.meshes) < min(leaf_count(m) for m in oak.meshes) / 3

    def test_berry_bush_has_berries(self):
        """Berry speckle pass: the leaf half contains the berry hue."""
        atlas = get("bush_berry").atlas
        hw = atlas.shape[1] // 2
        leaf = atlas[:, hw:]
        opaque = leaf[..., 3] == 255
        red = (leaf[..., 0].astype(int) - leaf[..., 1].astype(int)) > 40
        assert (opaque & red).sum() > 3
