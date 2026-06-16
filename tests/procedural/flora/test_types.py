"""
tests/procedural/flora/test_types.py
— Tests for fire_engine/procedural/flora/types.py.

Covers TreeSkeleton, validate_skeleton (re-exports), and TreeVariantSet.
Headless — no panda3d imports.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fire_engine.core.rng import for_domain, set_world_seed
from fire_engine.procedural.flora.types import (
    TreeSkeleton,
    TreeVariantSet,
    validate_skeleton,
)

# ---------------------------------------------------------------------------
# Re-export correctness
# ---------------------------------------------------------------------------


class TestReexports:
    def test_tree_skeleton_is_same_class(self):
        from fire_engine.procedural.flora.skeleton import TreeSkeleton as SkSkel

        assert TreeSkeleton is SkSkel

    def test_validate_skeleton_is_same_function(self):
        from fire_engine.procedural.flora.skeleton import validate_skeleton as skval

        assert validate_skeleton is skval


# ---------------------------------------------------------------------------
# TreeSkeleton dataclass
# ---------------------------------------------------------------------------


def _minimal_skeleton() -> TreeSkeleton:
    """Build a single-segment trunk skeleton for field tests."""
    set_world_seed(1)
    from fire_engine.procedural.flora.skeleton import SkeletonBuilder

    rng = for_domain("test", "types")
    sb = SkeletonBuilder(rng)
    sb.trunk(height_m=3.0, base_radius_m=0.2, segments=2, wobble_m=0.0)
    return sb.skeleton()


class TestTreeSkeletonDataclass:
    def test_instantiated_via_builder(self):
        sk = _minimal_skeleton()
        assert isinstance(sk, TreeSkeleton)

    def test_n_segments_property(self):
        sk = _minimal_skeleton()
        assert sk.n_segments == len(sk.parent)
        assert sk.n_segments > 0

    def test_sway_start_shape(self):
        sk = _minimal_skeleton()
        sw = sk.sway_start()
        assert sw.shape == (sk.n_segments,)
        assert sw.dtype == np.float32

    def test_tip_ids_all_leaves(self):
        set_world_seed(2)
        from fire_engine.procedural.flora.skeleton import SkeletonBuilder

        rng = for_domain("test", "types_tips")
        sb = SkeletonBuilder(rng)
        trunk = sb.trunk(height_m=4.0, base_radius_m=0.2)
        sb.branches(
            trunk,
            count=(2, 3),
            t_range=(0.5, 0.9),
            pitch_set=(math.radians(85),),
        )
        sk = sb.skeleton()
        tips = sk.tip_ids()
        parent_set = set(int(p) for p in sk.parent if p >= 0)
        for t in tips.tolist():
            assert t not in parent_set, f"tip_id {t} should not be a parent"

    def test_tip_ids_subset(self):
        set_world_seed(3)
        from fire_engine.procedural.flora.skeleton import SkeletonBuilder

        rng = for_domain("test", "types_tips_subset")
        sb = SkeletonBuilder(rng)
        trunk = sb.trunk(height_m=4.0, base_radius_m=0.2)
        limbs = sb.branches(trunk, count=(2, 3), t_range=(0.5, 0.9), pitch_set=(math.radians(85),))
        sk = sb.skeleton()
        # tip_ids filtered to limbs subset must be a subset of all tips
        all_tips = set(sk.tip_ids().tolist())
        limb_tips = set(sk.tip_ids(np.array(limbs)).tolist())
        assert limb_tips <= all_tips


# ---------------------------------------------------------------------------
# Tests for validate_skeleton re-export
# ---------------------------------------------------------------------------


class TestValidateSkeletonReexport:
    def test_passes_on_valid_skeleton(self):
        sk = _minimal_skeleton()
        validate_skeleton(sk)  # must not raise

    def test_radius_growth_raises(self):
        sk = _minimal_skeleton()
        # Make one segment's tip radius larger than start
        sk.radius_end[0] = sk.radius_start[0] + 0.5
        with pytest.raises(ValueError, match="radius"):
            validate_skeleton(sk)


# ---------------------------------------------------------------------------
# TreeVariantSet
# ---------------------------------------------------------------------------


class TestTreeVariantSet:
    def _dummy_mesh(self):
        """Create a minimal TreeMesh for TreeVariantSet construction."""
        from fire_engine.procedural.flora.mesher import TreeMesh

        return TreeMesh.empty()

    def test_instantiation(self):
        mesh = self._dummy_mesh()
        vs = TreeVariantSet(
            name="test_species",
            meshes=(mesh,),
            atlas=np.zeros((64, 64, 4), dtype=np.uint8),
            impostors=np.zeros((96, 64, 4), dtype=np.uint8),
            max_height_m=5.0,
            max_radius_m=1.5,
            impostor_width_m=3.0,
            impostor_height_m=6.0,
        )
        assert vs.name == "test_species"
        assert vs.n_variants == 1

    def test_frozen(self):
        """TreeVariantSet is frozen — attribute assignment must raise."""
        mesh = self._dummy_mesh()
        vs = TreeVariantSet(
            name="frozen_test",
            meshes=(mesh,),
            atlas=np.zeros((64, 64, 4), dtype=np.uint8),
            impostors=np.zeros((96, 64, 4), dtype=np.uint8),
            max_height_m=5.0,
            max_radius_m=1.0,
            impostor_width_m=2.0,
            impostor_height_m=5.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            vs.name = "other"  # type: ignore[misc]

    def test_n_variants_matches_meshes_length(self):
        mesh = self._dummy_mesh()
        vs = TreeVariantSet(
            name="multi",
            meshes=(mesh, mesh, mesh),
            atlas=np.zeros((64, 64, 4), dtype=np.uint8),
            impostors=np.zeros((96, 192, 4), dtype=np.uint8),
            max_height_m=6.0,
            max_radius_m=2.0,
            impostor_width_m=3.0,
            impostor_height_m=6.0,
        )
        assert vs.n_variants == 3
