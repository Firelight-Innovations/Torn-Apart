"""
tests/world/terrain/lod/test_desired.py — desired_node_set planner.

Headless, no panda3d.  The headline test is the REGRESSION LOCK: at
``max_rank=0`` / near_radius 6 / Z[-2..4] the near set equals the legacy
``ChunkManager.desired_set`` output exactly (the 1183-chunk square).  Also:
rank monotonic with distance, each coarse node covers ``(2**L)³`` unique chunk
columns, near/coarse disjointness, the meshgrid path runs at far_radius 64
within a time budget, and determinism.
"""

from __future__ import annotations

import itertools
import time

from fire_engine.core import EventBus, load_config
from fire_engine.core.math3d import Vec3
from fire_engine.core.rng import set_world_seed
from fire_engine.world.terrain.chunk_manager import ChunkManager
from fire_engine.world.terrain.lod.desired import NodePlan, desired_node_set
from fire_engine.world.terrain.lod.node import LodNode

_Z_BAND = (-2, 4)


def _legacy_desired(cm: ChunkManager, camera_chunk: tuple[int, int, int]) -> set:
    """Re-derive the legacy desired set straight from a camera CHUNK coord."""
    ccx, ccy, ccz = camera_chunk
    pos = Vec3(
        (ccx + 0.25) * cm._chunk_m,
        (ccy + 0.25) * cm._chunk_m,
        (ccz + 0.25) * cm._chunk_m,
    )
    assert cm.camera_chunk(pos) == camera_chunk
    return cm.desired_set(pos)


class TestRegressionLock:
    def test_near_equals_legacy_desired_set(self) -> None:
        set_world_seed(1337)
        cm = ChunkManager(load_config(), EventBus())
        for cc in [(0, 0, 0), (3, -2, 1), (-5, 7, 0), (100, -100, 2)]:
            plan = desired_node_set(cc, cm.config, _Z_BAND, max_rank=0, near_radius_chunks=6)
            assert plan.coarse_nodes == {}
            assert plan.near_chunks == _legacy_desired(cm, cc)

    def test_near_count_is_1183(self) -> None:
        plan = desired_node_set((0, 0, 0), None, _Z_BAND, max_rank=0, near_radius_chunks=6)
        assert len(plan.near_chunks) == 13 * 13 * 7  # 1183


class TestNearAuthoritativeWithCoarse:
    """The editable (rank-0) footprint must stay the full Chebyshev radius-near_r
    square even when coarse ranks are active — it must NOT shrink the ±near_r edge
    ring (regression lock for the P2 boundary fix; previously enter[1]==near_r with
    a ``>=`` keep silently swallowed the edge columns into rank-1 coarse nodes).
    """

    def test_near_contains_full_radius_square_at_max_rank_3(self) -> None:
        cfg = load_config()
        near_r = 6
        for cc in [(0, 0, 0), (2, -3, 1), (100, -100, 2), (-5, 7, 0)]:
            plan = desired_node_set(
                cc,
                cfg,
                _Z_BAND,
                max_rank=3,
                near_radius_chunks=near_r,
                far_radius_chunks=48,
            )
            near_cols = {(x, y) for (x, y, _z) in plan.near_chunks}
            full_square = {
                (cc[0] + dx, cc[1] + dy)
                for dx in range(-near_r, near_r + 1)
                for dy in range(-near_r, near_r + 1)
            }
            # Every column within Chebyshev near_r is editable (no edge-ring loss).
            assert full_square <= near_cols, (
                cc,
                sorted(full_square - near_cols),
            )

    def test_edge_ring_columns_are_near_not_coarse(self) -> None:
        # The exact ±near_r edge ring (Chebyshev distance == near_r) — the columns
        # that the old enter[1]==near_r boundary swallowed — must be near, and no
        # coarse node may cover any of them.
        cfg = load_config()
        near_r = 6
        plan = desired_node_set(
            (0, 0, 0),
            cfg,
            _Z_BAND,
            max_rank=3,
            near_radius_chunks=near_r,
            far_radius_chunks=48,
        )
        near_cols = {(x, y) for (x, y, _z) in plan.near_chunks}
        edge_ring = {
            (x, y)
            for x in range(-near_r, near_r + 1)
            for y in range(-near_r, near_r + 1)
            if max(abs(x), abs(y)) == near_r
        }
        assert len(edge_ring) == 4 * (2 * near_r)  # 24 boundary columns at r=6
        assert edge_ring <= near_cols
        coarse_cols = {
            (c[0], c[1])
            for nodes in plan.coarse_nodes.values()
            for key in nodes
            for c in LodNode(*key).covered_chunks()
        }
        assert edge_ring.isdisjoint(coarse_cols)


class TestRankMonotonic:
    def test_rank_non_decreasing_with_distance(self) -> None:
        # Build a rank lookup per column from the plan, walk outward along +X,
        # and assert the rank never decreases.
        cfg = load_config()
        plan = desired_node_set(
            (0, 0, 0),
            cfg,
            _Z_BAND,
            max_rank=3,
            near_radius_chunks=6,
            far_radius_chunks=40,
        )

        # column (cx, 0) at z=0 -> rank
        def rank_of(cx: int) -> int:
            if (cx, 0, 0) in plan.near_chunks:
                return 0
            for L, nodes in plan.coarse_nodes.items():
                node = LodNode.for_chunk((cx, 0, 0), L)
                if node.key in nodes:
                    return L
            raise AssertionError(f"column {cx} not in any rank")

        ranks = [rank_of(cx) for cx in range(0, 41)]
        assert ranks[0] == 0  # at the camera -> near
        assert all(b >= a for a, b in itertools.pairwise(ranks)), ranks
        assert max(ranks) >= 1  # at least one coarse rank appeared


class TestNodeCoverage:
    def test_each_node_covers_k_cubed_unique_columns(self) -> None:
        cfg = load_config()
        plan = desired_node_set(
            (0, 0, 0),
            cfg,
            _Z_BAND,
            max_rank=3,
            near_radius_chunks=6,
            far_radius_chunks=48,
        )
        for L, nodes in plan.coarse_nodes.items():
            for key in nodes:
                node = LodNode(*key)
                covered = node.covered_chunks()
                assert len(covered) == (1 << L) ** 3
                assert len(set(covered)) == (1 << L) ** 3


class TestDisjointness:
    def test_near_and_coarse_columns_disjoint(self) -> None:
        cfg = load_config()
        plan = desired_node_set(
            (2, -3, 1),
            cfg,
            _Z_BAND,
            max_rank=3,
            near_radius_chunks=6,
            far_radius_chunks=48,
        )
        near_cols = {(x, y) for (x, y, _z) in plan.near_chunks}
        for L, nodes in plan.coarse_nodes.items():
            for key in nodes:
                node = LodNode(*key)
                cols = {(c[0], c[1]) for c in node.covered_chunks()}
                # No covered column may also be a near column (hard band cut).
                assert near_cols.isdisjoint(cols), (L, key)

    def test_coarse_ranks_mutually_disjoint(self) -> None:
        cfg = load_config()
        plan = desired_node_set(
            (0, 0, 0),
            cfg,
            _Z_BAND,
            max_rank=3,
            near_radius_chunks=6,
            far_radius_chunks=48,
        )
        seen: set[tuple[int, int, int]] = set()
        for nodes in plan.coarse_nodes.values():
            for key in nodes:
                for c in LodNode(*key).covered_chunks():
                    assert c not in seen, f"column {c} covered by two coarse nodes"
                    seen.add(c)


class TestPerformanceAndDeterminism:
    def test_far_radius_64_within_budget(self) -> None:
        cfg = load_config()
        t0 = time.perf_counter()
        plan = desired_node_set(
            (0, 0, 0),
            cfg,
            _Z_BAND,
            max_rank=3,
            near_radius_chunks=6,
            far_radius_chunks=64,
        )
        elapsed = time.perf_counter() - t0
        assert isinstance(plan, NodePlan)
        # Generous budget — guards against an accidental O(r^3) regression while
        # tolerating a slow CI box (the meshgrid path runs in well under 1 s).
        assert elapsed < 3.0, f"desired_node_set too slow ({elapsed:.3f}s) — O(r^3)?"

    def test_deterministic(self) -> None:
        cfg = load_config()
        a = desired_node_set((1, 1, 0), cfg, _Z_BAND, max_rank=3, far_radius_chunks=40)
        b = desired_node_set((1, 1, 0), cfg, _Z_BAND, max_rank=3, far_radius_chunks=40)
        assert a.near_chunks == b.near_chunks
        assert a.coarse_nodes == b.coarse_nodes
