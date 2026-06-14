"""
tests/test_raycast.py — Golden-master / characterisation tests for raycast_voxel.

PURPOSE: Pin CURRENT behaviour (normal-sign convention, distance calculation,
inside-solid behaviour, boundary conditions).  Do NOT fix bugs; report
suspicions in the module docstring and in individual test comments.

Headless only — no panda3d / fire_engine.world / lighting.gpu imports.

SUSPECTED BUG (pinned, not fixed):
    _voxel_to_chunk uses plain `//` (floor division) which is correct for
    positive voxel coords but may be off-by-one for negative voxel coords
    when the voxel index is a negative multiple of chunk_size.
    e.g. voxel -32 → chunk -1 (correct), voxel -33 → chunk -2 (correct),
    but Python floor-div is correct for negatives so this may not be a bug.
    Tracked by test_chunk_coord_negative_voxel.

SUSPECTED BUG (pinned):
    Normal for origin-inside-solid is `normal = -d` (the negated unit
    direction vector), NOT a cardinal axis-aligned Vec3.  The doc says
    "the axis the ray stepped across to enter it" — an inside-solid hit
    has never stepped, so the normal is computed as `-normalised(direction)`.
    This means the normal will NOT be axis-aligned for diagonal rays.
    Pinned by test_inside_solid_normal_is_neg_direction.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fire_engine.core import load_config, EventBus
from fire_engine.core.math3d import Vec3
from fire_engine.core.rng import set_world_seed
from fire_engine.world.terrain.chunk import Chunk
from fire_engine.world.terrain.generation import generate_chunk
from fire_engine.world.terrain.raycast import raycast_voxel, Hit
from fire_engine.world.terrain.chunk_manager import ChunkManager


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg():
    return load_config()


def flat_provider(cfg, store=None):
    """
    Dict-backed chunk_provider that generates flat baseline terrain on miss.
    Mirrors the approach in test_terrain.py / test_brush.py.
    """
    store = {} if store is None else store

    def provider(coord):
        ch = store.get(coord)
        if ch is None:
            mat = generate_chunk(coord, cfg)
            ch = Chunk(coord, mat, chunk_size=cfg.chunk_size, voxel_size=cfg.voxel_size)
            store[coord] = ch
        return ch

    return provider, store


def single_voxel_provider(cfg, vx, vy, vz):
    """
    Provider with exactly one solid voxel at global voxel coords (vx, vy, vz).
    All other voxels are air.  Useful for testing normal directions in isolation.
    """
    store = {}
    vs = cfg.voxel_size
    n = cfg.chunk_size

    # Compute chunk coord (Python floor-div handles negatives correctly).
    cx = vx // n
    cy = vy // n
    cz = vz // n
    lx = vx - cx * n
    ly = vy - cy * n
    lz = vz - cz * n

    def provider(coord):
        ch = store.get(coord)
        if ch is None:
            ch = Chunk(coord, chunk_size=n, voxel_size=vs)
            store[coord] = ch
        if coord == (cx, cy, cz):
            ch.materials[lx, ly, lz] = 1
        return ch

    return provider


# ---------------------------------------------------------------------------
# 1. Basic downward (-Z) ray hitting flat generated terrain
# ---------------------------------------------------------------------------


class TestDownwardRayHitsTerrain:
    def test_hit_is_not_none(self, cfg):
        """A ray from above the surface pointing straight down MUST hit."""
        set_world_seed(1337)
        provider, _ = flat_provider(cfg)
        hit = raycast_voxel(
            Vec3(8.0, 8.0, 30.0),
            Vec3(0.0, 0.0, -1.0),
            provider,
        )
        assert hit is not None

    def test_hit_distance_positive(self, cfg):
        """Distance must be >= 0 for a ray that starts above terrain."""
        set_world_seed(1337)
        provider, _ = flat_provider(cfg)
        hit = raycast_voxel(Vec3(8.0, 8.0, 30.0), Vec3(0.0, 0.0, -1.0), provider)
        assert hit is not None
        assert hit.distance >= 0.0

    def test_hit_distance_within_max(self, cfg):
        """Distance must not exceed max_distance_m."""
        set_world_seed(1337)
        provider, _ = flat_provider(cfg)
        max_d = 100.0
        hit = raycast_voxel(
            Vec3(8.0, 8.0, 30.0),
            Vec3(0.0, 0.0, -1.0),
            provider,
            max_distance_m=max_d,
        )
        assert hit is not None
        assert hit.distance <= max_d

    def test_hit_point_near_ground_surface(self, cfg):
        """
        The hit point Z should be close to ground_height_m.
        The ray enters the topmost solid voxel, so hit.point.z is the world Z
        of the top face of that voxel: exactly ground_height_m (within 1 voxel).
        """
        set_world_seed(1337)
        provider, _ = flat_provider(cfg)
        hit = raycast_voxel(Vec3(8.0, 8.0, 30.0), Vec3(0.0, 0.0, -1.0), provider)
        assert hit is not None
        # The ray enters the top face of the topmost solid voxel.
        # Top of topmost solid voxel = ground_height_m (the surface).
        # hit.point.z should be within one voxel below ground_height_m.
        assert (
            cfg.ground_height_m - cfg.voxel_size
            <= hit.point.z
            <= cfg.ground_height_m + cfg.voxel_size
        )

    def test_hit_normal_points_up_for_top_face(self, cfg):
        """
        A ray from above going straight down (-Z) hits the TOP face of the surface
        voxel.  The normal should be +Z (pointing back toward the origin, i.e. up).
        Pin the sign convention: normal[last_axis] = -step[last_axis].
        For step[z]=-1 → normal_z = +1. We assert normal.z == 1.0.
        """
        set_world_seed(1337)
        provider, _ = flat_provider(cfg)
        hit = raycast_voxel(Vec3(8.0, 8.0, 30.0), Vec3(0.0, 0.0, -1.0), provider)
        assert hit is not None
        assert hit.normal.z == pytest.approx(1.0, abs=1e-5)
        assert hit.normal.x == pytest.approx(0.0, abs=1e-5)
        assert hit.normal.y == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 2. Six-axis normal sign convention (cardinal directions)
# ---------------------------------------------------------------------------


class TestNormalSignConvention:
    """
    Pin the exact normal a ray fired from each axis direction returns.
    Convention: normal[last_axis] = -step[last_axis], i.e. the normal
    always points back toward the origin (out of the face the ray entered).
    """

    def _hit_from(self, cfg, dx, dy, dz, offset=5.0):
        """
        Fire a ray from (offset*dx, offset*dy, offset*dz) toward the solid
        voxel at global voxel (10, 10, 10), which sits at world
        (5.25, 5.25, 5.25) centre with voxel_size=0.5.
        """
        # Place a single solid voxel at global voxel (10, 10, 10).
        provider = single_voxel_provider(cfg, 10, 10, 10)
        # voxel centre = (10+0.5)*0.5 = 5.25 m on each axis
        cx, cy, cz = (
            10 * cfg.voxel_size + cfg.voxel_size / 2,
            10 * cfg.voxel_size + cfg.voxel_size / 2,
            10 * cfg.voxel_size + cfg.voxel_size / 2,
        )
        origin = Vec3(cx + dx * offset, cy + dy * offset, cz + dz * offset)
        direction = Vec3(-dx, -dy, -dz)
        return raycast_voxel(origin, direction, provider, max_distance_m=offset + 5.0)

    def test_ray_from_plus_x_normal_is_plus_x(self, cfg):
        """Ray travelling -X enters +X face → normal should be +X."""
        hit = self._hit_from(cfg, 1, 0, 0)
        assert hit is not None, "expected a hit from +X"
        assert hit.normal.x == pytest.approx(1.0, abs=1e-5)
        assert hit.normal.y == pytest.approx(0.0, abs=1e-5)
        assert hit.normal.z == pytest.approx(0.0, abs=1e-5)

    def test_ray_from_minus_x_normal_is_minus_x(self, cfg):
        """Ray travelling +X enters -X face → normal should be -X."""
        hit = self._hit_from(cfg, -1, 0, 0)
        assert hit is not None, "expected a hit from -X"
        assert hit.normal.x == pytest.approx(-1.0, abs=1e-5)

    def test_ray_from_plus_y_normal_is_plus_y(self, cfg):
        """Ray travelling -Y enters +Y face → normal should be +Y."""
        hit = self._hit_from(cfg, 0, 1, 0)
        assert hit is not None
        assert hit.normal.y == pytest.approx(1.0, abs=1e-5)

    def test_ray_from_minus_y_normal_is_minus_y(self, cfg):
        """Ray travelling +Y enters -Y face → normal should be -Y."""
        hit = self._hit_from(cfg, 0, -1, 0)
        assert hit is not None
        assert hit.normal.y == pytest.approx(-1.0, abs=1e-5)

    def test_ray_from_plus_z_normal_is_plus_z(self, cfg):
        """Ray travelling -Z enters +Z (top) face → normal should be +Z."""
        hit = self._hit_from(cfg, 0, 0, 1)
        assert hit is not None
        assert hit.normal.z == pytest.approx(1.0, abs=1e-5)

    def test_ray_from_minus_z_normal_is_minus_z(self, cfg):
        """Ray travelling +Z enters -Z (bottom) face → normal should be -Z."""
        hit = self._hit_from(cfg, 0, 0, -1)
        assert hit is not None
        assert hit.normal.z == pytest.approx(-1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 3. Miss — ray pointing into open sky / away from terrain
# ---------------------------------------------------------------------------


class TestMiss:
    def test_ray_straight_up_from_terrain_returns_none(self, cfg):
        """A ray from ground level pointing +Z (into sky) finds no solid above."""
        set_world_seed(1337)
        provider, _ = flat_provider(cfg)
        hit = raycast_voxel(
            Vec3(8.0, 8.0, cfg.ground_height_m + 1.0),
            Vec3(0.0, 0.0, 1.0),
            provider,
            max_distance_m=50.0,
        )
        assert hit is None

    def test_ray_well_above_terrain_pointing_up_returns_none(self, cfg):
        """No solid voxels above the surface at all."""
        set_world_seed(1337)
        provider, _ = flat_provider(cfg)
        hit = raycast_voxel(
            Vec3(0.0, 0.0, 100.0),
            Vec3(0.0, 0.0, 1.0),
            provider,
        )
        assert hit is None

    def test_ray_outside_world_footprint_returns_none(self, cfg):
        """
        Beyond world_size_m/2 the terrain is air even below ground level.
        Fire a ray from far outside the footprint; it should miss.
        """
        set_world_seed(1337)
        provider, _ = flat_provider(cfg)
        far = cfg.world_size_m  # well outside the [-500,+500] footprint
        hit = raycast_voxel(
            Vec3(far, far, cfg.ground_height_m + 5.0),
            Vec3(0.0, 0.0, -1.0),
            provider,
            max_distance_m=50.0,
        )
        assert hit is None


# ---------------------------------------------------------------------------
# 4. max_distance_m boundary behaviour
# ---------------------------------------------------------------------------


class TestMaxDistance:
    def test_max_distance_zero_returns_none(self, cfg):
        """
        max_distance=0: the very first DDA iteration checks `t > max_distance_m`
        after advancing t.  At t=0 (first step check) this passes, so the
        origin voxel is still evaluated.  Pin current behaviour:
        - If origin is in air: returns None (no solid found before t > 0).
        - We test with origin above terrain (air): expect None.
        """
        set_world_seed(1337)
        provider, _ = flat_provider(cfg)
        hit = raycast_voxel(
            Vec3(8.0, 8.0, 30.0),
            Vec3(0.0, 0.0, -1.0),
            provider,
            max_distance_m=0.0,
        )
        # Pin current behaviour: t=0 fails `t > 0.0` check on first step
        # (it is NOT strictly greater), so origin voxel IS checked first.
        # Origin is air at z=30, so None.
        assert hit is None

    def test_short_max_distance_misses_terrain(self, cfg):
        """A max_distance shorter than the gap to terrain should return None."""
        set_world_seed(1337)
        provider, _ = flat_provider(cfg)
        origin_z = 30.0
        # Surface is at ground_height_m (~8 m); gap ≈ 22 m.  Use max_dist=5.
        hit = raycast_voxel(
            Vec3(8.0, 8.0, origin_z),
            Vec3(0.0, 0.0, -1.0),
            provider,
            max_distance_m=5.0,
        )
        assert hit is None

    def test_long_max_distance_hits_terrain(self, cfg):
        """A generous max_distance from far above should find the terrain."""
        set_world_seed(1337)
        provider, _ = flat_provider(cfg)
        hit = raycast_voxel(
            Vec3(8.0, 8.0, 90.0),
            Vec3(0.0, 0.0, -1.0),
            provider,
            max_distance_m=200.0,
        )
        assert hit is not None

    def test_exact_surface_max_distance_hits(self, cfg):
        """
        max_distance just long enough to reach the surface voxel should hit.
        Origin at ground_height_m + epsilon above the last air voxel.
        """
        set_world_seed(1337)
        provider, _ = flat_provider(cfg)
        origin_z = cfg.ground_height_m + 2.0
        # At most need to travel ~2 m + one voxel to get into solid.
        hit = raycast_voxel(
            Vec3(8.0, 8.0, origin_z),
            Vec3(0.0, 0.0, -1.0),
            provider,
            max_distance_m=origin_z,  # generously above the distance needed
        )
        assert hit is not None


# ---------------------------------------------------------------------------
# 5. Origin inside solid terrain
# ---------------------------------------------------------------------------


class TestOriginInsideSolid:
    def test_inside_solid_returns_immediate_hit(self, cfg):
        """
        When the ray origin is already inside a solid voxel, the DDA enters the
        hit branch on the very first iteration with t=0.  Pin: returns a Hit
        (not None) with distance == 0.0.
        """
        set_world_seed(1337)
        provider, _ = flat_provider(cfg)
        # Place origin below the ground surface (solid region).
        inside_z = cfg.ground_height_m - cfg.voxel_size  # definitely solid
        hit = raycast_voxel(
            Vec3(8.0, 8.0, inside_z),
            Vec3(0.0, 0.0, -1.0),
            provider,
        )
        assert hit is not None
        assert hit.distance == pytest.approx(0.0, abs=1e-6)

    def test_inside_solid_normal_is_neg_direction(self, cfg):
        """
        SUSPECTED BUG (pinned, not fixed):
        When origin is inside solid, last_axis=-1 so the code computes
        normal = -d (the negated unit direction vector).  For a pure -Z
        direction this incidentally gives (0,0,1), which looks correct.
        But for a diagonal ray this will NOT be axis-aligned.

        Pin the exact behaviour: normal == -normalise(direction), not a
        cardinal face normal.  This test fires a diagonal ray from inside
        solid and asserts the non-cardinal result.
        """
        set_world_seed(1337)
        provider, _ = flat_provider(cfg)
        inside_z = cfg.ground_height_m - cfg.voxel_size
        # Diagonal ray: equal X and Z components → 45°.
        raw_dir = np.array([1.0, 0.0, -1.0])
        d = raw_dir / np.linalg.norm(raw_dir)
        hit = raycast_voxel(
            Vec3(8.0, 8.0, inside_z),
            Vec3(float(d[0]), float(d[1]), float(d[2])),
            provider,
        )
        assert hit is not None
        # Normal = -d (negated normalised direction).
        expected = -d
        assert hit.normal.x == pytest.approx(float(expected[0]), abs=1e-5)
        assert hit.normal.y == pytest.approx(float(expected[1]), abs=1e-5)
        assert hit.normal.z == pytest.approx(float(expected[2]), abs=1e-5)

    def test_inside_solid_point_equals_origin(self, cfg):
        """When t=0 the hit point equals the ray origin."""
        set_world_seed(1337)
        provider, _ = flat_provider(cfg)
        ox, oy, oz = 8.0, 8.0, cfg.ground_height_m - cfg.voxel_size
        hit = raycast_voxel(
            Vec3(ox, oy, oz),
            Vec3(0.0, 0.0, -1.0),
            provider,
        )
        assert hit is not None
        assert hit.point.x == pytest.approx(ox, abs=1e-4)
        assert hit.point.y == pytest.approx(oy, abs=1e-4)
        assert hit.point.z == pytest.approx(oz, abs=1e-4)


# ---------------------------------------------------------------------------
# 6. Hit dataclass: field types and completeness
# ---------------------------------------------------------------------------


class TestHitDataclass:
    def test_hit_fields_exist(self, cfg):
        """All documented fields of Hit must be present."""
        set_world_seed(1337)
        provider, _ = flat_provider(cfg)
        hit = raycast_voxel(Vec3(8.0, 8.0, 30.0), Vec3(0.0, 0.0, -1.0), provider)
        assert hit is not None
        assert hasattr(hit, "point")
        assert hasattr(hit, "voxel")
        assert hasattr(hit, "chunk_coord")
        assert hasattr(hit, "normal")
        assert hasattr(hit, "distance")

    def test_hit_point_is_vec3(self, cfg):
        set_world_seed(1337)
        provider, _ = flat_provider(cfg)
        hit = raycast_voxel(Vec3(8.0, 8.0, 30.0), Vec3(0.0, 0.0, -1.0), provider)
        assert hit is not None
        assert isinstance(hit.point, Vec3)

    def test_hit_normal_is_vec3(self, cfg):
        set_world_seed(1337)
        provider, _ = flat_provider(cfg)
        hit = raycast_voxel(Vec3(8.0, 8.0, 30.0), Vec3(0.0, 0.0, -1.0), provider)
        assert hit is not None
        assert isinstance(hit.normal, Vec3)

    def test_hit_distance_is_float(self, cfg):
        set_world_seed(1337)
        provider, _ = flat_provider(cfg)
        hit = raycast_voxel(Vec3(8.0, 8.0, 30.0), Vec3(0.0, 0.0, -1.0), provider)
        assert hit is not None
        assert isinstance(hit.distance, float)

    def test_hit_voxel_is_int_triple(self, cfg):
        set_world_seed(1337)
        provider, _ = flat_provider(cfg)
        hit = raycast_voxel(Vec3(8.0, 8.0, 30.0), Vec3(0.0, 0.0, -1.0), provider)
        assert hit is not None
        assert isinstance(hit.voxel, tuple)
        assert len(hit.voxel) == 3
        assert all(isinstance(v, int) for v in hit.voxel)

    def test_hit_chunk_coord_is_int_triple(self, cfg):
        set_world_seed(1337)
        provider, _ = flat_provider(cfg)
        hit = raycast_voxel(Vec3(8.0, 8.0, 30.0), Vec3(0.0, 0.0, -1.0), provider)
        assert hit is not None
        assert isinstance(hit.chunk_coord, tuple)
        assert len(hit.chunk_coord) == 3
        assert all(isinstance(v, int) for v in hit.chunk_coord)

    def test_hit_is_frozen(self, cfg):
        """Hit is a frozen dataclass; assignment must raise."""
        set_world_seed(1337)
        provider, _ = flat_provider(cfg)
        hit = raycast_voxel(Vec3(8.0, 8.0, 30.0), Vec3(0.0, 0.0, -1.0), provider)
        assert hit is not None
        with pytest.raises((AttributeError, TypeError)):
            hit.distance = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 7. chunk_coord and voxel consistency
# ---------------------------------------------------------------------------


class TestCoordConsistency:
    def test_chunk_coord_contains_voxel(self, cfg):
        """
        The hit voxel coord must live inside the reported chunk_coord.
        voxel[i] // chunk_size must equal chunk_coord[i] for all axes.
        """
        set_world_seed(1337)
        provider, _ = flat_provider(cfg)
        hit = raycast_voxel(Vec3(8.0, 8.0, 30.0), Vec3(0.0, 0.0, -1.0), provider)
        assert hit is not None
        n = cfg.chunk_size
        cx = hit.voxel[0] // n
        cy = hit.voxel[1] // n
        cz = hit.voxel[2] // n
        assert (cx, cy, cz) == hit.chunk_coord

    def test_voxel_coord_consistent_with_hit_point(self, cfg):
        """
        The hit voxel must contain hit.point.
        voxel_min = voxel * voxel_size, voxel_max = (voxel+1) * voxel_size.
        """
        set_world_seed(1337)
        provider, _ = flat_provider(cfg)
        hit = raycast_voxel(Vec3(8.0, 8.0, 30.0), Vec3(0.0, 0.0, -1.0), provider)
        assert hit is not None
        vs = cfg.voxel_size
        vx, vy, vz = hit.voxel
        pt = hit.point
        # Allow one full voxel tolerance: hit.point is entry face — may be on the boundary.
        assert vx * vs - vs <= pt.x <= (vx + 1) * vs + vs
        assert vy * vs - vs <= pt.y <= (vy + 1) * vs + vs
        assert vz * vs - vs <= pt.z <= (vz + 1) * vs + vs

    def test_chunk_coord_negative_voxel(self, cfg):
        """
        Pin chunk-coord for a voxel at a negative global coordinate.
        Python floor division: voxel -1 → chunk -1 (since -1 // 32 = -1),
        voxel -32 → chunk -1 (since -32 // 32 = -1), voxel -33 → chunk -2.
        Fire a ray from negative X to check the mapping survives.
        """
        set_world_seed(1337)
        provider, _ = flat_provider(cfg)
        # Terrain is solid below ground_height_m and the footprint is [-500,+500].
        # A negative-X/Y position within [-500,+500] has solid terrain.
        hit = raycast_voxel(
            Vec3(-8.0, -8.0, 30.0),
            Vec3(0.0, 0.0, -1.0),
            provider,
        )
        assert hit is not None
        n = cfg.chunk_size
        cx = hit.voxel[0] // n
        cy = hit.voxel[1] // n
        cz = hit.voxel[2] // n
        assert (cx, cy, cz) == hit.chunk_coord


# ---------------------------------------------------------------------------
# 8. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_inputs_same_hit(self, cfg):
        """Two identical calls must return byte-identical Hit fields."""
        set_world_seed(1337)
        provider, store = flat_provider(cfg)
        origin = Vec3(8.0, -4.0, 30.0)
        direction = Vec3(0.0, 0.0, -1.0)
        h1 = raycast_voxel(origin, direction, provider)
        h2 = raycast_voxel(origin, direction, provider)
        assert h1 is not None
        assert h2 is not None
        assert h1.distance == h2.distance
        assert h1.voxel == h2.voxel
        assert h1.chunk_coord == h2.chunk_coord
        assert h1.normal.x == h2.normal.x
        assert h1.normal.y == h2.normal.y
        assert h1.normal.z == h2.normal.z
        assert h1.point.x == h2.point.x
        assert h1.point.y == h2.point.y
        assert h1.point.z == h2.point.z

    def test_different_origins_differ(self, cfg):
        """Two rays from different heights return different distances."""
        set_world_seed(1337)
        provider, _ = flat_provider(cfg)
        h1 = raycast_voxel(Vec3(8.0, 8.0, 20.0), Vec3(0.0, 0.0, -1.0), provider)
        h2 = raycast_voxel(Vec3(8.0, 8.0, 40.0), Vec3(0.0, 0.0, -1.0), provider)
        assert h1 is not None and h2 is not None
        assert h2.distance > h1.distance


# ---------------------------------------------------------------------------
# 9. Unnormalised direction vector
# ---------------------------------------------------------------------------


class TestUnnormalisedDirection:
    def test_unnormalised_dir_same_hit_as_normalised(self, cfg):
        """
        The implementation normalises direction internally, so a direction of
        different magnitude should produce the same hit point and normal.
        """
        set_world_seed(1337)
        provider, _ = flat_provider(cfg)
        h1 = raycast_voxel(Vec3(8.0, 8.0, 30.0), Vec3(0.0, 0.0, -1.0), provider)
        h2 = raycast_voxel(Vec3(8.0, 8.0, 30.0), Vec3(0.0, 0.0, -5.0), provider)
        assert h1 is not None
        assert h2 is not None
        assert h1.voxel == h2.voxel
        assert h1.normal.z == pytest.approx(h2.normal.z, abs=1e-5)

    def test_zero_direction_returns_none(self, cfg):
        """Zero-length direction must return None (degenerate ray)."""
        set_world_seed(1337)
        provider, _ = flat_provider(cfg)
        hit = raycast_voxel(Vec3(8.0, 8.0, 30.0), Vec3(0.0, 0.0, 0.0), provider)
        assert hit is None


# ---------------------------------------------------------------------------
# 10. ChunkManager as provider (integration path, as shown in docs example)
# ---------------------------------------------------------------------------


class TestChunkManagerProvider:
    def test_chunk_manager_as_provider(self, cfg):
        """
        ChunkManager is callable as a chunk_provider.  Verify the integration
        path from the demo loop in terrain.md: cm is passed directly.
        """
        set_world_seed(1337)
        cm = ChunkManager(cfg, EventBus())
        hit = raycast_voxel(Vec3(8.0, -4.0, 30.0), Vec3(0.0, 0.0, -1.0), cm)
        assert hit is not None
        assert hit.distance >= 0.0
        assert isinstance(hit.normal, Vec3)
