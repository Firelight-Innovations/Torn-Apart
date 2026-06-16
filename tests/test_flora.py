"""
tests/test_flora.py — sprite flora system: flower atlas + placement math.

Headless (no panda3d): the flower_sprite ProceduralTextureDef is pure
numpy, and the flora instance-count / hash-seed / attrib math lives in
``fire_engine/zones/flora_placement.py`` (the panda3d-free mirror of what
``FloraRendererComponent`` instances per volume).  Trees and bushes left
this system for 3-D meshes — see tests/test_tree_*.py.  This suite covers:

1. Texture determinism (same seed → identical bytes; different seed differs).
2. Texture shape / dtype / binary-alpha / atlas-variant invariants.
3. Flora instance-count math (area × density, caps, param override).
4. Flora hash-seed determinism, bound, and volume-distinctness.
5. flora_instance_attribs: bounds, scale range, variant range, determinism,
   and agreement with the grass chain on the shared links.
6. The GLSL mirror pin: flora.vert carries the same hash-chain constants.
"""

from __future__ import annotations

import pathlib

import numpy as np

from fire_engine.core import Config
from fire_engine.core.rng import for_domain, set_world_seed
from fire_engine.zones import (
    FLORA_KINDS,
    ZoneVolume,
    flora_hash_seed,
    flora_instance_attribs,
    flora_instance_count,
    instance_attribs,
)

_REPO = pathlib.Path(__file__).resolve().parents[1]

# (def name, atlas shape, variant cell width, variant count)
_SPRITES = [
    ("flower_sprite", (32, 128, 4), 32, 4),
]


def _gen(name: str, seed: int, **params) -> np.ndarray:
    """Generate a texture def fresh under a given world seed (bypasses the
    registry cache so seed changes actually re-generate)."""
    set_world_seed(seed)
    from fire_engine.procedural.textures.sprites.flower_sprite import FlowerSpriteDef

    cls = {"flower_sprite": FlowerSpriteDef}[name]
    return cls().generate(for_domain("procedural", name), **params)


# ---------------------------------------------------------------------------
# 1 & 2 — texture determinism + invariants
# ---------------------------------------------------------------------------


class TestFloraSpriteTextures:
    def test_shape_dtype(self):
        for name, shape, _, _ in _SPRITES:
            arr = _gen(name, 1337)
            assert arr.shape == shape, name
            assert arr.dtype == np.uint8, name

    def test_determinism_same_seed(self):
        for name, _, _, _ in _SPRITES:
            a = _gen(name, 1337)
            b = _gen(name, 1337)
            assert np.array_equal(a, b), name

    def test_different_seed_differs(self):
        for name, _, _, _ in _SPRITES:
            a = _gen(name, 1337)
            b = _gen(name, 9999)
            assert not np.array_equal(a, b), name

    def test_binary_alpha(self):
        # Discard-rendered cutouts: alpha is exactly 0 or 255, nothing between.
        for name, _, _, _ in _SPRITES:
            arr = _gen(name, 1337)
            assert ((arr[..., 3] == 0) | (arr[..., 3] == 255)).all(), name

    def test_variants_distinct(self):
        # Atlas cells must not be identical (different hues/silhouettes).
        for name, _, cell_w, n_var in _SPRITES:
            arr = _gen(name, 1337)
            cells = [arr[:, k * cell_w : (k + 1) * cell_w] for k in range(n_var)]
            for k in range(n_var - 1):
                assert not np.array_equal(cells[k], cells[k + 1]), name

    def test_bases_on_bottom_row(self):
        # Stems touch the bottom image row (V=0 after the upload flip) so
        # plants stand on the ground, not above it.
        for name, _, _, _ in _SPRITES:
            arr = _gen(name, 1337)
            assert (arr[-1, :, 3] == 255).any(), name

    def test_registered_via_get(self):
        set_world_seed(1337)
        from fire_engine.procedural import clear_cache, get

        clear_cache()
        for name, shape, _, _ in _SPRITES:
            assert get(name).shape == shape, name


# ---------------------------------------------------------------------------
# 3 — instance-count math
# ---------------------------------------------------------------------------


class TestFloraInstanceCount:
    def test_area_times_density(self):
        cfg = Config()
        v = ZoneVolume(1, "flowers", (0.0, 0.0, 0.0), (20.0, 20.0, 8.0))
        assert flora_instance_count(v, cfg, "flowers") == int(400 * cfg.flora_flower_density_per_m2)

    def test_density_param_override(self):
        cfg = Config()
        v = ZoneVolume(1, "flowers", (0.0, 0.0, 0.0), (10.0, 10.0, 4.0), params={"density": 3.0})
        assert flora_instance_count(v, cfg, "flowers") == 300

    def test_cap(self):
        cfg = Config()
        v = ZoneVolume(
            1, "flowers", (0.0, 0.0, 0.0), (1000.0, 1000.0, 8.0), params={"density": 100.0}
        )
        assert flora_instance_count(v, cfg, "flowers") == cfg.flora_flower_max_instances

    def test_negative_density_clamps_to_zero(self):
        cfg = Config()
        v = ZoneVolume(1, "flowers", (0.0, 0.0, 0.0), (10.0, 10.0, 4.0), params={"density": -5.0})
        assert flora_instance_count(v, cfg, "flowers") == 0


# ---------------------------------------------------------------------------
# 4 — hash seeds
# ---------------------------------------------------------------------------


class TestFloraHashSeed:
    def test_deterministic_same_seed_same_volume(self):
        v = ZoneVolume(7, "flowers", (0.0, 0.0, 0.0), (10.0, 10.0, 4.0))
        set_world_seed(1337)
        a = flora_hash_seed(v, "flowers")
        set_world_seed(1337)
        b = flora_hash_seed(v, "flowers")
        assert a == b

    def test_volumes_distinct(self):
        set_world_seed(1337)
        v1 = ZoneVolume(1, "flowers", (0.0, 0.0, 0.0), (10.0, 10.0, 4.0))
        v2 = ZoneVolume(2, "flowers", (0.0, 0.0, 0.0), (10.0, 10.0, 4.0))
        assert flora_hash_seed(v1, "flowers") != flora_hash_seed(v2, "flowers")

    def test_bounded_for_signed_int(self):
        set_world_seed(1337)
        v = ZoneVolume(4, "flowers", (0.0, 0.0, 0.0), (10.0, 10.0, 4.0))
        for k in FLORA_KINDS:
            s = flora_hash_seed(v, k)
            assert 0 <= s < 2**31


# ---------------------------------------------------------------------------
# 5 — instance attribs (the flora.vert mirror)
# ---------------------------------------------------------------------------


class TestFloraInstanceAttribs:
    MIN = (-12.0, -5.0, 6.0)
    MAX = (12.0, 25.0, 10.0)

    def _attrs(self, n=4096, seed=12345, variants=3, smin=0.8, sspan=0.8):
        return flora_instance_attribs(
            np.arange(n),
            seed,
            self.MIN,
            self.MAX,
            n_variants=variants,
            scale_min=smin,
            scale_span=sspan,
        )

    def test_positions_inside_bounds(self):
        a = self._attrs()
        assert (a["x"] >= self.MIN[0]).all() and (a["x"] <= self.MAX[0]).all()
        assert (a["y"] >= self.MIN[1]).all() and (a["y"] <= self.MAX[1]).all()

    def test_scale_range_parameterised(self):
        a = self._attrs(smin=0.8, sspan=0.8)
        assert (a["scale"] >= 0.8).all() and (a["scale"] < 1.6 + 1e-5).all()

    def test_variant_range_and_coverage(self):
        a = self._attrs(variants=3)
        assert a["variant"].dtype == np.int32
        assert set(np.unique(a["variant"])) == {0, 1, 2}

    def test_deterministic(self):
        a = self._attrs()
        b = self._attrs()
        for key in a:
            assert np.array_equal(a[key], b[key]), key

    def test_shared_links_match_grass_chain(self):
        # The first five hash links are the grass chain — same seed must give
        # identical placement, so a future refactor can't silently fork them.
        idx = np.arange(512)
        g = instance_attribs(idx, 999, self.MIN, self.MAX)
        f = flora_instance_attribs(idx, 999, self.MIN, self.MAX, n_variants=4)
        for key in ("x", "y", "rot", "scale", "phase"):
            assert np.array_equal(g[key], f[key]), key


# ---------------------------------------------------------------------------
# 6 — GLSL mirror pin
# ---------------------------------------------------------------------------


class TestShaderMirrorPin:
    def test_flora_vert_carries_chain_constants(self):
        src = (
            (_REPO / "fire_engine" / "render" / "shaders" / "flora.vert")
            .read_text(encoding="utf-8")
            .lower()
        )
        for const in (
            "0x9e3779b9u",
            "0x85ebca6bu",
            "0xc2b2ae35u",
            "0x27d4eb2fu",
            "0x165667b1u",
            "0x7feb352du",
            "0x846ca68bu",
        ):
            assert const in src, const

    def test_python_k5_matches_shader(self):
        from fire_engine.zones.flora_placement import _K5

        assert int(_K5) == 0x165667B1
