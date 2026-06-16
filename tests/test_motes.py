"""
tests/test_motes.py — WP4 wind particles: procedural textures + leaf placement.

Headless (no panda3d): the dust_mote / leaf_sprite ProceduralTextureDefs are
pure numpy, and the leaf instance-count / hash-seed math lives in
``fire_engine/zones/grass_placement.py`` (the panda3d-free mirror of what
``LeafLitterComponent`` instances per volume).  This suite covers:

1. Texture determinism (same seed → identical bytes; different seed → different).
2. Texture shape / dtype / channel invariants.
3. Leaf instance-count math (area × density, cap, density param override).
4. Leaf hash-seed determinism + bound.
5. No panda3d leaks into procedural/ or zones/ (AST import grep).
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

from fire_engine.core import Config
from fire_engine.core.rng import set_world_seed
from fire_engine.zones import ZoneVolume, leaf_hash_seed, leaf_instance_count

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gen(name: str, seed: int, **params) -> np.ndarray:
    """Generate a texture def fresh under a given world seed (bypasses the
    registry cache so seed changes actually re-generate)."""
    set_world_seed(seed)
    if name == "dust_mote":
        from fire_engine.core.rng import for_domain
        from fire_engine.procedural.textures.dust_mote import DustMoteDef

        return DustMoteDef().generate(for_domain("procedural", "dust_mote"), **params)
    if name == "leaf_sprite":
        from fire_engine.core.rng import for_domain
        from fire_engine.procedural.textures.leaf_sprite import LeafSpriteDef

        return LeafSpriteDef().generate(for_domain("procedural", "leaf_sprite"), **params)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# 1 & 2 — texture determinism + invariants
# ---------------------------------------------------------------------------


class TestDustMoteTexture:
    def test_shape_dtype(self):
        arr = _gen("dust_mote", 1337)
        assert arr.shape == (32, 32, 4)
        assert arr.dtype == np.uint8

    def test_determinism_same_seed(self):
        a = _gen("dust_mote", 1337)
        b = _gen("dust_mote", 1337)
        assert np.array_equal(a, b)

    def test_different_seed_differs(self):
        a = _gen("dust_mote", 1337)
        b = _gen("dust_mote", 9999)
        assert not np.array_equal(a, b)

    def test_soft_radial_falloff(self):
        # Centre alpha must exceed corner alpha (soft radial speck).
        arr = _gen("dust_mote", 1337)
        assert int(arr[16, 16, 3]) > int(arr[0, 0, 3])
        # Corners are fully transparent (outside the disc).
        assert arr[0, 0, 3] == 0

    def test_rgb_constant_warm_tint(self):
        # Additive blend reads a constant RGB tint; alpha carries the mask.
        arr = _gen("dust_mote", 1337)
        # All non-zero-alpha texels share the same RGB.
        rgb = arr[..., :3]
        assert rgb[..., 0].min() == rgb[..., 0].max()

    def test_param_override_size(self):
        arr = _gen("dust_mote", 1337, width=16, height=16)
        assert arr.shape == (16, 16, 4)


class TestLeafSpriteTexture:
    def test_shape_dtype(self):
        arr = _gen("leaf_sprite", 1337)
        assert arr.shape == (32, 96, 4)  # 3 cells × 32 wide
        assert arr.dtype == np.uint8

    def test_determinism_same_seed(self):
        a = _gen("leaf_sprite", 1337)
        b = _gen("leaf_sprite", 1337)
        assert np.array_equal(a, b)

    def test_different_seed_differs(self):
        a = _gen("leaf_sprite", 1337)
        b = _gen("leaf_sprite", 4242)
        assert not np.array_equal(a, b)

    def test_three_distinct_variants(self):
        # The 3 atlas cells should not be identical (different hues/shapes).
        arr = _gen("leaf_sprite", 1337)
        c0 = arr[:, 0:32]
        c1 = arr[:, 32:64]
        c2 = arr[:, 64:96]
        assert not np.array_equal(c0, c1)
        assert not np.array_equal(c1, c2)

    def test_leaf_body_opaque_background_clear(self):
        arr = _gen("leaf_sprite", 1337)
        # Centre of each leaf cell is opaque; the corner is background (clear).
        for k in range(3):
            cx = k * 32 + 16
            assert arr[16, cx, 3] == 255
            assert arr[0, k * 32, 3] == 0

    def test_registered_via_get(self):
        # The registry path (with the package import side-effect) returns it.
        set_world_seed(1337)
        from fire_engine.procedural import get

        arr = get("leaf_sprite")
        assert arr.shape == (32, 96, 4)


# ---------------------------------------------------------------------------
# 3 — leaf instance-count math (pure function mirror of the renderer)
# ---------------------------------------------------------------------------


class TestLeafInstanceCount:
    def _vol(self, sx=20.0, sy=20.0, params=None):
        return ZoneVolume(1, "trees", (0.0, 0.0, 0.0), (sx, sy, 8.0), params=params or {})

    def test_area_times_density(self):
        cfg = Config()  # wind_leaf_density_per_m2 = 0.15
        vol = self._vol(20.0, 20.0)  # 400 m²
        assert leaf_instance_count(vol, cfg) == int(400.0 * 0.15)  # 60

    def test_density_param_override(self):
        cfg = Config()
        vol = self._vol(10.0, 10.0, params={"leaf_density": 1.0})  # 100 m²
        assert leaf_instance_count(vol, cfg) == 100

    def test_cap_applied(self):
        cfg = Config()
        # A huge volume × the default density would exceed the cap.
        vol = self._vol(2000.0, 2000.0)
        assert leaf_instance_count(vol, cfg) == int(cfg.wind_leaf_max_instances)

    def test_zero_density_zero_count(self):
        cfg = Config()
        vol = self._vol(20.0, 20.0, params={"leaf_density": 0.0})
        assert leaf_instance_count(vol, cfg) == 0

    def test_negative_density_clamped_to_zero(self):
        cfg = Config()
        vol = self._vol(20.0, 20.0, params={"leaf_density": -5.0})
        assert leaf_instance_count(vol, cfg) == 0


# ---------------------------------------------------------------------------
# 4 — leaf hash-seed determinism
# ---------------------------------------------------------------------------


class TestLeafHashSeed:
    def test_deterministic_same_seed_same_volume(self):
        vol = ZoneVolume(7, "trees", (0.0, 0.0, 0.0), (10.0, 10.0, 4.0))
        set_world_seed(1337)
        a = leaf_hash_seed(vol)
        set_world_seed(1337)
        b = leaf_hash_seed(vol)
        assert a == b

    def test_distinct_volumes_distinct_seeds(self):
        v1 = ZoneVolume(1, "trees", (0.0, 0.0, 0.0), (10.0, 10.0, 4.0))
        v2 = ZoneVolume(2, "trees", (0.0, 0.0, 0.0), (10.0, 10.0, 4.0))
        set_world_seed(1337)
        s1 = leaf_hash_seed(v1)
        set_world_seed(1337)
        s2 = leaf_hash_seed(v2)
        assert s1 != s2

    def test_bound_signed_int_range(self):
        vol = ZoneVolume(3, "trees", (0.0, 0.0, 0.0), (10.0, 10.0, 4.0))
        set_world_seed(1337)
        s = leaf_hash_seed(vol)
        assert 0 <= s < 2**31


# ---------------------------------------------------------------------------
# 5 — no panda3d / direct leaks into procedural/ or zones/
# ---------------------------------------------------------------------------


class TestNoPandaLeak:
    @staticmethod
    def _imports(path: pathlib.Path) -> set[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
        return names

    def test_dust_and_leaf_defs_headless(self):
        root = pathlib.Path(__file__).resolve().parent.parent / "fire_engine"
        targets = [
            root / "procedural" / "textures" / "dust_mote.py",
            root / "procedural" / "textures" / "leaf_sprite.py",
            root / "zones" / "grass_placement.py",
        ]
        for path in targets:
            for mod in self._imports(path):
                assert not mod.startswith("panda3d"), f"{path.name} imports {mod}"
                assert not mod.startswith("direct"), f"{path.name} imports {mod}"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
