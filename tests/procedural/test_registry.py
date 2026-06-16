"""
tests/procedural/test_registry.py — Tests for fire_engine/procedural/registry.py.

Covers: register, get, clear_cache, reset_registry, cache identity, unknown name.
Headless — no panda3d imports.  Extracted from tests/test_procedural.py.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helper: create a fresh registry with a minimal test def
# ---------------------------------------------------------------------------


def _restore_builtins() -> None:
    """Re-register all built-in defs so later tests are not starved."""
    from tests.procedural.conftest import restore_builtins

    restore_builtins()


def _reset_with_minimal(seed: int = 42) -> None:
    from fire_engine.core.rng import set_world_seed
    from fire_engine.procedural.defs import ProceduralDef
    from fire_engine.procedural.registry import register, reset_registry

    set_world_seed(seed)
    reset_registry()

    class _MinimalDef(ProceduralDef):
        name = "_reg_test_def"

        def generate(self, rng, **params):
            w = int(params.get("width", 4))
            h = int(params.get("height", 4))
            out = np.empty((h, w, 4), dtype=np.uint8)
            out[..., :3] = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            out[..., 3] = 255
            return out

    register(_MinimalDef())


# ---------------------------------------------------------------------------
# Tests for register
# ---------------------------------------------------------------------------


class TestRegister:
    def setup_method(self):
        _reset_with_minimal()

    def teardown_method(self):
        _restore_builtins()

    def test_register_stores_def(self):
        from fire_engine.procedural.registry import get

        result = get("_reg_test_def")
        assert isinstance(result, np.ndarray)

    def test_register_non_def_raises_type_error(self):
        from fire_engine.procedural.registry import register

        with pytest.raises(TypeError, match="ProceduralDef"):
            register("not_a_def")  # type: ignore[arg-type]

    def test_register_replaces_and_evicts_cache(self):
        """Re-registering evicts cached results for that name."""
        from fire_engine.procedural.defs import ProceduralDef
        from fire_engine.procedural.registry import get, register

        arr1 = get("_reg_test_def")

        class _ReplaceDef(ProceduralDef):
            name = "_reg_test_def"

            def generate(self, rng, **params):
                return np.array([99], dtype=np.uint8)

        register(_ReplaceDef())
        arr2 = get("_reg_test_def")
        assert arr2[0] == 99
        assert arr1 is not arr2


# ---------------------------------------------------------------------------
# get() — cache identity
# ---------------------------------------------------------------------------


class TestGetCacheIdentity:
    def setup_method(self):
        _reset_with_minimal()

    def teardown_method(self):
        _restore_builtins()

    def test_same_call_returns_same_object(self):
        from fire_engine.procedural.registry import get

        arr1 = get("_reg_test_def")
        arr2 = get("_reg_test_def")
        assert arr1 is arr2

    def test_different_params_different_cache_slot(self):
        from fire_engine.procedural.registry import get

        arr_default = get("_reg_test_def")
        arr_custom = get("_reg_test_def", width=8, height=8)
        assert arr_default is not arr_custom
        assert arr_custom.shape == (8, 8, 4)

    def test_clear_cache_breaks_identity(self):
        from fire_engine.procedural.registry import clear_cache, get

        arr1 = get("_reg_test_def")
        clear_cache()
        arr2 = get("_reg_test_def")
        assert arr1 is not arr2

    def test_clear_cache_same_values_same_seed(self):
        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural.registry import clear_cache, get

        set_world_seed(7)
        _reset_with_minimal(seed=7)
        arr1 = get("_reg_test_def").copy()
        clear_cache()
        arr2 = get("_reg_test_def")
        assert np.array_equal(arr1, arr2)


# ---------------------------------------------------------------------------
# get() — unknown name
# ---------------------------------------------------------------------------


class TestGetUnknownName:
    def setup_method(self):
        _reset_with_minimal()

    def teardown_method(self):
        _restore_builtins()

    def test_unknown_name_raises_key_error(self):
        from fire_engine.procedural.registry import get

        with pytest.raises(KeyError, match="does_not_exist"):
            get("does_not_exist")


# ---------------------------------------------------------------------------
# Tests for reset_registry
# ---------------------------------------------------------------------------


class TestResetRegistry:
    def teardown_method(self):
        _restore_builtins()

    def test_reset_removes_all_defs(self):
        from fire_engine.procedural.registry import get, reset_registry

        _reset_with_minimal()
        reset_registry()
        with pytest.raises(KeyError):
            get("_reg_test_def")

    def test_reset_clears_cache(self):
        from fire_engine.procedural.registry import get, reset_registry

        _reset_with_minimal()
        get("_reg_test_def")  # prime cache
        reset_registry()
        # After reset the cache is gone; re-registering works fresh.
        _reset_with_minimal()
        arr = get("_reg_test_def")
        assert arr is not None


# ---------------------------------------------------------------------------
# Tests for clear_cache
# ---------------------------------------------------------------------------


class TestClearCache:
    def setup_method(self):
        _reset_with_minimal()

    def teardown_method(self):
        _restore_builtins()

    def test_clear_cache_leaves_registry_intact(self):
        from fire_engine.procedural.registry import clear_cache, get

        clear_cache()
        arr = get("_reg_test_def")
        assert isinstance(arr, np.ndarray)

    def test_clear_cache_multiple_slots(self):
        from fire_engine.procedural.registry import clear_cache, get

        a = get("_reg_test_def", width=4, height=4)
        b = get("_reg_test_def", width=8, height=8)
        assert a is not b
        clear_cache()
        a2 = get("_reg_test_def", width=4, height=4)
        b2 = get("_reg_test_def", width=8, height=8)
        assert a is not a2
        assert b is not b2


# ---------------------------------------------------------------------------
# Determinism through the registry
# ---------------------------------------------------------------------------


class TestRegistryDeterminism:
    def teardown_method(self):
        _restore_builtins()

    def test_same_seed_byte_identical(self):
        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural.registry import get

        set_world_seed(1337)
        _reset_with_minimal(seed=1337)
        arr1 = get("_reg_test_def").copy()

        set_world_seed(1337)
        _reset_with_minimal(seed=1337)
        arr2 = get("_reg_test_def")
        assert np.array_equal(arr1, arr2)

    def test_different_seeds_differ(self):
        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural.registry import get

        set_world_seed(1)
        _reset_with_minimal(seed=1)
        arr_a = get("_reg_test_def").copy()

        set_world_seed(2)
        _reset_with_minimal(seed=2)
        arr_b = get("_reg_test_def")
        assert not np.array_equal(arr_a, arr_b)
