"""
tests/procedural/test_defs.py — Tests for fire_engine/procedural/defs.py.

Covers ProceduralDef base class and the @register_def decorator.
Headless — no panda3d imports.
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.procedural.defs import ProceduralDef, register_def

# ---------------------------------------------------------------------------
# ProceduralDef — abstract base class
# ---------------------------------------------------------------------------


class TestProceduralDefABC:
    def test_cannot_instantiate_directly(self):
        """ProceduralDef is abstract — instantiating it must raise TypeError."""
        with pytest.raises(TypeError):
            ProceduralDef()  # type: ignore[abstract]

    def test_concrete_subclass_can_be_instantiated(self):
        class ConcreteDef(ProceduralDef):
            name = "_test_concrete"

            def generate(self, rng, **params):
                return np.array([1], dtype=np.uint8)

        obj = ConcreteDef()
        assert isinstance(obj, ProceduralDef)

    def test_generate_not_implemented_on_base(self):
        """Calling the abstract generate stub raises NotImplementedError."""

        class StubDef(ProceduralDef):
            name = "_test_stub"

            def generate(self, rng, **params):
                # Call super to hit the NotImplementedError branch.
                return super().generate(rng, **params)  # type: ignore[misc]

        obj = StubDef()
        rng = np.random.default_rng(0)
        with pytest.raises(NotImplementedError):
            obj.generate(rng)

    def test_name_attribute_required(self):
        """A concrete subclass must have a `name` attribute."""

        class MyDef(ProceduralDef):
            name = "_test_named"

            def generate(self, rng, **params):
                return None

        assert MyDef.name == "_test_named"

    def test_generate_receives_rng_and_params(self):
        """generate() is called with the rng and forwarded params."""
        received: dict = {}

        class RecordDef(ProceduralDef):
            name = "_test_record"

            def generate(self, rng, **params):
                received["rng"] = rng
                received["params"] = params
                return np.zeros(1, dtype=np.uint8)

        obj = RecordDef()
        rng = np.random.default_rng(42)
        obj.generate(rng, width=64, height=32)
        assert received["rng"] is rng
        assert received["params"] == {"width": 64, "height": 32}


# ---------------------------------------------------------------------------
# @register_def decorator
# ---------------------------------------------------------------------------


class TestRegisterDefDecorator:
    def setup_method(self):
        from fire_engine.core.rng import set_world_seed
        from fire_engine.procedural.registry import reset_registry

        set_world_seed(42)
        reset_registry()

    def teardown_method(self):
        """Re-register built-in defs so later tests are not starved."""
        from tests.procedural.conftest import restore_builtins

        restore_builtins()

    def test_decorator_registers_at_import(self):
        """A class decorated with @register_def is immediately retrievable."""
        from fire_engine.procedural.registry import get

        @register_def
        class MinimalDef(ProceduralDef):
            name = "_test_minimal_defs"

            def generate(self, rng, **params):
                return np.array([42], dtype=np.uint8)

        result = get("_test_minimal_defs")
        assert isinstance(result, np.ndarray)
        assert result[0] == 42

    def test_decorator_returns_original_class(self):
        """@register_def must return the class unchanged."""

        @register_def
        class ReturnCheckDef(ProceduralDef):
            name = "_test_return_check"

            def generate(self, rng, **params):
                return np.zeros(1, dtype=np.uint8)

        assert ReturnCheckDef is not None
        assert issubclass(ReturnCheckDef, ProceduralDef)

    def test_decorator_replaces_previous_registration(self):
        """Registering twice replaces the previous def."""
        from fire_engine.procedural.registry import get

        @register_def
        class FirstDef(ProceduralDef):
            name = "_test_replace"

            def generate(self, rng, **params):
                return np.array([1], dtype=np.uint8)

        @register_def
        class SecondDef(ProceduralDef):
            name = "_test_replace"

            def generate(self, rng, **params):
                return np.array([2], dtype=np.uint8)

        result = get("_test_replace")
        assert result[0] == 2

    def test_decorated_class_can_still_be_subclassed(self):
        """The decorated class retains its type so subclassing still works."""

        @register_def
        class BaseDef(ProceduralDef):
            name = "_test_base_cls"

            def generate(self, rng, **params):
                return np.zeros(1, dtype=np.uint8)

        class SubDef(BaseDef):
            name = "_test_sub_cls"

            def generate(self, rng, **params):
                return np.ones(1, dtype=np.uint8)

        assert issubclass(SubDef, BaseDef)
        assert issubclass(SubDef, ProceduralDef)
