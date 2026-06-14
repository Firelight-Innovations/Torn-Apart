"""
tests/test_simulation_stubs.py — Golden-master / characterization tests for
future-scope simulation stubs: ai, economy, politics.

These tests PIN CURRENT BEHAVIOR only. Do NOT fix bugs here — record
suspicions in comments instead and let the implementer decide.

Packages covered:
  fire_engine.ai       → NPCArchetype  (ARCHITECTURE.md §5.8)
  fire_engine.economy  → GoodDef       (ARCHITECTURE.md §5.9)
  fire_engine.politics → FactionDef    (ARCHITECTURE.md §5.10)
"""

from __future__ import annotations

import inspect
import types

import pytest

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import fire_engine.simulation.ai as _ai_pkg
import fire_engine.simulation.economy as _econ_pkg
import fire_engine.simulation.politics as _pol_pkg

from fire_engine.simulation.ai import NPCArchetype
from fire_engine.simulation.economy import GoodDef
from fire_engine.simulation.politics import FactionDef


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeRng:
    """Minimal stand-in for a numpy Generator so we can pass various objects."""
    pass


_RNG_A = _FakeRng()
_RNG_B = None          # also valid per current stub signature (no type check)
_RNG_C = object()


# ---------------------------------------------------------------------------
# 1. Package-level __all__ membership
# ---------------------------------------------------------------------------

class TestPackageExports:
    """Pin that each package's __all__ contains exactly the documented symbol."""

    def test_ai_all_contains_npcarchetype(self):
        assert "NPCArchetype" in _ai_pkg.__all__, (
            f"fire_engine.simulation.ai.__all__ = {_ai_pkg.__all__!r} — "
            "'NPCArchetype' is missing"
        )

    def test_economy_all_contains_gooddef(self):
        assert "GoodDef" in _econ_pkg.__all__, (
            f"fire_engine.simulation.economy.__all__ = {_econ_pkg.__all__!r} — "
            "'GoodDef' is missing"
        )

    def test_politics_all_contains_factiondef(self):
        assert "FactionDef" in _pol_pkg.__all__, (
            f"fire_engine.simulation.politics.__all__ = {_pol_pkg.__all__!r} — "
            "'FactionDef' is missing"
        )

    def test_ai_all_length(self):
        """Pin that __all__ has exactly 1 entry (no hidden extras)."""
        assert len(_ai_pkg.__all__) == 1

    def test_economy_all_length(self):
        assert len(_econ_pkg.__all__) == 1

    def test_politics_all_length(self):
        assert len(_pol_pkg.__all__) == 1


# ---------------------------------------------------------------------------
# 2. Class instantiability — pin that all three are concrete (not abstract)
# ---------------------------------------------------------------------------

class TestInstantiability:
    """Pin that the stub classes can be constructed with no arguments."""

    def test_npcarchetype_is_instantiable(self):
        obj = NPCArchetype()
        assert isinstance(obj, NPCArchetype)

    def test_gooddef_is_instantiable(self):
        obj = GoodDef()
        assert isinstance(obj, GoodDef)

    def test_factiondef_is_instantiable(self):
        obj = FactionDef()
        assert isinstance(obj, FactionDef)


# ---------------------------------------------------------------------------
# 3. generate() is an instance method (not classmethod / staticmethod)
# ---------------------------------------------------------------------------

class TestGenerateIsInstanceMethod:
    """Pin that generate is an ordinary instance method on all three classes."""

    def test_npcarchetype_generate_is_instance_method(self):
        raw = inspect.getattr_static(NPCArchetype, "generate")
        assert not isinstance(raw, (classmethod, staticmethod)), (
            "NPCArchetype.generate is currently a plain instance method; "
            f"found {type(raw)!r} instead"
        )

    def test_gooddef_generate_is_instance_method(self):
        raw = inspect.getattr_static(GoodDef, "generate")
        assert not isinstance(raw, (classmethod, staticmethod)), (
            f"GoodDef.generate is currently a plain instance method; "
            f"found {type(raw)!r} instead"
        )

    def test_factiondef_generate_is_instance_method(self):
        raw = inspect.getattr_static(FactionDef, "generate")
        assert not isinstance(raw, (classmethod, staticmethod)), (
            f"FactionDef.generate is currently a plain instance method; "
            f"found {type(raw)!r} instead"
        )


# ---------------------------------------------------------------------------
# 4. generate() raises NotImplementedError with the exact pinned message
# ---------------------------------------------------------------------------

# PINNED VERBATIM messages (copied directly from source):
_NPCARCHETYPE_MSG = (
    "NPCArchetype.generate is future scope — see ARCHITECTURE.md §5.8 "
    "(AI API). Not part of Session 1."
)
_GOODDEF_MSG = (
    "GoodDef.generate is future scope — see ARCHITECTURE.md §5.9 "
    "(Economy API). Not part of Session 1."
)
_FACTIONDEF_MSG = (
    "FactionDef.generate is future scope — see ARCHITECTURE.md §5.10 "
    "(Politics API). Not part of Session 1."
)


class TestGenerateRaisesNotImplementedError:
    """
    Pin that generate() raises NotImplementedError with the verbatim message
    regardless of what rng / params are passed.

    SUSPICION: The docstring convention says 'type hints mandatory on all
    public APIs' (CLAUDE.md Conventions), but generate() has no type hints
    and carries # noqa: ANN001, ANN003 suppressions.  This is a known tech
    debt the stubs carry — not a new bug, but worth flagging for implementers.
    """

    # -- NPCArchetype --

    def test_npcarchetype_raises_with_fake_rng(self):
        with pytest.raises(NotImplementedError, match="NPCArchetype.generate is future scope"):
            NPCArchetype().generate(_RNG_A)

    def test_npcarchetype_message_verbatim(self):
        with pytest.raises(NotImplementedError) as exc_info:
            NPCArchetype().generate(_RNG_A)
        assert str(exc_info.value) == _NPCARCHETYPE_MSG

    def test_npcarchetype_raises_with_none_rng(self):
        with pytest.raises(NotImplementedError):
            NPCArchetype().generate(_RNG_B)

    def test_npcarchetype_raises_with_kwargs(self):
        with pytest.raises(NotImplementedError):
            NPCArchetype().generate(_RNG_C, name="Drifter", level=5, faction="raiders")

    def test_npcarchetype_section_reference_present(self):
        """Pin that the §5.8 reference is in the message."""
        with pytest.raises(NotImplementedError, match=r"§5\.8"):
            NPCArchetype().generate(_RNG_A)

    # -- GoodDef --

    def test_gooddef_raises_with_fake_rng(self):
        with pytest.raises(NotImplementedError, match="GoodDef.generate is future scope"):
            GoodDef().generate(_RNG_A)

    def test_gooddef_message_verbatim(self):
        with pytest.raises(NotImplementedError) as exc_info:
            GoodDef().generate(_RNG_A)
        assert str(exc_info.value) == _GOODDEF_MSG

    def test_gooddef_raises_with_none_rng(self):
        with pytest.raises(NotImplementedError):
            GoodDef().generate(_RNG_B)

    def test_gooddef_raises_with_kwargs(self):
        with pytest.raises(NotImplementedError):
            GoodDef().generate(_RNG_C, name="scrap_metal", base_value=10)

    def test_gooddef_section_reference_present(self):
        """Pin that the §5.9 reference is in the message."""
        with pytest.raises(NotImplementedError, match=r"§5\.9"):
            GoodDef().generate(_RNG_A)

    # -- FactionDef --

    def test_factiondef_raises_with_fake_rng(self):
        with pytest.raises(NotImplementedError, match="FactionDef.generate is future scope"):
            FactionDef().generate(_RNG_A)

    def test_factiondef_message_verbatim(self):
        with pytest.raises(NotImplementedError) as exc_info:
            FactionDef().generate(_RNG_A)
        assert str(exc_info.value) == _FACTIONDEF_MSG

    def test_factiondef_raises_with_none_rng(self):
        with pytest.raises(NotImplementedError):
            FactionDef().generate(_RNG_B)

    def test_factiondef_raises_with_kwargs(self):
        with pytest.raises(NotImplementedError):
            FactionDef().generate(_RNG_C, name="Iron Brotherhood", alignment="hostile")

    def test_factiondef_section_reference_present(self):
        """Pin that the §5.10 reference is in the message."""
        with pytest.raises(NotImplementedError, match=r"§5\.10"):
            FactionDef().generate(_RNG_A)


# ---------------------------------------------------------------------------
# 5. NotImplementedError is a subclass of RuntimeError (stdlib invariant)
# ---------------------------------------------------------------------------

class TestNotImplementedErrorIsRuntimeError:
    """
    NotImplementedError is a subclass of RuntimeError in CPython.
    Pin this so implementers who swap to a custom exception class break
    loudly rather than silently.
    """

    def test_not_implemented_error_subclasses_runtime_error(self):
        assert issubclass(NotImplementedError, RuntimeError), (
            "CPython stdlib invariant broken: NotImplementedError must be a "
            "subclass of RuntimeError"
        )

    def test_npcarchetype_raises_runtime_error_too(self):
        with pytest.raises(RuntimeError):
            NPCArchetype().generate(_RNG_A)

    def test_gooddef_raises_runtime_error_too(self):
        with pytest.raises(RuntimeError):
            GoodDef().generate(_RNG_A)

    def test_factiondef_raises_runtime_error_too(self):
        with pytest.raises(RuntimeError):
            FactionDef().generate(_RNG_A)


# ---------------------------------------------------------------------------
# 6. No extra public attributes on stub instances (pin the surface area)
# ---------------------------------------------------------------------------

class TestStubPublicSurface:
    """
    Pin that the only non-dunder callable on each stub instance is `generate`.
    If an implementer adds attributes before this test is updated, it fails
    loudly and forces a conscious decision.

    NOTE: `__init_subclass__`, `__subclasshook__`, etc. are dunder — excluded.
    """

    def _public_methods(self, obj) -> list[str]:
        return sorted(
            name
            for name in dir(obj)
            if not name.startswith("_") and callable(getattr(obj, name))
        )

    def test_npcarchetype_public_methods(self):
        assert self._public_methods(NPCArchetype()) == ["generate"]

    def test_gooddef_public_methods(self):
        assert self._public_methods(GoodDef()) == ["generate"]

    def test_factiondef_public_methods(self):
        assert self._public_methods(FactionDef()) == ["generate"]
