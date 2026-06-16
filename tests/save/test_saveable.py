"""
tests/save/test_saveable.py — Tests for fire_engine/save/saveable.py.

Covers: Saveable Protocol (structural runtime checks), SaveIncompatibleError
re-export, and the isinstance() contract for compliant / non-compliant classes.
Headless: no panda3d.

Mirror of fire_engine/save/saveable.py.
"""

from __future__ import annotations

import pytest

from fire_engine.save.saveable import Saveable, SaveIncompatibleError

# ---------------------------------------------------------------------------
# Saveable Protocol — structural isinstance() checks
# ---------------------------------------------------------------------------


class TestSaveableProtocol:
    """Verify the @runtime_checkable Saveable Protocol enforces all three members."""

    def test_full_impl_is_saveable(self):
        """A class with save_key, get_delta, and apply_delta satisfies Saveable."""

        class Full:
            save_key = "full"

            def get_delta(self) -> dict:
                return {}

            def apply_delta(self, delta: dict) -> None:
                pass

        assert isinstance(Full(), Saveable)

    def test_missing_save_key_not_saveable(self):
        """A class without save_key does NOT satisfy isinstance(x, Saveable)."""

        class NoKey:
            def get_delta(self) -> dict:
                return {}

            def apply_delta(self, delta: dict) -> None:
                pass

        assert not isinstance(NoKey(), Saveable)

    def test_missing_get_delta_not_saveable(self):
        """A class without get_delta does NOT satisfy isinstance(x, Saveable)."""

        class NoGetDelta:
            save_key = "x"

            def apply_delta(self, delta: dict) -> None:
                pass

        assert not isinstance(NoGetDelta(), Saveable)

    def test_missing_apply_delta_not_saveable(self):
        """A class without apply_delta does NOT satisfy isinstance(x, Saveable)."""

        class NoApplyDelta:
            save_key = "x"

            def get_delta(self) -> dict:
                return {}

        assert not isinstance(NoApplyDelta(), Saveable)

    def test_empty_class_not_saveable(self):
        """A plain class with no members does NOT satisfy isinstance(x, Saveable)."""

        class Empty:
            pass

        assert not isinstance(Empty(), Saveable)

    def test_save_key_must_be_present_as_attr(self):
        """
        A class with get_delta and apply_delta methods but save_key only as a
        local variable (not an attribute) does NOT satisfy Saveable.
        """

        class BadKey:
            def get_delta(self) -> dict:
                _save_key = "oops"  # local var, not class attr
                return {}

            def apply_delta(self, delta: dict) -> None:
                pass

        assert not isinstance(BadKey(), Saveable)

    def test_instance_with_save_key_set_dynamically_is_saveable(self):
        """
        An instance that has save_key set on the instance (not the class) also
        satisfies Saveable — the Protocol checks the object, not the class.
        """

        class Dynamic:
            def __init__(self):
                self.save_key = "dynamic"

            def get_delta(self) -> dict:
                return {}

            def apply_delta(self, delta: dict) -> None:
                pass

        assert isinstance(Dynamic(), Saveable)


# ---------------------------------------------------------------------------
# SaveIncompatibleError re-export from saveable.py
# ---------------------------------------------------------------------------


class TestSaveIncompatibleErrorReexport:
    """
    saveable.py re-exports SaveIncompatibleError from save.types for backward
    compatibility.  Verify the symbol is importable here and is the same class.
    """

    def test_reexport_is_present(self):
        """SaveIncompatibleError is importable from fire_engine.save.saveable."""
        from fire_engine.save.saveable import SaveIncompatibleError as SIE

        assert SIE is SaveIncompatibleError

    def test_reexport_is_exception_subclass(self):
        """Re-exported SaveIncompatibleError is a proper Exception subclass."""
        exc = SaveIncompatibleError("test")
        assert isinstance(exc, Exception)

    def test_reexport_message_preserved(self):
        """Message passed to SaveIncompatibleError is preserved in str()."""
        msg = "world_seed mismatch"
        exc = SaveIncompatibleError(msg)
        assert msg in str(exc)

    def test_reexport_can_be_raised_and_caught(self):
        """SaveIncompatibleError from saveable.py can be raised and caught."""
        with pytest.raises(SaveIncompatibleError):
            raise SaveIncompatibleError("raised from saveable import path")


# ---------------------------------------------------------------------------
# Protocol contract: get_delta / apply_delta behavior on a real implementation
# ---------------------------------------------------------------------------


class TestSaveableContract:
    """Exercise the Saveable contract on a concrete minimal implementation."""

    def _make_impl(self, initial_health: int = 100):
        """Return a concrete Saveable-compliant instance."""

        class HealthSystem:
            save_key = "health"

            def __init__(self, hp: int) -> None:
                self.health = hp
                self._baseline = hp

            def get_delta(self) -> dict:
                if self.health == self._baseline:
                    return {}
                return {"health": self.health}

            def apply_delta(self, delta: dict) -> None:
                self.health = delta.get("health", self._baseline)

        return HealthSystem(initial_health)

    def test_unmodified_get_delta_is_empty(self):
        """An unmodified Saveable returns an empty delta (zero storage cost)."""
        sys = self._make_impl(100)
        assert sys.get_delta() == {}

    def test_modified_get_delta_contains_change(self):
        """After modifying state, get_delta() returns the changed field."""
        sys = self._make_impl(100)
        sys.health = 42
        delta = sys.get_delta()
        assert delta == {"health": 42}

    def test_apply_delta_restores_value(self):
        """apply_delta restores a value from the saved delta."""
        sys = self._make_impl(100)
        sys.apply_delta({"health": 77})
        assert sys.health == 77

    def test_apply_delta_empty_keeps_baseline(self):
        """apply_delta({}) leaves the system at its baseline value."""
        sys = self._make_impl(100)
        sys.apply_delta({})
        assert sys.health == 100

    def test_round_trip_get_apply(self):
        """get_delta then apply_delta restores the same value (correctness)."""
        sys = self._make_impl(100)
        sys.health = 55
        delta = sys.get_delta()

        sys2 = self._make_impl(100)
        sys2.apply_delta(delta)
        assert sys2.health == sys.health
