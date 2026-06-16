"""
tests/save/test_types.py — Tests for fire_engine/save/types.py.

Covers: SaveIncompatibleError construction, message, inheritance, raise/catch
behavior. Headless: no panda3d.

Mirror of fire_engine/save/types.py.
"""

from __future__ import annotations

import pytest

from fire_engine.save.types import SaveIncompatibleError

# ---------------------------------------------------------------------------
# SaveIncompatibleError — correctness tests
# ---------------------------------------------------------------------------


class TestSaveIncompatibleError:
    """Verify SaveIncompatibleError behaves as a proper Exception."""

    def test_is_exception(self):
        """SaveIncompatibleError is an instance of Exception."""
        exc = SaveIncompatibleError("test")
        assert isinstance(exc, Exception)

    def test_is_base_exception(self):
        """SaveIncompatibleError is also a BaseException (transitively)."""
        exc = SaveIncompatibleError("test")
        assert isinstance(exc, BaseException)

    def test_message_in_str(self):
        """The message passed at construction appears in str(exc)."""
        msg = "world_seed mismatch: saved=1 current=2"
        exc = SaveIncompatibleError(msg)
        assert msg in str(exc)

    def test_empty_message(self):
        """An empty string message is accepted without error."""
        exc = SaveIncompatibleError("")
        assert isinstance(exc, Exception)

    def test_args_preserved(self):
        """The args tuple is preserved (standard Exception behavior)."""
        exc = SaveIncompatibleError("reason A", "reason B")
        assert exc.args == ("reason A", "reason B")

    def test_raise_and_catch_specific(self):
        """SaveIncompatibleError can be raised and caught by its specific type."""
        with pytest.raises(SaveIncompatibleError):
            raise SaveIncompatibleError("specific catch")

    def test_raise_and_catch_as_exception(self):
        """SaveIncompatibleError can be caught as a generic Exception."""
        try:
            raise SaveIncompatibleError("generic catch")
        except Exception as exc:
            assert "generic catch" in str(exc)

    def test_message_preserved_through_re_raise(self):
        """Message is preserved when the exception is re-raised."""
        original_msg = "config_digest mismatch"
        try:
            try:
                raise SaveIncompatibleError(original_msg)
            except SaveIncompatibleError:
                raise
        except SaveIncompatibleError as exc:
            assert original_msg in str(exc)

    def test_distinct_from_value_error(self):
        """SaveIncompatibleError is not a subclass of ValueError."""
        exc = SaveIncompatibleError("test")
        assert not isinstance(exc, ValueError)

    def test_distinct_from_runtime_error(self):
        """SaveIncompatibleError is not a subclass of RuntimeError."""
        exc = SaveIncompatibleError("test")
        assert not isinstance(exc, RuntimeError)

    def test_world_seed_message_pattern(self):
        """A world_seed mismatch message contains the key phrase."""
        exc = SaveIncompatibleError(
            "Save file world_seed=1337 does not match current config world_seed=9999."
        )
        assert "world_seed" in str(exc)

    def test_config_digest_message_pattern(self):
        """A config_digest mismatch message contains the key phrase."""
        exc = SaveIncompatibleError(
            "Save file config_digest='abc' does not match current digest='def'."
        )
        assert "config_digest" in str(exc)

    def test_format_version_message_pattern(self):
        """A format_version message contains the key phrase."""
        exc = SaveIncompatibleError(
            "Save file format version 9999 is newer than this engine supports (max 1)."
        )
        assert "format" in str(exc).lower()
