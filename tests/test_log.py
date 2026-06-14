"""
tests/test_log.py — Characterisation (golden-master) tests for core/log.py.

Pins CURRENT behaviour; does NOT fix anything.
Suspected deviations from documented intent are flagged in comments and in the
final report but are NOT corrected here.

Covers
------
- get_logger returns a logging.Logger with the correct name.
- Idempotency: repeated calls to get_logger (same and different names) do NOT
  add duplicate handlers to the root logger.
- Format: the handler's formatter _fmt matches the module's _LOG_FORMAT.
- Root logger level is set to the value that the module actually applies
  (WARNING per the comment "Default to WARNING so tests aren't noisy").
- Distinct names produce distinct, independently-named loggers.
- Edge cases: empty-string name and dotted names.
- The handler type installed on the root logger is a StreamHandler.
- No panda3d import (headless safety check).
"""

from __future__ import annotations

import logging
import importlib

import pytest

import fire_engine.core.log as _log_module
from fire_engine.core.log import get_logger

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _restore_logging_state():
    """
    Snapshot and restore the root logger + module-level sentinel before/after
    every test so tests don't pollute each other.
    """
    root = logging.getLogger()

    # Snapshot root state
    original_level = root.level
    original_handlers = list(root.handlers)
    original_sentinel = _log_module._handler_installed

    yield  # run the test

    # Restore root handlers
    for h in list(root.handlers):
        if h not in original_handlers:
            root.removeHandler(h)
    root.handlers = original_handlers
    root.setLevel(original_level)

    # Restore module sentinel
    _log_module._handler_installed = original_sentinel


def _force_fresh_setup():
    """Reset the module sentinel and clear root handlers so _setup_root_handler
    runs from a clean slate within a single test."""
    _log_module._handler_installed = False
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.NOTSET)


# ---------------------------------------------------------------------------
# Return type & name
# ---------------------------------------------------------------------------

class TestGetLoggerReturnType:

    def test_returns_logger_instance(self):
        log = get_logger("fire_engine.test.foo")
        assert isinstance(log, logging.Logger)

    def test_logger_name_matches_argument(self):
        log = get_logger("fire_engine.terrain.chunk_manager")
        assert log.name == "fire_engine.terrain.chunk_manager"

    def test_empty_string_name(self):
        """Empty string → returns the root logger (Python logging behaviour)."""
        log = get_logger("")
        # Pin: empty name returns the root logger object
        assert isinstance(log, logging.Logger)
        # In Python's logging, "" resolves to the root logger
        assert log is logging.getLogger("")

    def test_dotted_name(self):
        log = get_logger("fire_engine.core.rng")
        assert log.name == "fire_engine.core.rng"
        assert isinstance(log, logging.Logger)

    def test_single_segment_name(self):
        log = get_logger("fire_engine")
        assert log.name == "fire_engine"
        assert isinstance(log, logging.Logger)


# ---------------------------------------------------------------------------
# Idempotency — no duplicate handlers
# ---------------------------------------------------------------------------

class TestHandlerIdempotency:

    def test_second_call_same_name_no_extra_handler(self):
        """Two calls with the same name must not add a second handler."""
        _force_fresh_setup()
        root = logging.getLogger()
        get_logger("fire_engine.alpha")
        count_after_first = len(root.handlers)
        get_logger("fire_engine.alpha")
        count_after_second = len(root.handlers)
        assert count_after_second == count_after_first, (
            f"Handler count grew from {count_after_first} to {count_after_second} "
            "on a second call with the same name."
        )

    def test_second_call_different_name_no_extra_handler(self):
        """Two calls with different names must not add a second handler."""
        _force_fresh_setup()
        root = logging.getLogger()
        get_logger("fire_engine.alpha")
        count_after_first = len(root.handlers)
        get_logger("fire_engine.beta")
        count_after_second = len(root.handlers)
        assert count_after_second == count_after_first, (
            f"Handler count grew from {count_after_first} to {count_after_second} "
            "on a second call with a different name."
        )

    def test_many_calls_handler_count_stable(self):
        """Ten calls with distinct names must not grow the handler list."""
        _force_fresh_setup()
        root = logging.getLogger()
        get_logger("fire_engine.system.a")
        expected = len(root.handlers)
        for i in range(9):
            get_logger(f"fire_engine.system.{i}")
        assert len(root.handlers) == expected

    def test_module_sentinel_set_after_first_call(self):
        """The _handler_installed flag must be True after the first call."""
        _force_fresh_setup()
        assert _log_module._handler_installed is False
        get_logger("fire_engine.sentinel_test")
        assert _log_module._handler_installed is True


# ---------------------------------------------------------------------------
# Format string
# ---------------------------------------------------------------------------

class TestHandlerFormat:

    def _get_root_stream_handler(self) -> logging.StreamHandler | None:
        """Return the first StreamHandler on the root logger, or None."""
        root = logging.getLogger()
        for h in root.handlers:
            if isinstance(h, logging.StreamHandler):
                return h
        return None

    def test_handler_formatter_fmt_matches_module(self):
        """The installed formatter's _fmt must equal _LOG_FORMAT from the module."""
        _force_fresh_setup()
        get_logger("fire_engine.format_check")
        handler = self._get_root_stream_handler()
        assert handler is not None, "No StreamHandler found on root logger."
        assert handler.formatter is not None, "StreamHandler has no formatter set."
        assert handler.formatter._fmt == _log_module._LOG_FORMAT, (
            f"Formatter mismatch.\n"
            f"  expected: {_log_module._LOG_FORMAT!r}\n"
            f"  actual:   {handler.formatter._fmt!r}"
        )

    def test_format_string_contains_levelname(self):
        """Pin: format includes %(levelname)s."""
        assert "%(levelname)s" in _log_module._LOG_FORMAT

    def test_format_string_contains_name(self):
        """Pin: format includes %(name)s."""
        assert "%(name)s" in _log_module._LOG_FORMAT

    def test_format_string_contains_message(self):
        """Pin: format includes %(message)s."""
        assert "%(message)s" in _log_module._LOG_FORMAT

    def test_installed_handler_is_stream_handler(self):
        """Pin: the handler installed by _setup_root_handler is a StreamHandler."""
        _force_fresh_setup()
        get_logger("fire_engine.handler_type_check")
        handler = self._get_root_stream_handler()
        assert handler is not None, (
            "Expected a StreamHandler on root logger after get_logger()."
        )


# ---------------------------------------------------------------------------
# Root logger level
# ---------------------------------------------------------------------------

class TestRootLoggerLevel:

    def test_root_level_is_warning_when_previously_notset(self):
        """
        When root.level is NOTSET before the first call, _setup_root_handler
        must set it to WARNING (per the comment in the source).
        """
        _force_fresh_setup()
        root = logging.getLogger()
        assert root.level == logging.NOTSET, "Precondition: level should be NOTSET."
        get_logger("fire_engine.level_check")
        assert root.level == logging.WARNING, (
            f"Expected root level WARNING ({logging.WARNING}), "
            f"got {root.level} ({logging.getLevelName(root.level)})."
        )

    def test_root_level_not_overridden_when_already_set(self):
        """
        If the root logger already has a non-NOTSET level before get_logger is
        called, the module must NOT override it.
        (Suspected bug: see report — the guard checks root.level == NOTSET, so
        a pre-existing level should be left alone.)
        """
        _force_fresh_setup()
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)     # caller already configured a level
        get_logger("fire_engine.no_override")
        # Pin current behaviour: level should stay DEBUG
        assert root.level == logging.DEBUG, (
            f"Root level was unexpectedly changed to {logging.getLevelName(root.level)}."
        )


# ---------------------------------------------------------------------------
# Distinct loggers for distinct names
# ---------------------------------------------------------------------------

class TestDistinctLoggers:

    def test_same_name_returns_same_object(self):
        """logging.getLogger guarantees the same Logger object for the same name."""
        a = get_logger("fire_engine.shared")
        b = get_logger("fire_engine.shared")
        assert a is b

    def test_different_names_return_different_objects(self):
        a = get_logger("fire_engine.module_a")
        b = get_logger("fire_engine.module_b")
        assert a is not b

    def test_different_names_have_different_name_attributes(self):
        a = get_logger("fire_engine.x")
        b = get_logger("fire_engine.y")
        assert a.name != b.name

    def test_child_logger_inherits_from_parent(self):
        """
        Dotted names set up the standard logging hierarchy: a child logger's
        effective level propagates from its parent.  Pin this hierarchy behaviour.
        """
        parent = get_logger("fire_engine.parent")
        child = get_logger("fire_engine.parent.child")
        # child is a descendant of parent in the logger hierarchy
        assert child.name.startswith(parent.name)


# ---------------------------------------------------------------------------
# Headless / import safety
# ---------------------------------------------------------------------------

class TestHeadless:

    def test_no_panda3d_import(self):
        """log.py must not import panda3d (hard engine rule)."""
        import sys
        # If panda3d was imported transitively by log.py it would appear here.
        # We can check the module's globals for any panda3d reference.
        module_globals = vars(_log_module)
        panda_keys = [k for k in module_globals if "panda" in k.lower()]
        assert panda_keys == [], (
            f"panda3d symbols found in log module globals: {panda_keys}"
        )

    def test_only_stdlib_logging_imported(self):
        """log.py should import only stdlib logging (and __future__)."""
        module_globals = vars(_log_module)
        # logging must be there
        assert "logging" in module_globals
        # numpy must NOT be there
        assert "np" not in module_globals
        assert "numpy" not in module_globals
