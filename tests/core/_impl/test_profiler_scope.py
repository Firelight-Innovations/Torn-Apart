"""
tests/core/_impl/test_profiler_scope.py — Mirror test for
fire_engine/core/_impl/profiler_scope.py.

Covers:
- NullScope: context-manager enter/exit is a no-op (no exception, no state)
- _NULL_SCOPE: module-level singleton is a NullScope instance
- _alloc_profiler_buffers: buffer shapes match config settings (enabled + disabled)
- _ScopeCtx: accessible via profiler factory (pools are used in normal flow)
- All symbols are re-exported from __all__
"""

from __future__ import annotations

import numpy as np
import pytest

from fire_engine.core._impl.profiler_scope import (
    _NULL_SCOPE,
    NullScope,
    _ScopeCtx,
)
from fire_engine.core.profiler import Profiler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_profiler(enabled: bool = True, history: int = 32, max_scopes: int = 16) -> Profiler:
    return Profiler(
        enabled=enabled,
        history_frames=history,
        max_scopes=max_scopes,
        max_counters=8,
        hitch_window=60,
    )


# ---------------------------------------------------------------------------
# NullScope
# ---------------------------------------------------------------------------


class TestNullScope:
    def test_context_manager_enter_returns_self(self):
        ns = NullScope()
        assert ns.__enter__() is ns

    def test_context_manager_exit_no_exception(self):
        ns = NullScope()
        ns.__enter__()
        ns.__exit__(None, None, None)  # must not raise

    def test_with_statement(self):
        """NullScope works correctly as a with-statement target."""
        ns = NullScope()
        entered = False
        with ns:
            entered = True
        assert entered

    def test_with_statement_suppresses_nothing(self):
        """NullScope does not swallow exceptions."""
        with pytest.raises(ValueError), NullScope():
            raise ValueError("test error")

    def test_slots_no_dict(self):
        """NullScope uses __slots__ — no __dict__."""
        ns = NullScope()
        assert not hasattr(ns, "__dict__")

    def test_null_scope_singleton(self):
        """_NULL_SCOPE is a NullScope instance."""
        assert isinstance(_NULL_SCOPE, NullScope)

    def test_multiple_uses_of_null_scope(self):
        """The singleton can be used multiple times without side effects."""
        for _ in range(5):
            with _NULL_SCOPE:
                pass  # should never raise


# ---------------------------------------------------------------------------
# _alloc_profiler_buffers — enabled path
# ---------------------------------------------------------------------------


class TestAllocProfilerBuffersEnabled:
    def test_frame_ms_shape(self):
        prof = _make_profiler(enabled=True, history=32)
        assert prof._frame_ms.shape == (32,)

    def test_scope_ms_shape(self):
        prof = _make_profiler(enabled=True, history=32, max_scopes=16)
        assert prof._scope_ms.shape == (32, 16)

    def test_frame_ms_dtype(self):
        prof = _make_profiler()
        assert prof._frame_ms.dtype == np.float64

    def test_scope_calls_dtype(self):
        prof = _make_profiler()
        assert prof._scope_calls.dtype == np.int32

    def test_write_index_starts_at_zero(self):
        prof = _make_profiler()
        assert prof._write_index == 0

    def test_frames_written_starts_at_zero(self):
        prof = _make_profiler()
        assert prof._frames_written == 0

    def test_scope_ids_starts_empty(self):
        prof = _make_profiler()
        assert prof._scope_ids == {}

    def test_scope_names_starts_empty(self):
        prof = _make_profiler()
        assert prof._scope_names == []


# ---------------------------------------------------------------------------
# _alloc_profiler_buffers — disabled path (zero-sized buffers)
# ---------------------------------------------------------------------------


class TestAllocProfilerBuffersDisabled:
    def test_frame_ms_zero_sized(self):
        prof = _make_profiler(enabled=False)
        assert prof._frame_ms.shape == (0,)

    def test_scope_ms_zero_sized(self):
        prof = _make_profiler(enabled=False)
        assert prof._scope_ms.shape == (0, 0)

    def test_ctx_pool_empty(self):
        prof = _make_profiler(enabled=False)
        assert prof._ctx_pool == []


# ---------------------------------------------------------------------------
# _ScopeCtx — accessible via normal profiler usage
# ---------------------------------------------------------------------------


class TestScopeCtx:
    def test_scope_ctx_exported(self):
        """_ScopeCtx must be importable from profiler_scope."""
        assert _ScopeCtx is not None

    def test_scope_ctx_is_context_manager(self):
        """_ScopeCtx instances must support the context manager protocol."""
        assert hasattr(_ScopeCtx, "__enter__")
        assert hasattr(_ScopeCtx, "__exit__")

    def test_profiler_scope_returns_context(self):
        """Profiler.scope() returns an object usable as a context manager."""
        prof = _make_profiler(enabled=True)

        class FakeClock:
            def __init__(self):
                self.t = 0

            def __call__(self):
                return self.t

            def advance_ms(self, ms):
                self.t += round(ms * 1_000_000)

        clk = FakeClock()
        prof._time = clk
        prof.begin_frame()
        ctx = prof.scope("TestScope")
        assert hasattr(ctx, "__enter__")
        assert hasattr(ctx, "__exit__")
        with ctx:
            clk.advance_ms(1.0)
        clk.advance_ms(2.0)
        prof.end_frame()
        # verify the scope was registered
        assert "TestScope" in prof._scope_ids
