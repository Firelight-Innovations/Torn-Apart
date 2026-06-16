"""
No-op and pooled timing scope context managers for the Torn Apart profiler.

Moved to :mod:`fire_engine.core._impl.profiler_scope` to satisfy the
per-directory module-count limit.  ``NullScope`` is the public class;
``_ScopeCtx`` and ``_NULL_SCOPE`` are private helpers re-exported for use by
:class:`~fire_engine.core.profiler.Profiler`.

Docs: docs/systems/core.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fire_engine.core.profiler import Profiler

__all__ = ["_NULL_SCOPE", "NullScope", "_ScopeCtx", "_alloc_profiler_buffers"]


class NullScope:
    """
    Shared no-op scope returned by a disabled :class:`~fire_engine.core.profiler.Profiler`.

    ``with profiler.scope(...)`` then costs one boolean check + this object's
    trivial enter/exit — no timing, no allocation.
    """

    __slots__ = ()

    def __enter__(self) -> NullScope:
        return self

    def __exit__(self, *exc: object) -> None:
        pass


#: Module-level singleton no-op scope (never holds state).
_NULL_SCOPE = NullScope()


def _alloc_profiler_buffers(prof: Profiler) -> None:
    """
    Allocate (or re-allocate) all preallocated numpy ring-buffer arrays on
    *prof* using its already-set scalar configuration attributes.

    Called by ``Profiler._configure`` after the scalar fields have been set so
    the allocation logic is factored out without changing public behaviour.
    When *prof* is disabled the buffers are zero-sized (no memory used).
    """
    H = prof.history_frames if prof.enabled else 0
    S = prof.max_scopes if prof.enabled else 0
    C = prof.max_counters if prof.enabled else 0
    W = prof.hitch_window if prof.enabled else 0

    # Scope / counter name registries (assigned lazily on first sight).
    prof._scope_ids = {}
    prof._scope_names = []
    prof._counter_ids = {}
    prof._counter_names = []
    prof._capacity_warned = False

    # Pool of reusable scope context objects (steady-state alloc-free).
    prof._ctx_pool = []

    # Ring buffers (history_frames rows).
    prof._frame_ms = np.zeros(H, dtype=np.float64)
    prof._scope_ms = np.zeros((H, S), dtype=np.float64)
    prof._scope_calls = np.zeros((H, S), dtype=np.int32)
    prof._counter_val = np.zeros((H, C), dtype=np.float64)
    prof._counter_seen_any = np.zeros(C, dtype=bool)

    prof._write_index = 0
    prof._frames_written = 0

    # Per-frame accumulators (reset each begin_frame).
    prof._cur_scope_ns = np.zeros(S, dtype=np.float64)
    prof._cur_scope_calls = np.zeros(S, dtype=np.int32)
    prof._cur_counter = np.zeros(C, dtype=np.float64)
    prof._cur_counter_seen = np.zeros(C, dtype=bool)

    # Active-scope bookkeeping (by scope id).
    prof._active_depth = np.zeros(S, dtype=np.int32)
    prof._active_start_ns = np.zeros(S, dtype=np.int64)
    prof._stack = []

    # Recent-frame ring for the rolling-median hitch threshold.
    prof._recent_ms = np.zeros(W, dtype=np.float64)
    prof._recent_index = 0
    prof._recent_count = 0

    # Hitch records (most-recent-first list, capped at recent_hitches).
    prof._hitches = []
    prof._hitch_count = 0
    prof._last_threshold_ms = 0.0

    # Frame timing state.
    prof._frame_start_ns = 0
    prof._pending = False  # a frame is open / awaiting commit
    prof._cpu_ms = 0.0  # loop-body time recorded by end_frame
    prof._start_wall = prof._time()  # for hitches-per-second
    prof._frame_cpu_ms_seen = False


class _ScopeCtx:
    """
    Pooled context manager for one active timing scope.

    Reused via the profiler's free-list so steady-state scope entry allocates
    nothing.  Holds only the owning profiler + the integer scope id.
    """

    __slots__ = ("_prof", "_sid")

    def __init__(self, prof: Profiler) -> None:
        self._prof = prof
        self._sid = -1

    def __enter__(self) -> _ScopeCtx:
        self._prof._start_sid(self._sid)
        return self

    def __exit__(self, *exc: object) -> None:
        self._prof._stop_sid(self._sid)
        self._prof._release_ctx(self)
